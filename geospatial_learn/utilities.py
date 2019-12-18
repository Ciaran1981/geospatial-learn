# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 22:35:39 2016
@author: Ciaran Robb
Aberytswyth Uni
Wales

If you use code to publish work cite/acknowledge me and authors of libs etc as 
appropriate 
"""

import numpy as np
from scipy.spatial import ConvexHull
#from scipy.ndimage.interpolation import rotate
from scipy import ndimage as ndi
import cv2
import matplotlib.pyplot as plt
from geospatial_learn.geodata import array2raster
import gdal, ogr
from tqdm import tqdm
from skimage.feature import match_template
from skimage.color import rgb2gray
from skimage import io
import matplotlib
matplotlib.use('Qt5Agg')
import napari
import dask.array as da
from skimage.measure import regionprops
from skimage import color
#import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from geospatial_learn.geodata import _copy_dataset_config





def temp_match(vector_path, raster_path, band, nodata_value=0):
    
    """ 
    Based on polygons return template matched images
    
    
    Parameters
    ----------
    
    vector_path : string
                  input shapefile
        
    raster_path : string
                  input raster

    band : int
           an integer val eg - 2
        
    nodata_value : numerical
                   If used the no data val of the raster
    Returns
    -------
    list of template match arrays same size as input
        
    """    

    
    rds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    #assert(rds)
    rb = rds.GetRasterBand(band)
    
    
    
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(vector_path, 1)  # TODO maybe open update if we want to write stats
   #assert(vds)
    vlyr = vds.GetLayer(0)

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    rejects = list()
    
    arList = []
    for label in features:

        if feat is None:
            continue
        geom = feat.geometry()

        src_offset = _bbox_to_pixel_offsets(rgt, geom)
        src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                               src_offset[3])
        if src_array is None:
            src_array = rb.ReadAsArray(src_offset[0]-1, src_offset[1], src_offset[2],
                               src_offset[3])
            if src_array is None:
                rejects.append(feat.GetFID())
                continue

        # calculate new geotransform of the feature subset
        new_gt = (
        (rgt[0] + (src_offset[0] * rgt[1])),
        rgt[1],
        0.0,
        (rgt[3] + (src_offset[1] * rgt[5])),
        0.0,
        rgt[5])

            
        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize it

        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
     
        rvds.SetGeoTransform(new_gt)
        rvds.SetProjection(rds.GetProjectionRef())
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        
        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
            

        #rejects.append(feat.GetField('DN'))
        masked = np.ma.MaskedArray(
            src_array,
            mask=np.logical_or(
                src_array == nodata_value,
                np.logical_not(rv_array)
            )
        )
        
        arList.append(masked)
        feat = vlyr.GetNextFeature()
        
        
    gray = rgb2gray(io.imread(raster_path))
    
    outList = []
    for a in tqdm(arList):
       result = match_template(gray, a, pad_input=True)
       outList.append(result)

    
    return outList


def test_gabor(im, size=100, stdv=4, angle=0, stripe_width=11, height=0,
               no_stripes=0, plot=True):
    
    def deginrad(degree):
        radiant = 2*np.pi/360 * degree
        return radiant
    
    img = rgb2gray(io.imread(im))

    theta = deginrad(angle)   # unit circle: left: -90 deg, right: 90 deg, straight: 0 deg
    g_kernel = cv2.getGaborKernel((size, size), stdv, theta, stripe_width, height, 
                                  no_stripes, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    
    theta2 = deginrad(angle+90)
    g_kernel2 = cv2.getGaborKernel((size, size), stdv, theta2, stripe_width, height, 
                                  no_stripes, ktype=cv2.CV_32F)
    filtered_img2 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel2)
    
    if plot == True:
        fig=plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(img)
        fig.add_subplot(1, 3, 2)
        plt.imshow(filtered_img)
        fig.add_subplot(1, 3, 3)
        plt.imshow(filtered_img2)
    
    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gabor kernel (resized)', g_kernel)
    
    
#    filtered_img[img==0]=0
#    filtered_img2[img==0]=0

    return  filtered_img, filtered_img2   


def accum_gabor(inRas, outRas, size=9, stdv=4, no_angles=16, stripe_width=11, height=0,
               no_stripes=0, blockproc=False):
    
    """ 
    Process with a custom gabor filter and output an raster containing each 
    kernel output as a band
    
    
    Parameters
    ----------
    
    inRas : string
                  input raster
        
    outRas : string
                  output raster

    size : int
           size of in gabor kerne
        
    stdv : int
           stdv of of gabor kernel
    
    no_angles : int
           number of angles  in gabor kerne

    stripe_width : int
           width of stripe in gabor kernel   
        
    no_stripes : int
           num of stripes in gabor kernel      
           
    height : int
          height/compactness of gabor kernel  
          
    blocproc : bool
          whether to process in chunks - necessary for large images!
    """  
    
    

    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats
    
     
    def build_filters():
         filters = []
         
         for theta in np.arange(0, np.pi, np.pi / no_angles):
             kern = cv2.getGaborKernel((size, size), stdv, theta, stripe_width, height, 
                                  no_stripes, ktype=cv2.CV_32F)
             kern /= 1.5*kern.sum()
             filters.append(kern)
         return filters
    
    
    thetaz = np.arange(0, np.pi, np.pi / no_angles)
    degrees = np.rad2deg(thetaz)


    def process(img, filters):

        accum = np.zeros_like(img)
        fmgList = []

        for i, kern in enumerate(filters):
             fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
             fmgList.append(fimg)   
             np.maximum(accum, fimg, accum)
             
        return accum, fmgList
    
    
    def plot_it(fmgList):
        
        """
        plt a grid of images

        """
       
        
        fig = plt.figure(figsize=(10., 10.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                         axes_pad=0.3,  # pad between axes in inch.
                         )

        for ax, im, d in zip(grid, fmgList, degrees):
            # Iterating over the grid returns the Axes.            
            ax.imshow(im)
            ax.set_title(str(d)+' degrees')
            ax.set_axis_off()
        
        plt.show()
        

    gfilters = build_filters()  


                  
    inDataset = gdal.Open(inRas)
    
    outDataset = _copy_dataset_config(inDataset, outMap = outRas,
                                     dtype = gdal.GDT_Byte, bands = no_angles)
    band = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    
    bands = inDataset.RasterCount
    
    if bands > 3:
        bands = 3

    blocksizeX = 256
    blocksizeY = 256
        
    if blockproc == True:            
        for i in tqdm(range(0, rows, blocksizeY)):
                if i + blocksizeY < rows:
                    numRows = blocksizeY
                else:
                    numRows = rows -i
            
                for j in range(0, cols, blocksizeX):
                    if j + blocksizeX < cols:
                        numCols = blocksizeX
                    else:
                        numCols = cols - j
                    if bands == 1:
                        band1 = inDataset.GetRasterBand(band)
                        data = band1.ReadAsArray(j, i, numCols, numRows)                        
                    else:
                        data = np.zeros((blocksizeX,blocksizeX, bands))
                                                
                        for band in range(1,bands+1):
                            band1 = inDataset.GetRasterBand(band)
                            data[:,:,band-1] = band1.ReadAsArray(j, i, numCols, numRows)
                        data = color.rgb2gray(data)
                    
                    _, fmgList = process(data, gfilters)
                    
                    # [:256, :256] this will pad it..... 
                    
                    [outDataset.GetRasterBand(k+1).WriteArray(f
                    , j,  i) for k, f in enumerate(fmgList)] 
    
                        
        outDataset.FlushCache()
        outDataset = None
                
   
    
    else:
        img  = io.imread(inRas)
        
        gabber, fmgList = process(img, gfilters)
        
        plot_it(fmgList)
        
        [outDataset.GetRasterBand(k+1).WriteArray(f) for k, f in enumerate(fmgList)]
        
        array2raster(gabber, 1, inRas, outRas, gdal.GDT_Byte)


def min_bound_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    Parameters
    ----------
    points : list
        An nx2 iterable of points
    
    Returns
    -------
    list
        an nx2 list of coordinates
    """
    points = np.asarray(points, dtype = np.float64)
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = list()#np.zeros((4, 2))
    rval.append(((x1,y2))) #np.dot([x1, y2], r)
    rval.append(((x2,y2)))#np.dot([x2, y2], r)
    rval.append(((x2,y1)))#np.dot([x2, y1], r)
    rval.append(((x1,y1)))#np.dot([x1, y1], r)
        
    
    return rval

def _bbox_to_pixel_offsets(rgt, geom):
    
    """ 
    Internal function to get pixel geo-locations of bbox of a polygon
    
    Parameters
    ----------
    
    rgt : array
          List of points defining polygon (?)
          
    geom : shapely.geometry
           Structure defining geometry
    
    Returns
    -------
    int
       x offset
           
    int
       y offset
           
    xcount : int
             rows of bounding box
             
    ycount : int
             columns of bounding box
    """
    
    xOrigin = rgt[0]
    yOrigin = rgt[3]
    pixelWidth = rgt[1]
    pixelHeight = rgt[5]
    ring = geom.GetGeometryRef(0)
    numpoints = ring.GetPointCount()
    pointsX = []; pointsY = []
    
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
            
    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1
#    originX = gt[0]
#    originY = gt[3]
#    pixel_width = gt[1]
#    pixel_height = gt[5]
#    x1 = int((bbox[0] - originX) / pixel_width)
#    x2 = int((bbox[1] - originX) / pixel_width) + 1
#
#    y1 = int((bbox[3] - originY) / pixel_height)
#    y2 = int((bbox[2] - originY) / pixel_height) + 1
#
#    xsize = x2 - x1
#    ysize = y2 - y1
#    return (x1, y1, xsize, ysize)
    return (xoff, yoff, xcount, ycount)  



def image_thresh(image):

#    image = rgb2gray(io.imread(im))
    
    def threshold(image, t):
        arr = da.from_array(image, chunks=image.shape)
        return arr > t
    
    all_thresholds = da.stack([threshold(image, t) for t in np.arange(255)])
    
    viewer = napari.view_image(image, name='input image')
    viewer.add_image(all_thresholds,
        name='thresholded', colormap='magenta', blending='additive'
    )

def colorscale(seg, prop):
    
    props = regionprops(seg)
    
    labels = np.unique(seg)
    propIm = np.zeros_like(seg, dtype=np.float64) 
    for label in labels:
        if label==0:
            continue
        propval=props[label-1][prop] 
        propIm[seg==label]=propval
    
    return propIm