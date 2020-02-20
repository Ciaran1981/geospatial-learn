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
from skimage import exposure
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

from scipy.ndimage import gaussian_filter

from skimage.transform import rescale
from skimage.feature import canny
from skimage.measure import LineModelND, ransac
from skimage.draw import line
#TODO
#def rgbind(inRas):
#    
#    
#    img = imread(inRas)
#    
#    
#    r = img[:,:,0] / (np.sum(img, axis=2))
#    g = img[:,:,1] / (np.sum(img, axis=2))
#    b = img[:,:,2] / (np.sum(img, axis=2))                    
#
#    exG = (g * 2) - (r - b)        
#           
#    exR = (r * 1.4) - g


    

def iter_ransac(image, sigma=3, no_iter=10, order = 'col', mxt=2500):
    
    # The plan here is to make the outliers inliers each time or summit
    
    outArray = np.zeros_like(image)
    
    #th = filters.threshold_otsu(inArray)
    
    bw = canny(image, sigma=sigma)
    
    
    inDex = np.where(bw > 0)
    
    if order =='col':       
    
        inData = np.column_stack([inDex[0], inDex[1]])
        
    if order == 'row':
        inData = np.column_stack([inDex[1], inDex[0]])
    
    for i in tqdm(range(0, no_iter)):
        
#        if orient == 'v':
        
        if order == 'col':
        
            #inData = np.column_stack([inDex[0], inDex[1]])
            
            
            model = LineModelND()
            model.estimate(inData)
        
            model_robust, inliers = ransac(inData, LineModelND, min_samples=2,
                                           residual_threshold=1, max_trials=mxt)
        
        
            outliers = np.invert(inliers)
        
            
            line_x = inData[:, 0]
            line_y = model.predict_y(line_x)
            line_y_robust = model_robust.predict_y(line_x)
        
            outArray[line_x, np.int64(np.round(line_y_robust))]=1
        
        if order == 'row':
        
#            inData = np.column_stack([inDex[1], inDex[0]])
        
        
            model = LineModelND()
            model.estimate(inData)
        
            model_robust, inliers = ransac(inData, LineModelND, min_samples=2,
                                       residual_threshold=1, max_trials=mxt)
        
        
            outliers = np.invert(inliers)
        
            
            line_x = inData[:,0]
            line_y = model.predict_y(line_x)
    
            line_y_robust = model_robust.predict_y(line_x)
            
            outArray[np.int64(np.round(line_y_robust)), line_x]=1

#    
        
        
    
        inData = inData[:,0:2][outliers==True]
        del model, model_robust, inliers, outliers
        
    
    return outArray

        

def temp_match(vector_path, raster_path, band, nodata_value=0, ind=None):
    
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
    ind : int
        The feature ID to use - if used this will use one feature and rotate it 90 for the second
        
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
    if ind != None:
        nArray = arList[ind]
        rotAr = np.rot90(nArray)
        arList = [nArray, rotAr]
        
    for a in tqdm(arList):
       result = match_template(gray, a, pad_input=True)
       np.where(gray==0, 0, result)
       outList.append(result)
       

    
    return outList

def test_gabor(im, size=9,  freq=0.1, angle=None, funct='cos', plot=True, 
                  smooth=True, interp='none'):
    """ 
    Process image with gabor filter bank of specified orientation or derived from
    image positive values bounding box - implemented from numpy with more intuitive 
    params 
    
    This is the numpy based one 
    
    Parameters
    ----------
    
    inRas: string
                  input raster

    size: int
           size of in gabor kernel in pixels (ksize)
        
    freq: float
           
           
        
    angles: int
           number of angles  in gabor kernel (theta)


    """  
    
    if funct == 'cos':
        func = np.cos
    if func == 'sin':
        func = np.sin
        
    


    def genGabor(sz, omega, theta, func=func, K=np.pi):
        
        sz = (sz,sz)
        radius = (int(sz[0]/2.0), int(sz[1]/2.0))
        [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))
    
        x1 = x * np.cos(theta) + y * np.sin(theta)
        y1 = -x * np.sin(theta) + y * np.cos(theta)
        
        gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))

        sinusoid = func(omega * x1) * np.exp(K**2 / 2)

        gabor = gauss * sinusoid
        return gabor
    

    
    def deginrad(degree):
        radiant = 2*np.pi/360 * degree
        return radiant
    
    if hasattr(im, 'shape'):
        img = im
    else:
        img = rgb2gray(io.imread(im))
    
    if smooth == True:
        
        img = gaussian_filter(img, 1)
    
    #TODO add a polygon argument to make it easier....
    if angle == None:
        # here we use the orientation to get the line of crops assuming the user has
        # cropped it well
        bw = img > 0
        props = regionprops(bw*1)
        orient = props[0]['Orientation']
        angle = 90 - np.degrees(orient)

    g = genGabor(size,  freq, np.radians(angle))
           
   
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g)
    
    theta2 = np.radians(angle+90)
    
    g2 = genGabor(size,  freq, theta2)

    filtered_img2 = cv2.filter2D(img, cv2.CV_8UC3, g2)
    
    if plot == True:
        fig=plt.figure()
        fig.add_subplot(1, 4, 1)
        plt.imshow(img)
        fig.add_subplot(1, 4, 2)
        plt.imshow(filtered_img)
        fig.add_subplot(1, 4, 3)
        plt.imshow(filtered_img2)
        fig.add_subplot(1, 4, 4)
        plt.imshow(g, interpolation=interp)
    


    return  filtered_img, filtered_img2   

def test_gabor_cv2(im, size=9,  stdv=1, angle=None, wave_length=3, eccen=1,
               phase_off=0, plot=True, smooth=True, interp='none'):
    """ 
    Process image with gabor filter bank of specified orientation or derived from
    image positive values bounding box
    
    This is the open cv based one
    
    Parameters
    ----------
    
    inRas: string
                  input raster

    size: int
           size of in gabor kernel in pixels (ksize)
        
    stdv: int
           stdv / of of gabor kernel (sigma/stdv)
           
        
    angles: int
           number of angles  in gabor kernel (theta)

    wave_length: int
           width of stripe in gabor kernel (lambda/wavelength)
           optional best to leave none and hence same as size
        
    phase_off: int
           the phase offset of the kernel      
           
    eccen: int
          the elipticity of the kernel when = 1 the gaussian envelope is circular (gamma)

    """  
    
        # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sunusoidal factor wave_length
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values that each pixel in the gabor kernel can hold
    

    

    def deginrad(degree):
        radiant = 2*np.pi/360 * degree
        return radiant
    if hasattr(im, 'shape'):
        img = im
    else:
        
        img = rgb2gray(io.imread(im))
    
    if smooth == True:
        
        img = gaussian_filter(img, 1)
    
    #TODO add a polygon argument to make it easier....
    if angle == None:
        # here we use the orientation to get the line of crops assuming the user has
        # cropped it well
        bw = img > 0
        props = regionprops(bw*1)
        orient = props[0]['Orientation']
        angle = 90 - np.degrees(orient)
    
    if wave_length==None:
        wave_length = 3
    
#    if width2 == None:
#        width2 = width
#                  
    theta = deginrad(angle)   # unit circle: left: -90 deg, right: 90 deg, straight: 0 deg
    g_kernel = cv2.getGaborKernel((size, size), stdv, theta, wave_length, eccen, 
                                  phase_off, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    
    theta2 = deginrad(angle+90)
    g_kernel2 = cv2.getGaborKernel((size, size), stdv, theta2, wave_length, eccen, 
                                  phase_off, ktype=cv2.CV_32F)
    filtered_img2 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel2)
    
    if plot == True:
        fig=plt.figure()
        fig.add_subplot(1, 4, 1)
        plt.imshow(img)
        fig.add_subplot(1, 4, 2)
        plt.imshow(filtered_img)
        fig.add_subplot(1, 4, 3)
        plt.imshow(filtered_img2)
        fig.add_subplot(1, 4, 4)
        plt.imshow(g_kernel, interpolation=interp)
    
    #h, w = g_kernel.shape[:2]
    #g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('gabor kernel (resized)', g_kernel)
    
    
    
#    filtered_img[img==0]=0
#    filtered_img2[img==0]=0

    return  filtered_img, filtered_img2   


def accum_gabor(inRas, outRas=None, size=(9,9), stdv=1, no_angles=16, wave_length=3, eccen=1,
               phase_off=0, pltgrid=(4,4), blockproc=False):
    
    """ 
    Process with custom gabor filters and output an raster containing each 
    kernel output as a band
    
    
    Parameters
    ----------
    
    inRas: string
                  input raster
        
    outRas: string
                  output raster

    size: tuple
           size of in gabor kernel in pixels (ksize)
        
    stdv: int
           stdv / of of gabor kernel (sigma/stdv)
    
    no_angles: int
           number of angles  in gabor kernel (theta)

    wave_length: int
           width of stripe in gabor kernel (lambda/wavelength)  
        
    phase_off: int
           the phase offset of the kernel      
           
    eccen: int
          the elipticity of the kernel when = 1 the gaussian envelope is circular
          
    blocproc: bool
          whether to process in chunks - necessary for very large images!
    """  
    
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sunusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values that each pixel in the gabor kernel can hold


    """
    Harmonic function consists of an imaginary sine function and a real cosine function. 
    Spatial frequency is inversely proportional to the wavelength of the harmonic 
    and to the standard deviation of a Gaussian kernel. 
    The bandwidth is also inversely proportional to the standard deviation.
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
             kern = cv2.getGaborKernel(size, stdv, theta, wave_length, eccen, 
                                  phase_off, ktype=cv2.CV_32F)
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
    
    
    def plot_it(fmgList, gFilters, pltgrid):
        
        """
        plt a grid of images for gab filters and outputs

        """
       
        
        fig = plt.figure(figsize=(10., 10.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=pltgrid,  # creates 2x2 grid of axes
                         axes_pad=0.3, share_all=True,  # pad between axes in inch.
                         )

        for ax, im, d in zip(grid, fmgList, degrees):
            # Iterating over the grid returns the Axes.            
            ax.imshow(im)
            ax.set_title(str(d)+' degrees')
            
            ax.set_axis_off()
        
        fig1 = plt.figure(figsize=(10., 10.))
        grid1 = ImageGrid(fig1, 111,  # similar to subplot(111)
                         nrows_ncols=pltgrid,  # creates 2x2 grid of axes
                         axes_pad=0.3,  # pad between axes in inch.
                         )

        for ax1, im1, d1 in zip(grid1, gFilters, degrees):
            # Iterating over the grid returns the Axes.            
            ax1.imshow(im1)
            ax1.set_title(str(d1)+' degrees')
            ax1.set_axis_off()
        
        plt.show()
        

    gfilters = build_filters()  


                  
    inDataset = gdal.Open(inRas)
    
    if outRas != None:
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
        
    if blockproc == True and outRas != None:            
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
                    
# TODO                    # [:256, :256] this will pad it if block is  bigger.....but still getting edge effect - why?
                    
                    [outDataset.GetRasterBand(k+1).WriteArray(f
                    , j,  i) for k, f in enumerate(fmgList)] 
    
                        
        outDataset.FlushCache()
        outDataset = None
                
   
    
    else:

        img  = io.imread(inRas)
        
        if len(img.shape) >1:
             img = rgb2gray(img)
            
            
            
        
        gabber, fmgList = process(img, gfilters)
        
        plot_it(fmgList, gfilters, pltgrid)
        
        if outRas != None:
        
            [outDataset.GetRasterBand(k+1).WriteArray(f) for k, f in enumerate(fmgList)]
            
            array2raster(gabber, 1, inRas, outRas[:-4]+'_comp.tif', gdal.GDT_Int32)
    return fmgList


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
    pixeleccen = rgt[5]
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
#    pixel_eccen = gt[5]
#    x1 = int((bbox[0] - originX) / pixel_width)
#    x2 = int((bbox[1] - originX) / pixel_width) + 1
#
#    y1 = int((bbox[3] - originY) / pixel_eccen)
#    y2 = int((bbox[2] - originY) / pixel_eccen) + 1
#
#    xsize = x2 - x1
#    ysize = y2 - y1
#    return (x1, y1, xsize, ysize)
    return (xoff, yoff, xcount, ycount)  


def image_thresh(image):

#    image = rgb2gray(io.imread(im))
    
    if image.shape[0] > 4000:
        image = rescale(image, 0.5, preserve_range=True, anti_aliasing=True)
        image = np.uint8(image)
    
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



def rotate_im(image, angle):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray
        Rotated Image
    
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def get_corners(bboxes):
    
    """Get corners of bounding boxes
    
    Parameters
    ----------
    
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
        
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners


def rotate_box(corners,angle,  cx, cy, h, w):
    
    """Rotate the bounding box.
    
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    
    angle : float
        angle by which the image is to be rotated
        
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
        
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
        
    h : int 
        height of the image
        
    w : int 
        width of the image
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated

def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final

def spinim(self, img, bboxes):

    angle = random.uniform(*self.angle)

    w,h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2

    img = rotate_im(img, angle)

    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:,4:]))


    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)


    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w,h))

    new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 

    bboxes  = new_bbox

    bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

    return img, bboxes