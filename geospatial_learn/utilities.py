# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 22:35:39 2016
@author: Ciaran Robb
The utilities module - things here don't have an exact theme or home yet so
may eventually move elsewhere


If you use code to publish work cite/acknowledge me and authors of libs etc as 
appropriate 
"""

import numpy as np
from scipy.spatial import ConvexHull
#from scipy.ndimage.interpolation import rotate
from skimage import exposure
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from geospatial_learn.raster import _copy_dataset_config, polygonize, array2raster
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
from scipy.ndimage import gaussian_filter
from phasepack.phasecong import phasecong
from skimage.transform import rescale
from skimage.feature import canny
from skimage.measure import LineModelND, ransac
from skimage.segmentation import relabel_sequential
#from skimage.draw import line
#from warnings import warn
#from scipy.interpolate import RectBivariateSpline
from skimage.util import img_as_float, invert
from skimage.morphology import dilation, remove_small_objects, remove_small_holes, medial_axis, skeletonize, binary_dilation, selem
import scipy.ndimage as nd
from skimage.feature import peak_local_max
from morphsnakes import morphological_geodesic_active_contour as gac
from morphsnakes import morphological_chan_vese as mcv
from morphsnakes import inverse_gaussian_gradient
#from multisnakes import MorphACWE, MorphGAC
#from multisnakes import multi_snakes as msn
import mahotas as mh
from plyfile import PlyData, PlyProperty#, PlyListProperty
from skimage.filters import sobel
from skimage.future import graph
#houghty chufty
from skimage.transform import hough_line, hough_line_peaks
from shapely.geometry import box, LineString
from skimage.draw import line

gdal.UseExceptions()
ogr.UseExceptions()


gdal.UseExceptions()
ogr.UseExceptions()

def _std_huff(inArray, outArray,  angl, valrange, interval, rgt):
    

    #Direct lift of scikit-image demo 
    
    tested_angles = np.linspace(angl - np.deg2rad(valrange), 
                                angl + np.deg2rad(valrange), num=interval)

    hh, htheta, hd = hough_line(inArray, theta=tested_angles)
    origin = np.array((0, inArray.shape[1]))

    height, width = inArray.shape
    
    # Shapely
    bbox = box(width, height, 0, 0)
    

    # Direct lift of scikit-image demo           
    # Here we use the skimage loop to draw a bw line into the image
    for _, angle, dist in tqdm(zip(*hough_line_peaks(hh, htheta, hd))):
    
        # here we obtain y extrema in our arbitrary coord system
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)     
        
        # shapely used to get the geometry         
        linestr = LineString([[origin[0], y0], [origin[1], y1]])
        
        in_coord= np.array(bbox.intersection(linestr).coords)
        
        coord = np.around(in_coord)
        
        # for readability sake
        x1 = int(coord[0][0])
        y1 = int(coord[0][1]) 
        x2 = int(coord[1][0])
        y2 = int(coord[1][1])
        
        if y1 == height:
            y1 = height-1
        if y2 == height:
            y2 = height-1
        if x1 == width:
            x1 = width-1
        if x2 == width:
            x2 = width-1
        
        #skimage
        cc, rr = line(x1, y1, x2, y2)
        
        outArray[rr, cc]=1
    

    return outArray

def houghseg(inRas, outShp, edge='canny', sigma=2, 
               thresh=None, ratio=2, n_orient=6, n_scale=5, hArray=True, vArray=True,
               valrange=1, interval=10, band=2,
               min_area=None):
    
        """
        Detect and write Hough lines to a line shapefile
        
        There two input arrays on the to keep line detection clean eg 2 orientations,
        such as vertical and horizontal
        
        Parameters
        ----------
    
        inRas : string
               path to an input raster from which the geo-reffing is obtained
    
        outShp: string
               path to the output shapefile
               
        edge: string
               edge method 'canny' or 'ph' 
               
        sigma: float
                scalar value for gaussian smoothing
                
        thresh: int/float
                 the high hysterisis threshold
        band: int
                the image band
        """
        # Standard GDAL I/O fair
        inDataset = gdal.Open(inRas, gdal.GA_ReadOnly)


        rgt = inDataset.GetGeoTransform()
        
        pixel_res = rgt[1]
        
        
        empty = np.zeros((inDataset.RasterYSize, inDataset.RasterXSize), dtype=np.bool)
        
            
        tempIm = inDataset.GetRasterBand(band).ReadAsArray()
        
        angleD, angleV, bw = imangle(tempIm)
        
        # mahotas
        perim = mh.bwperim(bw)
        
        hi_t = thresh
        low_t = np.round((thresh / ratio), decimals=1)

        if edge == 'phase':

            ph = do_phasecong(tempIm, low_t, hi_t, norient=n_orient, 
                               nscale=n_scale, sigma=sigma)
            
            ph[perim==1]=0
            
            if hArray is True:
                vArray = ph
            if hArray is True:
                hArray = ph
            del ph
           
        else: 

            inIm = tempIm.astype(np.float32)
            inIm[inIm==0]=np.nan 
        
            if hArray is True:

                hArray = canny(inIm, sigma=sigma, low_threshold=low_t,
                               high_threshold=hi_t)
            if vArray is True:

                vArray = canny(inIm, sigma=sigma, low_threshold=low_t,
                               high_threshold=hi_t)
            del inIm
                                  
        
            
        if hasattr(vArray, 'shape'):            
            empty =_std_huff(vArray, empty,  angleV, valrange, interval, rgt)
        if hasattr(hArray, 'shape'):
            empty =_std_huff(hArray, empty,  angleD, valrange, interval, rgt)          


        inv = np.invert(empty)
        inv[tempIm==0]=0
        if min_area != None:
            min_final = np.round(min_area/(pixel_res*pixel_res))
            if min_final <= 0:
                min_final=4

            remove_small_objects(inv, min_size=min_final, in_place=True)
        segRas=outShp[:-3]+"seg.tif"
        

        array2raster(inv, 1, inRas, segRas,  gdal.GDT_Int32)
        del tempIm, inv
        

        polygonize(segRas, outShp[:-4]+"_poly.shp", outField=None,  mask = True, band = 1)



def fixply(incloud, outcloud, field='scalar_label'): 
    
    
    
    # The labels should be contiguous ie -1,0,1,2,3 - counting from zero
    
    pf = PlyData.read(incloud)
    
    ar = np.array(pf.elements[0].data[field])

    # after cloud compare there are often spurious vales like -2564
    
    # do the nan to num in place 
    ar = np.nan_to_num(ar, nan=-1)
    

    ar[ar<-1]=-1
    
    ar = np.int32(ar)
    
#    ar[ar==1]=0
#
#    ar[ar==2]=1
#
#    ar[ar==3]=2
#
#    ar[ar==4]=3
    
    # All this modifies the original data
    new = pf['vertex']
    new.properties = ()
    new.data.dtype.names = ['x', 'y', 'z', 
                            'red', 'green', 'blue',
                            'nx', 'ny', 'nz',  'label']
    new.properties = (PlyProperty('x', 'double'),
                       PlyProperty('y', 'double'), 
                       PlyProperty('z', 'double'), 
                       PlyProperty('red', 'uchar'), 
                       PlyProperty('green', 'uchar'), 
                       PlyProperty('blue', 'uchar'), 
                       PlyProperty('nx', 'double'), 
                       PlyProperty('ny', 'double'), 
                       PlyProperty('nz', 'double'), 
                       PlyProperty('label', 'int'))
    
    pf.elements[0].data['label']=ar
    
    
    
    
    pf.write(outcloud)

def wipe_ply_field(incloud, outcloud, tfield='training' ,field='label'): 
    
    
    
    # The labels should be contiguous ie -1,0,1,2,3 - counting from zero
    
    pf = PlyData.read(incloud)
    
    ar = np.array(pf.elements[0].data[field])

    # after cloud compare there are often spurious vales like -2564
    
    # do the nan to num in place 
    #ar = np.nan_to_num(ar, nan=-1)
    
    

    #ar[ar<-1]=-1
    
    ar = np.int32(ar)
    ar[ar>=0]=-1
    
    
#    ar[ar==1]=0
#
#    ar[ar==2]=1
#
#    ar[ar==3]=2
#
#    ar[ar==4]=3
    
    # All this modifies the original data
    new = pf['vertex']
    new.properties = ()
    new.data.dtype.names = ['x', 'y', 'z', tfield, field,
                            'red', 'green', 'blue',
                            'nx', 'ny', 'nz']
    new.properties = (PlyProperty('x', 'double'),
                       PlyProperty('y', 'double'), 
                       PlyProperty('z', 'double'), 
                       PlyProperty(tfield, 'int'),
                       PlyProperty(field, 'int'),
                       PlyProperty('red', 'uchar'), 
                       PlyProperty('green', 'uchar'), 
                       PlyProperty('blue', 'uchar'), 
                       PlyProperty('nx', 'double'), 
                       PlyProperty('ny', 'double'), 
                       PlyProperty('nz', 'double'))
    
    pf.elements[0].data['label']=ar
    
    
    
    
    pf.write(outcloud)

#
def iou_score(inSeg, trueSeg):
    # The Intersection over Union (IoU) metric, also referred to as the Jaccard index
    intersection = np.logical_and(trueVals, predVals)
    union = np.logical_or(trueVals, predVals)
    iou_score = np.sum(intersection) / np.sum(union)



def _fix_overlapping_levelsets(levelsets):
    
    # many thanks to pmneila for this 
    # Find the areas where levelsets overlap
    mask = np.sum(levelsets, axis=0) > 1

    # Set overlapping regions to 0.
    for ls in levelsets:
        ls[mask] = 0

    return levelsets

def raster2array(inRas, bands=[1]):
    
    """
    Read a raster and return an array, either single or multiband

    
    Parameters
    ----------
    
    inRas: string
                  input  raster
        
    bands: list
                  a list of bands to return in the array
    
    """
    rds = gdal.Open(inRas)
   
   
    if len(bands) ==1:
        # then we needn't bother with all the crap below
        inArray = rds.GetRasterBand(bands[0]).ReadAsArray()
        
    else:
        #   The nump and gdal dtype (ints)
        #   {"uint8": 1,"int8": 1,"uint16": 2,"int16": 3,"uint32": 4,"int32": 5,
        #    "float32": 6, "float64": 7, "complex64": 10, "complex128": 11}
        
        # a numpy gdal conversion dict - this seems a bit long-winded
        dtypes = {"1": np.uint8, "2": np.uint16,
              "3": np.int16, "4": np.uint32,"5": np.int32,
              "6": np.float32,"7": np.float64,"10": np.complex64,
              "11": np.complex128}
        rdsDtype = rds.GetRasterBand(1).DataType
        inDt = dtypes[str(rdsDtype)]
        
        inArray = np.zeros((rds.RasterYSize, rds.RasterXSize, len(bands)), dtype=inDt) 
        for band in bands:  
            rA = rds.GetRasterBand(band).ReadAsArray()
            inArray[:, :, band-1]=rA
   
   
    return inArray


    
    
    
def ms_toposnakes(inSeg, inRas, outShp, iterations=100, algo='ACWE', band=2,
                  sigma=4, alpha=100, smooth=1, lambda1=1, lambda2=1, threshold='auto', 
                  balloon=-1):
    
    """
    Topology preserveing morphsnakes, implemented in python/numpy exclusively 
    by C.Robb
    
    This uses morphsnakes and explanations are from there.
    
    Parameters
    ----------
    
    inSeg: string
                  input segmentation raster
        
    raster_path: string
                  input raster whose pixel vals will be used

    band: int
           an integer val eg - 2

    algo: string
           either "GAC" (geodesic active contours) or "ACWE" (active contours without edges)
           
    sigma: the size of stdv defining the gaussian envelope if using canny edge
              a unitless value

    iterations: uint
        Number of iterations to run.
        
    smooth : uint, optional
    
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    
    lambda1: float, optional
    
        Weight parameter for the outer region. If `lambda1` is larger than
        `lambda2`, the outer region will contain a larger range of values than
        the inner region.
        
    lambda2: float, optional
    
        Weight parameter for the inner region. If `lambda2` is larger than
        `lambda1`, the inner region will contain a larger range of values than
        the outer region.
    
    threshold: float, optional
    
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
        
    balloon: float, optional
    
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
        
    """    


    rds1 = gdal.Open(inRas)
    img = rds1.GetRasterBand(band).ReadAsArray()
    
    rds2 = gdal.Open(inSeg)
    seg = rds2.GetRasterBand(1).ReadAsArray()
    
    # Don't convert 0 to nan or it won't work
    
    cnt = list(np.unique(seg))
    
    cnt.pop(0)
    
    iters = np.arange(iterations)
    
    orig = seg>0
#    
    # An implementation of the morphsnake turbopixel idea where the blobs
    # are prevented from merging by the skeleton of the background image which
    # is updated at every iteration - downside is that we always have a pixel gap
    # TODO rectify pixel gap issue 

    
    if algo=='GAC':
        
        gimg = inverse_gaussian_gradient(img, sigma=sigma, alpha=alpha)


#  using an approximation of the
#    homotopic skeleton to prevent merging of blobs       
        for i in tqdm(iters):          
            # get the skeleton of the background of the prev seg
            inv = invert(orig)
            sk = skeletonize(inv)
            bw = gac(gimg, iterations=1, init_level_set=orig, smoothing=smooth,
                     threshold=threshold)
            # approximation of homotopic skel in paper 
            # we still have endpoint issue at times but it is not bad...
            bw[sk==1]=0
            # why do this? I think seg=bw will result in a pointer....
            orig = np.zeros_like(bw, dtype=np.bool)
            orig[bw==1]=1
            del inv, sk
            
    else:

        for i in tqdm(iters):
            inv = invert(orig)
            sk = skeletonize(inv)            
            bw = mcv(img, iterations=1,init_level_set=orig, smoothing=smooth, lambda1=1,
                lambda2=1)
            bw[sk==1]=0
            # why do this? I think seg=bw will result in a pointer....
            orig = np.zeros_like(bw, dtype=np.bool)
            orig[bw==1]=1
            del inv, sk
   

    
    newseg, _ = nd.label(bw)
    

        

    
    array2raster(newseg, 1, inSeg, inSeg[:-4]+'tsnake.tif', gdal.GDT_Int32)
    
    
    
    polygonize(inSeg[:-4]+'tsnake.tif', outShp, outField=None,  mask = True, band = 1)    

def ms_toposeg(inRas, outShp, iterations=100, algo='ACWE', band=2, dist=30,
                se=3, usemin=False, imtype=None, useedge=True, burnedge=False,
                merge=False,close=True, sigma=4, hi_t=None, low_t=None, init=4,
                smooth=1, lambda1=1, lambda2=1, threshold='auto', 
                balloon=1):
    
    """
    Topology preserveing segmentation, implemented in python/nump inspired by 
    ms_topo and morphsnakes
    
    This uses morphsnakes level sets to make the segments and param explanations are mainly 
    from there.
    
    Parameters
    ----------
    
    inSeg: string
                  input segmentation raster
        
    raster_path: string
                  input raster whose pixel vals will be used

    band: int
           an integer val eg - 2

    algo: string
           either "GAC" (geodesic active contours) or "ACWE" (active contours without edges)
           
    sigma: the size of stdv defining the gaussian envelope if using canny edge
              a unitless value

    iterations: uint
        Number of iterations to run.
        
    smooth : uint, optional
    
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    
    lambda1: float, optional
    
        Weight parameter for the outer region. If `lambda1` is larger than
        `lambda2`, the outer region will contain a larger range of values than
        the inner region.
        
    lambda2: float, optional
    
        Weight parameter for the inner region. If `lambda2` is larger than
        `lambda1`, the inner region will contain a larger range of values than
        the outer region.
    
    threshold: float, optional
    
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
        
    balloon: float, optional
    
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
        
    """    

    
    
    
    rds1 = gdal.Open(inRas)
    img = rds1.GetRasterBand(band).ReadAsArray()
    
    img = np.float32(img)
    img[img==0]=np.nan
       
    maxIm = peak_local_max(img, min_distance=dist, indices=False)
    
    if useedge == True:
        #in case vals are ndvi or something
        imre = exposure.rescale_intensity(img, out_range='uint8')
        edge = canny(imre, sigma=sigma, low_threshold=low_t,
                               high_threshold=hi_t)
        edge = _skelprune(edge)
    if burnedge==True:
        img[edge==1]=0
        
    if usemin == True:
        minIm = peak_local_max(invert(img), min_distance=dist, indices=False)
        
    if algo=='ACWE':
        ste = selem.square(se)
        dilated = binary_dilation(maxIm, selem=ste)
        seg, _ = ndi.label(dilated)
        cnt = list(np.unique(seg))
        
        cnt.pop(0)
        #levelsets = [seg==s for s in cnt]
        
        iters = np.arange(iterations)
        
        orig = seg>0
#TODO - get fuse burner algo in this
    
    if algo=='GAC':
        
        ste = selem.square(se)
        gimg = inverse_gaussian_gradient(img)
        maxIm = peak_local_max(gimg, min_distance=dist, indices=False)
        dilated = binary_dilation(maxIm, selem=ste)
        seg, _ = ndi.label(dilated)
        cnt = list(np.unique(seg))
    
        cnt.pop(0)
        #levelsets = [seg==s for s in cnt]
        
        iters = np.arange(iterations)
    
        orig = seg>0


  
        for i in tqdm(iters):          
            # get the skeleton of the background of the prev seg
            inv = invert(orig)
            sk = skeletonize(inv)
            sk = _skelprune(sk)
            bw = gac(gimg, iterations=1, init_level_set=orig, smoothing=smooth,
                     threshold=threshold)
            # approximation of homotopic skel in paper 
            # we still have endpoint issue at times but it is not bad...
            bw[sk==1]=0
            if useedge == True and burnedge == False:
                bw[edge==1]=0
            # why do this? I think seg=bw will result in a pointer....
            orig = np.zeros_like(bw, dtype=np.bool)
            orig[bw==1]=1
            del inv, sk
            
        if usemin==True:
            
            minIm = peak_local_max(invert(gimg), min_distance=dist, indices=False)           
            dilated = binary_dilation(minIm, selem=ste)
            seg, _ = ndi.label(dilated)
            cnt = list(np.unique(seg))
        
            cnt.pop(0)        
            iters = np.arange(iterations)    
            orig = seg>0                
            e2 = mh.bwperim(bw)
            edge[e2==1]=1
    
            
            if init != None:
                initBw = orig
                orig = mcv(img, iterations=init,init_level_set=initBw, 
                           smoothing=smooth, lambda1=1,
                    lambda2=1)
                orig = orig>0
            
            for i in tqdm(iters):
                    inv = invert(orig)
                    sk = skeletonize(inv) 
                    sk = _skelprune(sk)
                    bw2 = mcv(img, iterations=1,init_level_set=orig, smoothing=smooth, lambda1=1,
                        lambda2=1)
                    bw2[sk==1]=0
                    if useedge == True and burnedge == False:
                        bw2[edge==1]=0
                    # why do this? I think seg=bw will result in a pointer....
                    orig = np.zeros_like(bw2, dtype=np.bool)
                    orig[bw2==1]=1
        
                    del inv, sk

            bw[bw2==1]=1        
            newseg, _ = nd.label(bw)  
            

            
    else:
        # let it run for a bit to avoid over seg
        if init != None:
            initBw = orig
            orig = mcv(img, iterations=init,init_level_set=initBw, 
                       smoothing=smooth, lambda1=1,
                lambda2=1)
            orig = orig>0

        for i in tqdm(iters):
            inv = invert(orig)
            
            sk = skeletonize(inv) 
            sk = _skelprune(sk)
            bw = mcv(img, iterations=1,init_level_set=orig, smoothing=smooth, lambda1=1,
                lambda2=1)
            bw[sk==1]=0
            if useedge == True and burnedge == False:
                bw[edge==1]=0
            # why do this? I think seg=bw will result in a pointer....
            orig = np.zeros_like(bw, dtype=np.bool)
            orig[bw==1]=1

            del inv, sk
            
     
            
            
        if usemin==True:
            
            
            
            dilated = binary_dilation(minIm, selem=ste)
            seg, _ = ndi.label(dilated)
            cnt = list(np.unique(seg))
        
            cnt.pop(0)        
            iters = np.arange(iterations)    
            orig = seg>0                
            e2 = mh.bwperim(bw)
            edge[e2==1]=1
    
            
            if init != None:
                initBw = orig
                orig = mcv(img, iterations=init,init_level_set=initBw, 
                           smoothing=smooth, lambda1=1,
                    lambda2=1)
                orig = orig>0
            
            for i in tqdm(iters):
                    inv = invert(orig)
                    sk = skeletonize(inv) 
                    sk = _skelprune(sk)
                    bw2 = mcv(img, iterations=1,init_level_set=orig, smoothing=smooth, lambda1=1,
                        lambda2=1)
                    bw2[sk==1]=0
                    if useedge == True and burnedge == False:
                        bw2[edge==1]=0
                    # why do this? I think seg=bw will result in a pointer....
                    orig = np.zeros_like(bw2, dtype=np.bool)
                    orig[bw2==1]=1
        
                    del inv, sk
#    remove_small_objects(bw, min_size=3, in_place=True)
            bw[bw2==1]=1        
            newseg, _ = nd.label(bw)        
                
        else:
            newseg, _ = nd.label(bw)
        
    if close==True:
        ste2 = selem.square(3)
        newseg = dilation(newseg, ste2)
        newseg+=1
        newseg, _, _ = relabel_sequential(newseg)
#        newseg, _ = nd.label(newseg)
    
#    for idx,l in enumerate(levelsets):
#        newseg[l>0]=cnt[idx]
            
    array2raster(newseg, 1, inRas, outShp[:-4]+'.tif', gdal.GDT_Int32)
    
    
    
    polygonize(outShp[:-4]+'.tif', outShp, outField=None,  mask = True, band = 1) 

def _weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def _merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass

def ragmerge(inSeg, inRas, outShp, band, thresh=0.02):
    
    img = raster2array(inRas, bands=[band])
    
    seg = raster2array(inSeg, bands=[1])
    
    
    edges = sobel(img)
    
    g = graph.rag_boundary(seg, edges)
    
    newseg = graph.merge_hierarchical(seg, g, thresh=thresh, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=_merge_boundary,
                                       weight_func=_weight_boundary)
    
    array2raster(newseg, 1, inSeg, outShp[:-4]+'.tif', gdal.GDT_Int32)
    
    
    
    polygonize(outShp[:-4]+'.tif', outShp, outField=None,  mask = True, band = 1) 
    
    

def combine_grid(inRas1, inRas2, outRas, outShp, min_area=None):
    
    
    rds1 = gdal.Open(inRas1, gdal.GA_ReadOnly)

    rb1 = rds1.GetRasterBand(1).ReadAsArray()
    
    rds2 = gdal.Open(inRas2, gdal.GA_ReadOnly)

    rb2 = rds2.GetRasterBand(1).ReadAsArray()
    
    rgt = rds1.GetGeoTransform()
        
    pixel_res = rgt[1]
    
    oot = rb1*rb2
    
    sg, _ = nd.label(oot)
    
    if min_area != None:
        min_final = np.round(min_area/(pixel_res*pixel_res))
        
        if min_final <= 0:
            min_final=4
        
        remove_small_objects(sg, min_size=min_final, in_place=True)
    
    
    array2raster(sg, 1, inRas1, outRas,  gdal.GDT_Int32)
    
    
    polygonize(outRas, outShp, outField=None,  mask = True, band = 1)
    

    
def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.
    
    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    
    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.
    
    """
    
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):
        
        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback

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

def _non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z
        
def do_phasecong(tempIm,  low_t=0, hi_t=0, norient=6, nscale=6, sigma=2):#, skel='medial'):
    """
    process phase congruency on an image 


    """
    ph = phasecong(tempIm, norient=norient, nscale=nscale, k=sigma)

    re = exposure.rescale_intensity(ph[0], out_range='uint8')
    
    nonmax = _non_max_suppression(re, ph[3])
    
    hyst = apply_hysteresis_threshold(nonmax, low_t, hi_t)
    
    hyst[tempIm==0]=0
    
#    if skel == 'medial':
#        skel = medial_axis(hyst)
#    else:
#        skel = skeletonize(hyst)
#    
    
    return hyst

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

def imangle(im):
    
    """
    Determine the orientation of non-zero vals in an image
    ---------- 
    
    im : np array
               input image
    Returns
    -------
    
    axes : tuple
              orientations of each side and binary array
    
    """
    
    # if the the binary box is pointing negatively along maj axis
    bw = im > 0
    props = regionprops(bw*1)
    orient = props[0]['Orientation']
        
        # we will need these.....
    perim = mh.bwperim(bw)
    #        bkgrnd = invert(bw)
        
        
    bw[perim==1]=0
    if orient < 0:
        orient += np.pi
    
    if orient < np.pi:
        axis1 = np.pi - orient
        axis2 = axis1 - np.deg2rad(90)
    else:
    # if the the binary box is pointing positively along maj axis
        axis1 = np.pi + orient
        axis2 = axis1 + np.deg2rad(90)
        
    return (axis1, axis2, bw)



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
    
    props = regionprops(np.int32(seg))
    propIm = np.zeros_like(seg, dtype=np.float64) 
    
    for p in props:
        propIm[np.where(seg==p.label)]=p[prop]
    
    return propIm





def otbMeanshift(inputImage, radius, rangeF, minSize, outShape):
    """ 
    OTB meanshift by calling the otb command line
    Written for convenience and due to otb python api being rather verbose 
    
    Notes:
    -----------        
    There is a maximum size for the .shp format otb doesn't seem to
    want to move beyond (2gb), so enormous rasters may need to be sub
    divided
        
    You will need to install OTB etc seperately
                
        
    Parameters
    -----------    
     
    inputImage : string
                 the input image 
        
    radius : int
             the kernel radius
        
    rangeF : int
             the kernel range
        
    minSize : int
              minimum segment size
        
    outShape : string
               the ouput shapefile

    
    """
    # Yes it is possible to do this with the otb python api, but it is way more
    # verbose, hence using the command line
    # the long winded version is greyed out as takes far too long to process
    print('segmenting image.... could be a little while!')
#    cmd1 = ('otbcli_MeanShiftSmoothing -in '+inputImage+ '
#            '-fout MeanShift_FilterOutput.tif -foutpos '
#            'MeanShift_SpatialOutput.tif -spatialr 16 -ranger 16 ' 
#            '-thres 0.1 -maxiter 100')
#    cmd2 = ('otbcli_LSMSSegmentation -in smooth.tif -inpos position.tif ' 
#            '-out segmentation.tif -ranger '+rangeF+' -spatialr '+radius+' 
#            ' -minsize '+minSize+'
#            ' -tilesizex 500 -tilesizey 500')
#    cmd3 = ('otbcli_LSMSSmallRegionsMerging -in smooth.tif '
#            '-inseg segmentation.tif -out merged.tif -minsize 20'
#            '-tilesizex 500 -tilesizey 500')
#    cmd4 = ('otbcli_LSMSVectorization -in avions.tif -inseg merged.tif '
#            '-out vector.shp -tilesizex 500 -tilesizey 500')
            
    cmd1 = ['otbcli_Segmentation', '-in', str(inputImage), '-filter meanshift',
            '-filter.meanshift.spatialr', str(radius),
            '-filter.meanshift.ranger', str(rangeF), 
            '-filter.meanshift.minsize', str(minSize), '-mode', 'vector',
            '-mode.vector.out', outShape]
    cmd1out = subprocess.check_output(cmd1)
    print(cmd1out)
#    print('filtering done')
#    os.system(cmd2)
#    print('raster seg done')
#    os.system(cmd3)
#    print('region merge done')
#    os.system(cmd4)
    print('vectorisation done - process complete - phew!')
#    output = subprocess.Popen([cmd], stdout=subprocess.PIPE).communicate()[0]
#    print(output)
#def ms_toposnakes2(inSeg, inRas, outShp, iterations=100, algo='ACWE', band=2,
#                  sigma=4, smooth=1, lambda1=1, lambda2=1, threshold='auto', 
#                  balloon=-1):
#    
#    """
#    Topology preserveing morphsnakes, implmented by Jirka Borovec version 
#    with C++/cython elements- credit to him!
#    
#    This is memory intensive so large images will likely fill RAM and produces
#    similar resuts to ms_toposnakes
#    
#    
#    This uses morphsnakes and explanations are from there.
#    
#    Parameters
#    ----------
#    
#    inSeg: string
#                  input segmentation raster
#        
#    raster_path: string
#                  input raster whose pixel vals will be used
#
#    band: int
#           an integer val eg - 2
#
#    algo: string
#           either "GAC" (geodesic active contours) or "ACWE" (active contours without edges)
#           
#    sigma: the size of stdv defining the gaussian envelope if using canny edge
#              a unitless value
#
#    iterations: uint
#        Number of iterations to run.
#        
#    smooth : uint, optional
#    
#        Number of times the smoothing operator is applied per iteration.
#        Reasonable values are around 1-4. Larger values lead to smoother
#        segmentations.
#    
#    lambda1: float, optional
#    
#        Weight parameter for the outer region. If `lambda1` is larger than
#        `lambda2`, the outer region will contain a larger range of values than
#        the inner region.
#        
#    lambda2: float, optional
#    
#        Weight parameter for the inner region. If `lambda2` is larger than
#        `lambda1`, the inner region will contain a larger range of values than
#        the outer region.
#    
#    threshold: float, optional
#    
#        Areas of the image with a value smaller than this threshold will be
#        considered borders. The evolution of the contour will stop in this
#        areas.
#        
#    balloon: float, optional
#    
#        Balloon force to guide the contour in non-informative areas of the
#        image, i.e., areas where the gradient of the image is too small to push
#        the contour towards a border. A negative value will shrink the contour,
#        while a positive value will expand the contour in these areas. Setting
#        this to zero will disable the balloon force.
#        
#    """    
#
#
#    rds1 = gdal.Open(inRas)
#    img = rds1.GetRasterBand(band).ReadAsArray()
#    
#    rds2 = gdal.Open(inSeg)
#    seg = rds2.GetRasterBand(1).ReadAsArray()
#    
#    
#    if algo=='GAC':
#
#        # class-based
#        mseg = msn.MultiMorphSnakes(img, seg, MorphGAC, 
#                               dict(smoothing=smooth, threshold=threshold,
#                                    balloon=balloon))
#        mseg.run(iterations)
#
#            
#    else:
#        mseg = msn.MultiMorphSnakes(img, seg, MorphACWE, 
#                               dict(smoothing=smooth, lambda1=lambda1,
#                                    lambda2=lambda2))
#        mseg.run(iterations)
#        
#        
#    outSeg = mseg.levelset
#    
#    array2raster(outSeg, 1, inSeg, inSeg[:-4]+'tsnake.tif', gdal.GDT_Int32)
#    
#    
#    
#    polygonize(inSeg[:-4]+'tsnake.tif', outShp, outField=None,  mask = True, band = 1)
