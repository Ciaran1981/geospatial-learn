# -*- coding: utf-8 -*-
"""
The shape module. 

Description
-----------

This module contains various functions for the writing of data in OGR vector 
formats. The functions are mainly concerned with writing geometric or pixel based attributes, with the view to them being classified in the learning module

"""
from skimage.measure import regionprops
from scipy.stats import entropy, skew, kurtosis
import scipy.ndimage as nd
from skimage import feature
#from sklearn import cluster
import shapefile
#from simpledbf import Dbf5
import os
import gdal
from json import dumps
import  ogr, osr
from tqdm import tqdm
import numpy as np
from scipy.stats.mstats import mode
from geospatial_learn.utilities import min_bound_rectangle
from shapely.wkt import loads
from shapely.geometry import Polygon, box, LineString, Point, LinearRing
from pandas import DataFrame
#from pysal.lib import io 
import pandas as pd
from skimage.segmentation import active_contour#, find_boundaries
#from shapely.affinity import affine_transform, rotate
import morphsnakes as ms
from geospatial_learn.raster import _copy_dataset_config,  array2raster, raster2array, polygonize
import warnings
from skimage.measure import LineModelND, ransac
from skimage.filters import gaussian
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola, apply_hysteresis_threshold
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
from skimage.transform import probabilistic_hough_line as phl
#from skimage.io import imread
from skimage.feature import canny
from skimage.morphology import remove_small_objects, remove_small_holes, medial_axis, skeletonize
from skimage.util import img_as_float, invert

import matplotlib
from shapely.affinity import rotate
#from geospatial_learn.geodata import rasterize
from math import ceil
from phasepack.phasecong import phasecong
import mahotas as mh
import cv2
#from centerline.geometry import Centerline

matplotlib.use('Qt5Agg')

gdal.UseExceptions()
ogr.UseExceptions()


def shp2gj(inShape, outJson):
    """
    Converts a geojson/json to a shapefile
    
    Parameters
    ----------
    
    inShape : string
              input shapefile
    

    outJson : string
              output geojson
    
    Notes
    -----
    
    Credit to person who posted this on the pyshp site
    """    
    
    fname = outJson
    
    
    # read the shapefile
    reader = shapefile.Reader(inShape)
    fields = reader.fields[1:]
    field_names = [field[0] for field in fields]
    buffer = []
    for sr in reader.shapeRecords():
        atr = dict(zip(field_names, sr.record))
        geom = sr.shape.__geo_interface__
        buffer.append(dict(type="Feature", 
                           geometry=geom, properties=atr)) 
       
       # write the GeoJSON file
       
    geojson = open(fname, "w")
    geojson.write(dumps({"type": "FeatureCollection", 
                         "features": buffer}, indent=2) + "\n")
    geojson.close()




def shape_props(inShape, prop, inRas=None,  label_field='ID'):
    """
    Calculate various geometric properties of a set of polygons
    Output will be relative to geographic units where relevant, but normalised where not (eg Eccentricity)
    
    Parameters 
    ----------
    
    inShape : string
              input shape file path

    
    inRas : string
            a raster to get the correct dimensions from (optional), required for
            scikit-image props
        
    
    prop : string
           Scikit image regionprops prop 
           (see http://scikit-image.org/docs/dev/api/skimage.measure.html)
        
    OGR is used to generate most of these as it is faster but the string
    keys are same as scikit-image see notes for which require raster
    
    Notes
    -----
    Only shape file needed (OGR / shapely / numpy based)
        
    'MajorAxisLength', 'MinorAxisLength', Area', 'Eccentricity', 'Solidity',
    'Extent': 'Extent', 'Perimeter': 'Perim'
    
    Raster required
        
    'Orientation' and the remainder of props calcualble with scikit-image. These
     process a bit slower than the above ones

    
    """


    #outData = list()
    print('Loading & prepping data')
    if inRas != None:    
        raster = gdal.Open(inRas, gdal.GA_ReadOnly)
        transform = raster.GetGeoTransform()
        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]
    
        # Reproject vector geometry to same projection as raster
        #sourceSR = lyr.GetSpatialRef()
        targetSR = osr.SpatialReference()
        targetSR.ImportFromWkt(raster.GetProjectionRef())
    shp = ogr.Open(inShape,1)
    
    lyr = shp.GetLayer()
    
    # here we create fields as this is a slow part of the process best outside 
    # of the main loops - this seems to be a pretty slow OGR function   
    #lyr.CreateField(ogr.FieldDefn(prop[0:5], ogr.OFTReal))
    # TODO Axis measurements are not quite right -
    propNames = {'MajorAxisLength': 'MjAxis', 'MinorAxisLength': 'MnAxis',
                 'Area': 'Area', 'Eccentricity':'Eccen', 'Solidity': 'Solid',
                 'Extent': 'Extent', 'Orientation': 'Orient', 
                 'Perimeter': 'Perim'}
    fldDef = ogr.FieldDefn(propNames[prop], ogr.OFTReal)
    lyr.CreateField(fldDef)
    fldName = propNames[prop]

    print('calculating stats')
    
    labels = np.arange(lyr.GetFeatureCount())
    for label in tqdm(labels):
        #print(label)
        # Get raster georeference info

        #coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
        feat = lyr.GetFeature(label)
        geom = feat.GetGeometryRef()
        iD = feat.GetField(label_field)
        # IMPORTANT length defines the perimeter of a polygon!!!
        wkt=geom.ExportToWkt()
        poly1 = loads(wkt)
        conv = poly1.convex_hull
        if prop == 'Area':
            stat = geom.Area()
            fldName = propNames[prop]
            feat.SetField(fldName, stat)
            lyr.SetFeature(feat)

        elif prop == 'MajorAxisLength':

            # this is a bit hacky at present but works!!
            #TODO: Make less hacky
            x,y=poly1.exterior.coords.xy
            xy = np.vstack((x,y))
            rec = min_bound_rectangle(xy.transpose())
            poly2 = Polygon(rec)
            minx, miny, maxx, maxy = poly2.bounds
            axis1 = maxx - minx
            axis2 = maxy - miny
            stats = np.array([axis1, axis2])
            feat.SetField(fldName, stats.max())
            lyr.SetFeature(feat)
        elif prop == 'MinorAxisLength':
            x,y = conv.exterior.coords.xy
            xy = np.vstack((x,y))
            rec = min_bound_rectangle(xy.transpose())
            poly2 = Polygon(rec)
            minx, miny, maxx, maxy = poly2.bounds
            axis1 = maxx - minx
            axis2 = maxy - miny
            stats = np.array([axis1, axis2])
            feat.SetField(fldName, stats.min())
            lyr.SetFeature(feat)
        elif prop == 'Eccentricity':
            x,y = conv.exterior.coords.xy
            xy = np.vstack((x,y))
            rec = min_bound_rectangle(xy.transpose())
            poly2 = Polygon(rec)
            minx, miny, maxx, maxy = poly2.bounds
            axis1 = maxx - minx
            axis2 = maxy - miny
            stats = np.array([axis1, axis2])
            ecc = stats.min() / stats.max()
            feat.SetField(fldName, ecc)
            lyr.SetFeature(feat)            
        elif prop == 'Solidity':
            #conv = poly1.convex_hull
            bbox = poly1.envelope
            stat = conv.area / bbox.area
            feat.SetField(fldName, stat)
            lyr.SetFeature(feat)
        elif prop == 'Extent':
            bbox = poly1.envelope
            stat = poly1.area / bbox.area
            feat.SetField(fldName, stat)
            lyr.SetFeature(feat)
        elif prop == 'Perimeter':
            bbox = poly1.envelope
            stat = poly1.length # important to note length means
            feat.SetField(fldName, stat)
            lyr.SetFeature(feat) 
            # TODO - this may not write to shape as a tuple
        elif prop == 'Centroid':
            cent=poly1.centroid
            stat = cent.coords[0]            
        else:
        #tqdm.write(str(iD))
        #geom.Transform(coordTrans)
            if inRas != None:
            # Get extent of feat - I assume this is where the slow down is
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
            
                else:
                    os.sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")
                
                xmin = min(pointsX)
                xmax = max(pointsX)
                ymin = min(pointsY)
                ymax = max(pointsY)
            
                # Specify offset and rows and columns to read
                xoff = int((xmin - xOrigin)/pixelWidth)
                yoff = int((yOrigin - ymax)/pixelWidth)
                xcount = int((xmax - xmin)/pixelWidth)+1
                ycount = int((ymax - ymin)/pixelWidth)+1
            
                # Create memory target raster
                target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Int32)
                target_ds.SetGeoTransform((
                    xmin, pixelWidth, 0,
                    ymax, 0, pixelHeight,
                ))
                
                # Create for target raster the same projection as for the value raster
                raster_srs = osr.SpatialReference()
                raster_srs.ImportFromWkt(raster.GetProjectionRef())
                target_ds.SetProjection(raster_srs.ExportToWkt())
                # Rasterize zone polygon to raster
                gdal.RasterizeLayer(target_ds, [1], lyr,
                                    options=["ATTRIBUTE=%s" % label_field ])
                
                # Read raster as arrays
                bandmask = target_ds.GetRasterBand(1)
                # (xoff, yoff, xcount, ycount) is required if reading from inRas
                datamask = bandmask.ReadAsArray(0, 0, xcount, ycount)
                if datamask is None:
                    continue
                dShape = datamask.shape
                if len(dShape) != 2 or dShape[0] < 2 or dShape[1] < 2:
                    continue
                datamask[datamask != iD]=0
                datamask[datamask>0]=iD
                #bwmask = np.zeros_like(dataraster)
                Props = regionprops(datamask)
                if len(Props) is 0:
                    continue
                stat = Props[0][prop]
                #print(label)
                fldName = propNames[prop]
                feat.SetField(fldName, stat)
                lyr.SetFeature(feat)

    lyr.SyncToDisk()
    shp.FlushCache()
    shp = None
        
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


def zonal_stats(vector_path, raster_path, band, bandname, stat = 'mean',
                write_stat=None, nodata_value=0):
    
    """ 
    Calculate zonal stats for an OGR polygon file
    
    Parameters
    ----------
    
    vector_path : string
                  input shapefile
        
    raster_path : string
                  input raster

    band : int
           an integer val eg - 2

    bandname : string
               eg - blue
        
    stat : string
           string of a stat to calculate, if omitted it will be 'mean'
           others: 'mode', 'min','mean','max', 'std',' sum', 'count','var',
           skew', 'kurt (osis)'
                     
    write_stat : bool (optional)
                If True, stat will be written to OGR file, if false, dataframe
                only returned (bool)
        
    nodata_value : numerical
                   If used the no data val of the raster
        
    """    
    # Inspired by Matt Perry's excellent script
    
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
    if write_stat != None:
        vlyr.CreateField(ogr.FieldDefn(bandname, ogr.OFTReal))

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    rejects = list()
    for label in tqdm(features):

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
        
        if stat == 'mode':
            feature_stats = mode(masked)[0]
        elif stat == 'min':
            feature_stats = float(masked.min())
        elif stat == 'mean':
            feature_stats = float(masked.mean())
        elif stat == 'max':
            feature_stats = float(masked.max())
        elif stat == 'median':
            feature_stats = float(np.median(masked[masked.nonzero()]))
        elif stat == 'std':
            feature_stats = float(masked.std())
        elif stat == 'sum':
            feature_stats = float(masked.sum())
#        elif stat is 'count':
#            feature_stats = int(masked.count())
        elif stat == 'var':
            feature_stats = float(masked.var())
        elif stat == 'skew':
            feature_stats = float(skew(masked[masked.nonzero()]))
        elif stat == 'kurt':
            feature_stats = float(kurtosis(masked[masked.nonzero()]))
        
        stats.append(feature_stats)
        if write_stat != None:
            feat.SetField(bandname, feature_stats)
            vlyr.SetFeature(feat)
        feat = vlyr.GetNextFeature()
    if write_stat != None:
        vlyr.SyncToDisk()
    #vds.FlushCache()


    vds = None
    rds = None
    frame = DataFrame(stats)
    
    if write_stat != None:
        return frame, rejects
    
def zonal_stats_all(vector_path, raster_path, bandnames, 
                    statList = ['mean', 'min', 'max', 'median', 'std',
                                'var', 'skew', 'kurt']):
    """ 
    Calculate zonal stats for an OGR polygon file
    
    Parameters
    ----------
    
    vector_path : string
                  input shapefile
        
    raster_path : string
                  input raster

    band : int
           an integer val eg - 2

    bandnames : list
               eg - ['b','g','r','nir']
        
    nodata_value : numerical
                   If used the no data val of the raster
        
    """    

# zonal stats
    for bnd,name in enumerate(bandnames):
    
        [zonal_stats(vector_path, raster_path, bnd+1, name+st, stat=st, write_stat = True) for st in statList]

def _set_rgb_ind(feat, rv_array, src_offset, rds, nodata_value):
    
    
    rgb = np.zeros((src_offset[3], src_offset[2], 3))
    
    for band in range(1, rds.RasterCount):
        
        rBnd = rds.GetRasterBand(band)
        
        rgb[:,:, band-1] = rBnd.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                                   src_offset[3])
        
        
        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
                   
    
    
    
    r = rgb[:,:,0] / (np.sum(rgb, axis=2))
    g = rgb[:,:,1] / (np.sum(rgb, axis=2))
    b = rgb[:,:,2] / (np.sum(rgb, axis=2)) 
    
    del rgb

    
    r = np.ma.MaskedArray(r, mask=np.logical_or(r == nodata_value,
                                                        np.logical_not(rv_array)))
    g = np.ma.MaskedArray(g, mask=np.logical_or(g == nodata_value,
                                                        np.logical_not(rv_array)))
    b = np.ma.MaskedArray(b, mask=np.logical_or(b == nodata_value,
                                                        np.logical_not(rv_array)))        
        
    
        
    # This all horrendously inefficient for now - must be addressed later
    # For some reason ogr won't accept the masked.mean() in the set feature function hence the float - must be python -> C++ type thing
    # otsu threshold works perfectly on green band - would be better stat representation
    exG = (g * 2) - (r - b)        
    feat.SetField('ExGmn', float(exG.mean()))            
    exR = (r * 1.4) - g
    feat.SetField('ExRmn',  float(exR.mean()))
    exGR = exG - exR
    feat.SetField('ExGRmn',  float(exGR.mean()))       
    cive = ((r * 0.441) - (g * 0.811)) + (b * 0.385) +18.78745
    feat.SetField('CIVEmn',  float(cive.mean()))
    # someting not right with this one!
    ndi = (g - r) / (g + r)
    feat.SetField('NDImn',  float(ndi.mean()))
    rgbvi = ((g**2 - b) * r) / ((g**2 + b) * r)
    feat.SetField('RGBVImn',  float(rgbvi.mean()))
    vari = ((g-r) / (g+r)- b)
    feat.SetField('VARImn',  float(vari.mean()))
    ari = 1 / (g * r)
    feat.SetField('ARImn',  float(ari.mean()))
    rgbi = r / g
    feat.SetField('RGBImn',  float(rgbi.mean()))
    gli = ((g-r) + (g-b)) / (2* g) + r + b
    feat.SetField('GLImn',  float(gli.mean())) 
    tgl = (g - 0.39) * (r - 0.61) * b
    feat.SetField('TGLmn',  float(tgl.mean()))
        

def zonal_rgb_idx(vector_path, raster_path, nodata_value=0):
    
    """ 
    Calculate RGB-based indicies per segment/AOI
    
    Parameters
    ----------
    
    vector_path : string
                  input shapefile
        
    raster_path : string
                  input raster
        
    nodata_value : numerical
                   If used the no data val of the raster
        
    """    
    #TODO ad other stat types - consider mask array for safety......
    
    rds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    #assert(rds)
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(vector_path, 1)  # TODO maybe open update if we want to write stats
   #assert(vds)
    vlyr = vds.GetLayer(0)
    #if write_stat != None:
    field_names = ['ExGmn', 'ExRmn', 'ExGRmn', 'CIVEmn', 'NDImn', 'RGBVImn', 'VARImn',
         'ARImn', 'RGBImn', 'GLImn', 'TGLmn']
    
    [vlyr.CreateField(ogr.FieldDefn(f, ogr.OFTReal)) for f in field_names]

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    rejects = list()
    
    for label in tqdm(features):

        if feat is None:
            continue
        geom = feat.geometry()

        src_offset = _bbox_to_pixel_offsets(rgt, geom)
    
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
        rv_array = rvds.ReadAsArray()>0
        
        _set_rgb_ind(feat, rv_array, src_offset, rds, nodata_value)
        
        vlyr.SetFeature(feat)
        feat = vlyr.GetNextFeature()

    vds.SyncToDisk()
    #vds.FlushCache()


    vds = None
    rds = None
#    
#    if write_stat != None:
#        return frame, rejects
    
def write_text_field(inShape, fieldName, attribute):
    
    """ Write a string to a ogr vector file
    
    Parameters
    ----------
    inShape : string
              input OGR vecotr file
        
    fieldName : string
                name of field being written
    
    attribute : string
                'text to enter in each entry of column'
        
    """
        
    vds = ogr.Open(inShape, 1)  # TODO maybe open update if we want to write stats
   #assert(vds)
    vlyr = vds.GetLayer(0)
    vlyr.CreateField(ogr.FieldDefn(fieldName, ogr.OFTString))
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    
    for label in tqdm(features):
        feat.SetField(fieldName, attribute)
        vlyr.SetFeature(feat)
        feat = vlyr.GetNextFeature()

    vlyr.SyncToDisk()
    vds = None

    

def texture_stats(vector_path, raster_path, band, gprop='contrast',
                  offset=2,angle=0, write_stat=None, nodata_value=0, mean=False):
    
    """ 
    Calculate and optionally write texture stats for an OGR compatible polygon
    based on underlying raster values
    
    
    Parameters
    ----------
    vector_path : string
                  input shapefile 
        
    raster_path : string 
                  input raster path
        
    gprop : string
            a skimage gclm property 
            entropy, contrast, dissimilarity, homogeneity, ASM, energy,
            correlation
        
    offset : int
             distance in pixels to measure - minimum of 2!!!
        
    angle : int
            angle in degrees from pixel (int) 
            
            135  90    45
            \    |    /
                 c    -  0         
     
    mean : bool
           take the mean of all offsets
     
    Important to note that the results will be unreliable for glcm 
    texture features if seg is true as non-masked values will be zero or
    some weird no data and will affect results
    
    Notes
    -----
    Important
    
    The texture of the bounding box is at present the "relible" measure
    
    Using the segment only results in potentially spurious results due to the 
    scikit-image algorithm measuring texture over zero/nodata to number pixels
    (e.g 0>54). The segment part will be developed in due course to overcome 
    this issue
    
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
    if write_stat != None:
        gname = gprop[:10]+str(band)
        vlyr.CreateField(ogr.FieldDefn(gname, ogr.OFTReal))


    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    rejects = list()
    for label in tqdm(features):
#        field = feat.GetField('DN')
#        print(field)

#        if not global_src_extent:
            # use local source extent
            # fastest option when you have fast disks and well indexed raster (ie tiled Geotiff)
            # advantage: each feature uses the smallest raster chunk
            # disadvantage: lots of reads on the source raster

        if feat is None:
            feat = vlyr.GetFeature(label)

        geom = feat.geometry()
        
        src_offset = _bbox_to_pixel_offsets(rgt, geom)
        
        src_offset = list(src_offset)
        
        
        src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                               src_offset[3])
        if src_array is None:
            src_array = rb.ReadAsArray(src_offset[0]-1, src_offset[1], src_offset[2],
                               src_offset[3])
            if src_array is None:
                rejects.append(feat.GetFID())
                continue
            if src_array.size == 1:
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
        #with warnings.catch_warnings():

        warnings.simplefilter("ignore")
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Int32)
     
        rvds.SetGeoTransform(new_gt)
        rvds.SetProjection(rds.GetProjectionRef())
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()




        zone = np.ma.MaskedArray(src_array,
                                 mask=np.logical_or(src_array == nodata_value, 
                                                    np.logical_not(rv_array)))

        
        if gprop == 'entropy':
            _, counts = np.unique(zone, return_counts=True)
            props = entropy(counts, base=2)
        elif mean is True and gprop != 'entropy':
            angles = np.radians([135,90,45,0])
            
            
            g = feature.greycomatrix(zone, [offset],
                                     angles, symmetric=True)
            props = feature.greycoprops(g, prop=gprop)
            props = props.mean()
        elif mean is False and gprop != 'entropy': 
            g = feature.greycomatrix(zone, [offset],
                                     [np.radians(angle)], 256, symmetric=True)
            props = feature.greycoprops(g, prop=gprop)
       
            
        stats.append(float(props))
    
        if write_stat != None:
            
            feat.SetField(gname, float(props))
            vlyr.SetFeature(feat)
        feat = vlyr.GetNextFeature()


    if write_stat != None:
        vlyr.SyncToDisk()
    #vds.FlushCache()


    vds = None
    rds = None
    frame = DataFrame(stats)
    return frame, rejects


def snake(inShp, inRas, outShp, band=1, buf=1, nodata_value=0,
          boundary='fixed', alpha=0.1, beta=30.0, w_line=0, w_edge=0, gamma=0.01,
          max_iterations=2500, smooth=True, eq=False, rgb=False):
    
    """ 
    Deform a line using active contours based on the values of an underlying
    
    raster - based on skimage at present so 
    
    not quick!
    
    Notes
    -----
    
    Param explanations for snake/active contour from scikit-image api
    
    Parameters
    ----------
    
    
    vector_path: string
                  input shapefile
        
    raster_path: string
                  input raster

    band: int
           an integer val eg - 2

    buf: int
            the buffer area to include for the snake deformation
            
    alpha: float
            Snake length shape parameter. Higher values makes snake contract faster.
            
    beta: float
        Snake smoothness shape parameter. Higher values makes snake smoother.
    
    w_line: float
    
           Controls attraction to brightness. Use negative values to attract toward dark regions.
           
    w_edge: float
            Controls attraction to edges. Use negative values to repel snake from edges.
    
    gamma: float
    
            Explicit time stepping parameter.
    
    max_iterations: int
            
            No of iterations to evolve snake        
    
    boundary: string
            Scikit-image text:
            Boundary conditions for the contour. Can be one of ‘periodic’, 
            ‘free’, ‘fixed’, ‘free-fixed’, or ‘fixed-free’. 
            ‘periodic’ attaches the two ends of the snake, ‘fixed’ 
            holds the end-points in place, 
            and ‘free’ allows free movement of the ends. 
            ‘fixed’ and ‘free’ can be combined by parsing ‘fixed-free’, 
            ‘free-fixed’. Parsing ‘fixed-fixed’ or ‘free-free’ 
            yields same behaviour as ‘fixed’ and ‘free’, respectively.
            
    nodata_value: numerical
                   If used the no data val of the raster
    rgb: bool
        read in bands 1-3 assuming them to be RGB
        
    """    
    
    # Partly inspired by the Heikpe paper...
    # TODO actually implement the Heipke paper properly
    
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    #assert(rds)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()
    
    cols = rds.RasterXSize
    rows = rds.RasterYSize

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats
   #assert(vds)
    vlyr = vds.GetLayer(0)
#    if write_stat != None:
#        vlyr.CreateField(ogr.FieldDefn(bandname, ogr.OFTReal))

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors

    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    
    
    # make a new vector to be writtent
    
    outShapefile = outShp
    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    
    # Remove output shapefile if it already exists
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)
    
    # get the spatial ref
    ref = vlyr.GetSpatialRef()
    
    # Create the output shapefile
    outDataSource = outDriver.CreateDataSource(outShapefile)
    outLayer = outDataSource.CreateLayer("OutLyr", geom_type=ogr.wkbMultiLineString,
                                         srs=ref)
    
    # Add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    outLayer.CreateField(idField)
    
#            for reference
#        xOrigin = rgt[0]
#        yOrigin = rgt[3]
#        pixelWidth = rgt[1]
#        pixelHeight = rgt[5]
    
#    rejects = list()
    for label in tqdm(features):

        if feat is None:
            continue
        geom = feat.geometry()
        
        buff = geom.Buffer(buf)
        
        wkt=buff.ExportToWkt()
        
        poly1 = loads(wkt)
        
        src_offset = _bbox_to_pixel_offsets(rgt, buff)
        
        # xoff, yoff, xcount, ycount
        
        src_offset = list(src_offset)
    
        
        
        
#        for idx, off in enumerate(src_offset):
#            if off <=0:
#                src_offset[idx]=0
            
#            if off > 
                
        if rgb == True:
            rgbList = []
            for band in range(1,4):
                arr = rds.GetRasterBand(band).ReadAsArray(src_offset[0], 
                                        src_offset[1], src_offset[2],
                                        src_offset[3])
                rgbList.append(arr)
                
            src_array = np.vstack((rgbList[0], rgbList[1], rgbList[2]))
        else:
            src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                               src_offset[3])
        if src_array is None:
            src_array = rb.ReadAsArray(src_offset[0]-1, src_offset[1], src_offset[2],
                               src_offset[3])
            if src_array is None:
#                rejects.append(feat.GetFID())
                continue

        # calculate new geotransform of the feature subset
        new_gt = (
        (rgt[0] + (src_offset[0] * rgt[1])),
        rgt[1],
        0.0,
        (rgt[3] + (src_offset[1] * rgt[5])),
        0.0,
        rgt[5])
                    

        
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('line', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())
#
#        # Rasterize it
        #
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        rvds.SetProjection(rds.GetProjectionRef())
        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        
        dist = nd.morphology.distance_transform_edt(np.logical_not(rv_array))
        
        # covert the dist raster to the actual units
        dist *= rgt[1]
        
        bw = dist <=buf
        
        rr, cc = rv_array.nonzero()
        
#        src_array[bw==0]=0
#        src_array = np.float32(src_array)
#        src_array[src_array==0]=np.nan
        
        init = np.array([rr, cc]).T
        
        if smooth == True:
            src_array = gaussian(src_array)
        if eq == True:
            src_array = exposure.equalize_hist(src_array)
            
    
        snake = active_contour(src_array, init, boundary_condition=boundary,
                           alpha=alpha, beta=beta, w_line=w_line, w_edge=w_edge,
                           gamma=gamma, max_iterations=max_iterations,
                           coordinates='rc')
        #dear skimage this function is deeply flawed.....grrrr
        # there should NOT be negative coordinate values in the output
        
        
        """
        for reference
        xOrigin = rgt[0]
        yOrigin = rgt[3]
        pixelWidth = rgt[1]
        pixelHeight = rgt[5]

       """ 
# FOR REF WHEN DEBUGGING DONT DEL
#        xlist = list(snake[:,1])
#        ylist = list(snake[:,0])
#        
#        #snakeLine = LineString(zip(xlist, ylist))     
       # snakeFinite = snake[snake[:,0]>=0] 
        
        #snakeFinite
        
#        sr = np.int32(snakeFinite[:,0])
#        sc = np.int32(snakeFinite[:,1])
        
        finite = snake[snake[:,0]>=0]
        snakeR = np.round(finite)
        snList=snakeR.tolist()
        outSnk = []
#                
#        
#        upper_left_x, x_res, x_rotation, upper_left_y, y_rotation, y_res = rgt
        
        
        for s in snList:
            x = s[1]
            y = s[0]
            xout = (x * new_gt[1]) + new_gt[0]
            yout = (y * new_gt[5]) + new_gt[3]
            
            outSnk.append([xout, yout])

        snakeLine2 = LineString(outSnk)
        
        
        geomOut = ogr.CreateGeometryFromWkt(snakeLine2.wkt)
        
        featureDefn = outLayer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        
        feature.SetGeometry(geomOut)
        feature.SetField("id", 1)
        outLayer.CreateFeature(feature)
        feature = None
        feat = vlyr.GetNextFeature()
        
        
        

    outDataSource.SyncToDisk()
      
    outDataSource=None
    vds = None    
        
def ms_snake(inShp, inRas, outShp, band=2, buf1=0, buf2=0, algo="ACWE", nodata_value=0,
          iterations=200,  smoothing=1, lambda1=1, lambda2=1, threshold='auto', 
          balloon=-1):
    
    """ 
    Deform a polygon using active contours on the values of an underlying raster.
    
    This uses morphsnakes and explanations are from there.
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster
    outShp: string
                  output shapefile
        
    band: int
           an integer val eg - 2

    algo: string
           either "GAC" (geodesic active contours) or the default "ACWE" (active contours without edges)
    buf1: int
           the buffer if any in map units for the bounding box of the poly which
           extracts underlying pixel values.
           
    buf2: int
           the buffer if any in map units for the expansion or contraction
           of the poly which will initialise the active contour. 
           This is here as you may wish to adjust the init polygon so it does not
           converge on a adjacent one or undesired area. 
          
    nodata_value: numerical
                   If used the no data val of the raster

    iterations: uint
        Number of iterations to run.
        
    smoothing : uint, optional
    
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
    
    # Partly inspired by the Heikpe paper...
   
    
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    #assert(rds)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()
    
    
    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats

    vlyr = vds.GetLayer(0)
#    if write_stat != None:
#        vlyr.CreateField(ogr.FieldDefn(bandname, ogr.OFTReal))

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors

    #feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())

    
    outDataset = _copy_dataset_config(rds, outMap = outShp[:-4]+'.tif',
                                     bands = 1, )
    
    outBnd = outDataset.GetRasterBand(1)
    
    

#    seg = np.zeros_like(rb.ReadAsArray())
#    tempRas = vector_path[:-4]+'.tif'
    
#    rasterize(vector_path, raster_path, tempRas)
#    sgbnd = gdal.Open(tempRas).GetRasterBand(1)
#    seg = sgbnd.ReadAsArray()
#    rejects = list()
    for label in tqdm(features):

        feat = vlyr.GetFeature(label)
#        if feat is None:
#            continue
        geom = feat.geometry()
        buff = geom.Buffer(buf1)
        
        src_offset = _bbox_to_pixel_offsets(rgt, buff)
        
        src_offset = list(src_offset)
        
        for idx, off in enumerate(src_offset):
            if off <=0:
                src_offset[idx]=0
               
        src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                               src_offset[3])
        if src_array is None:
            src_array = rb.ReadAsArray(src_offset[0]-1, src_offset[1], src_offset[2],
                               src_offset[3])
            if src_array is None:
#                rejects.append(feat.GetFID())
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
#
#        # Rasterize it
        
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        
        if buf2 < 0:
            dist = nd.morphology.distance_transform_edt(rv_array)
        else:
            dist = nd.morphology.distance_transform_edt(np.logical_not(rv_array))
        
        # covert the dist raster to the actual units
        dist *= rgt[1]
        
        # expand or contract the blob
        if buf2 != 0:
            if buf2 > 0:                
                rv_array = dist <=buf2
            else:
                rv_array = dist >=abs(buf2)
                       
        # this will stop it working as ti will converge on boundary!!!
        # kept so you are not tempted to reinstate!
        #src_array[bw==0]=0
    
        
        if algo == "ACWE":       
        
            bw = ms.morphological_chan_vese(src_array, iterations=iterations,
                                   init_level_set=rv_array,
                                   smoothing=smoothing, lambda1=lambda1,
                                   lambda2=lambda2)
        if algo == "GAC":
            gimg = ms.inverse_gaussian_gradient(src_array)
            bw = ms.morphological_geodesic_active_contour(gimg, iterations, rv_array,
                                             smoothing=smoothing, threshold=threshold,
                                             balloon=balloon)

        
        segoot = np.int32(bw)
        segoot*=int(label)+1
        
        # very important not to overwrite results
        if label > 0:
            ootArray = outBnd.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                               src_offset[3])
            ootArray[segoot==label+1]=label+1
            outBnd.WriteArray(ootArray, src_offset[0], src_offset[1])
        else:
    
            outBnd.WriteArray(segoot, src_offset[0], src_offset[1])
        
        del segoot, bw
        feat = vlyr.GetNextFeature()
        

        """
        for reference
        xOrigin = rgt[0]
        yOrigin = rgt[3]
        pixelWidth = rgt[1]
        pixelHeight = rgt[5]
        
       """
    
    outDataset.FlushCache()
    
    outDataset=None
    vds = None
    
    # This is a hacky solution for now really, but it works well enough!
    polygonize(outShp[:-4]+'.tif', outShp, outField='id',  mask = True, band = 1)    

def thresh_seg(inShp, inRas, outShp, band, buf=0, algo='otsu',
               min_area=4, nodata_value=0):
    
    """ 
    Use an image processing technique to threshold foreground and background in a polygon segment.
    
    This default is otsu's method.
    
    Parameters
    ----------
    
    vector_path : string
                  input shapefile
        
    raster_path : string
                  input raster

    band : int
           an integer val eg - 2

    algo : string
           'otsu', niblack, sauvola
          
    nodata_value : numerical
                   If used the no data val of the raster

 
    """    
    
    
   
    
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    #assert(rds)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats
   #assert(vds)
    vlyr = vds.GetLayer(0)
#    if write_stat != None:
#        vlyr.CreateField(ogr.FieldDefn(bandname, ogr.OFTReal))

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors

    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())

    
    outDataset = _copy_dataset_config(rds, outMap = outShp[:-4]+'.tif',
                                     bands = 1)
    
    outBnd = outDataset.GetRasterBand(1)
    pixel_res = rgt[1]

#    rejects = list()
    for label in tqdm(features):

        if feat is None:
            continue
        geom = feat.geometry()
        
        buff = geom.Buffer(buf)
        
        src_offset = _bbox_to_pixel_offsets(rgt, buff)
        
        src_offset = list(src_offset)
        
        
        src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                               src_offset[3])
        if src_array is None:
            src_array = rb.ReadAsArray(src_offset[0]-1, src_offset[1], src_offset[2],
                               src_offset[3])
            if src_array is None:
                rejects.append(feat.GetFID())
                continue
            if src_array.size == 1:
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
        #with warnings.catch_warnings():

        warnings.simplefilter("ignore")
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Int32)
     
        rvds.SetGeoTransform(new_gt)
        rvds.SetProjection(rds.GetProjectionRef())
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        
        src_array *= rv_array>0
        if src_array.max()==0:
            continue
#        zone = np.ma.MaskedArray(src_array,
#                                 mask=np.logical_or(src_array == nodata_value, 
#                                                    np.logical_not(rv_array)))
        
        if algo == 'otsu':       
            t = threshold_otsu(src_array)
        elif algo == 'niblack':
            t = threshold_niblack(src_array)
        elif algo == 'sauvola':
            t = threshold_sauvola(src_array)                            
                             
        bw = src_array > t
        
        remove_small_holes(bw, in_place=True, area_threshold=4)
        if min_area != None:
            min_final = np.round(min_area/(pixel_res*pixel_res))
        
            if min_final <= 0:
                min_final=4
        
            remove_small_objects(bw, min_size=min_final, in_place=True)

        segoot = np.int32(bw)        
        
        segoot*=int(label)+1
        outBnd.WriteArray(segoot, src_offset[0], src_offset[1])
        del segoot
        feat = vlyr.GetNextFeature()
        

        """
        for reference
        xOrigin = rgt[0]
        yOrigin = rgt[3]
        pixelWidth = rgt[1]
        pixelHeight = rgt[5]
        
       """
    
    outDataset.FlushCache()
    
    outDataset=None
    vds = None
    
    # This is a hacky solution for now really, but it works well enough!
    polygonize(outShp[:-4]+'.tif', outShp, outField=None,  mask = True, band = 1)    
    
def _std_huff(inArray, outArray, outLayer, angl, valrange, interval, rgt):#, mk=None):
    
    
    tested_angles = np.linspace(angl - np.deg2rad(valrange), 
                                angl + np.deg2rad(valrange), num=interval)

    hh, htheta, hd = hough_line(inArray, theta=tested_angles)
    origin = np.array((0, inArray.shape[1]))
    
    
    #empty = np.zeros_like(inArray, dtype=np.bool)
    
    height, width = inArray.shape
    
    bbox = box(width, height, 0, 0)
    
    #angl - np.radians(valrange), angl + np.radians(valrange)
    
    # opencv is simpler but dont get it yet
    #lines = cv2.HoughLines(re, 1, 150, None, 0, 0)
    
#    msk = ogr.Open(mk)
#    msklyr = msk.GetLayer(0)
#    mskFeat = msklyr.GetFeature(0)
#    geom = mskFeat.GetGeometryRef()
#    wkt=geom.ExportToWkt()
#    poly = loads(wkt)
#    bw = imread(mk)

                    
                    # Here we adapt the skimage loop to draw a bw line into the image
    for _, angle, dist in tqdm(zip(*hough_line_peaks(hh, htheta, hd))):
    
        # here we obtain y extrema in our arbitrary coord system
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        
        # shapely used to get the geom 
        
        linestr = LineString([[origin[0], y0], [origin[1], y1]])
        
        # just in case this has not been done
        #rotated = rotate(linestr, 90, origin='centroid')
        
        # here for readability visual query
        # shapely in-built converts to np via np.array(inter)
        #inter = bbox.intersection(linestr)
        
        in_coord= np.array(bbox.intersection(linestr).coords)
        
        coord = np.around(in_coord)
        
        # for readability just now
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
        
        cc, rr = line(x1, y1, x2, y2)
        
        # is the line tightly constrained to the desired angle?
#        test = np.zeros_like(outArray)
#        test[rr, cc]=1
#        props = regionprops(test*1)
#        orient = props[0]['Orientation']
#        if orient >= angl-valrange and orient <= angl+valrange:
        outArray[rr, cc]=1
        outSnk = []

    
        snList = np.arange(len(rr))
        
        for s in snList:
            y = rr[s]
            x = cc[s]
            xout = (x * rgt[1]) + rgt[0]
            yout = (y * rgt[5]) + rgt[3]
            
            outSnk.append([xout, yout])
        snakeLine2 = LineString(outSnk)

    
        geomOut = ogr.CreateGeometryFromWkt(snakeLine2.wkt)
    
        featureDefn = outLayer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
    
        feature.SetGeometry(geomOut)
        feature.SetField("id", 1)
        outLayer.CreateFeature(feature)
        feature = None
#        else:
#            continue
    
       
                    
        

    return outArray


def _phl_huff(inArray, outArray, outLayer, angl, valrange, interval, rgt, 
              line_length, line_gap, mask=None):
    
    tested_angles = np.linspace(angl - np.radians(valrange),
                                angl + np.radians(valrange), num=interval)
    huff = phl(inArray, line_length=line_length, line_gap=line_gap, 
                           theta=tested_angles)
    
    for linez in huff:
    
        # jeez that is ugly as feck
        x1 = linez[0][0]
        y1 = linez[0][1]
        x2 = linez[1][0]
        y2 = linez[1][1]
        rr, cc = line(y1, x1, y2, x2)

        outArray[rr, cc]=1
        

        
        outSnk = []
            
        snList = np.arange(len(cc))
        
        for s in snList:
            x = cc[s]
            y = rr[s]
            xout = (x * rgt[1]) + rgt[0]
            yout = (y * rgt[5]) + rgt[3]
            
            outSnk.append([xout, yout])
        
        snakeLine2 = LineString(outSnk)
        
        
        geomOut = ogr.CreateGeometryFromWkt(snakeLine2.wkt)
        
        featureDefn = outLayer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        
        feature.SetGeometry(geomOut)
        feature.SetField("id", 1)
        outLayer.CreateFeature(feature)
    return outArray

from numpy.lib.stride_tricks import as_strided as ast

def _block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)

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

def _do_phasecong(tempIm,  low_t=0, hi_t=0, norient=6, nscale=6, sigma=2):#, skel='medial'):
    """
    process phase congruency on an image returning vars for hough2line
    A subfunction used in hough2line. 
    
    At present skeletonising the result with the medial axis, though a better
    solution for final edges is needed ultimately
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
#    orientIm = ph[4]
#    anglez = np.array([p * (np.pi / 6) for p in range(1,7)])
#    
#    # hang on this is not working
#    ootV = np.abs(np.array(anglez) - angleV)
#    ootH = np.abs(np.array(anglez) - angleD)
#    
#    
#    vInd = np.where(ootV==ootV.min())
#    hInd = np.where(ootH==ootH.min())
#    
#    v = int(vInd[0])+1
#    h = int(hInd[0])+1
#    
#    if v>=6:
#        v -= 1
#    if h >=6:
#        h -= 1                
#
#    vT = threshold_otsu(orientIm[v])
#    hT = threshold_otsu(orientIm[h])
#    vArray = orientIm[v]>=vT
#    hArray = orientIm[h]>=hT     
#    return vArray, hArray

def hough2line(inRas, outShp, edge='canny', sigma=2, low_t=None, 
               hi_t=None, n_orient=6, n_scale=5, hArray=True, vArray=True,
               prob=False, line_length=100,
               line_gap=200, valrange=1, interval=50, band=2,
               min_area=None):
    
        """ 
        Detect and write Hough lines to a line shapefile
        
        There are optionally two input arrays on the to keep line detection clean eg 2 orientations,
        such as vertical and horizontal
        
        Parameters
        ----------
        
        inRaster: string
               path to an input raster from which the geo-reffing is obtained
    
        outShp: string
               path to the output line shapefile a corresponding polygon will also be written to disk
        
        edge: string
              the edge detection method - either phase congruency or canny
              phase is default.
        
        sigma: int
              the size of stdv defining the gaussian envelope if using canny edge or phase 
              a unitless value 
        
        n_orient: int
              the number of orientations used if using phase congruency edge
        
        n_scale: int
              the number of scales used if using phase congruency edge
                            
        vArray: bool
              whether to detect lines on approx vert axis
              
        hArray: bool
              whether to detect lines on approx horz axis             
        low_t: 
              the low hysteresis threshold
              the secondary low gradient threshold permitted if connected to 
              a high threshold pixel
        hi_t: 
              the high hysteresis threshold.
              the principal gradient threshold from which the low values are permitted 
              provided they are connected to pixels of this one
                                       
        prob: bool
               Whether to use a probabalistic hough - default is false
             
        line_length: int
               If using prob hough the min. line length threshold        
        line_gap: int
               If using prob hough the min. line gap threshold
        val_range: int
               The + - range around the orientation automatically chosen  
        interval: int
               The no of intervals of the range of values tested if using auto  
        """         
        #TODO this is FAR too long
        inDataset = gdal.Open(inRas, gdal.GA_ReadOnly)

#        rb = inRas.GetRasterBand(band)
        rgt = inDataset.GetGeoTransform()
        
        pixel_res = rgt[1]
        
        ref = inDataset.GetSpatialRef()
        
        outShapefile = outShp
        outDriver = ogr.GetDriverByName("ESRI Shapefile")
        
        # Remove output shapefile if it already exists
        if os.path.exists(outShapefile):
            outDriver.DeleteDataSource(outShapefile)
        
        # get the spatial ref
#        ref = vlyr.GetSpatialRef()
        
        # Create the output shapefile
        outDataSource = outDriver.CreateDataSource(outShapefile)
        outLayer = outDataSource.CreateLayer("OutLyr", geom_type=ogr.wkbMultiLineString,
                                         srs=ref)
    
        
        # Add an ID field
        idField = ogr.FieldDefn("id", ogr.OFTInteger)
        outLayer.CreateField(idField)
        
        empty = np.zeros((inDataset.RasterYSize, inDataset.RasterXSize), dtype=np.bool)
        
        #because I am thick
        #Degrees (°) 	Radians (rad) 	Radians (rad)
        #0° 	0 rad 	0 rad
        #30° 	π/6 rad 	0.5235987756 rad
        #45° 	π/4 rad 	0.7853981634 rad
        #60° 	π/3 rad 	1.0471975512 rad
        #90° 	π/2 rad 	1.5707963268 rad
        #120° 	2π/3 rad 	2.0943951024 rad
        #135° 	3π/4 rad 	2.3561944902 rad
        #150° 	5π/6 rad 	2.6179938780 rad
        #180° 	π rad 	3.1415926536 rad
        #270° 	3π/2 rad 	4.7123889804 rad
        #360° 	2π rad 	6.2831853072 rad

            
        tempIm = inDataset.GetRasterBand(band).ReadAsArray()
        bw = tempIm > 0
        
        bwRas = inRas[:-4]+'bw.tif'
        #maskShp = inRaster[:-4]+'bwmask.shp'
        
        #polygonize(bwRas, maskShp, outField=None,  mask = True, band = 1)   
        props = regionprops(bw*1)
        orient = props[0]['Orientation']
        
        # we will need these.....
        perim = mh.bwperim(bw)
#        bkgrnd = invert(bw)
        
        
        bw[perim==1]=0
        array2raster(bw, 1, inRas, bwRas,  gdal.GDT_Byte)
        

        # if the the binary box is pointing negatively along maj axis
        
        if orient < 0:
            orient += np.pi
        
        if orient < np.pi:
            angleD = np.pi - orient
            angleV = angleD - np.deg2rad(90)
        else:
        # if the the binary box is pointing positively along maj axis
            angleD = np.pi + orient
            angleV = angleD + np.deg2rad(90)
        

        
            

        if edge == 'phase':
            ph = _do_phasecong(tempIm, low_t, hi_t, norient=n_orient, 
                               nscale=n_scale, sigma=sigma)
            
            ph[perim==1]=0
            
            if hArray is True:
                vArray = ph
            if hArray is True:
                hArray = ph
            del ph
           
        else: 
            # else it is canny
            # We must have a float to get rid of  zero to nonzero image 
            # boundary, otherwise huff will only detect the non-zero boundary
            inIm = tempIm.astype(np.float32)
            inIm[inIm==0]=np.nan 
        
            if hArray is True:
                hArray = canny(inIm, sigma=sigma, low_threshold=low_t,
                               high_threshold=hi_t)
            if vArray is True:    
                vArray = canny(inIm, sigma=sigma, low_threshold=low_t,
                               high_threshold=hi_t)
            del inIm
                                  
        
        if prob is False:
            """
            Standard Hough ##############################################################
            """
            if hasattr(vArray, 'shape'):
                
                empty =_std_huff(vArray, empty, outLayer, angleV, valrange, interval, rgt)#, mk=bwRas)
            if hasattr(hArray, 'shape'):
                empty =_std_huff(hArray, empty, outLayer, angleD, valrange, interval, rgt)#, mk=bwRas)
           
        
        else:
            """
            Prob Hough ##############################################################
            """
            
            if hasattr(vArray, 'shape'):
                empty =_phl_huff(vArray, empty, outLayer, angleV, valrange,
                                 interval, rgt, line_length, line_gap)
            if hasattr(hArray, 'shape'):
                empty =_phl_huff(hArray, empty, outLayer, angleD, valrange,
                                 interval, rgt, line_length, line_gap)
            
            
                
        
        outDataSource.SyncToDisk()
          
        outDataSource=None
        
        if prob is True:
            array2raster(empty, 1, inRas, outShp[:-3]+"tif",  gdal.GDT_Int32)
        else:
            inv = np.invert(empty)
            inv[tempIm==0]=0
            if min_area != None:
                min_final = np.round(min_area/(pixel_res*pixel_res))
                if min_final <= 0:
                    min_final=4
                remove_small_objects(inv, min_size=min_final, in_place=True)
            #sg, _ = nd.label(inv)
            segRas=outShp[:-3]+"seg.tif"
            array2raster(inv, 1, inRas, segRas,  gdal.GDT_Int32)
            del tempIm, inv
        
        polygonize(segRas, outShp[:-4]+"_poly.shp", outField=None,  mask = True, band = 1)  

def cv_hough2line(inRas, outShp, edge='canny', sigma=2, low_t=0, 
               hi_t=0, n_orient=6, n_scale=5, increment=180, thresh=100, min_theta=None, bounds=1,
                  max_theta=None, min_area=None):
    
    img = cv2.imread(inRas) 
     
    gray = raster2array(inRas, bands=[2])
    
    bw = gray > 0
    
    inDataset = gdal.Open(inRas, gdal.GA_ReadOnly)

#        rb = inRas.GetRasterBand(band)
    rgt = inDataset.GetGeoTransform()
    
    pixel_res = rgt[1]
        
   
    props = regionprops(bw*1)
    orient = props[0]['Orientation']
    
    # we will need these.....
    perim = mh.bwperim(bw)

    
    bw[perim==1]=0

    # if the the binary box is pointing negatively along maj axis
#   
    if min_theta == None:
        if orient < 0:
            orient += np.pi
        
        if orient < np.pi:
            angleD = np.pi - orient
            angleV = angleD - np.deg2rad(90)
        else:
        # if the the binary box is pointing positively along maj axis
            angleD = np.pi + orient
            angleV = angleD + np.deg2rad(90)
        angles = np.array([angleD, angleV])
        
    
     
    if edge == 'phase':
            edges = _do_phasecong(gray, low_t, hi_t, norient=n_orient, 
                               nscale=n_scale, sigma=sigma)
            
            edges[perim==1]=0
            
           
    else: 
        # else it is canny
        # We must have a float to get rid of  zero to nonzero image 
        # boundary, otherwise huff will only detect the non-zero boundary
        inIm = gray.astype(np.float32)
        inIm[inIm==0]=np.nan 
    
        edges = canny(inIm, sigma=sigma, low_threshold=low_t,
                           high_threshold=hi_t)
        
 
      
    # This returns an array of r and theta values 
    lines1 = cv2.HoughLines(np.uint8(edges),1, np.pi/increment, thresh, min_theta=angles.min()-np.deg2rad(bounds), 
                           max_theta=angles.min()+np.deg2rad(bounds)) 
    
      
    # The below for loop runs till r and theta values  
    # are in the range of the 2d array 
    for l in tqdm(lines1): 
        
        # Right so unlike the cv docs we actually have to do this!!! 
        # see for i in lines:
        #        print(i) 
        # as to why
        r = l[0][0]
        theta = l[0][1]
        # Stores the value of cos(theta) in a 
        a = np.cos(theta) 
      
        # Stores the value of sin(theta) in b 
        b = np.sin(theta) 
          
        # x0 stores the value rcos(theta) 
        x0 = a*r 
          
        # y0 stores the value rsin(theta) 
        y0 = b*r 
          
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
        x1 = int(x0 + 1000*(-b)) 
          
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
        y1 = int(y0 + 1000*(a)) 
      
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
        x2 = int(x0 - 1000*(-b)) 
          
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
        y2 = int(y0 - 1000*(a)) 
          
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
        # (0,0,255) denotes the colour of the line to be  
        #drawn. In this case, it is red.  
        cv2.line(img,(x1,y1), (x2,y2), (0,0,255),1)
    
    lines2 = cv2.HoughLines(np.uint8(edges),1, np.pi/increment, thresh, min_theta=angles.max()-np.deg2rad(bounds), 
                           max_theta=angles.max()+np.deg2rad(bounds)) 
    for l in tqdm(lines2): 
        
        # Right so unlike the cv docs we actually have to do this!!! 
        # see for i in lines:
        #        print(i) 
        # as to why
        r = l[0][0]
        theta = l[0][1]
        # Stores the value of cos(theta) in a 
        a = np.cos(theta) 
      
        # Stores the value of sin(theta) in b 
        b = np.sin(theta) 
          
        # x0 stores the value rcos(theta) 
        x0 = a*r 
          
        # y0 stores the value rsin(theta) 
        y0 = b*r 
          
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
        x1 = int(x0 + 1000*(-b)) 
          
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
        y1 = int(y0 + 1000*(a)) 
      
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
        x2 = int(x0 - 1000*(-b)) 
          
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
        y2 = int(y0 - 1000*(a)) 
          
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
        # (0,0,255) denotes the colour of the line to be  
        #drawn. In this case, it is red.  
        cv2.line(img,(x1,y1), (x2,y2), (0,0,255),1) 
        
    #minLineLength = 5
    #maxLineGap = 10
    #linesP = cv2.HoughLinesP(np.uint8(edges),1,np.pi/180,50,minLineLength,maxLineGap)
    #for i in linesP:
    #    l = i[0]
    #    cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    
    #cv2.imwrite('houghlines5.jpg',img)
    
    
    oot= img[:,:,2]==255
    oot[perim==1]=1
 

    inv = np.invert(oot)
    inv[bw==0]=0
    if min_area != None:
        min_final = np.round(min_area/(pixel_res*pixel_res))
        if min_final <= 0:
            min_final=4
        remove_small_objects(inv, min_size=min_final, in_place=True)
    sg, _ = nd.label(inv)
    segRas=outShp[:-3]+"seg.tif"
    array2raster(sg, 1, inRas, segRas,  gdal.GDT_Int32)
    del img, inv
    polygonize(segRas, outShp, outField='DN',  mask = True, band = 1)        


def _do_ransac(inArray, order='col'):
    
    outArray = np.zeros_like(inArray)
    
    #th = filters.threshold_otsu(inArray)
    
    #bw = inArray > th
    
    
    inDex = np.where(inArray > 0)
    if order == 'col':
        
        inData = np.column_stack([inDex[0], inDex[1]])
        
        
        model = LineModelND()
        model.estimate(inData)
    
        model_robust, inliers = ransac(inData, LineModelND, min_samples=2,
                                       residual_threshold=1, max_trials=2500)
    
    
        outliers = inliers == False
    
        
        line_x = inData[:, 0]
        line_y = model.predict_y(line_x)
        line_y_robust = model_robust.predict_y(line_x)
    
        outArray[line_x, np.int64(np.round(line_y_robust))]=1
        
    if order == 'row':
        
        inData = np.column_stack([inDex[1], inDex[0]])
    
    
        model = LineModelND()
        model.estimate(inData)
    
        model_robust, inliers = ransac(inData, LineModelND, min_samples=2,
                                   residual_threshold=1, max_trials=2500)
    
    
        outliers = inliers == False
        
        line_x = inData[:,0]
        line_y = model.predict_y(line_x)

        line_y_robust = model_robust.predict_y(line_x)
        
        outArray[np.int64(np.round(line_y_robust)), line_x]=1

    
    return outArray
    
def ransac_lines(inRas, outRas, sigma=3, row=True, col=True, binwidth=40):


    inDataset = gdal.Open(inRas)
        
    outDataset = _copy_dataset_config(inDataset, outMap = outRas,
                                     dtype = gdal.GDT_Byte, bands = 1)
    band = inDataset.GetRasterBand(2)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    outBand = outDataset.GetRasterBand(1)
    
    
    blocksizeY = inDataset.RasterYSize
    
    blocksizeX = binwidth
    
    
    # vertical lines
    if col is True:
        
        
        for i in range(0, rows, blocksizeY):
            if i + blocksizeY < rows:
                numRows = blocksizeY
            else:
                numRows = rows -i
            
            for j in tqdm(range(0, cols, blocksizeX)):
                if j + blocksizeX < cols:
                    numCols = blocksizeX
                else:
                    numCols = cols - j
                
                inArray = band.ReadAsArray(j,i, numCols, numRows)
                #glim = nd.gaussian_laplace(inArray, sigma=4)
                edge = canny(inArray, sigma=sigma)
                oot = _do_ransac(edge, order='col')
                
                outBand.WriteArray(oot,j,i)
    
    if row is True:
    # horizontal lines
    
        blocksizeY = binwidth
        
        blocksizeX = inDataset.RasterXSize
        
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
                
                inArray = band.ReadAsArray(j,i, numCols, numRows)
                #glim = nd.gaussian_laplace(inArray, sigma=4)
                edge = canny(inArray, sigma=sigma)
                oot = _do_ransac(edge, order='row')
                
                outArray = outBand.ReadAsArray(j,i, numCols, numRows)
                
        
                outArray[oot==1]=1
                
                outBand.WriteArray(outArray,j,i)
    
            
    outDataset.FlushCache()
    outDataset = None
    tmpIm = gdal.Open(outRas)
    outIm = tmpIm.GetRasterBand(1).ReadAsArray()
    
    array2raster(np.invert(outIm), 1, inRas, inRas[:-4]+"seg.tif",  gdal.GDT_Int32)
        
    polygonize(inRas[:-4]+"seg.tif", inRas[:-4]+"seg.shp", outField=None,  mask = True, band = 1)
    return outIm        



def meshgrid(inRaster, outShp, gridHeight=1, gridWidth=1):

    #TODO - make alternating intervals and make it rotational
    
    
    # make a mask for non-zero vals for our mesh
    inRas = gdal.Open(inRaster)
    tempIm = inRas.GetRasterBand(1).ReadAsArray()
    
    bw = tempIm > 0
    
    props = regionprops(bw*1)
    orient = props[0]['Orientation']
    
    bwRas = inRaster[:-4]+'bw.tif'
    maskShp = inRaster[:-4]+'bwmask.shp'
    array2raster(bw, 1, inRaster, bwRas,  gdal.GDT_Byte)
    polygonize(bwRas, maskShp, outField=None,  mask = True, band = 1)
    
    inRas = None
    del bw, tempIm

    shape = ogr.Open(maskShp)
    
    lyr = shape.GetLayer()
    
    feat = lyr.GetFeature(0)
    
    geom = feat.GetGeometryRef()
    

    
    # not rotated!
#    numpoints = geom.GetPointCount()
#    pointsX = []; pointsY = []
#    
#    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
#        count = 0
#        pointsX = []; pointsY = []
#        for polygon in geom:
#            geomInner = geom.GetGeometryRef(count)
#            ring = geomInner.GetGeometryRef(0)
#            numpoints = ring.GetPointCount()
#            for p in range(numpoints):
#                    lon, lat, z = ring.GetPoint(p)
#                    pointsX.append(lon)
#                    pointsY.append(lat)
#            count += 1
#    elif (geom.GetGeometryName() == 'POLYGON'):
#        ring = geom.GetGeometryRef(0)
#        numpoints = ring.GetPointCount()
#        pointsX = []; pointsY = []
#        for p in range(numpoints):
#                lon, lat, z = ring.GetPoint(p)
#                pointsX.append(lon)
#                pointsY.append(lat)
#            
#    xmin = min(pointsX)
#    xmax = max(pointsX)
#    ymin = min(pointsY)
#    ymax = max(pointsY)
#    
    
    wkt=geom.ExportToWkt()
    poly1 = loads(wkt)
    
    if orient < np.pi:
        poly2 = rotate(poly1, np.pi-orient, use_radians=True)
    else:
        poly2 = rotate(poly1, np.pi+orient, use_radians=True)
    
    xmin, ymin, xmax, ymax = poly2.bounds
    
    



    # so here we spin the rectangle round to the vertical then measure it
#    x,y=poly2.exterior.coords.xy
#    xy = np.vstack((x,y))
#    rec = min_bound_rectangle(xy.transpose())
#    poly2 = Polygon(rec)
#    minx, miny, maxx, maxy = poly2.bounds
#    axis1 = maxx - minx
#    axis2 = maxy - miny
#    axes = np.array([axis1, axis2])
#    

    gridWidth = float(gridHeight)
    gridHeight = float(gridWidth)

    # get rows
    rows = ceil((ymax-ymin)/gridHeight)
    # get columns
    cols = ceil((xmax-xmin)/gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth 
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outShp):
        os.remove(outShp)
    
    ref = lyr.GetSpatialRef()
    outDataSource = outDriver.CreateDataSource(outShp)
    outLayer = outDataSource.CreateLayer(outShp, geom_type=ogr.wkbPolygon, srs=ref)
    featureDefn = outLayer.GetLayerDefn()

    # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            
            poly.AddGeometry(ring)
            g2 = poly.ExportToWkt()
            poly3 = loads(g2)
            poly4 = rotate(poly3, np.pi+orient, use_radians=True)
            
            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly4.to_wkt())
            outLayer.CreateFeature(outFeature)
            outFeature = None

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Save and close DataSources
    
    outDataSource.SyncToDisk()
    outDataSource = None



#def line2poly(inShp, outShp):
#    
#    
#    
#    vds = ogr.Open(inShp, 1)  
#   #assert(vds)
#    vlyr = vds.GetLayer(0)
#    
#    outShapefile = outShp
#    outDriver = ogr.GetDriverByName("ESRI Shapefile")
#        
#        # Remove output shapefile if it already exists
#    if os.path.exists(outShapefile):
#        outDriver.DeleteDataSource(outShapefile)
#        
#        # get the spatial ref
#    ref = vlyr.GetSpatialRef()
#        
#        # Create the output shapefile
#    outDataSource = outDriver.CreateDataSource(outShapefile)
#    outLayer = outDataSource.CreateLayer("OutLyr", geom_type=ogr.wkbPolygon,
#                                     srs=ref)
#    
#        
#        # Add an ID field
#    idField = ogr.FieldDefn("id", ogr.OFTInteger)
#    outLayer.CreateField(idField)
#    
#    feat = vlyr.GetNextFeature()
#    features = np.arange(vlyr.GetFeatureCount())
#    multiline = ogr.Geometry(ogr.wkbMultiLineString)
#    
##    rejects = list()
#    for label in tqdm(features):
#        
#        geom = feat.GetGeometryRef()
#        
#        wkt = geom.ExportToWkt()    
#    
#        line = ogr.CreateGeometryFromWkt(wkt)
#        
#        multiline.AddGeometry(line)
#            
#    polygon = ogr.BuildPolygonFromEdges(multiline)




def _dbf2DF(dbfile, upper=True): #Reads in DBF files and returns Pandas DF
    db = pio.open(dbfile) #Pysal to open DBF
    d = {col: db.by_col(col) for col in db.header} #Convert dbf to dictionary
    #pandasDF = pd.DataFrame(db[:]) #Convert to Pandas DF
    pandasDF = pd.DataFrame(d) #Convert to Pandas DF
    if upper == True: #Make columns uppercase if wanted 
        pandasDF.columns = map(str.upper, db.header) 
    db.close() 
    return pandasDF


    
##### make a new vector to be written for reference
    
#    outShapefile = outShp
#    outDriver = ogr.GetDriverByName("ESRI Shapefile")
#    
#    # Remove output shapefile if it already exists
#    if os.path.exists(outShapefile):
#        outDriver.DeleteDataSource(outShapefile)
#    
#    # Create the output shapefile
#    outDataSource = outDriver.CreateDataSource(outShapefile)
#    outLayer = outDataSource.CreateLayer("OutLyr", geom_type=ogr.wkbMultiPolygon)
#    
#    # Add an ID field
#    idField = ogr.FieldDefn("id", ogr.OFTInteger)
#    outLayer.CreateField(idField)