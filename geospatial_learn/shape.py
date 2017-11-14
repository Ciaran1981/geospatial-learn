# -*- coding: utf-8 -*-
"""
The shape module. 

Description
-----------

This module contains various functions for the writing of data in OGR vector 
formats. The functions are mainly concerned with writing geometric or pixel based attributes, with the view to them being classified in the learning module

"""
#from time import time
from skimage.measure import regionprops
from scipy.stats import entropy, skew, kurtosis
from skimage import feature
#from rasterstats import zonal_stats
from sklearn import cluster
import shapefile
from simpledbf import Dbf5
import os
import gdal
from json import dumps
import  ogr, osr
from tqdm import tqdm
import numpy as np
from scipy.stats.mstats import mode
from geospatial_learn.utilities import min_bound_rectangle
from shapely.wkt import loads
from shapely.geometry import Polygon

from pandas import DataFrame
import pysal as ps
import pandas as pd
#from osgeo.gdalconst import *
#import sys

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
        if prop is 'Area':
            stat = geom.Area()
            fldName = propNames[prop]
            feat.SetField(fldName, stat)
            lyr.SetFeature(feat)

        elif prop is 'MajorAxisLength':

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
        elif prop is 'MinorAxisLength':
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
        elif prop is 'Eccentricity':
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
        elif prop is 'Solidity':
            #conv = poly1.convex_hull
            bbox = poly1.envelope
            stat = conv.area / bbox.area
            feat.SetField(fldName, stat)
            lyr.SetFeature(feat)
        elif prop is 'Extent':
            bbox = poly1.envelope
            stat = poly1.area / bbox.area
            feat.SetField(fldName, stat)
            lyr.SetFeature(feat)
        elif prop is 'Perimeter':
            bbox = poly1.envelope
            stat = poly1.length # important to note length means
            feat.SetField(fldName, stat)
            lyr.SetFeature(feat) 
            # TODO - this may not write to shape as a tuple
        elif prop is 'Centroid':
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
           skew', 'kurtosis'
                     
    write_stat : bool (optional)
                If True, stat will be written to OGR file, if false, dataframe
                only returned (bool)
        
    nodata_value : numerical
                   If used the no data val of the raster
        
    """    
    # Inspired/Adapted from Matt Perry's excellent script
    
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
        
        
        # TODO This is horrible there must be a better way....
        if stat is 'mode':
            feature_stats = mode(masked)[0]
        elif stat is 'min':
            feature_stats = float(masked.min())
        elif stat is 'mean':
            feature_stats = float(masked.mean())
        elif stat is 'max':
            feature_stats = float(masked.max())
        elif stat is 'median':
            feature_stats = float(masked.median())
        elif stat is 'std':
            feature_stats = float(masked.std())
        elif stat is 'sum':
            feature_stats = float(masked.sum())
        elif stat is 'count':
            feature_stats = int(masked.count())
        elif stat is 'var':
            feature_stats = float(masked.var())
        elif stat is 'skew':
            feature_stats = float(skew(masked[masked.nonzero()]))
        elif stat is 'kurtosis':
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
    return frame, rejects


    
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

    

def texture_stats(vector_path, raster_path, band, gprop='contrast', offset=0,
                  angle=0, write_stat=None, nodata_value=None, mean=True):
    
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
             distance in pixels to measure 
        
    angle : int
            angle in degrees from pixel (int)
     
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
        gname = gprop[:10]
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
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        



        zone = np.ma.MaskedArray(src_array,
                                 mask=np.logical_or(src_array == nodata_value, 
                                                    np.logical_not(rv_array)))
        
        
        if gprop is 'entropy':
            _, counts = np.unique(zone.nonzero(), return_counts=True)
            props = entropy(counts, base=2)
        elif mean is True and gprop != 'entropy':
            angles = np.radians([135,90,45,0])
            
            g = feature.greycomatrix(np.uint8(zone.nonzero()), [1], 
                                     angles, symmetric=True)
            props = feature.greycoprops(g, prop=gprop)
            props = props.mean()
        elif mean is False and gprop != 'entropy': 
            g = feature.greycomatrix(np.uint8(zone.nonzero()), [offset], 
                                     [np.radians(angle)], symmetric=True)
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

    


def _dbf2DF(dbfile, upper=True): #Reads in DBF files and returns Pandas DF
    db = ps.open(dbfile) #Pysal to open DBF
    d = {col: db.by_col(col) for col in db.header} #Convert dbf to dictionary
    #pandasDF = pd.DataFrame(db[:]) #Convert to Pandas DF
    pandasDF = pd.DataFrame(d) #Convert to Pandas DF
    if upper == True: #Make columns uppercase if wanted 
        pandasDF.columns = map(str.upper, db.header) 
    db.close() 
    return pandasDF
    
