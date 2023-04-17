# -*- coding: utf-8 -*-
"""
The shape module. 

Description
-----------

This module contains various functions for the writing of data in OGR vector 
formats. 
The functions are mainly concerned with writing geometric or pixel based attributes, 
with the view to them being classified in the learning module

"""
from skimage.measure import regionprops
from scipy.stats import entropy, skew, kurtosis
import scipy.ndimage as nd
from skimage import feature
import json
import shapefile
import os
from json import dumps
from osgeo import gdal, ogr, osr
from tqdm import tqdm
import numpy as np
from scipy.stats.mstats import mode
from geospatial_learn.utilities import min_bound_rectangle
from shapely.wkt import loads
from shapely.geometry import Polygon, LineString
from pandas import DataFrame
from pysal.lib import io as pio
import pandas as pd
from skimage.segmentation import active_contour
import geopandas as gpd
import morphsnakes as ms
from geospatial_learn.raster import _copy_dataset_config,  array2raster, polygonize, raster2array
import warnings
from skimage.filters import gaussian
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage.morphology import remove_small_objects, remove_small_holes
import matplotlib
from shapely.affinity import rotate
#from geospatial_learn.geodata import rasterize
from math import ceil
#from centerline.geometry import Centerline

matplotlib.use('Qt5Agg')

gdal.UseExceptions()
ogr.UseExceptions()


def shp2gj(inShape, outJson):
    """
    Converts a geojson/json to a shapefile
    
    Parameters
    ----------
    
    inShape: string
              input shapefile
    
    outJson: string
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
    
def _feat2dict(feat):
    """
    convert an ogr feat to a dict
    """
    geom = feat.GetGeometryRef()
    js = geom.ExportToJson()
    geoj = json.loads(js)
    
    return geoj

def poly2dictlist(inShp):
    
    """
    convert an ogr to a list of json like dicts
    """
    vds = ogr.Open(inShp)
    lyr = vds.GetLayer()
    
    features = np.arange(lyr.GetFeatureCount()).tolist()
    
    feat = lyr.GetNextFeature() 

    oot = [_feat2dict(feat) for f in features]
    
    return oot
    

def _raster_extent(inras):
    
    """
    Parameters
    ----------
    
    inras: string
        input gdal raster (already opened)
    
    """
    rds = gdal.Open(inras)
    rgt = rds.GetGeoTransform()
    minx = rgt[0]
    maxy = rgt[3]
    maxx = minx + rgt[1] * rds.RasterXSize
    miny = maxy + rgt[5] * rds.RasterYSize
    ext = (minx, miny, maxx, maxy)
    spref = rds.GetSpatialRef()
    
    return ext, spref

def create_ogr_poly(outfile, spref, file_type="ESRI Shapefile", field="id", 
                     field_dtype=0):
    """
    Create an ogr dataset an layer (convenience)
    
    Parameters
    ----------
    
    outfile: string
                path to ogr file 
    
    spref: wkt or int
        spatial reference either a wkt or espg
    
    file_type: string
                ogr file designation
        
    field: string
            attribute field e.g. "id"
    
    field_type: int or ogr.OFT.....
            ogr dtype of field e.g. 0 == ogr.OFTInteger
        
             
    """   
    proj = osr.SpatialReference()
    #TODO if int assume espg - crude there will be a better way
    if spref is int:
        proj.ImportFromEPSG(spref)
    else:
        proj.ImportFromWkt(spref)
        
    out_drv = ogr.GetDriverByName(file_type)
    
    # remove output shapefile if it already exists
    if os.path.exists(outfile):
        out_drv.DeleteDataSource(outfile)
    
    # create the output shapefile
    ootds = out_drv.CreateDataSource(outfile)
    ootlyr = ootds.CreateLayer("extent", proj, geom_type=ogr.wkbPolygon)
    
    # add the fields
    # ogr.OFTInteger == 0, hence the arg
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    ootlyr.CreateField(idField)
    
    return ootds, ootlyr

def extent2poly(infile, filetype='raster', outfile=None, polytype="ESRI Shapefile", 
                   geecoord=False, lyrtype='ogr'):
    
    """
    Get the coordinates of a files extent and return an ogr polygon ring with 
    the option to save the  file
    
    
    Parameters
    ----------
    
    infile: string
            input ogr compatible geometry file or gdal raster
            
    filetype: string
            the path of the output file, if not specified, it will be input file
            with 'extent' added on before the file type
    
    outfile: string
            the path of the output file, if not specified, it will be input file
            with 'extent' added on before the file type
    
    polytype: string
            ogr comapatible file type (see gdal/ogr docs) default 'ESRI Shapefile'
            ensure your outfile string has the equiv. e.g. '.shp' or in case of 
            memory only 'Memory' (outfile would be None in that case)
    
    geecoord: bool
           optionally convert to WGS84 lat,lon
    
    lyrtype: string
            either 'gee' which means earth engine or 'ogr' which returns ds and lyr
           
    Returns
    -------
    
    a GEE polygon geometry or ogr dataset and layer
    
    """
    # gdal/ogr read in etc
    if filetype == 'raster':
        ext, rstref = _raster_extent(infile)
        
    else:
        # tis a vector
        vds = ogr.Open(infile)
        lyr = vds.GetLayer()
        ext = lyr.GetExtent()
    
    # make the linear ring 
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(ext[0],ext[2])
    ring.AddPoint(ext[1], ext[2])
    ring.AddPoint(ext[1], ext[3])
    ring.AddPoint(ext[0], ext[3])
    ring.AddPoint(ext[0], ext[2])
    
    # drop the geom into poly object
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    if geecoord == True:
        # Getting spatial reference of input 
        srs = lyr.GetSpatialRef()
    
        # make WGS84 projection reference3
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
    
        # OSR transform
        transform = osr.CoordinateTransformation(srs, wgs84)
        # apply
        poly.Transform(transform)
        
        tproj = wgs84
    if filetype == 'raster':
        tproj = rstref
    else:
        tproj = lyr.GetSpatialRef()
    
    # in case we wish to write it for later....    
#    if outfile != None:
#        outfile = infile[:-4]+'extent.shp'
    
    out_drv = ogr.GetDriverByName(polytype)
    
    # remove output shapefile if it already exists
    if outfile != None and polytype != 'Memory':
        if os.path.exists(outfile):
            out_drv.DeleteDataSource(outfile)
        ootds = out_drv.CreateDataSource(outfile)
    else:
        ootds = out_drv.CreateDataSource('out')

    ootlyr = ootds.CreateLayer("extent", tproj, geom_type=ogr.wkbPolygon)
    
    # add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    ootlyr.CreateField(idField)
    
    # create the feature and set values
    featureDefn = ootlyr.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(poly)
    feature.SetField("id", 1)
    ootlyr.CreateFeature(feature)
    feature = None
    
    # Save and close if not a memory driver
    
    ootds.FlushCache()
    ootds = None
    
    if lyrtype == 'gee':
        poly.FlattenTo2D()
        return poly
    elif lyrtype == 'ogr':
        return ootds, ootlyr


def shape_props(inShape, prop, inRas=None,  label_field='ID'):
    """
    Calculate various geometric properties of a set of polygons
    Output will be relative to geographic units where relevant, but normalised 
    where not (eg Eccentricity)
    
    Parameters 
    ----------
    
    inShape: string
              input shape file path

    inRas: string
            a raster to get the correct dimensions from (optional), required for
            scikit-image props
        
    prop: string
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
                 'Perimeter': 'Perim', 'AverageWidth':'AvWidth' }
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
        elif prop == 'AverageWidth':
            #(Diameter of a circle with the same perimeter as the polygon) 
            # * Area / (Area of a circle with the same perimeter as the polygon)
            # (perimeter / pi) * area / (perimeter**2 / (4*pi)) = 4 * area / perimeter
            # TODO - this may not write to shape as a tuple
            #((perim/pi) * area) / (perim**2 / (4 * pi))
            # seemingly simplified to 
            # area / perimeter * 4
            stat = (poly1.area / poly1.length) * 4 
                  
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
                if len(Props) == 0:
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

    x offset: int
           
    y offset: int
           
    xcount: int
             rows of bounding box
             
    ycount: int
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
    xcount = int((xmax - xmin)/pixelWidth)#+1
    ycount = int((ymax - ymin)/pixelWidth)#+1
#    originX = rgt[0]
#    originY = rgt[3]
#    pixel_width = rgt[1]
#    pixel_height = rgt[5]
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

def sqlfilter(inShp, sql):
    
    """ 
    Return an OGR layer via sql statement
    for some further analysis
    
    See https://gdal.org/user/ogr_sql_dialect.html for examples
    
    Notes
    -----

    An OS Master map example
    
    "SELECT * FROM TopographicArea WHERE DescriptiveGroup='General Surface'"
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    sql: string
                  sql expression (ogr dialect)
    Returns
    -------
    
    ogr lyr
          
    """
    vds = ogr.Open(inShp, 1)
    
    lyr = vds.ExecuteSQL(sql)
    
    return lyr
    
    
def filter_shp(inShp, expression, outField, outLabel):
    
    """ 
    Filter and index an OGR polygon file features by attribute
    
    Potentially useful for rule sets or prepping a subsiduary underlying
    raster operation
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    expression: string
                  expression e.g. "DN >= 168"
    
    outField: string
                  the field in which the label will reside
                  
    outLabel: int
        the label identifying the filtered features
    """
    
    vds = ogr.Open(inShp, 1) 

    lyr = vds.GetLayer(0)
    
    lyr.SetAttributeFilter(expression)
    
    feat = lyr.GetNextFeature()
    features = np.arange(lyr.GetFeatureCount())
    
    lyr.CreateField(ogr.FieldDefn(outField, ogr.OFTInteger))
    
    for label in tqdm(features):
        feat.SetField(outField, outLabel)
        lyr.SetFeature(feat)
        feat = lyr.GetNextFeature()
        
    lyr.SyncToDisk()

    vds = None
    
#def _deletefield(inShp, field):
#    
#    "dump a field"
#    cdir = os.getcwd()
#    os.chdir(hd)
#    
#    hd, tl = os.path.split(inShp)
#    
#    ds = gdal.OpenEx(inShp, gdal.OF_VECTOR | gdal.OF_UPDATE)
#    cmd = "ALTER TABLE "+tl+" DROP COLUMN "+field
#    ds.ExecuteSQL(cmd)
    
def _fieldexist(vlyr, field):
    """
    check a field exists
    """
    
    lyrdef = vlyr.GetLayerDefn()

    fieldz = []
    for i in range(lyrdef.GetFieldCount()):
        fieldz.append(lyrdef.GetFieldDefn(i).GetName())
    return field in fieldz

def geom2pixelbbox(inshp, inras, label="Tree", outfile=None):
    
    """
    Convert shapefile geometries to a df of pixel bounding boxes
    Projections must be the same!
    
    Parameters
    ----------
    
    inshp: string
                    input ogr compatible geometry
    
    inras: string
                    input raster
        
    label: string
                    label name def. 'Tree'
                    
    outfile: string
                    path to save annotation csv 
    """
    # inputs
    rds = gdal.Open(inras, gdal.GA_ReadOnly)
    rgt = rds.GetGeoTransform()
    # for reference as always forget
    originX = rgt[0]
    originY = rgt[3]
    pixel_width = rgt[1] # usually square but oh well
    pixel_height = rgt[5]

    # Should we need to revert to OGR
    vds = ogr.Open(inshp, 1) 
    vlyr = vds.GetLayer()
    # mem_drv = ogr.GetDriverByName('Memory')
    # driver = gdal.GetDriverByName('MEM')
    # Loop through vectors
    #feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    
    # # via ogr it may be quicker with gpd....here for ref
    
    # we only need the relative path not the full
    rds_path = os.path.split(inras)[1]
    
    ootlist = []
    for f in tqdm(features):
        feat = vlyr.GetFeature(f)
        if feat is None:
            continue
    #        debug
    #        wkt=geom.ExportToWkt()
    #        poly1 = loads(wkt)
        geom = feat.geometry()
        
        # tuple is (xmin, xmax, ymin, ymax)
        bbox = geom.GetEnvelope()
        
        # so (xmin - rasteroriginX) / pixel_width
        xmin = int((bbox[0] - originX) / pixel_width) #xmin
        xmax = int((bbox[1] - originX) / pixel_width) + 1 #xmax
    
        ymin = int((bbox[3] - originY) / pixel_height) #ymin
        ymax = int((bbox[2] - originY) / pixel_height) + 1 #ymax
        
        # order should be thus for annotation
        # image_path, xmin, ymin, xmax, ymax, label
        ootlist.append([rds_path, xmin, ymin, xmax, ymax, label])
        
    
    df = pd.DataFrame(data=ootlist, 
                      columns=["image_path", "xmin", "ymin",
                               "xmax", "ymax", 'label'])
    if outfile != None:
        df.to_csv(outfile)
        
    return df
        
        # to check reading raster
        #src_offset = _bbox_to_pixel_offsets(rgt, geom)
    
    # todo so much shorter and vectorised
    # gdf = gpd.read_file(inshp) 
    
    # bboxes = gdf.bounds
    
    # df["xmin"] = (bboxes['minx'] - originX) / pixel_width
    # df["xmax"] = (bboxes['maxx'] - originX) / pixel_width #+1
    # df["ymin"] = (bboxes['miny'] - originX) / pixel_height
    # df["ymax"] = (bboxes['maxy'] - originX) / pixel_height #+1
    # floating pint issues here - line below not working
    # df["xmin", "ymin", "xmax", "ymax"].round(0).astype(int)
    
    
def rasterext2poly(inras):
    
    ext, rstref = _raster_extent(inras)
        
    
    # make the linear ring 
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(ext[0],ext[2])
    ring.AddPoint(ext[1], ext[2])
    ring.AddPoint(ext[1], ext[3])
    ring.AddPoint(ext[0], ext[3])
    ring.AddPoint(ext[0], ext[2])
    
    # drop the geom into poly object
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    return poly

def zonal_stats(inShp, inRas, band, bandname, layer=None, stat = 'mean',
                write_stat=True, nodata_value=0, all_touched=True, 
                expression=None):
    
    """ 
    Calculate zonal stats for an OGR polygon file
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster

    band: int
           an integer val eg - 2

    bandname: string
               eg - blue
    layer: string
           if using a db type format with multi layers, specify the name of the
           layer in question
           
    stat: string
           string of a stat to calculate, if omitted it will be 'mean'
           others: 'mode', 'min','mean','max', 'std',' sum', 'count','var',
           skew', 'kurt (osis)', 'vol'
                     
    write_stat: bool (optional)
                If True, stat will be written to OGR file, if false, dataframe
                only returned (bool)
        
    nodata_value: numerical
                   If used the no data val of the raster
    
    all_touched: bool
                    whether to use all touched when raterising the polygon
                    if the poly is smaller/comaparable to the pixel size, 
                    True is perhaps the best option
    expression: string
                     process a selection only eg expression e.g. "DN >= 168"    
    """    
    # gdal/ogr-based zonal stats
    
    if all_touched == True:
        touch = "ALL_TOUCHED=TRUE"
    else:
        touch = "ALL_TOUCHED=FALSE"
        
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    #assert(rds)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1) 
    
    # if we are using a db of some sort gpkg etc where we have to choose
    if layer !=None:
        vlyr = vds.GetLayerByName(layer)
    else:
        vlyr = vds.GetLayer()
    
    if expression != None:
        vlyr.SetAttributeFilter(expression)
        fcount = str(vlyr.GetFeatureCount())    
        print(expression+"\nresults in "+fcount+" features to process")
    
    if write_stat != None:
        # if the field exists leave it as ogr is a pain with dropping it
        # plus can break the file
        if _fieldexist(vlyr, bandname) == False:
            vlyr.CreateField(ogr.FieldDefn(bandname, ogr.OFTReal))

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    rejects = list()
    
    #create a poly of raster bbox to test for within raster
    poly = rasterext2poly(inRas)
    
    #TODO FAR too many if statements in this loop.
    # This is FAR too slow
   # offs = []
   
    for label in tqdm(features):

        if feat is None:
            continue
#        debug
#        wkt=geom.ExportToWkt()
#        poly1 = loads(wkt)
        geom = feat.geometry()
        
        # if outside the raster
        src_offset = _bbox_to_pixel_offsets(rgt, geom)
        
       # This does not seem to be fullproof
       # This is a hacky mess that needs fixed
        if poly.Contains(geom) == False:
            #print(src_offset[0],src_offset[1])
            #offs.append()
            feat = vlyr.GetNextFeature()
            continue
        elif src_offset[0] > rds.RasterXSize:
            feat = vlyr.GetNextFeature()
            continue
        elif src_offset[1] > rds.RasterYSize:
            feat = vlyr.GetNextFeature()
            continue
        elif src_offset[0] < 0 or src_offset[1] < 0:
            feat = vlyr.GetNextFeature()
            continue
        
        if src_offset[0] + src_offset[2] > rds.RasterXSize:
                # needs to be the diff otherwise neg vals are possble
                xx = abs(rds.RasterXSize - src_offset[0])
                
                src_offset = (src_offset[0], src_offset[1], xx, src_offset[3])
        
        if src_offset[1] + src_offset[3] > rds.RasterYSize:
                 yy = abs(rds.RasterYSize - src_offset[1])
                 src_offset = (src_offset[0], src_offset[1],  src_offset[2], yy)
        
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
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1], options=[touch])
        rv_array = rvds.ReadAsArray()
        
        # Mask the source data array with our current feature using np mask     

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
        elif stat == 'count':
            feature_stats = int(masked.count())
        elif stat == 'perc':
            total = masked.shape[0]* masked.shape[1]
            perc = masked.count() / total
            feature_stats = int(np.round(perc*100))  
        elif stat == 'var':
            feature_stats = float(masked.var())
        elif stat == 'skew':
            feature_stats = float(skew(masked[masked.nonzero()]))
        elif stat == 'kurt':
            feature_stats = float(kurtosis(masked[masked.nonzero()]))
        elif stat == 'vol':
            # get vol per cell
            src_array[src_array==nodata_value]=0
            cellvol = src_array*rgt[1]*rgt[1]
            # then sum them
            feature_stats = float(cellvol.sum())
        else:
            raise ValueError("Must be one of mode, min, mean, max,"
                             "std, sum, count, var, skew, kurt, vol")               
        # You can't have the stat of a single value - this is not an ideal
        # solution - should be flagged somehow but
        if src_array.shape == (1,1):
            feature_stats=float(src_array[0])
            
        stats.append(feature_stats)
        if write_stat != None:
            feat.SetField(bandname, feature_stats)
            vlyr.SetFeature(feat)
        feat = vlyr.GetNextFeature()
        
    if write_stat != None:
        vlyr.SyncToDisk()



    vds = None
    rds = None
    frame = DataFrame(stats)
    
    if write_stat != None:
        return frame, rejects
    
def zonal_stats_all(inShp, inRas, bandnames, 
                    statList = ['mean', 'min', 'max', 'median', 'std',
                                'var', 'skew', 'kurt']):
    """ 
    Calculate zonal stats for an OGR polygon file
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster

    band: int
           an integer val eg - 2

    bandnames: list
               eg - ['b','g','r','nir']
        
    nodata_value: numerical
                   If used the no data val of the raster
        
    """    

# zonal stats
    for bnd,name in enumerate(bandnames):
    
        [zonal_stats(inShp, inRas, bnd+1, name+st, stat=st, write_stat = True) for st in statList]

def _set_rgb_ind(feat, rv_array, src_offset, rds, nodata_value):
    
    
    rgb = np.zeros((src_offset[3], src_offset[2], 3))
    
    for band in range(1, rds.RasterCount):
        
        rBnd = rds.GetRasterBand(band)
        
        rgb[:,:, band-1] = rBnd.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                                   src_offset[3])
        
                   
    
    
    
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
        

def zonal_rgb_idx(inShp, inRas, nodata_value=0):
    
    """ 
    Calculate RGB-based indicies per segment/AOI
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster
        
    nodata_value: numerical
                   If used the no data val of the raster
        
    """    
    #TODO ad other stat types - consider mask array for safety......
    
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    #assert(rds)
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats

    vlyr = vds.GetLayer(0)

    field_names = ['ExGmn', 'ExRmn', 'ExGRmn', 'CIVEmn', 'NDImn', 'RGBVImn', 'VARImn',
         'ARImn', 'RGBImn', 'GLImn', 'TGLmn']
    
    [vlyr.CreateField(ogr.FieldDefn(f, ogr.OFTReal)) for f in field_names]

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
#    stats = []
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
#    rejects = list()
    
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



    vds = None
    rds = None

def write_text_field(inShape, fieldName, attribute):
    
    """ Write a string to a ogr vector file
    
    Parameters
    ----------
    inShape: string
              input OGR vecotr file
        
    fieldName: string
                name of field being written
    
    attribute: string
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

def write_id_field(inShape, fieldName='id'):
    
    """ Write a string to a ogr vector file
    
    Parameters
    ----------
    inShape: string
              input OGR vecotr file
        
    fieldName: string
                name of field being written
    
        
    """
        
    vds = ogr.Open(inShape, 1)  # TODO maybe open update if we want to write stats
   #assert(vds)
    vlyr = vds.GetLayer(0)
    vlyr.CreateField(ogr.FieldDefn(fieldName, ogr.OFTInteger))
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    
    for label in tqdm(features):
        feat.SetField(fieldName, int(label))
        vlyr.SetFeature(feat)
        feat = vlyr.GetNextFeature()

    vlyr.SyncToDisk()
    vds = None    

def texture_stats(inShp, inRas, band, gprop='contrast',
                  offset=2,angle=0, write_stat=None, nodata_value=0, mean=False):
    
    """ 
    Calculate and optionally write texture stats for an OGR compatible polygon
    based on underlying raster values
    
    
    Parameters
    ----------
    inShp: string
                  input shapefile 
        
    inRas: string 
                  input raster path
        
    gprop: string
            a skimage gclm property 
            entropy, contrast, dissimilarity, homogeneity, ASM, energy,
            correlation
        
    offset: int
             distance in pixels to measure - minimum of 2!!!
        
    angle: int
            angle in degrees from pixel (int) 
                    
    mean: bool
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

    
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    #assert(rds)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats

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
            
            
        # Temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize

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
    
    inShp: string
                  input shapefile
        
    inRas: string
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

    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats

    vlyr = vds.GetLayer(0)

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
        
#        poly1 = loads(wkt)
        
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
        # dear skimage this function is deeply flawed.....grrrr
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
        
def ms_snake(inShp, inRas, outShp,  band=2, buf1=0, buf2=0, algo="ACWE", nodata_value=0,
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
    # TODO read rgb/3band in and convert 2 gray
    
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
#    tempRas = inShp[:-4]+'.tif'
    
#    rasterize(inShp, inRas, tempRas)
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
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster

    band: int
           an integer val eg - 2

    algo: string
           'otsu', niblack, sauvola
          
    nodata_value: numerical
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
    

    

    
    wkt=geom.ExportToWkt()
    poly1 = loads(wkt)
    
    if orient < np.pi:
        poly2 = rotate(poly1, np.pi-orient, use_radians=True)
    else:
        poly2 = rotate(poly1, np.pi+orient, use_radians=True)
    
    xmin, ymin, xmax, ymax = poly2.bounds
    

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

def zonal_point(inShp, inRas, field, band=1, nodata_value=0, write_stat=True):
    
    """ 
    Get the pixel val at a given point and write to vector
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster
    
    field: string
                    the name of the field

    band: int
           an integer val eg - 2
                            
    nodata_value: numerical
                   If used the no data val of the raster
        
    """    
    
   

    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats
    vlyr = vds.GetLayer(0)
    
    if write_stat != None:
        # if the field exists leave it as ogr is a pain with dropping it
        # plus can break the file
        if _fieldexist(vlyr, field) == False:
            vlyr.CreateField(ogr.FieldDefn(field, ogr.OFTReal))
    
    
    
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    
    for label in tqdm(features):
    
            if feat is None:
                continue
            
            # the vector geom
            geom = feat.geometry()
            
            #coord in map units
            mx, my = geom.GetX(), geom.GetY()  

            # Convert from map to pixel coordinates.
            # No rotation but for this that should not matter
            px = int((mx - rgt[0]) / rgt[1])
            py = int((my - rgt[3]) / rgt[5])
            
            
            src_array = rb.ReadAsArray(px, py, 1, 1)

            if src_array is None:
                # unlikely but if none will have no data in the attribute table
                continue
            outval =  int(src_array.max())
            
#            if write_stat != None:
            feat.SetField(field, outval)
            vlyr.SetFeature(feat)
            feat = vlyr.GetNextFeature()
        
    if write_stat != None:
        vlyr.SyncToDisk()



    vds = None
    rds = None
    
def mesh_from_raster(inras, outshp=None, band=1):
    
    img = raster2array(inras)
    
    cnt = img.shape[0]*img.shape[1]
    
    # must have non zero vals
    newarr = np.arange(1, cnt+1)
    newarr = newarr.reshape(img.shape)
    ootras = inras[:-4]+'mesh.tif'
    array2raster(newarr, 1, inras, ootras, dtype=5)
    
    if outshp == None:
        outshp = ootras[:-3]+'shp'
    polygonize(ootras, outshp, outField=None,  mask=True, band=1)

#essentially cookbook version

def buffer(inShp, outfile, dist):
    
    """ 
    Buffer a shapefile by a given distance outputting a new one
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    outfile: string
                  output shapefile
                  
    dist: float
                the distance in map units to buffer
    """
    
    inputds = ogr.Open(inShp)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outfile):
        shpdriver.DeleteDataSource(outfile)
    outputBufferds = shpdriver.CreateDataSource(outfile)
    bufferlyr = outputBufferds.CreateLayer(outfile, geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()

    for feat in tqdm(inputlyr):
        ingeom = feat.GetGeometryRef()
        geomBuffer = ingeom.Buffer(dist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None
    
    return outfile


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
