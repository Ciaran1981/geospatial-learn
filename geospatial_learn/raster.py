
# -*- coding: utf-8 -*-
"""
The raster module. 

Description
-----------

A series of tools for the manipulation of geospatial imagery/rasters such as
masking or raster algebraic type functions and the conversion of Sentinel 2 
data to gdal compatible formats.  

"""
from osgeo import gdal, ogr,  osr
import os
import numpy as np
from glob import glob
from geospatial_learn.gdal_merge import _merge
import tempfile
from tqdm import tqdm
import scipy.ndimage as nd
import subprocess
from skimage.morphology import  remove_small_objects#, remove_small_holes#disk, square, binary_dilation
from skimage.filters import rank
from skimage.exposure import rescale_intensity
import warnings
from os import sys, path
from shapely.wkt import loads
from shapely.geometry import Polygon, LineString
import matplotlib
from owslib.wms import WebMapService
from io import BytesIO#, StringIO
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from subprocess import run, PIPE, Popen
from subprocess import Popen
import geopandas as gpd
import imageio
import cv2
from PIL import Image, ImageFont, ImageDraw
#matplotlib.use('Qt5Agg')


gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

def arc_gdb_convert(gdb, outdir, virt=True):
    
    """
    Convert an esri raster gdb to a set of tifs and optionaly also virt
    requires https://github.com/r-barnes/ArcRasterRescue to be compiled
    seperately
    
    Parameters
    ----------
    
    gdb: string
         input gdb
         
    outdir: string
           output dir in which rasters and potentially the virtual will reside
    
    virt: bool
          whether to write a virtual raster
    
    """
    listcmd = ['arc_raster_rescue.exe', gdb+'/']

    # as out is bytes have to decode it
    varlist = run(listcmd, stdout=PIPE).stdout.decode('utf-8')
    
    tmplst = varlist.split()
    #  'Rasters'
    tmplst = tmplst[2:]
    
    #this is outstanding - must remember these things
    num = tmplst[::2]  # Start at first element, then every other.
    tiles = tmplst[1::2] # Start at second element, then every other.
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    tiles = [os.path.join(outdir, t+'.tif') for t in tiles]
    
    # we add the args to this in the loop
    execmd = ['arc_raster_rescue.exe', gdb+'/']
    
    # add to the cmds
    cmdlist = [execmd+[i,t] for i, t in zip(num,tiles)]
    

    # doesn't wait with run
    [run(c) for c in cmdlist]
    
    # the old way
#    procs = [Popen(c) for c in cmdlist]
#    for p in procs:
#       p.wait()
    
    vrt = os.path.split(gdb)[1][:-3] + 'vrt'
    write_vrt(tiles, os.path.join(outdir, vrt))
    
    
    

def batch_wms_download(gdf, wms, layer, outdir, attribute='id',
                       espg='27700', res=0.25):
    
    """
    Download a load of wms tiles with georeferencing
    
    Parameters
    ----------
    
    gdf: geopandas gdf
    
    wms: string 
        the wms addresss
    
    layer: string
        the wms layer
    
    espg: string
            the proj espg
    
    outfile: string
              path to outfile
    
    res: int
            per pixel resolution of imagery in metres
    
    """

    
    rng = np.arange(0, gdf.shape[0])
    
    # assuming each tile is the same size
    bbox = gdf.bounds.iloc[0].tolist()
    # for the img_size
    div = int(1 / res) # must be an int otherwise wms doesn't accept
    # in case it is not a fixed tile size for our aoi
    img_size = (int(bbox[2]-bbox[0])*div,  int(bbox[3]-bbox[1])*div)
    
    outfiles = [os.path.join(outdir, a+'.tif') for a in gdf.id.to_list()]
    
    _ = Parallel(n_jobs=gdf.shape[0],
             verbose=2)(delayed(wmsGrabber)(gdf.bounds.iloc[i].tolist(),
                        img_size, wms, layer,
                        outfiles[i], espg=espg, res=res) for i in rng)

def wmsGrabber(bbox, image_size, wms, layer, outfile, espg='27700', 
               res=0.25):
    
    """
    Return a wms tile from a given source and optionally write to disk with 
    georef
    
    Parameters
    ----------
    
    bbox: list or tuple
            xmin, ymin, xmax, ymax
    
    image_size: tuple
                image x,y dims 
    
    wms: string 
        the wms addresss
        
    layer: string 
        the wms (sub)layer    
    
    espg: string
            the proj espg
    
    outfile: string
              path to outfile, if None only array is returned
    
    """
    
    wms = WebMapService(wms, version='1.1.1')
    
    wms_img = wms.getmap(layers=[layer],
                        srs='EPSG:'+espg,
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        size=image_size,
                        format='image/png',
                        transparent=True
                        )

    f_io = BytesIO(wms_img.read())
    img = plt.imread(f_io)
    
    np2gdal = {"uint8": 1,"int8": 1,"uint16": 2,"int16": 3,
               "uint32": 4,"int32": 5, "float32": 6, 
               "float64": 7, "complex64": 10, "complex128": 11}
    
    
    if outfile != None:
        
        dtpe = np2gdal[str(img.dtype)]
        
        bbox2raster(img, img.shape[2], bbox, outfile, pixel_size=res,  
                    proj=int(espg), dtype=dtpe, FMT='Gtiff')
    
    return img

def bbox2raster(array, bands, bbox, outras, pixel_size=0.25,  proj=27700,
                dtype=5, FMT='Gtiff'):
    
    """
    Using a bounding box and other information georef an image and write to disk
    
    Parameters
    ----------      
    array: np array
            a numpy array.
    
    bands: int
            the no of bands.
    
    bbox: list or tuple
        xmin, ymin, xmax, ymax
    
    pixel_size: int
                pixel size in metres (unless proj is degrees!)
    
    outras: string
             the path of the output raster.
    
    proj: int
         the espg code eg 27700 for osgb
    
    dtype: int 
            though you need to know what the number represents!
            a GDAL datatype (see the GDAL website) e.g gdal.GDT_Int32 = 5
    
    FMT: string 
           (optional) a GDAL raster format (see the GDAL website) eg Gtiff, KEA.
    
    """
    # dimensions & ref coords
    x_pixels = array.shape[1]
    y_pixels = array.shape[0] 
    
    x_min = bbox[0]
    y_max = bbox[3]
    
    driver = gdal.GetDriverByName(FMT)
    
    # Set params for output raster
    ds = driver.Create(
         outras, 
         x_pixels,
         y_pixels,
         bands,
         dtype)

    ds.SetGeoTransform((
        x_min,        # rgt[0]
        pixel_size,   # rgt[1]
        0,            # rgt[2]
        y_max,        # rgt[3]
        0,            # rgt[4]
        -pixel_size)) # rgt[5]
    
    # georef
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(proj)    
    ds.SetProjection(srs.ExportToWkt())
    # Write 
    if bands == 1:
        ds.GetRasterBand(1).WriteArray(array)

    else:
    # Loop through bands - not aware of quicker way when writing
        for band in range(1, bands+1):
            ds.GetRasterBand(band).WriteArray(array[:, :, band-1])
    # Flush to disk
    ds.FlushCache()  
    ds=None
    
    


def array2raster(array, bands, inRaster, outRas, dtype, FMT=None):
    
    """
    Save a raster from a numpy array using the geoinfo from another.
    
    Parameters
    ----------      
    array: np array
            a numpy array.
    
    bands: int
            the no of bands. 
    
    inRaster: string
               the path of a raster.
    
    outRas: string
             the path of the output raster.
    
    dtype: int 
            though you need to know what the number represents!
            a GDAL datatype (see the GDAL website) e.g gdal.GDT_Int32
    
    FMT: string 
           (optional) a GDAL raster format (see the GDAL website) eg Gtiff, HFA, KEA.
        
    
    """

    if FMT == None:
        FMT = 'Gtiff'
        
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'    
    
    inras = gdal.Open(inRaster, gdal.GA_ReadOnly)    
    
    x_pixels = inras.RasterXSize  # number of pixels in x
    y_pixels = inras.RasterYSize  # number of pixels in y
    geotransform = inras.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    #if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inras.GetProjection()
    geotransform = inras.GetGeoTransform()   

    driver = gdal.GetDriverByName(FMT)

    dataset = driver.Create(
        outRas, 
        x_pixels,
        y_pixels,
        bands,
        dtype)

    dataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))    

    dataset.SetProjection(projection)
    if bands == 1:
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.FlushCache()  # Write to disk.
        dataset=None
        #print('Raster written to disk')
    else:
    # TODO - the entire n-dim array can be written rather than loop
    # Here we loop through bands
        for band in range(1,bands+1):
            Arr = array[:,:,band-1]
            dataset.GetRasterBand(band).WriteArray(Arr)
        dataset.FlushCache()  # Write to disk.
        dataset=None
        #print('Raster written to disk')
        
def raster2array(inRas, bands=None):
    
    """
    Read a raster and return an array, either single or multiband

    
    Parameters
    ----------
    
    inRas: string
                  input  raster 
                  
    bands: list or None
                  a list of bands to return in the array 
                  if None all bands are read and the axes will be (bands,x,y)
                  
    
    """
    rds = gdal.Open(inRas)
   
    if bands is None:
        #unsure if this is worth it - gdal order is fine right??
        inArray = np.moveaxis(rds.ReadAsArray(), 0, 2) 
        
    elif len(bands) ==1:
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
        for idx, band in enumerate(bands):  
            rA = rds.GetRasterBand(band).ReadAsArray()
            inArray[:, :, idx]=rA
   
   
    return inArray

def write_vrt(infiles, outfile):
    
    """
    Parameters
    ----------
    
    infiles: list
              a list of raster files
                             
    outfile: string
                the output .vrt

    """
    
    
    virtpath = outfile
    outvirt = gdal.BuildVRT(virtpath, infiles)
    outvirt.FlushCache()
    outvirt=None


def tile_rasters(inRas, outDir, tilesize = ["256", "256"], overlap='0'): 
    
    """ 
    Split a large raster into smaller ones
    
    Parameters
    ----------        
    inRas: string
              the path to input raster
    
    outDir: string
                  the path to the output dir
    
    tilesize: list of str
               the sides of a square tile ["256", "256"]
    overlap: string
            should a overlap per tile be required
        
    """
    

    #TODO use the API directly
    cmd = ["gdal_retile.py", inRas, "-ps", tilesize[0], tilesize[1], "-overlap",
           overlap,
                              "-targetDir", outDir]
    subprocess.call(cmd)


def batch_translate(folder, wildcard, FMT='Gtiff'):
    """ 
    Using the gdal python API, this function translates the format of files
    to commonly used formats
    
    Parameters
    -----------         
    folder: string
             the folder containing the rasters to be translated
    
    wildcard: string
               the format wildcard to search for e.g. '.tif'
    
    FMT : string (optional)
          a GDAL raster format (see the GDAL website) eg Gtiff, HFA, KEA
        
    
    """
    
    if FMT == None:
        FMT = 'HFA'
        fmt = 'img'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'    
        
    fileList = glob2.glob(path.join(folder,'*'+wildcard))
    outList = list()
    files = np.arange(len(fileList))
    
    
    for file in tqdm(files): 
        src_filename = fileList[file]

        #Open existing datasetbatch
        src_ds = gdal.Open(src_filename)
        
        #Open output format driver, see gdal_translate --formats for list
        driver = gdal.GetDriverByName(FMT)
        outList.append(src_filename[:-4]+fmt)
        dst_filename = outList[file]
        #Output to new format
        dst_ds = driver.CreateCopy(dst_filename, src_ds, 0)
        dst_ds.FlushCache()
        #Properly close the datasets to flush to disk
        dst_ds = None
        src_ds = None
        #print(outList[file]+' done')

def batch_translate_adf(inlist):
    
    """
    batch translate a load of adf (arcgis) files from some format to tif
    
    Parameters
    ----------
    
    inlist: string
        A list of raster paths
    
    Returns
    -------
    
    List of file paths
    
    """
    outpths = []
    
    for i in tqdm(inlist):
        hd, _ = os.path.split(i)
        ootpth = hd+".tif"
        srcds = gdal.Open(i)
        out = gdal.Translate(ootpth, srcds)
        out.FlushCache()
        out = None
        outpths.append(ootpth)
    return outpths

def batch_gdaldem(inlist, prop='aspect'):
    
    """
    batch dem calculation a load of gdal files from some format to tif
    
    Parameters
    ----------
    
    inlist: string
        A list of raster paths
    
    prop: string
        one of "hillshade", "slope", "aspect", "color-relief", "TRI",
        "TPI", "Roughness"
    
    Returns
    -------
    
    List of file paths
    
    """
    
    outpths = []
    
    for i in tqdm(inlist):
        
        ootpth = i[:-4]+prop+".tif"
        srcds = gdal.Open(i)
        out = gdal.DEMProcessing(ootpth, srcds, prop)
        out.FlushCache()
        out = None
        outpths.append(ootpth)
    return outpths

def srtm_gdaldem(inlist, prop='aspect'):
    
    
    """
    Batch dem calculation a load of srtm files 
    
    SRTM scale & z factor vary across the globe so this calculates based on 
    latitude
    
    Parameters
    ----------
    
    inlist: string
        A list of raster paths
    
    prop: string
        one of "hillshade", "slope", "aspect", "color-relief", "TRI",
        "TPI", "Roughness"
    
    Returns
    -------
    
    List of file paths
    
    """
    
    
    # There is likely a more susinct way to do this but it works....
    # eg pretty sure the centroid can be done with numpy....
    outpths = []
    
    for i in tqdm(inlist):
        
        ootpth = i[:-4]+prop+".tif"
        srtm = gdal.Open(i)
        
        
        # the geotransform contains the top left, bottom right coords we need
        ext = srtm.GetGeoTransform()
        
        # make an OGR geom from the layer extent
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(ext[0],ext[2])
        ring.AddPoint(ext[1], ext[2])
        ring.AddPoint(ext[1], ext[3])
        ring.AddPoint(ext[0], ext[3])
        ring.AddPoint(ext[0], ext[2])
        
        # drop the geom into poly object
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        # load as a shapely geometry as OGR centroid returns nowt
        poly1 = loads(poly.ExportToWkt())
        # get the centroid
        cent=poly1.centroid
        lon, lat = cent.coords[0]
        
        # the scale factor based on the latitude derived from the centre of the 
        # tile (111320 is len of 1 degree at equator)
        s = 111320 * np.cos(lat * np.pi/180)
        z = 1 / (111320 * np.cos(lat * np.pi/180))
        
        # create the output dataset and cal slope
        out = gdal.DEMProcessing(ootpth, srtm, prop, zFactor=z, scale=s)
        # to disk
        out.FlushCache()
        # deallocate
        out = None
        outpths.append(ootpth)
        

def _bbox_to_pixel_offsets(rgt, geom):
    
    """ 
    Internal function to get pixel geo-locations of bbox of a polygon
    
    Parameters
    ----------
    
    rgt: array
         raster geotransform
          
    geom: shapely.geometry
           Structure defining geometry
    
    Returns
    -------
    xoffset: int
           
    yoffset: iny
           
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
    xcount = int((xmax - xmin)/pixelWidth)#+1#??
    ycount = int((ymax - ymin)/pixelWidth)#+1#?? - was this a hack?

    return (xoff, yoff, xcount, ycount)

def _raster_extent2poly(inras):
    
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
    
    # make the linear ring -
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minx, miny)
    ring.AddPoint(maxx, miny)
    ring.AddPoint(maxx, maxy)
    ring.AddPoint(minx, maxy)
    ring.AddPoint(minx, miny) # the 5th is here as the ring must be closed
    
    # drop the geom into poly object
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    return poly, spref, ext


    tproj = lyr.GetSpatialRef()
    
def _extent2lyr(infile, filetype='raster', outfile=None, 
                polytype="ESRI Shapefile", geecoord=False, lyrtype='ogr'):
    
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
        _, rstref, ext = _raster_extent2poly(infile)
        # where ext is 
        xmin, ymin, xmax, ymax = ext # readable
        
    else:
        # tis a vector
        vds = ogr.Open(infile)
        lyr = vds.GetLayer()
        ext = lyr.GetExtent()
    
    # make the linear ring -
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xmin, ymin)
    ring.AddPoint(xmax, ymin)
    ring.AddPoint(xmax, ymax)
    ring.AddPoint(xmin, ymax)
    ring.AddPoint(xmin, ymin) # the 5th is here as the ring must be closed
    
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
    
    if outfile != None and polytype != 'Memory':
        ootds = None
    
    if lyrtype == 'gee':
        poly.FlattenTo2D()
        return poly
    elif lyrtype == 'ogr':
        return ootds, ootlyr
    

def mask_with_poly(inshp, inras, layer=True, value=0, mtype='inside'):
    
    """ 
    Change raster values inside a polygon and update the raster
    
    Geometries must intersect!!!
    
    If done with layer=True entire raster will be read in at once and
    masked with layer, otherwise (False), each geometry will be read seperately
    and only that area read in. 
    
    Parameters
    ----------
    
    vector_path: string
                  input shapefile
        
    raster_path: string
                  input raster
    
    layer: bool
           whether to use the entire vector file as a mask (True) or loop through
           geometries seperately (False)
    
    value: int
            the value to alter
    
    mtype: str
            either area 'inside' or 'outside' of polygon is masked
    """    
    
    rds = gdal.Open(inras, gdal.GA_Update)
    rgt = rds.GetGeoTransform()
    bands = rds.RasterCount
    
    vds = ogr.Open(inshp, 1)  
    vlyr = vds.GetLayer(0)

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    features = np.arange(vlyr.GetFeatureCount())
    
    rds_ext, spref, ext = _raster_extent2poly(inras)
    
    if layer == True:
        
        # What a mission - but more efficient for sure
        # TODO make clip shp with shp func from this
        ootds, ootlyr = _extent2lyr(inras, polytype='Memory')
        clipds, cliplyr = create_ogr_poly('out', spref.ExportToWkt(),
                                 file_type="Memory", field="id", 
                                 field_dtype=0)
        #self method result
        ogr.Layer.Clip(vlyr, ootlyr, cliplyr) # it works.....
        
        # debug
        #poly1 = loads(rds_ext.ExportToWkt())
        #feat = cliplyr.GetFeature(0)
        # geom2 = feat.GetGeometryRef()
        #wkt=geom2.ExportToWkt()
        #poly2 = loads(wkt)
        
        # dataset to put the rasterised into
        rvlyr = _copy_dataset_config(rds, FMT='MEM', outMap='copy',
                                   dtype=gdal.GDT_Byte, bands=1)

        gdal.RasterizeLayer(rvlyr, [1], cliplyr, burn_values=[1])
        rv_array = rvlyr.ReadAsArray()
        src_array = rds.ReadAsArray()
        # broadcast to 3d - nice!
        d_mask = np.broadcast_to(rv_array==1, src_array.shape)
        if mtype == 'inside':
            src_array[d_mask==1]=value
        elif mtype == 'outside':
            src_array[d_mask!=1]=value       
        rds.WriteArray(src_array)
    
    else:
        # TODO perhaps get rid or adapt to where a select poly is wanted
    
        for label in tqdm(features):
            feat = vlyr.GetNextFeature()
    
            if feat is None:
                continue
            geom = feat.geometry()
            
            # the poly may be partially outside the raster
            if rds_ext.Intersects(geom) == False:
                continue
            interpoly = rds_ext.Intersection(geom)
    
            src_offset = _bbox_to_pixel_offsets(rgt, interpoly)
            
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
            featureDefn = mem_layer.GetLayerDefn()
            feature = ogr.Feature(featureDefn)
            feature.SetGeometry(interpoly)
            mem_layer.CreateFeature(feature)
            #mem_layer.CreateFeature(feat.Clone()) # if were using orig geom
    
            # Rasterize it
    
            rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
         
            rvds.SetGeoTransform(new_gt)
            rvds.SetProjection(rds.GetProjectionRef())
            rvds.SetGeoTransform(new_gt)
            gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
            rv_array = rvds.ReadAsArray()
            
            # This could get expensive mem wise for big images
            for band in range(1, bands+1):
                bnd = rds.GetRasterBand(band)
                src_array = bnd.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                                   src_offset[3])
                
                src_array[rv_array>0]=value
                bnd.WriteArray(src_array, src_offset[0], src_offset[1])
            
    rds.FlushCache()
        

    vds = None
    rds = None
    
    


def mask_raster(inputIm, mval, maskIm=None, outputIm=None,
                    blocksize=256, FMT=None):
    """ 
    Perform a numpy masking operation on a raster where all values
    corresponding to mask value are retained - does this in blocks for
    efficiency on larger rasters. Use of an external mask is optional. 
    
    Parameters 
    ----------- 
    
    inputIm: string
              the input raster
        
    mval: int
           the mask value eg 1, 2 etc
    
    mask: string
           path to optional mask image must be same size and extent, is assumed
           to be 2d e.g. one band (will be broadcast to 3D if required)

    FMT: string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    outputIm: string (optional)
               optionally write a separate output image, if None, will mask the input
        
    blocksize: int
                the chunk of raster to read in
        
    Returns
    ----------- 
    string
          A string of the output file path
        
    """
    #TODO - horrid old mess of if else statements
    if FMT == None:
        FMT = 'Gtiff'
        fmt = '.tif'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'
    
    if outputIm is None:
        print('Overwriting input file')
        inDataset = gdal.Open(inputIm, gdal.GA_Update)

    else:
        print('Creating output file')
        inDataset = gdal.Open(inputIm)
        dt = inDataset.GetRasterBand(1).DataType
        outDataset = _copy_dataset_config(inDataset, outMap=outputIm, dtype=dt,
                                     bands=inDataset.RasterCount)

    if maskIm is not None:
        print('Using external mask raster')
        mskds = gdal.Open(maskIm)
        mskbnd = mskds.GetRasterBand(1)
        
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = bnd.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize
    
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
                # read all bands as a block
                array = inDataset.ReadAsArray(j, i, numCols, numRows)
                
                if maskIm is not None:
                    marray = mskbnd.ReadAsArray(j, i, numCols, numRows)
                    marray = np.broadcast_to(marray, array.shape)
                    array[marray != mval]=0
                else:
                    array[array != mval]=0
                if outputIm is None:
                    inDataset.WriteArray(array, j, i)
                else:
                    outDataset.WriteArray(array, j, i)

    if outputIm is None:
        inDataset.FlushCache()
        inDataset = None
    else:                        
        outDataset.FlushCache()
        outDataset = None
     
def mask_raster_multi(inputIm,  mval=1, rule='==', outval = None, mask=None,
                    blocksize = 256, FMT = None, dtype=None):
    """ 
    Perform a numpy masking operation on a raster where all values
    corresponding to, less than or greater than the mask value are retained 
    - does this in blocks for efficiency on larger rasters
    
    Parameters 
    ----------- 
    
    inputIm: string
              the granule folder 
        
    mval: int
           the masking value that delineates pixels to be kept
    
    rule: string
            the logic for masking either '==', '<' or '>'
        
    outval: numerical dtype eg int, float
              the areas removed will be written to this value default is 0
        
    mask: string
            the mask raster to be used (optional)
        
    FMT: string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    mode: string
           None > 10m data, '20' >20m
        
    blocksize: int
                the chunk of raster read in & write out

    """

    if FMT == None:
        FMT = 'Gtiff'
        fmt = '.tif'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'
    
    
    if outval == None:
        outval = 0
    
    inDataset = gdal.Open(inputIm, gdal.GA_Update)
    bands = inDataset.RasterCount
    
    bnnd = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize


    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = bnnd.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize
    
    if mask != None:
        msk = gdal.Open(mask)
        maskRas = msk.GetRasterBand(1)
        
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
                mask = maskRas.ReadAsArray(j, i, numCols, numRows)
                if mval not in mask:
                    array = np.zeros(shape=(numRows,numCols), dtype=np.int32)
                    for band in range(1, bands+1):
                        bnd = inDataset.GetRasterBand(band)
                        bnd.WriteArray(array, j, i)
                else:
                    
                    for band in range(1, bands+1):
                        bnd = inDataset.GetRasterBand(band)
                        array = bnd.ReadAsArray(j, i, numCols, numRows)
                        if rule == '==':
                            array[array != mval]=0
                        elif rule == '<':
                            array[array < mval]=0
                        elif rule == '>':
                            array[array > mval]=0
                        bnd.WriteArray(array, j, i)
                        
    else:
             
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
                    for band in range(1, bands+1):
                        bnd = inDataset.GetRasterBand(1)
                        array = bnd.ReadAsArray(j, i, numCols, numRows)
                        if rule == '==':
                            array[array != mval]=0
                        elif rule == '<':
                            array[array < mval]=0
                        elif rule == '>':
                            array[array > mval]=0
                        if outval != None:
                            array[array == mval] = outval     
                            bnd.WriteArray(array, j, i)

           
        inDataset.FlushCache()
        inDataset = None

def calc_ndvi(inputIm, outputIm, bandsList, blocksize = 256, FMT = None, dtype=None):
    """ 
    Create a copy of an image with an ndvi band added
    
    Parameters 
    ----------- 
    
    inputIm: string
              the granule folder 
        
    bands: list
            a list of band indicies to be used, eg - [3,4] for Sent2 data

    FMT: string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    blocksize: int
                the chunk of raster read in & write out
    

    """

    if FMT == None:
        FMT = 'Gtiff'
        fmt = '.tif'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'
    
    if dtype is None:
        dtype = gdal.GDT_Float32
        
    inDataset = gdal.Open(inputIm, gdal.GA_Update)
    
    bands = int(inDataset.RasterCount+1)
    
    bnnd = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize

    outDataset = _copy_dataset_config(inDataset, outMap=outputIm,
                                     bands=bands, dtype=dtype)
    
    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = bnnd.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize
         
    bR = inDataset.GetRasterBand(bandsList[0])
    bNIR = inDataset.GetRasterBand(bandsList[1])
    
                
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
                for band in range(1, bands):
                    inbnd = inDataset.GetRasterBand(band)
                    array = inbnd.ReadAsArray(j, i, numCols, numRows)
                    outbnd = outDataset.GetRasterBand(band)
                    outbnd.WriteArray(array, j , i)
        
                del inbnd, array, outbnd 
                
                aR = bR.ReadAsArray(j, i, numCols, numRows)
                aNIR = bNIR.ReadAsArray(j, i, numCols, numRows)
                ndvi = (aNIR - aR) / (aNIR + aR) 
                
                bnd = outDataset.GetRasterBand(bands)
                bnd.WriteArray(ndvi, j, i)
           
    outDataset.FlushCache()
    outDataset = None

def rgb_ind(inputIm, outputIm, blocksize = 256, FMT = None, 
            dtype=gdal.GDT_Int32):
    """ 
    Create a copy of an image with rgb indices added
    Parameters 
    ----------- 
    
    inputIm: string
              the input rgb image
        
    outputIm: string
            the output image

    FMT: string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    blocksize: int
                the chunk of raster read in & write out
    

    """

    if FMT == None:
        FMT = 'Gtiff'
        fmt = '.tif'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'
    
    if dtype is None:
        dtype = gdal.GDT_Byte
        
    inDataset = gdal.Open(inputIm, gdal.GA_Update)
    
    bands = int(inDataset.RasterCount)
    
    bnnd = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize

    outDataset = _copy_dataset_config(inDataset, outMap=outputIm,
                                     bands=bands, dtype=dtype)
    
    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = bnnd.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize         
    
                
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
                rgb = np.zeros((numRows, numCols, 3))
                
                for band in range(1, bands+1):
                    inbnd = inDataset.GetRasterBand(band)
                    rgb[:,:, band-1] = inbnd.ReadAsArray(j, i, numCols, numRows)
                
                # ugly and efficient for now
                r = rgb[:,:,0]
                g = rgb[:,:,1]
                b = rgb[:,:,2]
    
                del rgb
    
                r = r / (r+g+b) * 10000
                g = g / (r+g+b) * 10000
                b = b / (r+g+b) * 10000
                outList = [r,g,b]
                del r, g, b
                for band in range(1, bands+1):
                    outbnd = outDataset.GetRasterBand(band)
                    outbnd.WriteArray(outList[band-1], j , i)

    outDataset.FlushCache()
    outDataset = None



def stack_ras(rasterList, outFile):
    """ 
    Stack some rasters 
    
    Parameters
    ----------- 
        
    rasterList: string
             the input image 
        
    outFile: string
              the output file path including file extension
        

    """
    _merge(names=rasterList, out_file=outFile)
    
def combine_scene(scl, c_scn, blocksize = 256):
    """ 
    Combine another scene classification with the sen2cor one
    
    Parameters 
    ----------
    scl: string
        the sen2cor one

    c_scn: string
        the independently derived one - this will be modified
    
    blocksize: string
        chunck to process
        
    """

    
    inDataset = gdal.Open(c_scn, gdal.GA_Update)
    
    sclDataset = gdal.Open(scl)
    bnnd = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize

    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = bnnd.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize
    
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
#                for band in range(1, bands+1):
                bnd = inDataset.GetRasterBand(1)
                scnBnd = sclDataset.GetRasterBand(1)
                scArray = scnBnd.ReadAsArray(j, i, numCols, numRows)
                array = bnd.ReadAsArray(j, i, numCols, numRows)
                # cloud
                array[scArray > 6]=2
                # converting water to land to avoid loss of pixels in buffer
                # later when getting rid of cloud/shadow
                array[np.logical_or(scArray == 6, array==4)]=3
                bnd.WriteArray(array, j, i)

    inDataset.FlushCache()
    inDataset = None
    
def polygonize(inRas, outPoly, outField=None,  mask = True, band = 1, 
               filetype="ESRI Shapefile"):
    
    """ 
    Polygonise a raster

    Parameters
    -----------   
      
    inRas: string
            the input image   
        
    outPoly: string
              the output polygon file path 
        
    outField: string (optional)
             the name of the field containing burnded values

    mask: bool (optional)
            use the input raster as a mask

    band: int
           the input raster band
            
    """    
    
    #TODO investigate ways of speeding this up   

    options = []
    src_ds = gdal.Open(inRas)
    if src_ds is None:
        print('Unable to open %s' % inRas)
        sys.exit(1)
    
    try:
        srcband = src_ds.GetRasterBand(band)
    except RuntimeError as e:
        # for example, try GetRasterBand(10)
        print('Band ( %i ) not found')
        print(e)
        sys.exit(1)
    if mask == True:
        maskband = src_ds.GetRasterBand(band)
        options.append('-mask')
    else:
        mask = False
        maskband = None
    
#    srs = osr.SpatialReference()
#    srs.ImportFromWkt( src_ds.GetProjectionRef() )
    
    ref = src_ds.GetSpatialRef()
    #
    #  create output datasource
    #
    dst_layername = outPoly
    drv = ogr.GetDriverByName(filetype)
    dst_ds = drv.CreateDataSource( dst_layername)
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = ref )
    
    if outField is None:
        dst_fieldname = 'DN'
        fd = ogr.FieldDefn( dst_fieldname, ogr.OFTInteger )
        dst_layer.CreateField( fd )
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(dst_fieldname)

    
    else: 
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(outField)

    gdal.Polygonize(srcband, maskband, dst_layer, dst_field,
                    callback=gdal.TermProgress)
    dst_ds.FlushCache()
    
    srcband = None
    src_ds = None
    dst_ds = None
    
def raster2points(inras, outpoints=None, field="value", field_dtype=0, band=1,
                  no_data=0):
    
    """
    Convert a raster to a points shapefile
    
    Parameters
    -----------   
      
    inras: string
            the input image   
        
    outpoints: string
              the output points file path 
        
    field: string (optional)
             the name of the field to write the values to
    
    field_dtype: int or ogr.OFT.....
            ogr dtype of field e.g. 0 == ogr.OFTInteger, 2 == OFTReal
             
    band: int
           the input raster band
    
    no_data: float / list of floats
            a value to ignore (e.g -9999 = nodata in the raster)
    
    """
    
    rds = gdal.Open(inras)
    
    cols = int(rds.RasterXSize)
    rows = int(rds.RasterYSize)
    
    rgt = rds.GetGeoTransform()
    
    # It is easier to derive coords for the whole raster than figuring
    # it out from a chunk as the chunk starts from zero (but in the middle
    # of the image somewhere)
    rowind, colind = np.indices((rows, cols))

    # Flatten ahin first
    xcoord = colind * rgt[1] + rgt[0]
    xcoord.shape = (cols*rows)
    ycoord = rowind * rgt[5] + rgt[3]
    ycoord.shape = (cols*rows)
    
    img = rds.GetRasterBand(band).ReadAsArray()
    img.shape = (cols*rows)
    
    
    # Now we remove nodata vals prior to loop
    # A mess but there could be more thgan one value
    if isinstance(no_data, list):
        indlist = []
        for n in no_data:
            indz = np.where(img==n)
            indlist.append(indz)
        inds  = np.hstack(indlist)  
    else:        
        inds = np.where(img==no_data)
    img = np.delete(img, inds, axis=0)
    xcoord = np.delete(xcoord, inds, axis=0)
    ycoord = np.delete(ycoord, inds, axis=0)
    
    if outpoints is None:
        outpoints = inras[:-3]+'shp'
        
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vds = driver.CreateDataSource(outpoints)
    
    ref = rds.GetSpatialRef()
    lyr = vds.CreateLayer('ogr_pts', ref, ogr.wkbPoint)
    lyrDef = lyr.GetLayerDefn()
    fldDef = ogr.FieldDefn(field, field_dtype)
    lyr.CreateField(fldDef)
    
    cnt = np.arange(0, img.shape[0])
    
    for c in tqdm(cnt):
        #coords
        point = ogr.Geometry(ogr.wkbPoint)
        point.SetPoint(0, xcoord[c], ycoord[c])
        # feat admin
        feat = ogr.Feature(lyrDef)
        feat.SetGeometry(point)
        feat.SetFID(c+1)
        feat.SetField(field, img[c])
        lyr.CreateFeature(feat)
    
    vds.FlushCache()
    vds = None
    rds = None

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

def set_bandnames(inras, names):
    
    """
    Set bandnames from a list
    
    Parameters
    ----------
    
    inras: str
            input raster path
    
    names: list
            list of band names
    
    """
    rds = gdal.Open(inras, gdal.GA_Update)
    bands = np.arange(1, rds.RasterCount+1).tolist()
    for b, n in zip(bands, names):
        rb = rds.GetRasterBand(b)
        rb.SetDescription(n)
    rds.FlushCache()
    rds = None
    
def rasterize(inShp, inRas, outRas, field=None, fmt="Gtiff"):
    
    """ 
    Rasterize a polygon to the extent & geo transform of another raster


    Parameters
    -----------   
      
    inRas: string
            the input image 
        
    outRas: string
              the output polygon file path 
        
    field: string (optional)
             the name of the field containing burned values, if none will be 1s
    
    fmt: the gdal image format
    
    """
    
    
    
    inDataset = gdal.Open(inRas)
    
    # the usual 
    
    outDataset = _copy_dataset_config(inDataset, FMT=fmt, outMap=outRas,
                         dtype = gdal.GDT_Int32, bands=1)
    
    
    vds = ogr.Open(inShp)
    lyr = vds.GetLayer()
    
    
    if field == None:
        gdal.RasterizeLayer(outDataset, [1], lyr, burn_values=[1])
    else:
        gdal.RasterizeLayer(outDataset, [1], lyr, options=["ATTRIBUTE="+field])
    
    outDataset.FlushCache()
    
    outDataset = None

def tile_raster(inRas, inShp, outdir, attribute='TILE_NAME', tiles=None,
                virt=True):
    
    """
    Parameters
    ----------
    
    """
    
           
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    
    # easier with gpd?  
    gdf = gpd.read_file(inShp)
    extent = gdf.bounds
    
    # gpd & ogr differences
    #gpd minx miny maxx maxy
    #ogr minx maxx miny maxy
    
    # oddly the warp function takes it in the gpd order, whereas if reading from
    # ogr we have to swap about - here for ref
    #         # minx      miny        maxx       maxy 
    # for ogr
    #extent = [extent[0], extent[2], extent[1], extent[3]]
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    if tiles is not None:
        outlist = tiles
    else:
        outlist = gdf[attribute].to_list()
    
    finalist = [os.path.join(outdir, o+'.tif') for o in outlist]
    
    # SLOW........but shared mem object
    for i, f in tqdm(enumerate(finalist)):
        ext = extent.iloc[i].to_list()
        ootds = gdal.Warp(f,
                          rds,
                          format='GTiff',
                          outputBounds=ext)
        ootds.FlushCache()
        ootds = None
    
    
    if virt is True:
        vrt = os.path.split(inRas)[1][:-3] + 'vrt'
        write_vrt(finalist, os.path.join(outdir, vrt))
    
def clip_raster_sel(inRas, inShp, outRas, field, attribute):

    """
    Clip a raster with a polygon selected by field (column) and attribute
    
    Parameters
    ----------
        
    inRas: string
            the input image 
            
    outPoly: string
              the input polygon file path 
        
    outRas: string
             the clipped raster
             
    field: string 
             the field/column to select by    

    attribute: string 
             the attribute/row value to select by              
   
    """
    #TODO - merge with below? - poss require sql sel from shp module...

    gdf = gpd.read_file(inShp)
    extent = gdf.bounds
           
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    
    #ugly
    ext = extent.iloc[gdf[gdf[field]==attribute].index[0]].tolist()        

    print('Clipping')
    ootds = gdal.Warp(outRas,
              rds,
              format='GTiff', 
              outputBounds=ext,
              callback=gdal.TermProgress)
              
        
    ootds.FlushCache()
    ootds = None
    rds = None 
    

def clip_raster(inRas, inShp, outRas, cutline=False, fmt='GTiff'):

    """
    Clip a raster
    
    Parameters
    ----------
        
    inRas: string
            the input image 
            
    inShp: string
              the input polygon file path 
        
    outRas: string (optional)
             the clipped raster
             
    cutline: bool (optional)
             retain raster values only inside the polygon       
            
   
    """
    
    if cutline != True:
        
        vds = ogr.Open(inShp)
        rds = gdal.Open(inRas, gdal.GA_ReadOnly)
        lyr = vds.GetLayer()
              
        rds_ext, spref, ext = _raster_extent2poly(inRas)
        
        ootds, ootlyr = _extent2lyr(inRas, polytype='Memory')
        clipds, cliplyr = create_ogr_poly('out', spref.ExportToWkt(),
                                 file_type="Memory", field="id", 
                                 field_dtype=0)
        #self method result
        ogr.Layer.Clip(lyr, ootlyr, cliplyr) # it works.....
        
        # debug
        # cliplyr.GetExtent()
        #poly1 = loads(rds_ext.ExportToWkt())
        #feat = cliplyr.GetFeature(0)
        #geom2 = feat.GetGeometryRef()
        #wkt=geom2.ExportToWkt()
        #poly2 = loads(wkt)
        
        # VERY IMPORTANT - TO AVOID COMING ACROSS THIS AGAIN
        # The problem iootrns[0]s that the bounds are not in the 'correct' order for gdalWarp
        # if taken from GetExtent() - they should in fact be in shapely order
        wrng = cliplyr.GetExtent()
        extent = [wrng[0], wrng[2], wrng[1], wrng[3]]
    
    
        print('cropping')
        ootds = gdal.Warp(outRas,
                  rds,
                  format=fmt, 
                  outputBounds=extent)
                  
            
        ootds.FlushCache()
        ootds = None
        rds = None

    
    else:
        #cutline == True:
        
        rds1 = gdal.Open(outRas, gdal.GA_Update)
        rasterize(inShp, outRas, outRas[:-4]+'mask.tif', field=None,
                  fmt="Gtiff")
        
        mskds = gdal.Open(outRas[:-4]+'mask.tif')
        
        mskbnd = mskds.GetRasterBand(1)

        cols = mskds.RasterXSize
        rows = mskds.RasterYSize

        blocksizeX = 256
        blocksizeY = 256
        
        bands = rds1.RasterCount
        
        mskbnd = mskds.GetRasterBand(1)
        
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
                    #for band in range(1, bands+1):
                        
                    #bnd = rds1.GetRasterBand(band)
                    array = rds1.ReadAsArray(j, i, numCols, numRows)
                    mask = mskbnd.ReadAsArray(j, i, numCols, numRows)
                    if len(array.shape)==2:
                        array[mask!=1]=0
                    else:
                        d_mask = np.broadcast_to(mask==1, array.shape)
                        array[d_mask!=1]=0
                    rds1.WriteArray(array, j, i)
                        
        rds1.FlushCache()
        rds1 = None
        

def fill_nodata(inRas, maxSearchDist=5, smoothingIterations=1, 
                bands=[1]):
    
    """
    fill no data using gdal
    
    Parameters
    ----------
    
    inRas: string
              the input image 
            
    maxSearchDist: int
              max search dist to fill
        
    smoothingIterations: int 
             the clipped raster
             
    bands: list of ints
            the bands to process      
    
    """
    
    rds = gdal.Open(inRas, gdal.GA_Update)
    
    for band in tqdm(bands):
        bnd = rds.GetRasterBand(band)
        gdal.FillNodata(targetBand=bnd, maskBand=None, 
                         maxSearchDist=maxSearchDist, 
                         smoothingIterations=smoothingIterations)
    
    rds.FlushCache()
    
    rds=None

def color_raster(inRas, color_file, output_file):
    """ 
    Generate a txt colorfile and make a RGB image from a grayscale one
    
    Parameters
    ---------- 
    
    inRas: string
            Path to input raster (single band greyscale)
        
    color_file: string
                 Path to output colorfile.txt     
        
    """


        
    fp, color_file = tempfile.mkstemp(suffix='.txt')

    raster = gdal.Open(inRas)
    phase_data = raster.ReadAsArray()

    max_ph = np.nanmax(phase_data)
    min_ph = np.nanmin(phase_data)
    range_ph = max_ph-min_ph
    colors = ['black', 'blue', 'yellow', 'orange', 'red', 'white']
    with open(color_file, 'w') as f:
        for i, c in enumerate(colors[:-1]):
            f.write(str(int(min_ph + (i + 1)*range_ph/len(colors))) +
                    ' ' + c + '\n')
        f.write(str(int(max_ph - range_ph/len(colors))) +
                ' ' + colors[-1] + '\n')
    os.close(fp)

    
    cmd = ['gdaldem', 'color-relief', inRas, color_file, output_file]
    subprocess.check_call(cmd)    


def _ecdf(x):
    
    """convenience function for computing the empirical CDF
    in hist_match below"""
    vals, counts = np.unique(x, return_counts=True)
    _ecdf = np.cumsum(counts).astype(np.float64)
    _ecdf /= _ecdf[-1]
        
    return vals, _ecdf

def hist_match(inputImage, templateImage):
    
    
    
    # TODO VERY SLOW
    
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image. 
    
    Writes to the inputImage dataset so that it matches
    
    Notes: 
    -----------
        
    As the entire band histogram is required this can become memory
    intensive with big rasters eg 10 x 10k+
    
    Inspire by/adapted from something on stack on image processing - credit to
    that author

    Parameters
    -----------
    
    inputImage: string
                 image to transform; the histogram is computed over the flattened array
            
    templateImage: string
                    template image can have different dimensions to source    
    
    """
    # TODO - cythinis or numba this one
    sourceRas = gdal.Open(inputImage, gdal.GA_Update)
    templateRas = gdal.Open(templateImage)
    #Bands = list()
    bands = sourceRas.RasterCount
    
    oldshape = ((sourceRas.RasterXSize, sourceRas.RasterYSize))
    
    for band in tqdm(range(1, bands+1)):
        #print(band)
        sBand = sourceRas.GetRasterBand(band)
        # seems to be issue with combining properties as with templateRas hence
        # separate lines
        source = sBand.ReadAsArray()
        
        template = templateRas.GetRasterBand(band).ReadAsArray()
        
        
        #source = source.ravel()
        #template = template.ravel()
                
        
        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source.ravel(), return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template.ravel(), return_counts=True)
    
        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
    
        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
        out_array = interp_t_values[bin_idx].reshape(oldshape)
        
        # reuse the var from earlier
        sBand.WriteArray(out_array)
    
    sourceRas.FlushCache()
    templateRas = None
    sourceRas = None

def multi_temp_filter(inRas, outRas, bands=None, windowSize=None):
    
    """ 
    The multi temp filter for SAR data as outlined & published by
    Quegan et al
    
    This is only suitable for small images, as it holds intermediate data in memory
        
    Parameters 
    ----------
    
    inRas: string
            the input raster
    
    outRas: string
             the output raster
    
    blocksize: int
                the chunck processed 
    
    windowsize: int
                 the filter window size
    
    FMT: string
          gdal compatible (optional) defaults is tif

 
    """
    selem = np.ones(shape=((7,7)))
    
    inDataset = gdal.Open(inRas)
    if bands==None:
        bands = inDataset.RasterCount
    
    outDataset = _copy_dataset_config(inDataset, outMap=outRas,
                                     bands=bands)
    


    rStack = np.zeros(shape = (outDataset.RasterYSize, outDataset.RasterXSize,
                               bands))
    mStack = np.zeros(shape = (outDataset.RasterYSize, outDataset.RasterXSize,
                               bands))
    
    for band in tqdm(range(1,bands+1)):
        band1 = inDataset.GetRasterBand(band)
        stats = band1.GetStatistics(True, True)
        if stats[1]==0:
            continue
        data = band1.ReadAsArray()
        data = rescale_intensity(data, in_range='image',
                                 out_range=(0,255))
        mStack[:,:,band-1] = rank.subtract_mean(data.astype(np.uint8), selem=selem)
        rStack[:,:,band-1]= np.float64(data) / np.float64(mStack[:,:,band-1])
    # mean on the band axis
    ovMean = np.nanmean(rStack, axis=2)     
    imFinal = np.empty((outDataset.RasterYSize, outDataset.RasterXSize, bands))    
    #imFinal = np.empty((data.shape[0], data.shape[1], bands))
    for band in range(1,bands+1):
        imFinal[:,:,band-1] = mStack[:,:,band-1] * ovMean
        outBand = outDataset.GetRasterBand(band)
        outBand.WriteArray(imFinal[:,:,band-1])
                    #print(i,j)
        


    outDataset.FlushCache()

def temporal_comp(fileList, outMap, stat='percentile', q = 95, folder=None,
                  blocksize=256,
                  FMT=None,  dtype=gdal.GDT_Int32):
    
    """
    Calculate an image beased on a time series collection of imagery (eg a years woth of S2 data)
            
    Parameters 
    ---------- 
    
    FileList: list of strings
               the files to be inputed, if None a folder must be specified
    
    outMap: string
             the output raster calculated

    stat: string
           the statisitc to be calculated         

    blocksize: int
                the chunck processed 

    q: int
        the ith percentile if percentile is the stat used         
    
    FMT: string
          gdal compatible (optional) defaults is tif

    dtype: string
            gdal datatype (default gdal.GDT_Int32)
    """
    
    
    if fileList is None:
        rasterList = glob2.glob(path.join(folder,'*.tif'))
    else:
        rasterList = fileList
    openList = [gdal.Open(i) for i in rasterList]
    
    inDataset = gdal.Open(rasterList[0])
    bands = inDataset.RasterCount
    
    outDataset = _copy_dataset_config(inDataset, outMap=outMap,
                                     bands=bands)
        
    band = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    
    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = band.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize

    def statChoose(X, stat, q):
        if stat == 'mean':
            stats = np.nanmean(X, axis=2)
        if stat == 'std':
            stats = np.nanstd(X, axis=2)
        if stat == 'percentile':
            stats = np.nanpercentile(X, q, axis=2)    
        if stat == 'median':
            stats = np.nanmedian(X, axis=2)
            
        
        return stats
    

    for band in range(1,bands):
         
        
        if blocksizeY==1:
            rows = np.arange(outDataset.RasterYSize, dtype=np.int)                
            for row in tqdm(rows):
                i = int(row)
                j = 0
                X = np.zeros(shape = (blocksizeY , blocksizeX, len(rasterList)))

                for ind, im in enumerate(openList):
                    array = im.GetRasterBand(band).ReadAsArray(j,i,
                                                blocksizeX, j)
                    array.shape = (1, blocksizeX)
                    X[:,:,ind] = array
                        
                stArray = statChoose(X, stat, q)

                outDataset.GetRasterBand(band).WriteArray(stArray,j,i)

    
        # else it is a block            
        else:
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
                    X = np.zeros(shape = (numRows, numCols, len(openList)))

                    for ind, im in enumerate(openList):
                        array = im.GetRasterBand(band).ReadAsArray(j,i,
                                                numCols, numRows)
                        
                        X[:,:,ind] = array
                        
                    stArray = statChoose(X, stat, q)

                    outDataset.GetRasterBand(band).WriteArray(stArray,j,i)

                    #print(i,j)
    outDataset.FlushCache()
    outDataset = None
    
    
def stat_comp(inRas, outMap, bandList = None,  stat = 'percentile', q = 95, 
                  blocksize=256,
                  FMT=None,  dtype = gdal.GDT_Float32):
    
    """
    Calculate depth wise stat on a multi band raster with selected or all bands
            
    Parameters 
    ---------- 
    
    inRas: string
               input Raster
    
    outMap: string
             the output raster calculated

    stat: string
           the statisitc to be calculated make sure there 
           are no nans as nan percentile is far too slow        

    blocksize: int
                the chunck processed 

    q: int
        the ith percentile if percentile is the stat used         
    
    FMT: string
          gdal compatible (optional) defaults is tif

    dtype: string
            gdal datatype (default gdal.GDT_Int32)
    """
    
    

    inDataset = gdal.Open(inRas)
    
    
    ootbands = len(bandList)
    
    bands = inDataset.RasterCount
    
    outDataset = _copy_dataset_config(inDataset, outMap=outMap,
                                     bands=1)
        
    band = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    
    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = band.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize

    def statChoose(X, stat, q):
        if stat == 'mean':
            stats = np.nanmean(X, axis=2)
        if stat == 'std':
            stats = np.nanstd(X, axis=2)
        if stat == 'percentile':
            # slow as feck
            stats = np.percentile(X, q, axis=2)    
        if stat == 'median':
            stats = np.nanmedian(X, axis=2)
            
        
        return stats
    
        
    if blocksizeY==1:
        rows = np.arange(outDataset.RasterYSize, dtype=np.int)                
        for row in tqdm(rows):
            i = int(row)
            j = 0
            X = np.zeros(shape = (blocksizeY , blocksizeX, ootbands))

            for ind, im in enumerate(bandList):
                array = inDataset.GetRasterBand(im).ReadAsArray(j,i,
                                            blocksizeX, j)
                array.shape = (1, blocksizeX)
                X[:,:,ind] = array
                    
            stArray = statChoose(X, stat, q)

            outDataset.GetRasterBand(band).WriteArray(stArray,j,i)


    # else it is a block            
    else:
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
                X = np.zeros(shape = (numRows, numCols, ootbands))

                for ind, im in enumerate(bandList):
                    array = inDataset.GetRasterBand(im).ReadAsArray(j,i,
                                            numCols, numRows)
                    
                    X[:,:,ind] = array
                    
                stArray = statChoose(X, stat, q)

                outDataset.GetRasterBand(1).WriteArray(stArray,j,i)

                    #print(i,j)
    outDataset.FlushCache()
    outDataset = None    



def _copy_dataset_config(inDataset, FMT = 'Gtiff', outMap = 'copy',
                         dtype=gdal.GDT_Int32, bands = 1):
    """Copies a dataset without the associated rasters.

    """

    
    x_pixels = inDataset.RasterXSize  # number of pixels in x
    y_pixels = inDataset.RasterYSize  # number of pixels in y
    geotransform = inDataset.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    #if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are the "top left" corner.
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()   
    #dtype=gdal.GDT_Int32
    driver = gdal.GetDriverByName(FMT)
    
    # Set params for output raster
    outDataset = driver.Create(
        outMap, 
        x_pixels,
        y_pixels,
        bands,
        dtype)

    outDataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))
        
    outDataset.SetProjection(projection)
    
    return outDataset

def _gdalwarp(inRas, outRas, **kwargs):
    
    """gdalwarp a dataset

    """
    ootRas = gdal.Warp(outRas, inRas, **kwargs)
    ootRas.FlushCache()
    ootRas=None

def batchwarp(inlist, outdir, xres, yres, cores=16, fmt='Gtiff'):
    
    """
    Gdal warp a load of datasets 
    
    Parameters
    ----------
    
    inlist: list
            list of files
    
    outdir: string
            output directory for warped files
    
    xres: int
          pixel x resolution in map units
    
    yres: int
          pixel y resolution in map units
    
    cores: int
          no of processors to use in parallel -1 will indicate all available
    
    """
    
    outlist = [os.path.join(outdir, os.path.split(i)[1]) for i in inlist]
    
    #finalist = zip(inlist, outlist)
    
    if cores == 1:
        _= [_gdalwarp(i, o, xRes=xres, yRes=xres, format=fmt) for i, o in tqdm(zip(inlist, outlist))]
    
    else:
    
        _ = Parallel(n_jobs=cores,
                 verbose=2)(delayed(_gdalwarp)(i,
                                                o,
                                                xRes=xres,
                                                yRes=yres,  
                                                format=fmt) for i,o in zip(inlist, outlist))
    
    
def _quickwarp(inRas, outRas, proj='EPSG:27700', **kwargs):
    
    """gdalwarp a dataset

    """
    ootRas = gdal.Warp(outRas, inRas, dstSRS=proj, format='Gtiff', 
                       callback=gdal.TermProgress, **kwargs)
    ootRas.FlushCache()
    ootRas=None

def multiband2gif(inras, outgif=None, duration=1, loop=0):
    
    """
    Write a multi band image to a animated gif
    
    Parameters
    ----------
    
    inras: string
           input raster
           
    outgif: string
            output gif
    
    """
    
    
    rds = gdal.Open(inras)
    
    bands = np.arange(1, rds.RasterCount+1, 1).tolist()
    
    # shame I couldn't just give it the complete block - func suggests it
    # will only accept up to 4 bands....
    
    images = [_read_rescale(rds, b) for b in bands]
    
    if outgif is None:
        outgif = inras[:-3]+'gif'
        
    imageio.mimsave(outgif, images, duration=duration, loop=loop)

    
def _read_rescale(rds, band):
    
    img = rds.GetRasterBand(band).ReadAsArray()
    
    img = rescale_intensity(img, out_range='uint8')
    
    # can't see it....
#    img = cv2.putText(img, text=str(band),
#                               org=(0,0),
#                               fontFace=3,
#                               fontScale=200,
#                               color=(0,0,0),
#                               thickness=5)
    
    # long winded but works
    img = Image.fromarray(img)
    
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 200)
    
    draw = ImageDraw.Draw(img)
    
    draw.text((0,0), str(band), font=font)
    
    img = np.array(img)
    
    return img
    


# Can't remember if this works   
#def temporal_comp2(inRasSet, outRas, stat, q=5,  window = None, blockSize = None):
#    """
#    
#    Calculate an image beased on a time series collection of imagery (eg a years woth of S2 data)
#
#	Parameters 
#    ---------- 
#    
#    inRasSet : list of strings
#               the files to be inputed, if None a folder must be specified
#    
#    outRas : string
#             the output raster calculated
#
#    stat : string
#           the statisitc to be calculated         
#
#    blocksize : int
#                the chunck processed 
#
#    q : int
#        the  ith percentile if percentile is the stat used         
#            
#    
#    """
#    #use multtemp filter block and classify_pixel_bloc as inspiration
#    #Watch out for block processing    
#
#    inDatasets = [gdal.Open(raster) for raster in inRasSet]
#    
#    bands = inDatasets[0].RasterCount
#    x_pixels = inDatasets[0].RasterXSize  # number of pixels in x
#    y_pixels = inDatasets[0].RasterYSize  # number of pixels in y
#    
#    outDataset = _copy_dataset_config(inDatasets[1], outMap = outRas, bands = bands)
#    
#    def statChoose(bandCube, stat, q=None):
#        if stat == 'mean':
#            outDataset.GetRasterBand(band).WriteArray(bandCube.mean(0))
#        elif stat == 'stdev':
#            outDataset.GetRasterBand(band).WriteArray(bandCube.std(0))
#        elif stat == 'median':
#            outDataset.GetRasterBand(band).WriteArray(np.median(bandCube, 0))
#        elif stat == 'percentile':
#            outDataset.GetRasterBand(band).WriteArray(np.percentile(bandCube, q, 0))
#    
#    
#    bandCube = np.empty([bands, x_pixels, y_pixels])
#    for band in tqdm(range(1, bands)): #gdal why
#        for i,dataset in enumerate(inDatasets):
#            cubeView = bandCube[i,:,:]    #Exposes a view of bandCube; any changes made are reflected in bandCube
#            np.copyto(cubeView, dataset.ReadAsArray(0)[band,:,:])
#            statChoose(bandCube, stat, q)
#        
#        
#    outDataset.FlushCache()
#    outDataset = None #gdal. please.