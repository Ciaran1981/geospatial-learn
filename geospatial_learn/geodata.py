
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:36:15 2015

author: Ciaran Robb
Research Associate in Earth Observation
Centre for Landscape and Climate Research (CLCR)
Department of Geography, University of Leicester, University Road, Leicester, 
LE1 7RH, UK 

If you use code to publish work cite/acknowledge me and authors of libs etc as 
appropriate 

Description
-----------
A series of tools for the manipulation of geospatial imagery/rasters

"""
import gdal, ogr,  osr
#from PIL import Image, ImageDraw

import os
import numpy as np
import glob2
from geospatial_learn.data import _get_S2_geoinfo
from geospatial_learn.shape import _bbox_to_pixel_offsets
import tempfile
#from pyrate.shared import DEM
import glymur
from tqdm import tqdm
#from skimage.util import pad

#from skimage.segmentation import mark_boundaries
#from skimage.io import imread
#from scipy import misc
import scipy.ndimage as nd
from more_itertools import unique_everseen
import subprocess
from skimage.morphology import  remove_small_objects#, remove_small_holes#disk, square, binary_dilation
from skimage.filters import rank
from skimage.exposure import rescale_intensity
import warnings
from os import sys
import re


#from pathlib import Path

#import xmltodict
#from c_utils.misc import chanvese

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

# TODO - cythonise loop block processing as separate function then 
# use an index in similar way to stack_S2

def array2raster(array, bands, inRaster, outRas, dtype, FMT=None):
    
    """Save a raster from a numpy array using the geoinfo from another.
    
    Parameters
    ----------      
    array : np array
            a numpy array.
    
    bands : int
            the no of bands. 
    
    inRaster : string
               the path of a raster.
    
    outRas : string
             the path of the output raster.
    
    dtype : int 
            though you need to know what the number represents!
            a GDAL datatype (see the GDAL website) e.g gdal.GDT_Int32
    
    FMT  : string 
           (optional) a GDAL raster format (see the GDAL website) eg Gtiff, HFA, KEA.
        
    
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
        outRas+fmt, 
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
    # Here we loop through bands
        for band in range(1,bands+1):
            Arr = array[:,:,band-1]
            dataset.GetRasterBand(band).WriteArray(Arr)
        dataset.FlushCache()  # Write to disk.
        dataset=None
        #print('Raster written to disk')


def tile_rasters(inImage, outputImage, tilesize): 
    
    """ Split a large raster into smaller ones
    
    Parameters
    ----------        
    inImage : string
        the path to input raster
    
    outputImage : string
        the path to the output image
    
    tilesize : int
        the side of a square tile
        
    """
    

    inputImage = gdal.Open(inImage)
    #outputImage = gdal.Open(inImage)
    
    inputImage = gdal.Open(inImage)
 
    width = inputImage.RasterXSize
    height = inputImage.RasterYSize
    
    procList = []

    for i in tqdm(range(0, width, tilesize)):
        for j in range(0, height, tilesize):
            w = min(i+tilesize, width) - i
            h = min(j+tilesize, height) - j
            
            gdaltranString = ['gdal_translate', '-of', 'Gtiff', '-srcwin',
                              str(i), str(j), str(w), str(h), inImage,
                                 outputImage, str(i), str(j)]
            p = subprocess.Popen(gdaltranString)
            procList.append(p)
            
    exit_codes = [p.wait() for p in procList]
            #print(i)

def batch_translate(folder, wildcard, FMT=None):
    """ Using the gdal python API, this function translates the format of files
    to commonly used formats
    
    Where:
    -----------         
    folder : string
        the folder containing the rasters to be translated
    
    wildcard : string
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
        
    fileList = glob2.glob(path.join(folder,'**', '**','*', wildcard))
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

def jp2_translate(folder, FMT=None, mode='L1C'):
    
    """ translate all files from S2 download to a useable format 
    
        default FMT is GTiff (leave blank), for .img FMT='HFA', for .vrt FMT='VRT'
        
        If you posses a gdal compiled with the corrext openjpg support use that
        
        Where:
        ----------- 
        folder : string
            S2 granule dir
    
        mode : string
            'L2A' , '20', '10', L1C (default)  
        
        FMT : string (optional)
             a GDAL raster format (see the GDAL website) eg Gtiff, HFA, KEA
                    
        Notes:
        ----------- 
        This function might be useful if you wish to retain seperate rasters,
        but the use of stack_S2 is recommended
            
    """
    if FMT == None:
        FMT = "GTiff"
        fmt = '.tif'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
#    if FMT == 'VRT':
#        fmt = '.vrt'
    xml = glob2.glob(folder+'/*.xml')
    xml = xml[0]
    if mode == 'L1C':
        # this is used for the 1C data 
        fileList = glob2.glob(path.join(folder,'IMG_DATA','**', '**', 
                                        '*MSI*_B*.jp2'))
        fileList = list(unique_everseen(fileList))
        pixelDict = {'01' : 60, '02': 10, '03': 10, '04': 10, '05': 20,
               '06': 20, '07': 20, '8A': 20, '08': 10, '09': 60, '10': 60,
               '11': 20, '12': 20}
        geoinfo= get_S2_geoinfo(xml, mode = None)
    elif mode == 'L2A':
        # this is a level 2A product
        fileList = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                        '*MSI*_B*.jp2'))
        SCL = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**', 
                                   '*SCL*.jp2'))    
        fileList = list(unique_everseen(fileList))
        fileList.append(str(SCL[0]))
        pixelDict = {'60' : 60, '20': 20, '10': 10}
        geoinfo= get_S2_geoinfo(xml, mode = 'L2A')
    elif mode=='20':
        fileList = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                        '*MSI*20*.jp2'))
        SCL = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                   '*SCL*20m.jp2'))
        fileList = list(unique_everseen(fileList))
        fileList.append(str(SCL[0]))
        geoinfo= get_S2_geoinfo(xml)
    elif mode=='10':
        fileList = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                        '*MSI*10*.jp2'))
        SCL = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                   '*SCL*20m.jp2'))
        fileList = list(unique_everseen(fileList))
        fileList.append(str(SCL[0]))
        geoinfo= get_S2_geoinfo(xml)
    elif mode == 'scene':
        geoinfo= get_S2_geoinfo(xml, mode = 'L2A')
        fileList = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                        '*SCL*20m.jp2'))
    
    outList = list()
    #files = np.arange(len(fileList))
    
    #length = len(fileList)
    #files = np.arange(length)
    driver = gdal.GetDriverByName(FMT)
    
    dtype = gdal.GDT_Int32
    if mode=='20' or mode == 'scene':
#        x_min = int(geoinfo['ulx20'])
#        y_max = int(geoinfo['uly20'])
        #x_pixels =   int(geoinfo['cols20'])
        #y_pixels =   int(geoinfo['rows20'])
        pixelSize = 20
    if mode == '10':
#        x_min = int(geoinfo['ulx10'])
#        y_max = int(geoinfo['uly10'])
        #x_pixels =   int(geoinfo['cols10'])
        #y_pixels =   int(geoinfo['rows10'])
        pixelSize = 10
    
    # They are the same for all  grid res
    x_min = int(geoinfo['ulx10'])
    y_max = int(geoinfo['uly10'])
    
    count = np.arange(len(fileList))
    
    for file in tqdm(count): 
        src_filename = fileList[file]
        outList.append(src_filename[:-4]+fmt)
        #if os.path.isfile(src_filename[:-4]+fmt):
            #continue
        # These if rules work, but they are a bit risky going forward
        # Xml reading might be safer.....
        if mode == 'L1C':
            # this one relies on SCL being last, which it is 
            pixelSize = pixelDict[src_filename[-6:-4]]            
        if mode == 'L2A':
            pixelSize = pixelDict[src_filename[-7:-5]]
        if mode == 'scene':
            pixelSize = 20
            
        # this is no good for 10m imagery, takes up mass of ram and time    
        #fullres = imread(src_filename)
        kwargs = {"tilesize": (2048, 2048), "prog": "RPCL"}
        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            jp2 = glymur.Jp2k(src_filename, **kwargs)
        #jp2 = gdal.Open(src_filename).ReadAsArray()
        #Open existing datasetbatch
        #src_ds = gdal.Open(src_filename)
        fullres = jp2.read()
        #Open output format driver, see gdal_translate --formats for list
        #FMT = "GTiff"
        
        dst_filename = outList[file]
        dataset = driver.Create(dst_filename, 
                                fullres.shape[0],
                                fullres.shape[1],
                                1,
                                dtype)

        dataset.SetGeoTransform((
                x_min,    # 0
                pixelSize,  # 1
                0,                      # 2
                y_max,    # 3
                0,                      # 4
                -pixelSize))  
        dataset.GetRasterBand(1).WriteArray(fullres)
        dataset.GetRasterBand(1).GetStatistics(0, 1)
        #Output to new format
        dataset.FlushCache()
        dataset = None
        return dst_filename

        #Properly close the datasets to flush to disk

def jp2_translate_batch(mainFolder, FMT=None, mode=None):
    
    """
    Batch version of jp2translate
    
    Perhaps only useful for the old tile format
    
    Where:
    -----------         
    mainFolder : string
        the path to S2 tile folder to process
    
    FMT : string
        a GDAL raster format (see the GDAL website) eg Gtiff, HFA, KEA
    
    mode : string (optional)
        'L2A' , '20', '10', L1C (default)  
        
        
    """
    
    
    if FMT == None:
        FMT = None
    if mode == None:
        mode = None
        
    paths =  glob2.glob(path.join(mainFolder, 'GRANULE'))
    paths = list(unique_everseen(paths))
    #noDirs = np.arange(len(paths))
    
    for pth in paths:
        jp2_translate(pth, FMT=FMT, mode=mode)

def stack_S2(granule, inFMT = 'jp2', FMT = None, mode = None, blocksize=2048,
             overwrite=True):
    """ Stacks S2 bands downloaded from ESA site
        Can translate directly from jp2 format (this is recommended and is 
        default). 
        
        If you possess gdal 2.1 with jp2k support then alternatively use 
        gdal_translate
                    
        Where:
        ----------- 
        granule : string
            the granule folder 
        
        inFMT : string (optional)
            the format of the bands will likely be jp2
        
        FMT : string (optional)
            the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
        mode : string (optional)
            None, '10'  '20' 
        
        blocksize : int (optional)
            the chunk of jp2 to read in - glymur seems to work fastest with 2048
        
        Returns:
        ----------- 
        A string of the output file path
            
        Notes:
        -----------             
        Uses glymur to read in raster chuncks (until I write something better).
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
#    if inFMT==None:
#        #inFMT = fmt
    #paths =  glob2.glob(mainFolder+'/GRANULE/*/')
    #paths = list(unique_everseen(paths))
    #noDirs = np.arange(len(paths))    
    
#    for file in noDirs:
    
    xml = glob2.glob(path.join(granule,'*MTI*.xml'))
    if len(xml) is 0:
        xml = glob2.glob(granule+'/*MTD*.xml')
        if len(xml) is 0:
            print('Error: \nXml metadata file does not exist ')
            sys.exit(1)
    
    xml = xml[0]
    #pixelDict = {'60' : 60, '20': 20, '10': 10}
    dtype = gdal.GDT_Int32
    geoinfo= get_S2_geoinfo(xml)   
    kwargs = {"tilesize": (blocksize, blocksize), "prog": "RPCL"}
    
    if mode == None:
        bands = 4
        fileList = glob2.glob(path.join(granule,'IMG_DATA', 'R10m','*MSI*.jp2'))
        if len(fileList) is 0:
        # the following if is for S2 since format change
            fileList = glob2.glob(granule+'IMG_DATA/R10m/*B0*.jp2')
        #        x_min = int(geoinfo['ulx10'])
#        y_max = int(geoinfo['uly10'])
        x_pixels =   int(geoinfo['cols10'])
        y_pixels =   int(geoinfo['rows10'])
        cols = x_pixels
        rows = y_pixels
        PIXEL_SIZE = 10
        #tiles = np.arange(36)
        fileList.sort()
        fileList = list(unique_everseen(fileList))
        #needed due to apparent bug in glob2 - removes duplicate entries

        

    if mode == '20':
        fileList = glob2.glob(path.join(granule,'IMG_DATA','R20m','*MSI*.jp2'))
        # the following if is for S2 since format change
        if len(fileList) is 0:
            fileList = glob2.glob(path.join(granule,'IMG_DATA','R20m','*B*.jp2'))
        bands = len(fileList)
        #        x_min = int(geoinfo['ulx20'])
#        y_max = int(geoinfo['uly20'])
        x_pixels =   int(geoinfo['cols20'])
        y_pixels =   int(geoinfo['rows20'])
        cols = x_pixels
        rows = y_pixels
        PIXEL_SIZE = 20
        #tiles = np.arange(6)
        fileList.sort()
        fileList = list(unique_everseen(fileList))

        
    
    dataList = list()
    for file in fileList:
        inDataset = glymur.Jp2k(file, **kwargs)
        dataList.append(inDataset)

    outFile= fileList[0][:-9]+'_stk_'+fmt
    
   
    #if inFMT == 'jp2':
   
    proj = osr.SpatialReference()
    
    # if int is required
    #cs = geoinfo['cs']
    
    # This below suddenly doesn't work???!!!    
    #espgCode = re.findall('\d+', cs)
    #proj.ImportFromEPSG(int(espgCode[0]))
    
    #replaced by

    csCode = geoinfo['cs_code'].split()
    if csCode[4][2] is 'S':
        flag = 0
    else:
        flag = 1
    proj.SetProjCS(geoinfo['cs_code'])
    proj.SetWellKnownGeogCS( "WGS84" )
    proj.SetUTM(int(csCode[4][:-1]), flag)
    projection = proj.ExportToWkt()
    #files = np.arange(length)    
        
    # They are the same for all  grid res
    x_min = int(geoinfo['ulx10'])
    y_max = int(geoinfo['uly10'])

        
    datasetList = list()

    for band in range(0, bands):
        data = glymur.Jp2k(fileList[band], **kwargs)
        datasetList.append(data)
            
    dtype=gdal.GDT_Int32
    
    driver = gdal.GetDriverByName(FMT)
#    if os.path.isfile(outRas+fmt):
#            os.remove(outRas+fmt)
    # Set params for output raster
    outDataset = driver.Create(
        outFile, 
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

    blocksizeX = blocksize
    blocksizeY = blocksize
    # This index is different as the bbox coords are required
    #index = []
    

    for i in tqdm(range(0, rows, blocksizeY)):
        if i + blocksizeY < rows:
            numRows = blocksizeY
            brightY = i+numRows 
        else:
            numRows = rows -i
            brightY = i+numRows 
    
        for j in range(0, cols, blocksizeX):
            if j + blocksizeX < cols:
                numCols = blocksizeX
                brightX = j+numCols 
            else:
                numCols = cols - j
                brightX = j+numCols

#            tiles = np.arange(len(index))  
#            for tile in tqdm(tiles):         
            for band in range(0, bands):
                with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            array1 = dataList[band].read(area=(i,j, brightY, brightX))

                outDataset.GetRasterBand(band+1).WriteArray(array1, j, i)
            #utDataset.GetRasterBand(band).ComputeStatistics(0)                            
        
                        
    outDataset.FlushCache() 
    outDataset = None
    
    return outFile

def mask_raster(inputIm, mval, overwrite=True, outputIm=None,
                    blocksize = None, FMT = None):
    """ perform a numpy masking operation on a raster where all values
    corresponding to  mask value are retained - does this in blocks for
    efficiency on larger rasters
    
    Where: 
    ----------- 
    inputIm : string
        the input raster
        
    mval : int
        the mask value eg 1, 2 etc
        
    FMT : string
        the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    outputIm : string (optional)
        optionally write a separate output image, if None, will 
        mask the input
        
    blocksize : int
        the chunk of raster to read in
        
    Returns:
    ----------- 
    A string of the output file path
        
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
    
    if overwrite is True:
        inDataset = gdal.Open(inputIm, gdal.GA_Update)
        outBand = inDataset.GetRasterBand(1)
        bnd = inDataset.GetRasterBand(1)
    else:
        
        inDataset = gdal.Open(inputIm)
    
        x_pixels = inDataset.RasterXSize  # number of pixels in x
        y_pixels = inDataset.RasterYSize  # number of pixels in y
        geotransform = inDataset.GetGeoTransform()
        PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
        #if not would need w x h
        x_min = geotransform[0]
        y_max = geotransform[3]
        # x_min & y_max are like the "top left" corner.
        projection = inDataset.GetProjection()  
        dtype=gdal.GDT_Int32
        driver = gdal.GetDriverByName(FMT)
        
        # Set params for output raster
        outDataset = driver.Create(
            outputIm+fmt, 
            x_pixels,
            y_pixels,
            1,
            dtype)
    
        outDataset.SetGeoTransform((
            x_min,    # 0
            PIXEL_SIZE,  # 1
            0,                      # 2
            y_max,    # 3
            0,                      # 4
            -PIXEL_SIZE))
            
        outDataset.SetProjection(projection)    
        bnd = inDataset.GetRasterBand(1)
        
        
        outBand = outDataset.GetRasterBand(1)
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
#                for band in range(1, bands+1):
               
                array = bnd.ReadAsArray(j, i, numCols, numRows)
                array[array != mval]=0
                outBand.WriteArray(array, j, i)
    # This is annoying but necessary as the stats need updated and cannot be 
    # done in above band loop due as this would be very inefficient
    #for band in range(1, bands+1):
    #inDataset.GetRasterBand(1).ComputeStatistics(0)
    if overwrite is True:
        inDataset.FlushCache()
        inDataset = None
    else:                        
        outDataset.FlushCache()
        outDataset = None
     
def mask_raster_multi(inputIm,  mval=1, outval = None, mask=None,
                    blocksize = 256, FMT = None, dtype=None):
    """ perform a numpy masking operation on a raster where all values
    corresponding to  mask value are retained - does this in blocks for
    efficiency on larger rasters
    
    Where: 
    ----------- 
    inputIm : string
        the granule folder 
        
     mval : int
         the masking value that delineates pixels to be kept
        
     outval : numerical dtype eg int, float
         the areas removed will be written to this value
     default is 0
        
     mask : string
         the mask raster to be used (optional)
        
     FMT : string
         the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
     mode : string
         None > 10m data, '20' >20m
        
     blocksize : int
         the chunk of raster read in & write out
    
    Returns:
    ----------- 

    nowt

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
                        array[mask != mval]=0
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
                        array[array != mval]=0
                        if outval != None:
                            array[array == mval] = outval     
                            bnd.WriteArray(array, j, i)
        # This is annoying but necessary as the stats need updated and cannot be 
        # done in above band loop due as this would be very inefficient
        #for band in range(1, bands+1):
        #inDataset.GetRasterBand(1).ComputeStatistics(0)
           
        inDataset.FlushCache()
        inDataset = None

def calc_ndvi(inputIm, outputIm, bandsList, blocksize = 256, FMT = None, dtype=None):
    """ Create a copy of an image with an ndvi band added
    
    Where: 
    ----------- 
    inputIm : string
        the granule folder 
        
     bands : list
         a list of band indicies to be used, eg - [3,4] for Sent2 data

     FMT : string
         the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
     blocksize : int
         the chunk of raster read in & write out
    
    Returns:
    ----------- 

    nowt

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
        dtype = gdal.GDT_Int32
        
    inDataset = gdal.Open(inputIm, gdal.GA_Update)
    
    bands = inDataset.RasterCount+1
    
    bnnd = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize

    x_pixels = inDataset.RasterXSize  # number of pixels in x
    y_pixels = inDataset.RasterYSize  # number of pixels in y
    geotransform = inDataset.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    #if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()   

    driver = gdal.GetDriverByName(FMT)
    
    outDataset = driver.Create(
            outputIm+fmt, 
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

def stack_ras(inRas1, inRas2, outFile,  FMT = None, mode = None,
              blocksize=None):
    """ Stack some rasters for change classification - must be same size!!!
    
    Parameters
    ----------         
    inRas1 : string
        the input image 
        
    inRas2 : string
        the second image 
        
    outFile : string
        the output file path (no file extension required)
        
    FMT : string
        the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    min_size : int
        size in pixels to retain of cloud mask
        
    blocksize : int
        the square chunk processed at any one time
            
    Returns
    -------
    String of output file path
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
    
    inras1 = gdal.Open(inRas1, gdal.GA_ReadOnly)
    
    inras2 = gdal.Open(inRas2, gdal.GA_ReadOnly)
    
    bands = inras1.RasterCount + inras2.RasterCount
    
    x_pixels = inras1.RasterXSize  # number of pixels in x
    y_pixels = inras1.RasterYSize  # number of pixels in y
    geotransform = inras1.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    #if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inras1.GetProjection()
    geotransform = inras1.GetGeoTransform()  
    driver = gdal.GetDriverByName(FMT)
    bnd = inras1.GetRasterBand(1)
    dtype = bnd.DataType
    del bnd

#    if os.path.isfile(outRas+fmt):
#            os.remove(outRas+fmt)
    # Set params for output raster
    outDataset = driver.Create(
        outFile+fmt, 
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
    
    rows = x_pixels
    cols = y_pixels
    
    if blocksize is None:
        bnd = inras1.GetRasterBand(1)
        blocksize = bnd.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
        del bnd
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
            # This is extremely messy & inefficcient   
            for band in range(1, bands+1):
                    if band > inras1.RasterCount:
                        bnd = inras2.GetRasterBand(band-inras1.RasterCount)
                    else:
                        bnd = inras1.GetRasterBand(band)
                    
                    array = bnd.ReadAsArray(j, i, numCols, numRows)
                    outDataset.GetRasterBand(band).WriteArray(array, j, i)
            
    outDataset.FlushCache() 
    outDataset = None
    

def polygonize(inRas, outPoly, outField=None,  mask = True, band = 1):
    
    """ Lifted straight from the cookbook http://pcjericks.github.io/py-gdalogr-cookbook
    and gdal func docs. Very slow......
    Parameters
    ----------         
    inRas : string
        the input image 
    
        
    outPoly : string
        the output polygon file path 
        
    outField : string (optional)
        size in pixels to retain of cloud mask
        
    blocksize : int
        the square chunk processed at any one time
        
    band : int
        the input raster band
            
    """    
    
        
    # My goodness this is SO SLOW - it's just the gdal function that's slow
    # nowt else
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
        maskband = srcband.GetMaskBand()
        options.append('-mask')
    else:
        mask == False
        maskband = None
    

    #
    #  create output datasource
    #
    dst_layername = outPoly
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( dst_layername + ".shp" )
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = None )
    if outField is None:
        dst_fieldname = 'DN'
        fd = ogr.FieldDefn( dst_fieldname, ogr.OFTInteger )
        dst_layer.CreateField( fd )
        dst_field = 0 
    
    else: 
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(outField)

    outShape = gdal.Polygonize(srcband, maskband, dst_layer,dst_field, options,
                    callback=gdal.TermProgress)
    outShape.FlushCache()
    
    srcband = None
    src_ds = None
    dst_ds = None
    #mask_ds = None    

    
def otbMeanshift(inputImage, radius, rangeF, minSize, outShape):
    """ Convenience function for OTB meanshift by calling the otb command line
        written for convenience and due to otb python api being rather verbose 
        and Py v2.7 (why do folk still use it??).
        
        You will need to install OTB etc seperately
        
        
        
    Parameters
    ----------         
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


    Notes
    -----      
    There is a maximum size for the .shp format otb doesn't seem to
    want to move beyond (2gb), so enormous rasters may need to be sub
    divided
    
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



def clip_raster(inRas, inShape, outRas, nodata_value=None, blocksize=None, 
                blockmode = True):

    """
    Clip a raster
    
    Parameters
    ----------         
    inRas : string
        the input image 
    
    outPoly : string
        the input polygon file path 
        
    outRas : string (optional)
        the clipped raster
        
    nodata_value : numerical (optional)
        self explanatory
        
    blocksize : int (optional)
        the square chunk processed at any one time
        
    blockmode : bool (optional)
        whether the raster will be clipped entirely in memory or by chunck

     
    Notes
    -----
    This just calls the gdal cmd line at present and was just written for 
    convenience, quicker solution is currently not finished....
   
    """
    # Polygon shapefile used to clip
    vds = ogr.Open(inShape)
    

    # The input raster        
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    
    print('getting geo-info')
    
    #driver = gdal.GetDriverByName('Gtiff')
    # This first section gets the necessary coords/pixel dimensions for the 
    # new sub-raster
    rb = rds.GetRasterBand(1)
   # dtype = rb.DataType
    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)
       
    
    rgt = rds.GetGeoTransform()

    lyr = vds.GetLayer()
    feat = lyr.GetFeature(0)
    geom = feat.geometry()
            
    src_offset = _bbox_to_pixel_offsets(rgt, geom)
    # 'offset = xoff, yoff, xcount, ycount'

    new_gt = (
    (rgt[0] + (src_offset[0] * rgt[1])),
    rgt[1],
    0.0,
    (rgt[3] + (src_offset[1] * rgt[5])),
    0.0,
    rgt[5])
    
    cmd = ['gdalwarp', '-q', '-cutline', inShape, '-crop_to_cutline', 
           '-tr', str(new_gt[1]), str(new_gt[5]),
           '-of', 'GTiff', inRas, outRas] 
    subprocess.call(cmd)

#    TODO - sort the code below as it is very fast - just something wrong with
#       offsets being written to outraster     
#    bands = rds.RasterCount
#    # Now we have the geo info necessary to write the new raster to the correct
#    # coordinates etc, extra
#    outDataset = driver.Create(
#        outRas+'tif', 
#        src_offset[2],
#        src_offset[3],
#        bands,
#        dtype)
#    
#    outDataset.SetGeoTransform(new_gt)
#    outDataset.SetProjection(rds.GetProjection())
#    
#    if blocksize == None:   
#        blocksizeX = src_offset[3]
#        blocksizeY = 1
#    else:
#        blocksizeX = blocksize
#        blocksizeY = blocksize
#    
#    # Each array is read in and out in scanlines or blocks for memory and speed
#    # with big rasters    
#    # Scanline
#
#        
#    if blocksizeY==1:
#        rows = np.arange(src_offset[3], dtype=np.int)        
#        for row in tqdm(rows):
#            i = int(row)
#            j = 0    
#            for band in range(1,bands+1):
#                rb = rds.GetRasterBand(band)
#                src_array = rb.ReadAsArray(j, i, blocksizeX, 1)
#            
#            outDataset.GetRasterBand(band).WriteArray(src_array, j, i)
#        outDataset.FlushCache()        
#        
#    else:
#    #Block
#        rows = src_offset[3]
#        cols = src_offset[2]
#        
#        # These two vars here are for the new raster which obviously begins at 
#        # 0,0
#                
#        
#        index = []
#        for i in tqdm(range(0, rows, blocksizeY)):
#            if i + blocksizeY < rows:
#                numRows = blocksizeY
#            else:
#                numRows = rows -i
#        
#                for j in range(0, cols, blocksizeX):
#                    if j + blocksizeX < cols:
#                        numCols = blocksizeX
#                    else:
#                        numCols = cols - j
#            index.append(((j,i)))
#                        
#
#                        
#                        
#                    
#                    for band in range(1,bands+1):
#                        rb = rds.GetRasterBand(band)
#                        src_array = rb.ReadAsArray(j, i, numCols, numRows)                     
#                        # The i & j here are not the same as the source dataset
#                        # as they will start from zero as the are relative to
#                        # the new raster
#
#                        outDataset.GetRasterBand(band).WriteArray(src_array,
#                                                                  tgt_j, tgt_i)
#      
#        outDataset.FlushCache()
#     
#    outDataset = None



def color_raster(inRas, color_file, output_file):
    """ Generate a txt colorfile and make a RGB image from a grayscale one
    
    Parameters
    ----------       
    inRas : string
        Path to input raster (single band greyscale)
        
    color_file : string
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

#=======================2 multi temp filters now==============================#
#=============================================================================#
def multi_temp_filter_block(inRas, outRas, bands=None, blocksize=256, 
                            windowsize=7, FMT=None):
    
    """ Multi temporal filter implementation for radar data 
        See Quegan et al., Uni of Sheffield for paper
        Requires an installation of OTB
        
        Parameters 
        ----------- 
        inRas : string
            the input raster
        
        outRas : string
            the output raster
        
        blocksize : int
            the chunck processed 
        
        windowsize : int
            the filter window size
        
        FMT : string
            gdal compatible (optional) defaults is tif
    """
    #selem = square(7)
    if FMT == None:
        FMT = 'Gtiff'
        fmt = '.tif'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'
    
    inDataset = gdal.Open(inRas)
    if bands==None:
        bands = inDataset.RasterCount
    x_pixels = inDataset.RasterXSize  # number of pixels in x
    y_pixels = inDataset.RasterYSize  # number of pixels in y
    geotransform = inDataset.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    #if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()   
    dtype=gdal.GDT_Float64

    driver = gdal.GetDriverByName(FMT)

    
    # Set params for output raster
    outDataset = driver.Create(
        outRas+fmt, 
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
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    #outBand = outDataset.GetRasterBand(1)
    PIXEL_SIZE = geotransform[1] 
    # So with most datasets blocksize is a row scanline
    if blocksize==None:
        blocksize=256
    blocksizeX = blocksize
    blocksizeY = blocksize

    # Key issue now is to speed this part up 
    tempRas = inRas[:-4]+'av_.tif'
    print('filtering image')
    cmd = ('otbcli_Smoothing -in '+inRas+' -out '+tempRas+' -type mean'
           ' -type.mean.radius '+windowsize)
    os.system(cmd) 
    meanRas = gdal.Open(tempRas)       
    print('filtering done')
    for i in tqdm(range(1, rows-1, blocksizeY)):
        if i + blocksizeY < rows:
            numRows = blocksizeY
        else:
            numRows = rows -i
    
        for j in range(1, cols-1, blocksizeX):
            if j + blocksizeX < cols:
                numCols = blocksizeX
            else:
                numCols = cols - j
            rStack = np.zeros(shape = (numRows, numCols, bands))
            mStack = np.zeros(shape = (numRows, numCols, bands))
                        
            for band in range(1,bands+1):

                band1 = inDataset.GetRasterBand(band)
                band1A = meanRas.GetRasterBand(band)
                stats = band1.GetStatistics(True, True)
                if stats[1]==0:
                    continue
                
                data = band1.ReadAsArray(j, i, numCols, numRows)
#                with warnings.catch_warnings():
#                    warnings.simplefilter("ignore")
#                    #data = rescale_intensity(data, in_range='image',
#                    #                         out_range=(0,255))
#                    
                meanIm = band1A.ReadAsArray(j, i, numCols, numRows)

                mStack[:,:,band-1] = meanIm
                ratioIm = np.float64(data) / np.float64(meanIm)
                rStack[:,:,band-1]=ratioIm
            # mean on the band axis
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")    
                ovMean = np.nanmean(rStack, axis=2)     
           
            imFinal = np.empty((data.shape[0], data.shape[1], bands))
            for band in range(1,bands+1):
                imFinal[:,:,band-1] = mStack[:,:,band-1] * ovMean
                outBand = outDataset.GetRasterBand(band)
                outBand.WriteArray(imFinal[:,:,band-1],j,i)
                #print(i,j)
        

    #outBand.FlushCache()
    outDataset.FlushCache()
    outDataset = None

def _ecdf(x):
    
    """convenience function for computing the empirical CDF
    in hist_match below"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
        
    return vals, ecdf

def hist_match(inputImage, templateImage):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image. 

    Parameters
    ----------
    inputImage : string
        image to transform the histogram is computed over the flattened array
            
    templateImage : string
        emplate image can have different dimensions to source
    
    Notes: 
    -----------
        
    As the entire band histogram is required this can become memory
    intensive with big rasters eg 10 x 10k+
    
    Inspire by/adapted from something on stack on image processing - credit to
    that author
    
    """
    #
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
    
    """ The multi temp filter for radar data as outlined & published by
        Quegan et al, Uni of Sheffield - this is only suitable for small images,
        as it holds intermediate data in memory
        
        Parameters 
        ---------- 
        inRas : string
            the input raster
        
        outRas : string
            the output raster
        
        blocksize : int
            the chunck processed 
        
        windowsize : int
            the filter window size
        
        FMT : string
            gdal compatible (optional) defaults is tif



    """
    selem = np.ones(shape=((7,7)))
    
    
    
    inDataset = gdal.Open(inRas)
    if bands==None:
        bands = inDataset.RasterCount
    x_pixels = inDataset.RasterXSize  # number of pixels in x
    y_pixels = inDataset.RasterYSize  # number of pixels in y
    geotransform = inDataset.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    #if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()   
    dtype=gdal.GDT_Float64
    FMT='HFA'
    driver = gdal.GetDriverByName(FMT)

    
    # Set params for output raster
    outDataset = driver.Create(
        outRas, 
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
    #outBand = outDataset.GetRasterBand(1)
    PIXEL_SIZE = geotransform[1] 
    # So with most datasets blocksize is a row scanline
     # size of the pixel...they are square so thats ok.
    #if not would need w x h
    #If the block is a row, this simplifies things a bit
    # Key issue now is to speed this part up 
    # 
    rStack = np.zeros(shape = (y_pixels, x_pixels, bands))
    mStack = np.zeros(shape = (y_pixels, x_pixels, bands))
    
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
    imFinal =np.empty((y_pixels, x_pixels, bands))    
    #imFinal = np.empty((data.shape[0], data.shape[1], bands))
    for band in range(1,bands+1):
        imFinal[:,:,band-1] = mStack[:,:,band-1] * ovMean
        outBand = outDataset.GetRasterBand(band)
        outBand.WriteArray(imFinal[:,:,band-1])
                    #print(i,j)
        

    #outBand.FlushCache()
    outDataset.FlushCache()

