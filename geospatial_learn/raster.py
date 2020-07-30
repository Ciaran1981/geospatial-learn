
# -*- coding: utf-8 -*-
"""
The raster module. 

Description
-----------

A series of tools for the manipulation of geospatial imagery/rasters such as
masking or raster algebraic type functions and the conversion of Sentinel 2 
data to gdal compatible formats.  

"""
import gdal, ogr,  osr
import os
import numpy as np
import glob2
#from geospatial_learn.data import _get_S2_geoinfo
from geospatial_learn.gdal_merge import _merge
import tempfile
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
from os import sys, path
#import re
import matplotlib
matplotlib.use('Qt5Agg')


gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()


def array2raster(array, bands, inRaster, outRas, dtype, FMT=None):
    
    """
    Save a raster from a numpy array using the geoinfo from another.
    
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
    # Here we loop through bands
        for band in range(1,bands+1):
            Arr = array[:,:,band-1]
            dataset.GetRasterBand(band).WriteArray(Arr)
        dataset.FlushCache()  # Write to disk.
        dataset=None
        #print('Raster written to disk')
        
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

def tile_rasters(inImage, outputImage, tilesize): 
    
    """ 
    Split a large raster into smaller ones
    
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

def batch_translate(folder, wildcard, FMT='Gtiff'):
    """ 
    Using the gdal python API, this function translates the format of files
    to commonly used formats
    
    Parameters
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

def jp2_translate(folder, FMT=None, mode='L1C'):
    
    """ 
    Translate all files from S2 download to a useable format 
    
    Default FMT is GTiff (leave blank), for .img FMT='HFA', for .vrt FMT='VRT'
        
    If you posses a gdal compiled with the corrext openjpg support use that
    
    This function might be useful if you wish to retain seperate rasters,
    but the use of stack_S2 is recommended
        
    Parameters
    ----------- 
    
    folder : string
        S2 granule dir

    mode : string
        'L2A' , '20', '10', L1C (default)  
    
    FMT : string (optional)
         a GDAL raster format (see the GDAL website) eg Gtiff, HFA, KEA
                

            
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
        geoinfo= _get_S2_geoinfo(xml, mode = None)
    elif mode == 'L2A':
        # this is a level 2A product
        fileList = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                        '*MSI*_B*.jp2'))
        SCL = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**', 
                                   '*SCL*.jp2'))    
        fileList = list(unique_everseen(fileList))
        fileList.append(str(SCL[0]))
        pixelDict = {'60' : 60, '20': 20, '10': 10}
        geoinfo= _get_S2_geoinfo(xml, mode = 'L2A')
    elif mode=='20':
        fileList = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                        '*MSI*20*.jp2'))
        SCL = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                   '*SCL*20m.jp2'))
        fileList = list(unique_everseen(fileList))
        fileList.append(str(SCL[0]))
        geoinfo= _get_S2_geoinfo(xml)
    elif mode=='10':
        fileList = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                        '*MSI*10*.jp2'))
        SCL = glob2.glob(path.join(folder, 'IMG_DATA', '**', '**',
                                   '*SCL*20m.jp2'))
        fileList = list(unique_everseen(fileList))
        fileList.append(str(SCL[0]))
        geoinfo= _get_S2_geoinfo(xml)
    elif mode == 'scene':
        geoinfo= _get_S2_geoinfo(xml, mode = 'L2A')
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
    
    Parameters
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

def stack_S2(granule, inFMT = 'jp2', FMT = None, mode = None, old_order=False,
             blocksize=2048, overwrite=True):
    """ 
    Stacks S2 bands downloaded from ESA site
    
    Can translate directly from jp2 format (this is recommended and is 
    default). 
        
    If you possess gdal 2.1 with jp2k support then alternatively use 
    gdal_translate
                
    Parameters
    ----------- 
    
    granule : string
              the granule folder 
    
    inFMT : string (optional)
            the format of the bands will likely be jp2
    
    FMT : string (optional)
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
    
    mode : string (optional)
           None, '10'  '20' 
    
    old_order : bool (optional)
                this function used to order the 20m imagery 2,3,4,5,6,7,11,12,8a
                if false ordered like this 2,3,4,5,6,7,8a,11,12
    
    blocksize : int (optional)
                the chunk of jp2 to read in - glymur seems to work fastest with 2048
    
    Returns
    ----------- 
    string
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
    geoinfo= _get_S2_geoinfo(xml)   
    kwargs = {"tilesize": (blocksize, blocksize), "prog": "RPCL"}
    
    if mode == None:
        bands = 4
        fileList = glob2.glob(path.join(granule,'IMG_DATA', 'R10m','*MSI*.jp2'))
        if len(fileList) is 0:
        # the following if is for S2 since format change
            fileList = glob2.glob(path.join(granule,'IMG_DATA/R10m/*B0*.jp2'))
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
        bandNames = ['2','3','4','8']

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
        # to cover the old order where band 8a is last
        if old_order is False:            
            eight = fileList[8]
            fileList.insert(3, eight)
            fileList.pop(9)
            bandNames = ['2','3','4','5','6','7','8a','11','12']
        else:
            bandNames = ['2','3','4','5','6','7','11','12', '8a']
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

    for file in fileList:
        data = glymur.Jp2k(file, **kwargs)
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

    for b,desc in enumerate(bandNames):
        outDataset.GetRasterBand(b+1).SetDescription(desc)
    
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
            for band, file in enumerate(dataList):
                
                with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            array1 = file.read(area=(i,j, brightY, brightX))
                outDataset.GetRasterBand(band+1).WriteArray(array1, j, i)
            #utDataset.GetRasterBand(band).ComputeStatistics(0)                            
        
                        
    outDataset.FlushCache() 
    outDataset = None
    
    return outFile

def mask_raster(inputIm, mval, overwrite=True, outputIm=None,
                    blocksize = None, FMT = None):
    """ 
    Perform a numpy masking operation on a raster where all values
    corresponding to  mask value are retained - does this in blocks for
    efficiency on larger rasters
    
    Parameters 
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
        
    Returns
    ----------- 
    string
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
    
        
        outDataset = _copy_dataset_config(inputIm, outMap = outputIm,
                                     bands = inDataset.RasterCount)
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
    """ 
    Perform a numpy masking operation on a raster where all values
    corresponding to  mask value are retained - does this in blocks for
    efficiency on larger rasters
    
    Parameters 
    ----------- 
    
    inputIm : string
              the granule folder 
        
    mval : int
           the masking value that delineates pixels to be kept
        
    outval : numerical dtype eg int, float
              the areas removed will be written to this value default is 0
        
    mask : string
            the mask raster to be used (optional)
        
    FMT : string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    mode : string
           None > 10m data, '20' >20m
        
    blocksize : int
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
    """ 
    Create a copy of an image with an ndvi band added
    
    Parameters 
    ----------- 
    
    inputIm : string
              the granule folder 
        
    bands : list
            a list of band indicies to be used, eg - [3,4] for Sent2 data

    FMT : string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    blocksize : int
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
    Create a copy of an image with an ndvi band added
    
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


def remove_cloud_S2(inputIm, sceneIm,
                    blocksize = 256, FMT = None, min_size=4, dist=1):
    """ 
    Remove cloud using the a scene classification
    
    This saves back to the input raster by default
        
    Parameters
    ----------- 
    
    inputIm : string
              the input image 
        
    sceneIm : string
              the scenemap to use as a mask for removing cloud
              It is assumed the scene map consists of 1 shadow, 2 cloud, 3 land, 4 water 
        
    FMT : string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    min_size : int
               size in pixels to retain of cloud mask
        
    blocksize : int
                the square chunk processed at any one time
        

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
    
    sceneRas = gdal.Open(sceneIm)
    inDataset = gdal.Open(inputIm, gdal.GA_Update)
    tempBand = inDataset.GetRasterBand(1)
    dtypeCode = gdal.GetDataTypeName(tempBand.DataType)
    # common gdal datatypes - when calling eg GDT_Int32 it returns an integer
    # hence the dict with the integer codes - not really required but allows me
    # to see the data type
    dtypeDict = {'Byte':1, 'UInt16':2, 'Int16':3, 'UInt32':4, 'Int32':5,
                'Float32':6, 'Float64':7}
#    dtype = dtypeDict[dtypeCode]
    tempBand = None
    
    bands = inDataset.RasterCount

#        
#    outDataset.SetProjection(projection)    
  
    band = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    if blocksize == None:
        blocksize = band.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize
     # size of the pixel...they are square so thats ok.
    #if not would need w x h
    #If the block is a row, this simplifies things a bit
    # Key issue now is to speed this part up 
    # 
    
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
                mask = sceneRas.ReadAsArray(j, i, numCols, numRows)
                mask = mask>6
                with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            mask = remove_small_objects(mask, min_size=min_size)
                mask= nd.distance_transform_edt(mask==0)<=1
                for band in range(1, bands+1):
                    bnd = inDataset.GetRasterBand(band)
                    array = bnd.ReadAsArray(j, i, numCols, numRows)
                    array[mask==1]=0
                    inDataset.GetRasterBand(band).WriteArray(array, j, i)
    # This is annoying but necessary as the stats need updated and cannot be 
    # done in above band loop due as this would be very inefficient
    for band in range(1, bands+1):
        inDataset.GetRasterBand(band).ComputeStatistics(0)

    inDataset.FlushCache()
    inDataset = None     

def remove_cloud_S2_stk(inputIm, sceneIm1, sceneIm2=None, baseIm = None,
                    blocksize = 256, FMT = None, max_size=10,
                    dist=1):
    """ remove cloud using the the c_utils scene classification
        the KEA format is recommended, .tif is the default,

        no need to add the file extension this is done automatically

        Parameters
        -----------

        inputIm : string
            the input image

        sceneIm1, 2 : string
            the classification rasters used to mask out the areas in
        the input image

        baseIm : string
            Another multiband raster of same size extent as the inputIm
            where the baseIm image values are used rather than simply converting
            to zero (in the use case of 2 sceneIm classifications)

        Returns:
        -----------
        nowt

        Notes:
        -----------
        Useful if you have a base image whic is a cloudless composite, which
        you intend to replace with the current image for the next round of
        classification/ change detection

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

    sceneRas1 = gdal.Open(sceneIm1)
    if sceneIm2 != None:
        sceneRas2 = gdal.Open(sceneIm2)
    if baseIm != None:
        baseRas = gdal.Open(baseIm)
    inDataset = gdal.Open(inputIm, gdal.GA_Update)
    tempBand = inDataset.GetRasterBand(1)
    dtypeCode = gdal.GetDataTypeName(tempBand.DataType)
    # common gdal datatypes - when calling eg GDT_Int32 it returns an integer
    # hence the dict with the integer codes - not really required but allows me
    # to see the data type
    dtypeDict = {'Byte':1, 'UInt16':2, 'Int16':3, 'UInt16':4, 'Int32':5,
                'Float32':6, 'Float64':7}
#    dtype = dtypeDict[dtypeCode]
#    tempBand = None
#
    bands = inDataset.RasterCount

    band = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    if blocksize == None:
        blocksize = band.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize
     # size of the pixel...they are square so thats ok.
    #if not would need w x h
    #If the block is a row, this simplifies things a bit
    # Key issue now is to speed this part up
    #
    # TODO - this is VERY ugly fix mess of if statements


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
                mask1 = sceneRas1.ReadAsArray(j, i, numCols, numRows)
                if sceneIm2 != None:
                    mask2 = sceneRas2.ReadAsArray(j, i, numCols, numRows)
                    mask1 = np.logical_not(mask1==3)
                    mask2 = np.logical_not(mask2==3)
                    mask1[mask2==1]=1
                else:
                    mask1 =np.logical_or(mask1 ==3, mask1==4)
                with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # both pointless provided it gets sivved by gdal
                            remove_small_objects(mask1, min_size=max_size,
                                                 in_place=True)
#                            remove_small_holes(mask1, min_size=max_size,
#                                                 in_place=True)
                            mask1= nd.distance_transform_edt(mask1==1)<=dist
                for band in range(1, bands+1):
                    bnd = inDataset.GetRasterBand(band)
                    if baseIm != None:
                        bnd2 = baseRas.GetRasterBand(band)
                        array2 = bnd2.ReadAsArray(j, i, numCols, numRows)
                        array2[mask1==0]=0
                        array = bnd.ReadAsArray(j, i, numCols, numRows)
                        array[mask1==1]=0
                        array += array2
                    else:
                        array = bnd.ReadAsArray(j, i, numCols, numRows)
                        array[mask1==1]=0
                    inDataset.GetRasterBand(band).WriteArray(array, j, i)
    # This is annoying but necessary as the stats need updated and cannot be
    # done in above band loop as this would be very inefficient
    for band in range(1, bands+1):
        inDataset.GetRasterBand(band).ComputeStatistics(0)

    inDataset.FlushCache()
    inDataset = None



def stack_ras(rasterList, outFile):
    """ 
    Stack some rasters for change classification
    
    Parameters
    ----------- 
        
    rasterList : string
             the input image 
        
    outFile : string
              the output file path including file extension
        

    """
    _merge(names = rasterList, out_file = outFile)
    
def combine_scene(scl, c_scn, blocksize = 256):
    """ 
    combine another scene classification with the sen2cor one
    
    Where: 
    ----------- 
    scl : string
        the sen2cor one

    c_scn : string
        the independently derived one - this will be modified
    
    blocksize : string
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
    # This is annoying but necessary as the stats need updated and cannot be 
    # done in above band loop due as this would be very inefficient
    #for band in range(1, bands+1):
    #inDataset.GetRasterBand(1).ComputeStatistics(0)
                        
    inDataset.FlushCache()
    inDataset = None
    
def polygonize(inRas, outPoly, outField=None,  mask = True, band = 1, filetype="ESRI Shapefile"):
    
    """ 
    Lifted straight from the cookbook and gdal func docs.

    http://pcjericks.github.io/py-gdalogr-cookbook


    Parameters
    -----------   
      
    inRas : string
            the input image 
    
        
    outPoly : string
              the output polygon file path 
        
    outField : string (optional)
             the name of the field containing burnded values

    mask : bool (optional)
            use the input raster as a mask

    band : int
           the input raster band
            
    """    
    
    #TODO investigate ways of speeding this up   
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
            
   
    """
    

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
    
    #TODO drop the subprocess call
    
    cmd = ['gdalwarp', '-q', '-cutline', inShape, '-crop_to_cutline', 
           '-tr', str(new_gt[1]), str(new_gt[5]),
           '-of', 'GTiff', inRas, outRas] 
    subprocess.call(cmd)




def color_raster(inRas, color_file, output_file):
    """ 
    Generate a txt colorfile and make a RGB image from a grayscale one
    
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
    
    """ 
    Multi temporal filter implementation for radar data 
    
    See Quegan et al., for paper
        
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
        bands = inDataset.RasterCountl
        inDataset = gdal.Open(inRas)
    if bands==None:
        bands = inDataset.RasterCount
    
    outDataset = _copy_dataset_config(inRas, outMap = outRas,
                                     bands = bands)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    #outBand = outDataset.GetRasterBand(1)
    # So with most datasets blocksize is a row scanline
    if blocksize==None:
        blocksize=256
    blocksizeX = blocksize
    blocksizeY = blocksize

    # Key issue now is to speed this part up 
    tempRas = inRas[:-4]+'av_.tif'
    print('filtering image')
    cmd = ['otbcli_Smoothing', '-in', inRas, '-out', tempRas, '-type', 'mean',
           '-type.mean.radius', windowsize]
    subprocess.call(cmd) 
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
    _ecdf = np.cumsum(counts).astype(np.float64)
    _ecdf /= _ecdf[-1]
        
    return vals, _ecdf

def hist_match(inputImage, templateImage):
    
    
    
    # TODO optimise with either cython or numba
    
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
    
    inputImage : string
                 image to transform; the histogram is computed over the flattened array
            
    templateImage : string
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
    The multi temp filter for radar data as outlined & published by
    Quegan et al, Uni of Sheffield
    
    This is only suitable for small images, as it holds intermediate data in memory
        
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
    
    outDataset = _copy_dataset_config(inDataset, outMap = outRas,
                                     bands = bands)
    


    # So with most datasets blocksize is a row scanline
     # size of the pixel...they are square so thats ok.
    #if not would need w x h
    #If the block is a row, this simplifies things a bit
    # Key issue now is to speed this part up 
    # 
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
    imFinal =np.empty((outDataset.RasterYSize, outDataset.RasterXSize, bands))    
    #imFinal = np.empty((data.shape[0], data.shape[1], bands))
    for band in range(1,bands+1):
        imFinal[:,:,band-1] = mStack[:,:,band-1] * ovMean
        outBand = outDataset.GetRasterBand(band)
        outBand.WriteArray(imFinal[:,:,band-1])
                    #print(i,j)
        

    #outBand.FlushCache()
    outDataset.FlushCache()

def temporal_comp(fileList, outMap, stat = 'percentile', q = 95, folder=None,
                  blocksize=None,
                  FMT=None,  dtype = gdal.GDT_Int32):
    
    """
    Calculate an image beased on a time series collection of imagery (eg a years woth of S2 data)
            
    Parameters 
    ---------- 
    
    FileList : list of strings
               the files to be inputed, if None a folder must be specified
    
    outMap : string
             the output raster calculated

    	stat : string
           the statisitc to be calculated         

    blocksize : int
                the chunck processed 

    q : int
        the ith percentile if percentile is the stat used         
    
    FMT : string
          gdal compatible (optional) defaults is tif

    dtype : string
            gdal datatype (default gdal.GDT_Int32)
    """
    
    
    if fileList is None:
        rasterList = glob2.glob(path.join(folder,'*.tif'))
    else:
        rasterList = fileList
    openList = [gdal.Open(i) for i in rasterList]
    
    inDataset = gdal.Open(rasterList[0])
    bands = inDataset.RasterCount
    
    outDataset = _copy_dataset_config(rasterList[1], outMap = outMap,
                                     bands = bands)
        
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
        elif stat == 'std':
            stats = np.nanstd(X, axis=2)
        elif stat == 'percentile':
            stats = np.nanpercentile(X, q, axis=2)    
        elif stat == 'median':
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
    
    inRas : string
               input Raster
    
    outMap : string
             the output raster calculated

    stat : string
           the statisitc to be calculated make sure there 
           are no nans as nan percentile is far too slow        

    blocksize : int
                the chunck processed 

    q : int
        the ith percentile if percentile is the stat used         
    
    FMT : string
          gdal compatible (optional) defaults is tif

    dtype : string
            gdal datatype (default gdal.GDT_Int32)
    """
    
    

    inDataset = gdal.Open(inRas)
    
    
    ootbands = len(bandList)
    
    bands = inDataset.RasterCount
    
    outDataset = _copy_dataset_config(inDataset, outMap = outMap,
                                     bands = 1)
        
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
        elif stat == 'std':
            stats = np.nanstd(X, axis=2)
        elif stat == 'percentile':
            # slow as feck
            stats = np.percentile(X, q, axis=2)    
        elif stat == 'median':
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
                         dtype = gdal.GDT_Int32, bands = 1):
    """Copies a dataset without the associated rasters.

    """

    
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