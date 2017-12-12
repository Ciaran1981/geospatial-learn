#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:34:56 2017

Script for classifying change over a period of time with image pairs
author: Ciaran Robb
Research Associate in Earth Observation
Centre for Landscape and Climate Research (CLCR)
Department of Geography, University of Leicester, University Road, Leicester, 
LE1 7RH, UK 

If you use code to publish work cite/acknowledge me and authors of libs as 
appropriate 
"""

from geospatial_learn import learning, geodata,  shape #, handyplots, data 
import os
#import glob2
import argparse
#import numpy as np
#from datetime import datetime
from os import  path, mkdir
import gdal
from glob2 import glob
import re
import subprocess


gdal.UseExceptions()
#ogr.UseExceptions()

parser = argparse.ArgumentParser()




parser.add_argument("-folder", "--flder", type=str, required=True,
                    help="folder with imagery L2A to be classified")

parser.add_argument("-model", "--mdl", type=str, required=True, 
                    help="model path '.gz file'")

parser.add_argument("-granule", "--granule_nm", type=str, required=False, 
                    help="name of granule eg 36MYE")


parser.add_argument("-polygon", "--clpPoly", type=str, required=False, 
                    help="polygon to clip S2 scene (optional)")

parser.add_argument("-mask", "--mask", type=bool, required=False, 
                    help="Raster forest mask (optional)")


args = parser.parse_args() 


modelPth = args.mdl

    
scratch = args.flder


outputData  = path.join(scratch, 'outputData')
changeMaps = path.join(scratch, 'changeMaps')

dirs = [scratch, outputData, changeMaps]

for fld in dirs:
    if os.path.exists(fld):
        continue
    mkdir(fld)
    
tileId = args.granule_nm
#baseImage = path.join(baseDir, 'T'+tileId+'.tif')




clipShape = args.clpPoly

stackList = glob(path.join(scratch, '*.tif'))
stackList.sort()

#items = np.arange(len(stackList))

for image in stackList:
    
    dr, name =os.path.split(image)
    outMap = path.join(changeMaps, name+'_10m_ch')
    probMap = path.join(changeMaps, name+'_10m_prob')
    clipRas = path.join(changeMaps, name+'_10m_ch_clip_.tif')
    
    json = path.join(outputData, name+'_10m_ch_clip.geojson')
    outKml = path.join(outputData, name+'_10m_ch_clip')
       

    print('commencing change classification')
    #    if os.path.exists(outMapList[item]+'.tif'):
    #        print('change map '+str(item)+' exists moving on')
    
    #    else:

    learning.classify_pixel_bloc(modelPth, image, 8, outMap, 
                                 blocksize = 256, FMT='Gtiff')
    
    
    

    print('producing model probability map')
    ##for item in items:
    #    if os.path.exists(probMapList[item]+'.tif'):
    #        print('probability map '+str(item)+' exists moving on')
    #        pass
    ##    elif os.path.exists(probMapList[item]+'.tif') and item is len(items):
    ##        print('probability map '+str(item)+' exists moving on')
    ##        pass
    #    else:

    learning.prob_pixel_bloc(modelPth, image, 8, probMap,
                             7, blocksize=256)
    
    
    #==============================================================================
    #==============================================================================

    print('sieving change map')
    
    # Have replaced subprocess with api now eliminating the need for sievelist
    noiseRas = gdal.Open(outMap+'.tif', gdal.GA_Update)
    noiseBand = noiseRas.GetRasterBand(1)
    prog_func = gdal.TermProgress
    result = gdal.SieveFilter(noiseBand, None, noiseBand, 4, 4,
                              callback = prog_func)
    
    
    noiseRas.FlushCache()
    noiseRas = None
    noiseBand = None
    result = None
    
    
    print('producing deforest only raster')
    dF = outMap[:-4]+'_DF'
    
    geodata.mask_raster(outMap+'.tif', 1, overwrite=False, 
                        outputIm = dF[:-4])
    
    geodata.mask_raster(probMap+'.tif', 1, overwrite=False, 
                        outputIm = dF[:-4]+'_prob.tif')
    
    outDShp = dF+'.shp'   
    geodata.polygonize(dF+'.tif', outDShp)
    
    # mask arg is required for speed
#    print('polygonising deforest map')
#    
#    polyCmd = ['gdal_polygonize.py', '-mask', dF+'.tif',
#               dF+'.tif', '-f', "ESRI Shapefile", 
#                outDShp]

#    subprocess.call(polyCmd) 
    

    fld, file = os.path.split(dF)    
    date = re.findall('\d+', file)
    date = date[0]
    
    dateNew = int(date[6:8]+date[4:6]+date[2:4])
    
    # TODO
    # reinstate dateraster - wasn't working last time
    #        geodata.mask_raster(dateRas, 1, overwrite=True)
    #    else:
    #        geodata.date_raster(dateRas, dF[item]+'.tif')    
    # Here a load of zonal/shape stats are calculated
    
    shape.zonal_stats(outDShp, probMap+'.tif', 1, 'Prob', write_stat = True)
    shape.shape_props(outDShp,'Area', label_field = 'DN')
    shape.shape_props(outDShp,'Perimeter', label_field = 'DN')

    shape.shape_props(outDShp,'MajorAxisLength', label_field = 'DN')
    shape.shape_props(outDShp,'MinorAxisLength', 
                      label_field = 'DN')
    #shape.write_text_field(outDShp,'County', county)
    shape.write_text_field(outDShp,'Date', str(date))
           
    
        
    print('producing json for database')
    
    
    
    polyJscmd = ['ogr2ogr', '-f', '"Geojson"', 
                 json[:-4]+'.geojson', outDShp,
                 '-s_srs', 'EPSG:32736',  '-t_srs', 'EPSG:4326']
    
    subprocess.call(polyJscmd) 
