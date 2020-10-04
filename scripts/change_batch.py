#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:34:56 2017

Script for classifying change over a period of time with image pairs
This assumes you have a sklearn-based model, folder organised via stk_sc_granule.py & a polygon to clip the scene to AOI
author: Ciaran Robb
Research Associate in Earth Observation
Centre for Landscape and Climate Research (CLCR)
Department of Geography, University of Leicester, University Road, Leicester, 
LE1 7RH, UK 

If you use code to publish work cite/acknowledge me and authors of libs as 
appropriate 
"""

from geospatial_learn import learning, raster,  shape #, handyplots, data 
import os
import argparse
from os import  path
import gdal
from glob2 import glob
import re
import subprocess


gdal.UseExceptions()


parser = argparse.ArgumentParser()


args = parser.parse_args() 


parser.add_argument("-folder", "--flder", type=str, required=True, 

                    help="folder with imagery L2A to be classified")

parser.add_argument("-model", "--mdl", type=str, required=True, 
                    help="model path '.gz file'")


parser.add_argument("-polygon", "--clpPoly", type=str, required=False, 
                    help="polygon to clip S2 scene (optional)")

parser.add_argument("-mask", "--mask", type=bool, required=False, 
                    help="Raster forest mask (optional)")




modelPth = args.mdl

    

parentFolder= args.flder



parentFolder= args.aoi

aoi = args.aoinm


scratch = path.join(parentFolder,'scratch')
stacks = path.join(scratch,'stacks')
baseDir  = path.join(parentFolder,'baseImage')
outputData  = path.join(parentFolder, 'outputData')
changeMaps = path.join(parentFolder, 'changeMaps')

dirs = [scratch, stacks, baseDir, outputData, changeMaps]


tileId = args.granule_nm
baseImage = path.join(baseDir, tileId+'.tif')


clipShape = args.clpPoly

stackList = glob(stacks+'*clip*.tif')
stackList.sort()


for image in stackList:
    
    dr, name =os.path.split(image)
    outMap = path.join(changeMaps, name,'_10m_ch')
    probMap = path.join(changeMaps, name,'_10m_prob')
    clipRas = path.join(changeMaps, name,'_10m_ch_clip_.tif')
    
    json = path.join(outputData, name,'_10m_ch_clip.geojson')
    outKml = path.join(outputData, name,'_10m_ch_clip')
       

    print('commencing change classification')

    learning.classify_pixel_bloc(modelPth, image, 8, outMap, 
                                 blocksize = 256, FMT='Gtiff')
    
    
    

    print('producing model probability map')


    learning.prob_pixel_bloc(modelPth, image, 8, probMap,
                             7, blocksize=256, one_class =1)
    

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
    
    raster.mask_raster(outMap+'.tif', 1, overwrite=False, 
                        outputIm = dF[:-4])
    
    raster.mask_raster(probMap+'.tif', 1, overwrite=False, 
                        outputIm = dF[:-4]+'_prob.tif')
    
    
    outDShp = dF+'_DF.shp'
    raster.polygonize(dF+'.tif', outDShp)
    

    fld, file = os.path.split(dF)    
    date = re.findall('\d+', file)
    date = date[0]
    
    dateNew = int(date[6:8]+date[4:6]+date[2:4])
    
    
    shape.zonal_stats(outDShp, probMap+'.tif', 1, 'Prob', write_stat = True)
    shape.shape_props(outDShp,'Area', label_field = 'DN')
    shape.shape_props(outDShp,'Perimeter', label_field = 'DN')

    shape.shape_props(outDShp,'MajorAxisLength', label_field = 'DN')
    shape.shape_props(outDShp,'MinorAxisLength', 
                      label_field = 'DN')
    shape.write_text_field(outDShp,'Date', str(date))
           
    
        
#    print('producing json for database')
    
    
    
##    polyJscmd = ['ogr2ogr', '-f', '"Geojson"', 
#                 json[:-4]+'.geojson', outDShp,
#                 '-s_srs', 'EPSG:32736',  '-t_srs', 'EPSG:4326']
#    
#    subprocess.call(polyJscmd) 
