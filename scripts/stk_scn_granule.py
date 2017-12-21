#!/home/ubuntu/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
@author: Ciaran Robb
author: Ciaran Robb
Research Associate in Earth Observation
Centre for Landscape and Climate Research (CLCR)
Department of Geography, University of Leicester, University Road, Leicester, 
LE1 7RH, UK 

If you use code to publish work cite/acknowledge me and authors of libs as 
appropriate 
"""
#Context hack for now

#if __name__ == '__main__':
#    import os, sys
#    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geospatial_learn import learning, geodata #shape #, handyplots, data 
import os
import glob2
import argparse
import numpy as np
#from datetime import datetime
#import os
#import gdal
from more_itertools import unique_everseen
#from tqdm import tqdm
#from skimage.morphology import  remove_small_objects
#from datetime import datetime
from joblib import Parallel, delayed
#import warnings
#from sentinelsat.sentinel import SentinelAPI, get_coordinates
import re
#import json
import subprocess
from os import mkdir, path

parser = argparse.ArgumentParser()
parser.add_argument("-folder", "--aoi", type=str, required=True, 
                    help="folder")

parser.add_argument("-aoi", "--aoinm", type=str, required=True, 
                    help="name of aoi")

parser.add_argument("-granule", "--granule_nm", type=str, required=True, 
                    help="name of granule eg 36MYE")

parser.add_argument("-model", "--scnmdl", type=str, required=True, 
                    help="model path .gz")

parser.add_argument("-polygon", "--clpPoly", type=str, required=False, 
                    help="polygon to clip S2 scene (ogr compatible)")

parser.add_argument("-mask", "--mask", type=bool, required=False, 
                    help="Raster forest mask (optional)")


args = parser.parse_args() 


parentFolder= args.aoi

aoi = args.aoinm
scratch = path.join(parentFolder, 'scratch')
stacks = path.join(scratch,'stacks')
baseDir  = path.join(parentFolder,'baseImage')

dirs = [scratch, stacks, baseDir]

for fld in dirs:
    if os.path.exists(fld):
        continue
    mkdir(fld)
    
tileId = args.granule_nm
baseImage = baseDir + tileId+'.tif'



clipShape = args.clpPoly



cloudModel = args.scnmdl


"""
###############################################################################
Create raster stacks of both the 10 & 20m imagery
###############################################################################
"""
#print('stacking 10 & 20m bands')
# Might this be better run in parallel from bash?
l2aList = glob2.glob(parentFolder+'/*L2A*'+tileId+'*.SAFE')
if l2aList == []:
    l2aList = glob2.glob(parentFolder+'/*/*L2A*'+tileId+'*.SAFE') 

l2aList.sort()



    
# TODO Perhaps a bit inefficient - glob2 bug partly responsible
paths = list()
for item in l2aList:
    granuleSet = list(unique_everseen(glob2.glob(item+'/GRANULE/*/')))
    paths.append(granuleSet[0])

print('making base image')
outBse = geodata.stack_S2(paths[0], blocksize = 2048)

print('classifying scene and masking cloud')

ootBseScn = outBse[:-4]+'_scn'+'.tif'

learning.classify_pixel_bloc(cloudModel, outBse, 9, ootBseScn[:-4],
                             blocksize=256)

sen_scnFile = geodata.jp2_translate(paths[0], FMT=None, mode='scene')
    
geodata.combine_scene(sen_scnFile, ootBseScn)

geodata.remove_cloud_S2(outBse, ootBseScn, blocksize=256)


bscmd = ['gdal_translate', '-outsize', '100%', '100%', '-of', 'GTiff',
               outBse, baseImage]

subprocess.call(bscmd)
# There will only ever be a couple of tiles at a time so this is quick enough
# but separate scripts are written just in case there is an issue
# Both taking arounf 1 min per granule - this need

# 10m 

def stk20(inRas):
    kwargs = {'mode':'20', 'blocksize': 2048}
    stk = geodata.stack_S2(inRas, **kwargs)
    return stk


changeNames = []

nms = np.arange(len(l2aList))

# This is clumsy but does the job
for i in range(0, len(l2aList)-1):
    changeNames.append((l2aList[i],l2aList[i+1])) 

    
cnms = np.arange(len(changeNames))    
for item in cnms:
    

        
    l2aList = changeNames[item][1]
        
    # TODO Perhaps a bit inefficient - glob2 bug partly responsible
    paths = list()
    granuleSet = glob2.glob(l2aList+'/GRANULE/*/')
    #paths.append(granuleSet)
    sclFile = geodata.jp2_translate(granuleSet[0], FMT=None, mode='scene')

         
    kwargList = [{'mode':None, 'blocksize': 2408},
         {'mode':'20', 'blocksize': 2048}]
    things = np.arange(len(kwargList)) 
    print('stacking 10 & 20m bands')            
    stkList10m = Parallel(n_jobs=-1,verbose=5)(delayed
                         (geodata.stack_S2)(granuleSet[0],
                          **kwargList[i]) for i in things)
    stkList20m = stkList10m[1]

    stkList10m  = stkList10m[0]

    """
    ###############################################################################
    The next wee bit is to collect the corresponding base rasters (composites)
    from lists the rasters addresses
    As the availability of granules varies with each aquisition the functionality 
    has to be flexible, ensuring incorrect pixel values are not added to the
    temporal composites
    ###############################################################################
    """
    print('creating file lists')

    match = re.search(r'\d{4}\d{2}\d{2}', stkList10m)
    end_date = match.group(0)
        

    changeName = stacks+aoi+'_changeto_'+end_date+'_'+tileId+'.tif'

    """
    ###############################################################################
    Scene Classification - this is to remove cloud and shadow-----------------
    Previous attempts have used sen2cor's own scene classification and active
    contours, but machine learning approach (RF) appears to be most effective, 
    albeit adding overall processing time
    
    ###############################################################################
    """
    print('processing scene classification & cloud removal')
    


    dr, file = os.path.split(baseImage)
    sceneRas20 =  stkList20m+'_scene.tif'

    sceneRas10 = stkList10m[:-4]+'_scene.tif'

        

    stuff = np.arange(len(stkList20m))
    
    
#    for item in stuff:
#        if os.path.exists(sceneRas20):
#            print('scene map '+str(item)+' exists moving on')            
#            continue
#        elif os.path.exists(sceneRasList20[item]+'.tif') and item is len(stuff):
#            print('scene map exists '+str(item)+' moving on')
#            break       
#        else:
    learning.classify_pixel_bloc(cloudModel, stkList20m, 9,
                                 sceneRas20[:-4], blocksize=256)
    
    geodata.combine_scene(sclFile, sceneRas20)
        
        
    
    print('resampling scene-map from 20 to 10m')
    
    #TODO Cut out the serial subprocess usage eugh!!!
    res_cmd = ['gdal_translate', '-outsize', '200%', '200%', '-of', 'GTiff',
               sceneRas20, sceneRas10]
    subprocess.call(res_cmd)

    
    # Now to create a cloud free composite with the 2 scene classifications

    
    print('removing cloud')

    geodata.remove_cloud_S2_stk(stkList10m, sceneRas10, 
                                baseIm=baseImage, dist=2, max_size=20)
        #print('matching histogram')
        #geodata.hist_match(stkList10m[item], templateRas)
        
    print('stacking base and new images')
    


    geodata.stack_ras([baseImage, stkList10m], changeName)
    
    subprocess.call(['rm', '-rf', baseImage])

    bWrite = ['gdal_translate', '-of', 'Gtiff', stkList10m,
              baseImage]
    subprocess.call(bWrite)

    


    if clipShape != None:
        
        fld, file = path.split(changeName[:-4])
    
        clipped = fld+file+'_clip.tif'
    
        geodata.clip_raster(changeName, clipShape, clipped, 
                        nodata_value=0)
        print(file+'  clipped')
    else:
        print(file+' done')
#    if args.mask is True:
#        geodata.mask_raster_multi(clipped, mval=2, mask=maskRas)
#        print(file+'masked, all done')
