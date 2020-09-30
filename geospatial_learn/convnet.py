#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The convnet module. 

Description
-----------

A module for using pytorch to classify EO data for semantic segmentation and 
object identification

"""
from geospatial_learn import handyplots as hp
from geospatial_learn import raster as rs
from geospatial_learn.gdal_merge import _merge 
from geospatial_learn.convutils import *
import numpy as np
import os
from glob2 import glob
import matplotlib.pyplot as plt

import random
import shutil
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import gdal

cudnn.benchmark = True


#mainDir = '/media/ciaran/Storage4A/Phenomics/Planet/DEEP'
## Firstly, I need to have an OS master map topo of fields and planet imagery
## here a number index is added to the outfile name
#inRas = '/media/ciaran/Storage4A/Phenomics/Planet/PlanetSpring/SpringMergeosgb.tif'
## Note that the vector was first bufferred by 3m (a planet pixel), as otherwise
## there was no clear boundary between fields
#inLabel = '/media/ciaran/Storage4A/Phenomics/Planet/DEEP/FieldsRaster.tif'
#
#outRasDir = '/media/ciaran/Storage4A/Phenomics/Planet/DEEP/planet'
#
#outLabelDir =  '/media/ciaran/Storage4A/Phenomics/Planet/DEEP/label'
#
#outMap = '/media/ciaran/Storage4A/Phenomics/Planet/DEEP/Epoch50Big.tif'
#
#
#ootDir = '/media/ciaran/Storage4A/Phenomics/Planet/DEEP/classified'


def semantic_seg(mainDir, inRas, inLabel, outMap, plot=False, bands=[1,2,3], 
                 trainPercent=0.3, f1=False, unet=True,
                 params = {"model": "UNet11",
                           "device": "cuda",
                           "lr": 0.001,
                           "batch_size": 16,
                           "num_workers": 2,
                           "epochs": 50}):
    """
    Semantic Segmentation of EO-imagery - an early version things are to be 
    changed in the near future
    
    Parameters 
    ----------
    
    mainDir: string
              the working directory where everything is done

    model: string
              a choice of "unet, "fcn_resnet50", "fcn_resnet101","deeplabv3_resnet50"
              "deeplabv3_resnet50", "deeplabv3_resnet101"
    
    inRas: string
            the input raster
        
    
    inLabel: string
           the input label raster - at present this the same size as the input
           raster with the class labels, though this will be improved
    
    outMap: string
           the output classification map 
    
    plot: bool
          whether to plot intermediate data results eg visualise the image aug,
          test results etc. 
    
    bands: list of ints
            the image bands to use

    trainPercent: 
                 the percentage of to use as training
    
    f1: bool
        whether to svae a classification report (will be a plot in working dir)
    
    params: dict
          the convnet model params
    
    Notes
    -----
    
    This is an early version with some stuff to add/change eg
    
    The training and classification will be split into two functions
    
    Non intersecting training data to be done
    
    Image augmentations are to be made a param
    
    """
    
    
    outRasDir = os.path.join(mainDir, 'ImageChips')
    
    if os.path.isdir(outRasDir):
        cleanupdir(outRasDir)
    else:
        os.mkdir(outRasDir)
        
    outLabelDir = os.path.join(mainDir, 'LabelChips')
    if os.path.isdir(outLabelDir):
        cleanupdir(outLabelDir)
    else:
        os.mkdir(outLabelDir)
    
    print("prepping data...")
    
    rs.tile_rasters(inRas, outRasDir, ["256", "256"])

    #For the labels
    rs.tile_rasters(inLabel, outLabelDir, ["256", "256"])
    
    print("...done")
    
    # Now that all the data has been 'chipped' a random subset is required for 
    # training and perhaps testing
    # list all files in dir using glob
    
    # TODO this could be made into a fucntion
    trainInit = glob(os.path.join(outLabelDir, "*.tif"))
    trainInit.sort()
    
    # again so updated
    trainInit = glob(os.path.join(outLabelDir, "*.tif"))
    trainInit.sort()
       
    planetInit = glob(os.path.join(outRasDir, "*.tif"))
    planetInit.sort()
    
    # so garbage is gone and all is well behaved
    rename_labels(trainInit, planetInit, inRas, inLabel)
    # repeat so we have the new ones
    trainInit = glob(os.path.join(outLabelDir, "*.tif"))
    trainInit.sort()
    
    # select 0.3 of the files randomly using numpy
    randomInit = np.random.choice(trainInit, int(len(trainInit)*trainPercent))
    trainList = randomInit.tolist()
    trainList = test256(trainList)
    
    randomValid = np.random.choice(planetInit, int(len(trainInit)*.1))
    valList = randomValid.tolist()
    valList = test256(valList)
    
    
    randomTest = np.random.choice(planetInit, int(len(trainInit)*.1))
    testList = randomTest.tolist()
    
    # Here I need to use albumentations to further expand the training set, which,
    # according to various (including them) improves model results
    
    imgNms = [os.path.split(p)[1] for p in planetInit]
    imgNms.sort()
    
    train_files = [os.path.split(p)[1] for p in trainList]
    train_files.sort() 
    
    valNames = [os.path.split(v)[1] for v in valList]
    valNames.sort()
    
    
    test_imgNms =  [os.path.split(t)[1] for t in testList]
    test_imgNms.sort()

    
    trainTfrm = A.Compose(
    [   #A.ToFloat(max_value=65535.0),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        #A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
    )
    
    class planetDataset(Dataset):
        def __init__(self, imgNms, outRasDir, outLabelDir,
                     transform=None):
            self.imgNms = imgNms
            self.outRasDir = outRasDir
            self.outLabelDir = outLabelDir
            self.transform = transform
    
        def __len__(self):
            return len(self.imgNms)
    
        def __getitem__(self, idx):
            imgNm = self.imgNms[idx]
            image = rs.raster2array(os.path.join(self.outRasDir, imgNm),
                                   bands=bands)
            mask = rs.raster2array(os.path.join(self.outLabelDir, imgNm),
                             bands=[1])
            
            mask = prep_mask(mask)
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            return image, mask


    trainData = planetDataset(train_files, outRasDir,
                                  outLabelDir, transform=trainTfrm)
    valTfrm = A.Compose(
        [A.Normalize(), ToTensorV2()]
    )
    
    valData = planetDataset(valNames, outRasDir, 
                                outLabelDir, transform=valTfrm)
    random.seed(42)

    if plot == True:
       visAug(trainData, idx=6)
       
    model = create_model(params)

    model = train_and_validate(model, trainData, valData, params)

        
    
    class planetInf(Dataset):
        def __init__(self, imgNms, outRasDir, transform=None):
            self.imgNms = imgNms
            self.outRasDir = outRasDir
            self.transform = transform
    
        def __len__(self):
            return len(self.imgNms)
    
        def __getitem__(self, idx):
            imgNm = self.imgNms[idx]
            image = rs.raster2array(os.path.join(self.outRasDir, imgNm),
                                   bands=bands)
            original_size = tuple(image.shape[:2])
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image, original_size
    
    testTfrm = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
                    
    testData = planetInf(test_imgNms, outRasDir,
                             transform=testTfrm)
    
    predTest = predict(model, params, testData, batch_size=16)
    
    
    predMasks = []
    for predicted_256x256_mask, original_height, original_width in predTest:
        full_sized_mask = F.resize(
            predicted_256x256_mask, height=original_height, width=original_width, interpolation=cv2.INTER_NEAREST
        )
        predMasks.append(full_sized_mask)

    
    # For the entire dataset
    
    wholeLot = planetInf(imgNms, outRasDir,
                             transform=testTfrm)
    
    predFinal = predict(model, params, wholeLot, batch_size=16)
    
    predicted_Geo = []
    for predicted_256x256_mask, original_height, original_width in predFinal:
        full_sized_mask = F.resize(
            predicted_256x256_mask, height=original_height, width=original_width, interpolation=cv2.INTER_NEAREST
        )
        predicted_Geo.append(full_sized_mask)
    
    # Write back to georeff'd raster
    # Here we could use original tile structure to write the result then merge it 
    # using gdal merge and clear up the mess   
    
    ootDir = os.path.join(mainDir, 'classifChips')
    os.mkdir(ootDir)
    
    # hahaha list comp mania
    imOot = [os.path.join(ootDir, i[:-4]+'cls.tif') for i in imgNms]
    
    
    for idx, p in enumerate(predicted_Geo):
        rs.array2raster(p, 1, planetInit[idx], imOot[idx], gdal.GDT_Byte)
    
    
    
    _merge(names=imOot, out_file=outMap, separate=0)
    
    rs.polygonize(outMap, outMap[:-3]+"shp")
    
    # remove cak
    shutil.rmtree(ootDir)  
    
    shutil.rmtree(outRasDir)
    
    shutil.rmtree(outLabelDir)
    
    # May as well do the accuracy too!
        
    val = rs.raster2array(inLabel)>0
    seg = rs.raster2array(outMap)>0
    
    if f1 == True:
        
        phRp = hp.plot_classif_report(val.flatten(), seg.flatten(),
                                      labels=[True, False], 
                                      target_names=['Segment', 'Background'], 
                                      colmap=plt.cm.gray, 
                                      save=os.path.join(mainDir, 
                                                        'classifreport.png')) 



###############################################################################
#Worth keeping but the above renders it not required
# using this random selection, get the corresponding planet imagery by string
# replacement
# the old
#r1 = os.path.join(outLabelDir, "FieldsRaster")
## the replacement
#r2 = os.path.join(outRasDir, "osgbplanetsemantic")

#planetList = [i.replace(r1, r2) for  i in trainList]
## Finished lists - these will again be re-defined and used later
#trainList.sort()
#planetList.sort()

## Now send all the randomly selected images to other dirs for torch etc later
#trainPth = os.path.join(mainDir, "trainMask")
#os.mkdir(trainPth)
#[cp(t, trainPth) for t in trainList]
## The final input list for the processing
#trainF = glob(os.path.join(trainPth, "*.tif"))
#
#imPth = os.path.join(mainDir, "trainImg")
#os.mkdir(imPth)
#[cp(t, imPth) for t in planetList]
## The final input list for the processing
#imgF = glob(os.path.join(imPth, "*.tif"))
    
# Prob not required as just resize in train transform
