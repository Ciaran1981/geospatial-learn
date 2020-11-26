#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The convnet utils module. 

Description
-----------
Admin utils for convmets using pytorch

"""
from geospatial_learn import raster as rs
import numpy as np
from skimage.exposure import rescale_intensity 
import os
from glob2 import glob
import matplotlib.pyplot as plt
# Albumentations
from collections import defaultdict
import copy
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ternausnet.models
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import segmentation
import gdal
import segmentation_models_pytorch as smp
cudnn.benchmark = True

# handy to know
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

def raster2arrayc(inRas):
    
    """
    Read a raster and return an array of all bands

    
    Parameters
    ----------
    
    inRas: string
                  input  raster 
                  
    """
    rds = gdal.Open(inRas)
    
   

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
    
    rcount = rds.RasterCount+1
    
    inArray = np.zeros((rds.RasterYSize, rds.RasterXSize, rcount-1),
                       dtype=inDt) 
    
    
    for band in range(0, rcount-1):  
        rA = rds.GetRasterBand(band+1).ReadAsArray()
        inArray[:, :, band]=rA
   
   
    return inArray
def cleanupdir(inDir):
    files = glob(os.path.join(inDir, "*.tif"))
    [os.remove(f) for f in files]


def test256(inList):   
    outList = []
    for idx, i in enumerate(inList):
        inR = gdal.Open(i)

        xSize = inR.RasterXSize
        ySize = inR.RasterYSize
        if xSize != 256:
            continue
        if ySize != 256:
            continue
        outList.append(i)
    return outList

# rename label files if different from 
def rename_labels(trainInit, planetInit, inRas, inLabel):
    
    _, replaceStr = os.path.split(inRas)
    _, origStr = os.path.split(inLabel)

    [os.rename(f, f.replace(origStr[:-4], 
               replaceStr[:-4])) for f in trainInit]

    
def prep_mask(mask):
    mask = mask.astype(np.float32)

    return mask


def display_image_grid(imgNms, outRasDir,  mask,
                       outLabelDir, predMasks=None, maxIm=3, bands=[1,2,3]):
    
    cols = 3 if predMasks else 2
    rows = maxIm
    
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
    for i, imgNm in enumerate(imgNms):

        image = rs.raster2array(os.path.join(outRasDir, imgNm),
                               bands=bands)
    
        image = rescale_intensity(image, out_range="uint8")

        mask = rs.raster2array(os.path.join(outLabelDir, imgNm),
                               bands=[1])
        mask = prep_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Train/Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        

        if predMasks:
            predicted_mask = predMasks[i]
            ax[i, 2].imshow(predicted_mask, interpolation="nearest")
            ax[i, 2].set_title("Predicted mask")
            ax[i, 2].set_axis_off()
        if i == maxIm-1:
            break
    plt.tight_layout()
    plt.show()



def visAug(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        if len(image.shape) > 2:
            image = image[:,:,0:3]       
        image = rescale_intensity(image, out_range="uint8")
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )

def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            metric_monitor.update("Loss", loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

# The method of doing this in torch vision is pretty similar            
def create_model(params, proc="cuda:0"):
#    os.environ['CUDA_VISIBLE_DEVICES'] = proc

    if params["model"] == "UNet11" or params["model"] == "UNet16":
        model = getattr(ternausnet.models, params["model"])(pretrained=True)
        hrdWare = torch.device(proc)
        model = model.to(hrdWare)
        
    else:
        #Unet, UNet11, UNet16, ULinknet, FPN, PSPNet,PAN, DeepLabV3 and DeepLabV3+
        if params["model"] == 'Unet':
            model = smp.Unet(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels'])     
        if params["model"] == 'Linknet':
            model = smp.Linknet(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels']) 
        if params["model"] == 'FPN':
            model = smp.FPN(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels']) 
        if params["model"] == 'PSPNet':
            model = smp.PSPNet(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels']) 
        if params["model"] == 'PAN':
            model = smp.PAN(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels']) 
        if params["model"] == 'DeepLabV3':
            model = smp.DeepLabV3(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels']) 
        if params["model"] == 'DeepLabV3+':
            model = smp.DeepLabV3(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels'])
        hrdWare = torch.device(proc)
        model = model.to(hrdWare)        
    
    return model


def train_and_validate(model, trainData, valData, params):
    train_loader = DataLoader(
        trainData,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        valData,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    criterion = nn.BCEWithLogitsLoss().to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params)
        validate(val_loader, model, criterion, epoch, params)
    return model

def predict(model, params, testData, batch_size):
    test_loader = DataLoader(
        testData, batch_size=batch_size, shuffle=False, num_workers=params["num_workers"], pin_memory=True,
    )
    model.eval()
    predTest = []
    with torch.no_grad():
        for images, (original_heights, original_widths) in test_loader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predMasks = (probabilities >= 0.5).float() * 1
            predMasks = predMasks.cpu().numpy()
            for predicted_mask, original_height, original_width in zip(
                predMasks, original_heights.numpy(), original_widths.numpy()
            ):
                predTest.append((predicted_mask, original_height, original_width))
    return predTest

