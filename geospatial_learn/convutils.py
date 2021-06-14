#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The convnet utils module. 

Description
-----------
Admin utils for convmets using pytorch

"""
from PIL import Image
from geospatial_learn import raster as rs
import cv2
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
import skimage.morphology as skm
import gdal
import pandas as pd
gdal.UseExceptions()
cudnn.benchmark = True

# handy to know
#numpy_type_map = {
#    'float64': torch.DoubleTensor,
#    'float32': torch.FloatTensor,
#    'float16': torch.HalfTensor,
#    'int64': torch.LongTensor,
#    'int32': torch.IntTensor,
#    'int16': torch.ShortTensor,
#    'int8': torch.CharTensor,
#    'uint8': torch.ByteTensor,
#}

def close_mask(inRas, noclasses=1):
    
    
    rds = gdal.Open(inRas, gdal.GA_Update)
    
    #lazy
    inbw=rs.raster2array(inRas)
    
    if noclasses>1:
       bw = skm.closing(inbw)
    else:
        bw = skm.binary_closing(inbw)
    rds.GetRasterBand(1).WriteArray(bw)
    rds.FlushCache()
    rds = None
    

# for mask to be returned to a readable state
def torch2np(outputs):
    outputs = outputs.squeeze(0).permute(1, 2, 0).numpy()
    return outputs

def filtermask(inList):
    
    outList = []
    
    for i in inList:
        img = Image.open(i)
        if img.getextrema()[1] == 0:
            continue
        else:
            outList.append(i)
        del img
    return outList

def matchImgList(trainList, imgFolder):
    
    outList = []
    
    for i in trainList:
        hd, tl = os.path.split(i)
        imgpath = os.path.join(imgFolder, tl)
        outList.append(imgpath)
       
    return outList
# TODO - see notebook
# For chip classifications 

def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()


def train_chip(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True).float().view(-1, 1)
        output = model(images)
        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )

def validate_chip(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True).float().view(-1, 1)
            output = model(images)
            loss = criterion(output, target)
            accuracy = calculate_accuracy(output, target)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

def display_chips(images_filepaths, predicted_labels, true_label = "noclouds", 
                  predicted_label="clouds", cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


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


def testtile(inList, tileSize=256):   
    outList = []
    for idx, i in enumerate(inList):
        inR = gdal.Open(i)

        xSize = inR.RasterXSize
        ySize = inR.RasterYSize
        if xSize != tileSize:
            continue
        if ySize != tileSize:
            continue
        outList.append(i)
    return outList

def get_classes(inList):   
    
    nList = []
    cList = []
    lList = []
    
    for idx, i in enumerate(inList):
        arr = rs.raster2array(i)
        vals = np.unique(arr)
        nList.append(i)
        cList.append(vals[vals!=0])
        lList.append(len(vals))
    
    df = pd.DataFrame(columns=["File", "Classes", "noClasses"])
    
    df["File"]=nList
    df["Classes"]=cList
    df["noClasses"]=lList

    return df



# rename label files if different from 
def rename_labels(trainInit, planetInit, inRas, inLabel):
    
    _, replaceStr = os.path.split(inRas)
    _, origStr = os.path.split(inLabel)

    [os.rename(f, f.replace(origStr[:-4], 
               replaceStr[:-4])) for f in trainInit]

    
def prep_mask(mask):
    mask = mask.astype(np.float32)

    return mask

def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def image_grid(imgNms, mskNms, predMasks=None, maxIm=3, bands=[1,2,3]):
    
    cols = 3 if predMasks else 2
    rows = maxIm
    
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
    for i, imgNm in enumerate(imgNms):

        image = rs.raster2array(imgNm, bands=bands)
    
        image = rescale_intensity(image, out_range="uint8")

        mask = rs.raster2array(mskNms[i], bands=[1])
        
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
def create_model(params, activation, proc="cuda:0"):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if params["model"] == "UNet11" or params["model"] == "UNet16":
        model = getattr(ternausnet.models, params["model"])(pretrained=True)
        if torch.cuda.device_count() > 1: 
            #consider also DistributedDataParallel
            model= nn.DataParallel(model)
        hrdWare = torch.device(proc)
        model = model.to(hrdWare)
        
    else:
        #Unet,  UNet16, ULinknet, FPN, PSPNet,PAN, DeepLabV3 and DeepLabV3+
        if params["model"] == 'Unet':
            model = smp.Unet(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels'],
                        activation=activation)     
        if params["model"] == 'Linknet':
            model = smp.Linknet(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels'],
                        activation=activation) 
        if params["model"] == 'FPN':
            model = smp.FPN(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels'],
                        activation=activation) 
        if params["model"] == 'PSPNet':
            model = smp.PSPNet(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels'],
                        activation=activation) 
        if params["model"] == 'PAN':
            model = smp.PAN(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels'],
                        activation=activation) 
        if params["model"] == 'DeepLabV3':
            model = smp.DeepLabV3(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels'],
                        activation=activation) 
        if params["model"] == 'DeepLabV3+':
            model = smp.DeepLabV3(encoder_name=params['encoder'], 
                        classes=params['classes'],in_channels=params['in_channels'],
                        activation=activation)
        if torch.cuda.device_count() > 1: 
            #consider also DistributedDataParallel
            model= nn.DataParallel(model)
            model = model.to(device)
        else:
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

def predict(model, testData, batch_size, nw, device):
    
    # TODO parallelize prediction? 
    test_loader = DataLoader(
        testData, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    model.eval()
    predTest = []
    with torch.no_grad():
        for images, (original_heights, original_widths) in test_loader:
            
            images = images.to(device, non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predMasks = (probabilities >= 0.5).float() * 1
            predMasks = predMasks.cpu().numpy()
            for predicted_mask, original_height, original_width in zip(
                predMasks, original_heights.numpy(), original_widths.numpy()
            ):
                predTest.append((predicted_mask, original_height, original_width))
    return predTest

def pad_predict(inRas, outputIm, model, classes, preprocessing,
                    blocksize = 256, FMT ='Gtiff', bands=[1,2,3],
                    device='cuda'):
    """ 
    Perform a perblock prediction on a raster
    
    Parameters 
    ----------- 
    
    inputIm: string
              the input raster
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

    outDataset = rs._copy_dataset_config(inDataset, outMap = outputIm,
                                 bands = 1)
    ootBnd = outDataset.GetRasterBand(1)
        
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize

    blocksizeX = blocksize
    blocksizeY = blocksize
    
    # the number of block in a row is simply rounding up the division
    # us np floor to ensure round down
    # TODO  use this to predict entire rows rather than chip wise
    #blkrow = np.floor(rows / blocksize) +1
    #blkcol = np.floor(cols / blocksize) +1
    
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
                
                image = mb2array(inDataset, j, i, numCols, numRows, 
                                 bands=bands)
                
                pred = pred_img(image, preprocessing, model, device,
                         classes, blocksize)

                ootBnd.WriteArray(pred, j, i)
    
               
    outDataset.FlushCache()
    outDataset = None 

def chip_pad_predict(inRas, outputIm, model, classes, preprocessing,
                    blocksize = 256, FMT ='Gtiff', bands=[1,2,3],
                    device='cuda'):
    """ 
    Perform a perblock prediction on a raster, classifying the entire block 
    as a single class
    
    Parameters 
    ----------- 
    
    inputIm: string
              the input raster
    FMT: string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    outputIm: string (optional)
               optionally write a separate output image, if None, will mask the input
    inList:  list
           the list (of arrays) of predicted masks
        
    blocksize: int
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
    
   
        
    inDataset = gdal.Open(inRas)

    outDataset = rs._copy_dataset_config(inDataset, outMap = outputIm,
                                 bands = 1)
    ootBnd = outDataset.GetRasterBand(1)
        
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize

    blocksizeX = blocksize
    blocksizeY = blocksize
    
    # the number of block in a row is simply rounding up the division
    # us np floor to ensure round down
    # TODO  use this to predict entire rows rather than chip wise
    #blkrow = np.floor(rows / blocksize) +1
    #blkcol = np.floor(cols / blocksize) +1
    
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
                
                image = mb2array(inDataset, j, i, numCols, numRows, 
                                 bands=bands)
                
                pred = pred_img(image, preprocessing, model, device,
                         classes, blocksize)

                ootBnd.WriteArray(pred, j, i)
    
               
    outDataset.FlushCache()
    outDataset = None 

def maskblock(inRas, outRas, blocksize=256):
    
    
    """
    Collect and save chips of both mask and image from a list of images
    
    The list of images must correspond/be in the same order
    
    Parameters
    ----------
    
    inRas: string
                Path to a mask image containing drawn masks to be converted
                to blocks
    
    outRas: string
                A list of images containing the corresponding spectral info
                
    """
    
    
    rds = gdal.Open(inRas)
    
    inBnd = rds.GetRasterBand(1)
    
    outrds = rs._copy_dataset_config(rds, outMap = outRas,
                                 bands = 1)
    ootBnd = outrds.GetRasterBand(1)
        
    cols = rds.RasterXSize
    rows = rds.RasterYSize

    blocksizeX = blocksize
    blocksizeY = blocksize
    
    # the number of block in a row is simply rounding up the division
    # us np floor to ensure round down
    # TODO  use this to predict entire rows rather than chip wise
    #blkrow = np.floor(rows / blocksize) +1
    #blkcol = np.floor(cols / blocksize) +1
    
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
                
                image = inBnd.ReadAsArray(j, i, numCols, numRows)
                

                if image.max() == 0:
                    continue
                else:
                    oot = np.ones_like(image, dtype=np.uint8)
                    

                ootBnd.WriteArray(oot, j, i)
    
               
    outrds.FlushCache()
    outrds = None 
    
    return 


def to_tensor(x, **kwargs):
    """
    ingestible by torch
    """
    return x.transpose(2, 0, 1).astype('float32')
    
def get_preprocessing_p(preprocessing_fn, tilesize):
    
    """
    custom transforms and pad if needed - p denotes prediction
    """
    
    _transform = [
            A.PadIfNeeded(tilesize, tilesize),
            A.Lambda(image=preprocessing_fn),
            A.Lambda(image=to_tensor, mask=to_tensor),
            ]
    return A.Compose(_transform)    

def convert_pred(inpred, tilesize, classes):
    
    """
    bit of reshaping to get the result 'image ready'
    """
    
    oot = np.zeros(shape=(tilesize, tilesize))
    count = inpred.shape[0]
    for i in range(0, count):
        oot[inpred[i,:,:]==1]=int(classes[i])
    return oot


def pred_img(image, preprocessing, model, device, classes, tilesize):
    """
    convert an image to an appropriate form, predict, unpad, then return
    """
    
    if image.shape[1:2] != (tilesize, tilesize):
        oldshp = image.shape[0:2]
        
    sample = preprocessing(image=image)
    image = sample['image']
    
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    
    if len(classes) > 1: 
        pred = convert_pred(pr_mask, tilesize, classes)
    else:
        # For binary sometimes get weird results here - 
        # neg values and 0 which should be 1
        # hack is to get rid of neg values and make 0 (which should be 1)
        # likel to be dumped as mask should be binary already!
        pred = pr_mask
        pred[pred>0]=1
        pred[pred<0]=0
    
    # this is on the premise the old shape < pred.shape which should always be case
    # as well as the 0th coordinate being top left 
    if oldshp != pred.shape:
        # take a view of the pred shape to return
        pred = pred[:oldshp[0], :oldshp[1]]
        
    
    
    return pred
    



def chip_writer(array, j, i, n_cols, n_rows, rgt, inras, outfile, fmt='Gtiff'):
    """
    convert pixels to geo coords to extract a subset and write to file
    retaining correct positional info
    """
    # Top left of the raster is the x/ymin
    xmin = j * rgt[1] + rgt[0]
    ymin = i * rgt[5] + rgt[3]

    # size of pixel
    pixel_sz = rgt[1]
    
    #Get the info from the input raster
    projection = inras.GetProjection()
    
    # usual gdal setup
    driver = gdal.GetDriverByName(fmt)
    
    bands = inras.RasterCount
    
    dtype = inras.GetRasterBand(1).DataType
    
    # Set the information in the new subset
    dataset = driver.Create(outfile, n_cols, n_rows, bands, dtype)

    dataset.SetGeoTransform((xmin, pixel_sz, 0, ymin, 0, -pixel_sz))    

    dataset.SetProjection(projection)
    
    if bands == 1:
        dataset.GetRasterBand(1).WriteArray(array)
        # Write to disk, deallocate
        dataset.FlushCache()  
        dataset=None
    else:
    # Per band....
        for band in range(1,bands+1):
            arr = array[:,:,band-1]
            dataset.GetRasterBand(band).WriteArray(arr)
        # Write to disk, deallocate
        dataset.FlushCache() 
        dataset=None

def mb2array(rds, j, i, n_cols, n_rows, bands=[1,2,3]):
    
    """
    multi band2array
    """
    # gdal returns a weird shape reading all at once, hence this function
    # np transpose put the img upside down/on side
    
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
    
    inArray = np.zeros((n_rows, n_cols, len(bands)), dtype=inDt) 
    for idx, band in enumerate(bands):  
        rA = rds.GetRasterBand(band).ReadAsArray(j, i, n_cols, n_rows)
        inArray[:, :, idx]=rA
    
    return inArray

def split_rand_tile(tilelist):
    
    final_test = list(np.random.choice(tilelist, 1))
    tl = os.path.split(final_test[0])[1]
    final_test.append(os.path.join(label_t, tl))
    
    return final_test
