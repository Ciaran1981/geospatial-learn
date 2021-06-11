#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The convnet module. 

Description
-----------

A module for using pytorch to classify EO data for semantic segmentation and 
object identification

"""
#from geospatial_learn import handyplots as hp
from geospatial_learn import raster as rs
from geospatial_learn.gdal_merge import _merge 
from geospatial_learn.convutils import *
import numpy as np
import os
from glob2 import glob
#import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
#import random
import shutil
import albumentations as A
import albumentations.augmentations.functional as F
import cv2
#import cv2
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.backends.cudnn as cudnn
import gdal
import torch
from tqdm import tqdm
cudnn.benchmark = True

def makewrkdir(directory):
    if os.path.isdir(directory):
        cleanupdir(directory)
    else:
        os.mkdir(directory)


def train_semseg_binary(maindir, bands=[1,2,3], train_percent=0.8, f1=False, 
                 proc="cuda:0", tilesize=256, redo=False, activation=None,
                 modelpth='./best_model.pth',
                 params={'model': 'Unet',
                         'encoder': 'resnet34',
                         'in_channels': 3,
                         'classes' : 1,
                         "lr": 0.001,
                         "device": "cuda",
                         "batch_size": 16,
                         "num_workers": 2,"epochs": 50}):
    """
    Train a semantic seg model for binary classification
    
    Based on segmentation_models.pytorch & albumentations
    
    Parameters 
    ----------
    
    maindir: string
              the directory containing the training masks and labels as 
              produced by collect_train is stored
    
    bands: list of ints
            the image bands to use

    train_percent: 
                 the percentage of to use as training
    
    f1: bool
        whether to svae a classification report (will be a plot in working dir)
    
    params: dict
          the convnet model params
          models: 
          Unet, UNet11, UNet16, U Linknet, FPN, PSPNet,PAN, DeepLabV3 and DeepLabV3+
          encoders:
          'resnet18','resnet34','resnet50', 'resnet101','resnet152','resnext50_32x4d',
          'resnext101_32x4d','resnext101_32x8d','resnext101_32x16d','resnext101_32x32d',
          'resnext101_32x48d','dpn68','dpn68b','dpn92','dpn98','dpn107','dpn131','vgg11',
          'vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn','senet154',
          'se_resnet50','se_resnet101','se_resnet152','se_resnext50_32x4d','se_resnext101_32x4d',
          'densenet121','densenet169','densenet201','densenet161','inceptionresnetv2',
          'inceptionv4','efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3',
          'efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7',
          'mobilenet_v2','xception','timm-efficientnet-b0','timm-efficientnet-b1',
          'timm-efficientnet-b2','timm-efficientnet-b3','timm-efficientnet-b4','timm-efficientnet-b5',
          'timm-efficientnet-b6','timm-efficientnet-b7','timm-efficientnet-b8','timm-efficientnet-l2'

    Notes
    -----
    
    This is an early version with some stuff to add/change eg
    
    """
     

    mskdir = os.path.join(maindir, 'masks')
    imgdir = os.path.join(maindir, 'images')

    # Now that all the data has been 'chipped' a random subset is required for 
    # training and perhaps testing
    # list all files in dir using glob
    
    # TODO this could be made into a fucntion
    
    train_init = glob(os.path.join(mskdir, "*.tif"))
    train_init.sort()

    
    planet_init = glob(os.path.join(imgdir,  "*.tif"))
    planet_init.sort()

    random_init = np.random.choice(train_init, int(len(train_init)*train_percent))
    
    
    trainlist = random_init.tolist()
    
    # handled by albumentations
    #trainList = testtile(trainInit, tilesize=tilesize)
    
    #for later- I think I need to replace this with set theory func
    planet_train =  matchImgList(trainlist, imgdir)
    #planetTrain = testtile(planetTrain, tilesize=tilesize)
    
    random_valid = np.random.choice(trainlist, int(len(train_init)*.1))
    vallist = random_valid.tolist()
    #vallist = testtile(vallist, tilesize=tilesize)
    val_img = matchImgList(vallist, imgdir)
    
    
    random_test = np.random.choice(train_init, int(len(train_init)*.1))
    #testlist = testtile(randomTest, tilesize=tilesize)
    testlist = random_test.tolist()
    test_im = matchImgList(testlist, imgdir)
    
    # This and the class structure need addressed also
    img_nms = [os.path.split(p)[1] for p in planet_init]
    img_nms.sort()
    
    train_files = [os.path.split(p)[1] for p in trainlist]
    train_files.sort() 
    
    val_nms = [os.path.split(v)[1] for v in vallist]
    val_nms.sort()
    
    
    test_img_nms =  [os.path.split(t)[1] for t in testlist]
    test_img_nms.sort()

    # Other augs in the various material online do not appear to offer better results
    # than this simple one!
    
    # TODO
    # Note!!! Albu will support 4-band images, but the default normalisation 
    # must be changed
    # Just an extension of the defaults at present.....
    if len(bands) > 3:
        mean=(0.485, 0.456, 0.406, 0.485)
        std=(0.229, 0.224, 0.225, 0.229)
    else:
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
    
    trainTfrm = A.Compose(
    [   #A.ToFloat(max_value=65535.0),
        A.PadIfNeeded(min_height=tilesize, min_width=tilesize),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        #A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # Above 0.5 is shite
        A.Normalize(mean=mean,  std=std),
        ToTensorV2(),
    ]
    )
    
    
    class imgDataset(Dataset):
        def __init__(self, img_nms, imgdir, mskdir,
                     transform=None):
            self.img_nms = img_nms
            self.imgdir = imgdir
            self.mskdir = mskdir
            self.transform = transform
    
        def __len__(self):
            return len(self.img_nms)
    
        def __getitem__(self, idx):
            imgNm = self.img_nms[idx]
            image = rs.raster2array(os.path.join(self.imgdir, imgNm),
                                   bands=bands)
            # now seperate cos of some albu issue
            #image = normalize(image, 0.5, 0.5)
            mask = rs.raster2array(os.path.join(self.mskdir, imgNm),
                             bands=[1])
            
            mask = prep_mask(mask)
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            return image, mask


    trainData = imgDataset(train_files, imgdir,
                                  mskdir, transform=trainTfrm)
    valTfrm = A.Compose(
        [A.Normalize(mean=mean,  std=std), 
         A.PadIfNeeded(min_height=tilesize, min_width=tilesize),
         ToTensorV2()]
    )
    
    valData = imgDataset(val_nms, imgdir, 
                                mskdir, transform=valTfrm)
    random.seed(42)

       
    model = create_model(params, proc=proc, activation=activation)

    modelout = train_and_validate(model, trainData, valData, params)
    
    torch.save(model, modelpth)
    
    
    return modelout    

        
def train_semantic_seg(maindir, plot=False, bands=[1,2,3], 
                 train_percent=0.8, tilesize=256, f1=False, preTrain=True,
                 proc="cuda:0",  activation='softmax2d', classes=['1'],
                 weights='imagenet', modelpth='./best_model.pth',
                 params={'model': 'Unet',
                         'encoder': 'resnet34',
                         'in_channels': 3,
                         'classes' : 1,
                         "lr": 0.0001,
                         "device": "cuda",
                         "batch_size": 16,
                         "num_workers": 2,"epochs": 50}):
    """
    multi-class Semantic Segmentation of EO-imagery - an early version things are to be 
    changed in the near future
    Based on segmentation_models.pytorch & albumentations
    
    Parameters 
    ----------
    
    mainDir: string
              the working directory where everything is done

    modelpth: string
                where to save the model eg 'dir/best_model.pth'
            
    plot: bool
          whether to plot intermediate data results eg visualise the image aug,
          test results etc. 
    
    bands: list of ints
            the image bands to use

    train_percent: string
                 the percentage of to use as training
                 
    tilesize: int
              the size of the image tile used in training and thus classification
              
    classes: list
                a list of strings with the classes eg ['1', '2', '3'] as labelled
                in raster
    weights: string
                the encoder weights, typically imagenet or None for rand init
    
    f1: bool
        whether to svae a classification report (will be a plot in working dir)
    
    activation: string
               the neural net activation function e.e 'sigmoid', softmax2d
    
    params: dict
          the convnet model params
          models: 
          Unet, UNet11, UNet16, U Linknet, FPN, PSPNet,PAN, DeepLabV3 and DeepLabV3+
          encoders:
          'resnet18','resnet34','resnet50', 'resnet101','resnet152','resnext50_32x4d',
          'resnext101_32x4d','resnext101_32x8d','resnext101_32x16d','resnext101_32x32d',
          'resnext101_32x48d','dpn68','dpn68b','dpn92','dpn98','dpn107','dpn131','vgg11',
          'vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn','senet154',
          'se_resnet50','se_resnet101','se_resnet152','se_resnext50_32x4d','se_resnext101_32x4d',
          'densenet121','densenet169','densenet201','densenet161','inceptionresnetv2',
          'inceptionv4','efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3',
          'efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7',
          'mobilenet_v2','xception','timm-efficientnet-b0','timm-efficientnet-b1',
          'timm-efficientnet-b2','timm-efficientnet-b3','timm-efficientnet-b4','timm-efficientnet-b5',
          'timm-efficientnet-b6','timm-efficientnet-b7','timm-efficientnet-b8','timm-efficientnet-l2'

    Notes
    -----
    
    This is an early version with some stuff to add/change 
   
    """

    def makewrkdir(directory):
        if os.path.isdir(directory):
            cleanupdir(directory)
        else:
            os.mkdir(directory)
            
    mskdir = os.path.join(maindir, 'masks')
    imgdir = os.path.join(maindir, 'images')

    # Now that all the data has been 'chipped' a random subset is required for 
    # training and perhaps testing
    # list all files in dir using glob
    
    # TODO this could be made into a fucntion
    
    train_init = glob(os.path.join(mskdir, "*.tif"))
    train_init.sort()

    
    planet_init = glob(os.path.join(imgdir,  "*.tif"))
    planet_init.sort()

    random_init = np.random.choice(train_init, int(len(train_init)*train_percent))
    
    
    trainlist = random_init.tolist()
    
    # handled by albumentations
    #trainList = testtile(trainInit, tilesize=tilesize)
    
    #for later- I think I need to replace this with set theory func
    planet_train =  matchImgList(trainlist, imgdir)
    #planetTrain = testtile(planetTrain, tilesize=tilesize)
    
    random_valid = np.random.choice(trainlist, int(len(train_init)*.1))
    vallist = random_valid.tolist()
    #vallist = testtile(vallist, tilesize=tilesize)
    #val_img = matchImgList(vallist, imgdir)
    
    
    random_test = np.random.choice(train_init, int(len(train_init)*.1))
    #testlist = testtile(randomTest, tilesize=tilesize)
    testlist = random_test.tolist()
    test_im = matchImgList(testlist, imgdir)

    
    # TODO as always, how does one strat sample
    
    # Here we obtain the unique classes to ensure we training evenly
    #    df = get_classes(trainList)
        
        # as a quick hack for now lets take only those with all classes
        # TODO - this needs to change to a strat sample!!!
    #    if params['classes'] > 1:
    #        all_class=df.loc[df['noClasses']==params['classes']]
    #    else:
    #        all_class=df
        
    #    trainList = all_class['File'].to_list()
    #    trainList.sort()
        
     
    # get norm elswhere now
    #    if len(bands) > 3:
    #        mean=(0.485, 0.456, 0.406, 0.485)
    #        std=(0.229, 0.224, 0.225, 0.229)
    #    else:
    #        mean=(0.485, 0.456, 0.406)
    #        std=(0.229, 0.224, 0.225)
    #        

    # Ultimately for smp version
    makewrkdir(os.path.join(maindir, 'trainMsk'))
    makewrkdir(os.path.join(maindir, 'trainImg'))
    makewrkdir(os.path.join(maindir, 'validMsk'))
    makewrkdir(os.path.join(maindir, 'validImg'))
    makewrkdir(os.path.join(maindir, 'testImg'))
    makewrkdir(os.path.join(maindir, 'testMsk'))
    
    # here to stop a list related bug
    val_img = matchImgList(vallist, imgdir)
   
    
    [shutil.copy(t, os.path.join(maindir, 'trainMsk')) for t in trainlist]
    [shutil.copy(i, os.path.join(maindir, 'trainImg')) for i in planet_train]
    [shutil.copy(v, os.path.join(maindir, 'validMsk')) for v in vallist]
    [shutil.copy(m, os.path.join(maindir, 'validImg')) for m in val_img]
    [shutil.copy(e, os.path.join(maindir, 'testMsk')) for e in testlist]
    [shutil.copy(s, os.path.join(maindir, 'testImg')) for s in test_im]
    
    
    
    class Dataset(BaseDataset):

        # a relic to be altered
        CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                   '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

       
        
        def __init__(
                self, 
                images_dir, 
                masks_dir,
                classes=None, 
                augmentation=None, 
                preprocessing=None,
        ):
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
            
            # convert str names to class values on masks
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
            
            self.augmentation = augmentation
            self.preprocessing = preprocessing
        
        def __getitem__(self, i):
            
            
            image = rs.raster2array(self.images_fps[i],
                                       bands=bands)
            
            #image = cv2norm(image)
            mask = rs.raster2array(self.masks_fps[i],
                             bands=[1])
            
            
            # extract certain classes from mask 
            
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
            
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
                
            return image, mask
            
        def __len__(self):
            return len(self.ids)

    def get_training_augmentation():
        train_transform = [
            
  #          A.Normalize(mean=mean,  std=std),
            A.PadIfNeeded(min_height=tilesize, min_width=tilesize),
            A.HorizontalFlip(p=0.5),
    
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.3,
                                  p=1, border_mode=0)]
        return A.Compose(train_transform)
    
    
    def get_validation_augmentation():
        """Add paddings to make image shape divisible by..."""
        test_transform = [
 #           A.Normalize(mean=mean,  std=std),
            A.PadIfNeeded(tilesize, tilesize)
        ]
        return A.Compose(test_transform)
    
    
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
    
    def cv2norm(img):
        
        norm = cv2.normalize(img, None, alpha=0, beta=1, 
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm
    
    def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform
        
        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: Amentations.Compose
        
        """
        
        _transform = [
            A.Lambda(image=preprocessing_fn), # should this be reinstated?
            A.Lambda(image=to_tensor, mask=to_tensor),
        ]
        return A.Compose(_transform)
    

    # param proc is now redundant as it counts the GPUs and spreads the proc
    model = create_model(params, activation, proc="cuda:0")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(params['encoder'],
                                                        weights)
    
    y_train_dir = os.path.join(maindir, 'trainMsk')
    x_train_dir = os.path.join(maindir, 'trainImg')
    
    y_valid_dir = os.path.join(maindir, 'validMsk')
    x_valid_dir = os.path.join(maindir, 'validImg')
    

    
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir,
        classes=classes,
        augmentation=get_training_augmentation(), 

        preprocessing=get_preprocessing(preprocessing_fn),
    )
    
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=classes,
        augmentation=get_validation_augmentation(), 

        preprocessing=get_preprocessing(preprocessing_fn),
        
    )
    
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'],
                              shuffle=True, num_workers=params["num_workers"])
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'],
                              shuffle=False, num_workers=params["num_workers"])
    

    # Something weird here with the losses or activation etc.  
#    if len(classes) > 1:
#        loss = smp.utils.losses.DiceLoss()
#
#    else:
#        # was using dice/f1 but wasn't working well.....
#        loss = smp.utils.losses.BCEWithLogitsLoss()
        
    
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
   # params["lr"]
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=params["lr"]),
    ])
    
    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=params['device'],
        verbose=True,
    )
    
    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=params['device'],
        verbose=True,
    )
    
    
    # train model for x epochs 
    
    max_score = 0
    
    for i in range(0, params['epochs']):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, modelpth)
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
            
    # load best saved checkpoint
    best_model = torch.load(modelpth)
    
    print('running test dataset')
    x_test_dir = os.path.join(maindir, 'testImg')
    y_test_dir = os.path.join(maindir, 'testMsk')
    # create test dataset
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=classes,
    )
    
    test_dataloader = DataLoader(test_dataset)
    
    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=params['device'],
    )
    
    logs = test_epoch.run(test_dataloader)
    
    # test dataset without transformations for image visualization
#    test_dataset_vis = Dataset(
#        x_test_dir, y_test_dir, 
#        classes=classes,
#    )
    
    
    
    return best_model, logs





    
def semseg_pred(inRas, model, outMap, encoder, classes=['1'], tilesize=256,
                bands=[1,2,3],  weights='imagenet', device='cuda'):
    """
    Semantic Segmentation of EO-imagery - an early version things are to be 
    changed in the near future
    Based on segmentation_models.pytorch & albumentations
    
    Parameters 
    ----------
    
    inRas: string
            the input raster
            
    model: string or pytorch model
             the model to predict

    outMap: string
           the output classification map 
           
    encoder: string
           the encoder component of the CNN e.g. resnet34
    
    tilesize: int
          the image chip/tile size that will be processed def 256
    
    bands: list of ints
            the image bands to use

    Notes
    -----
    
    This is an early version with some stuff to add/change 
    
    """
    # move these and above outside the func
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, weights)
    
    # from utils
    preprocessing=get_preprocessing_p(preprocessing_fn, tilesize)
    
    
    if type(model) is str:
        model = torch.load(model)
    
    # if we have trained in parallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    # from utils  do the block proc
    # This is VERY slow with larger images - suspect GDAL I/O of blocks
    # is bottle neck
    pad_predict(inRas, outMap, model, classes, preprocessing,
                    blocksize = tilesize, FMT ='Gtiff', bands=bands,
                    device=device)
        
def chip_pred(inRas, model, outMap, encoder, classes=['1'], tilesize=256,
                bands=[1,2,3],  weights='imagenet', device='cuda'):
    """
    Chip-based prediction of EO imagery 
    Based on pytorch & albumentations
    
    Parameters 
    ----------
    
    inRas: string
            the input raster
            
    model: string or pytorch model
             the model to predict

    outMap: string
           the output classification map 
           
    encoder: string
           the encoder component of the CNN e.g. resnet34
    
    tilesize: int
          the image chip/tile size that will be processed def 256
    
    bands: list of ints
            the image bands to use

    Notes
    -----
    
    This is an early version with some stuff to add/change 
    
    """
    # move these and above outside the func
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, weights)
    
    # from utils
    preprocessing=get_preprocessing_p(preprocessing_fn, tilesize)
    
    def convert_pred(inpred):
        oot = np.zeros(shape=(tilesize, tilesize))
        count = inpred.shape[0]
        for i in range(0, count):
            oot[inpred[i,:,:]==1]=i
        return oot
    
    if type(model) is str:
        model = torch.load(model)
    
    # if we have trained in parallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    # from utils  do the block proc
    # This is VERY slow with larger images - suspect GDAL I/O of blocks
    # is bottle neck
    pad_predict(inRas, outMap, model, classes, preprocessing,
                    blocksize = tilesize, FMT ='Gtiff', bands=bands,
                    device=device)
            
            
def collect_train(masklist, tilelist, outdir, chip_size=256, bands=[1,2,3]):
    
    """
    Collect and save chips of both mask and image from a list of images for a
     semantic segmentation task
    
    The list of images must correspond/ be in the same order
    
    Parameters
    ----------
    
    masklist: list
                A list of images containing the training masks
    
    tilelist: list
                A list of images containing the corresponding spectral info
    
    outdir: string
                Where the training chips will be written
    
    chip_size: int
                the training "chip" size e.g. 256x256 pixels  dependent on the 
                nnet used
    Returns
    -------
    
    A tuple of lists of the respective paths of both masks and corresponding 
    images
    

    """
    # we know this so not in loop
    block_X = chip_size
    block_Y = chip_size
    
    # make the dirs if not there
    mskdir = os.path.join(outdir, 'masks')
    makewrkdir(mskdir)
    tiledir = os.path.join(outdir, 'images')
    makewrkdir(tiledir)
    # TODO  this could be parallelized per raster pair
    for t,m in zip(tilelist, masklist):
        
        tile_rds =  gdal.Open(t)
        
        mask_rds = gdal.Open(m)
        
        cols = mask_rds.RasterXSize
        rows = mask_rds.RasterYSize
        
        mskbnd = mask_rds.GetRasterBand(1)
        
        rgt = mask_rds.GetGeoTransform()

        block_Y = chip_size
        block_X = chip_size
             
        mskout = []
        imgout = []
        
        # basic tile name
        basename = os.path.split(t)[1]
        # extended for each 
        msk_bsnm = os.path.join(mskdir, basename)
        tl_bsnm = os.path.join(tiledir, basename) 
        
        # block proc
        # no running off end of the pier....
        # As with image proc convention i = y, j = x 
        for i in tqdm(range(0, rows, block_Y)):
                if i + block_Y < rows:
                    n_rows = block_Y
                else:
                    n_rows = rows - i
            
                for j in range(0, cols, block_X):
                    if j + block_X < cols:
                        n_cols = block_X
                    else:
                        n_cols = cols - j
                    msk = mskbnd.ReadAsArray(j, i, n_cols, n_rows)
                    
                    # check if it is blank
                    if msk.max() == 0:
                        continue
                        
                    else:
                        
                        img = mb2array(tile_rds, j, i, n_cols, n_rows, bands=bands)
                    
                        # ugly but its friday make names and store them
                        outmsk = msk_bsnm.replace('.tif', str(j)+str(i)+'.tif')
                        mskout.append(outmsk)
                        
                        outimg = tl_bsnm.replace('.tif', str(j)+str(i)+'.tif')
                        imgout.append(outimg)
                        
                        chip_writer(msk, j, i, n_cols, n_rows, rgt,
                                    mask_rds, outmsk, fmt='Gtiff')
                        chip_writer(img, j, i, n_cols, n_rows, rgt,
                                    tile_rds, outimg, fmt='Gtiff')
        
    return mskout, imgout
        
def collect_train_chip(masklist, tilelist, outdir, chip_size=256, include_zero=True,
                       bands=[1,2,3]):
    
    """
    Collect and save chips of an image from a list of masks and images
    
    for a chip-based CNN (i.e. we are simply labelling a chip NOT segmenting anything)
    
    Please note that areas of 0 (no mask) will count as a class
    
    The list of images must correspond/ be in the same order.
    
    Parameters
    ----------
    
    masklist: list
                A list of images containing the training masks
    
    tilelist: list
                A list of images containing the corresponding spectral info
    
    outdir: string
                Where the training chips will be written
    
    chip_size: int
                the training "chip" size e.g. 256x256 pixels  dependent on the 
                nnet used
                
    include_zero: bool
                whether to include a non-masked area as class 0
                
    Returns
    -------
    
    A tuple of lists of the respective paths of both masks and corresponding 
    images
    

    """
    # TODO ultimately this could be replaced with a text file denoting the 
    # img name, coords, class etc, rather than writing the chips out  
    
    # we know this so not in loop
    block_X = chip_size
    block_Y = chip_size
    
    # make the dirs if not there

    tiledir = os.path.join(outdir, 'images')
    makewrkdir(tiledir)
    # TODO  this could be parallelized per raster pair
    for t,m in zip(tilelist, masklist):
        
        tile_rds =  gdal.Open(t)
        
        mask_rds = gdal.Open(m)
        
        cols = mask_rds.RasterXSize
        rows = mask_rds.RasterYSize
        
        mskbnd = mask_rds.GetRasterBand(1)
        
        rgt = mask_rds.GetGeoTransform()

        block_Y = chip_size
        block_X = chip_size
             
        imgout = []
        
        # basic tile name
        basename = os.path.split(t)[1]
        # extended for each 

        tl_bsnm = os.path.join(tiledir, basename) 
        
        # block proc
        # no running off end of the pier....
        # As with image proc convention i = y, j = x 
        for i in tqdm(range(0, rows, block_Y)):
                if i + block_Y < rows:
                    n_rows = block_Y
                else:
                    n_rows = rows - i
            
                for j in range(0, cols, block_X):
                    if j + block_X < cols:
                        n_cols = block_X
                    else:
                        n_cols = cols - j
                    msk = mskbnd.ReadAsArray(j, i, n_cols, n_rows)
                    
                    
                    #this messy clause to be replaced
                    if msk.max() == 0 and include_zero == False:
                        continue
                    else:
                        # use the maximum value to label the class
                        idx = '_'+str(int(msk.max()))+'.tif'
                    outimg = tl_bsnm.replace('.tif', str(j)+str(i)+idx)

                    imgout.append(outimg)
                        
                    img = mb2array(tile_rds, j, i, n_cols, n_rows, bands=bands)

                    chip_writer(img, j, i, n_cols, n_rows, rgt,
                                tile_rds, outimg, fmt='Gtiff')
        
    return imgout
    

