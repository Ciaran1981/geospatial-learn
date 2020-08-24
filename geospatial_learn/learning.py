# -*- coding: utf-8 -*-
"""
the learning module

Description
-----------

The learning module set of functions provide a framework to optimise and classify
EO data for both per pixel or object properties

"""

# This must go first or it causes the error:
#ImportError: dlopen: cannot load any more object with static TLS
try:
    #import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    import xgboost as xgb
except ImportError:
    pass
    print('xgb not available')



from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import OrderedDict


import glob
from sklearn import svm
import gdal, ogr#,osr
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
import joblib
#from sklearn.externals import joblib
from sklearn import metrics
import joblib as jb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from geospatial_learn.raster import array2raster
from geospatial_learn.shape import _bbox_to_pixel_offsets#, zonal_stats
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from tpot import TPOTClassifier, TPOTRegressor
import warnings
from geospatial_learn.raster import _copy_dataset_config

import pandas as pd
import simpledbf
from plyfile import PlyData, PlyProperty, PlyListProperty
from pyntcloud import PyntCloud

from keras.models import Sequential

from keras.models import load_model, save_model
# if still not working try:
from keras.layers.core import Dense, Dropout, Flatten, Activation

from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from keras.utils import multi_gpu_model
import os
gdal.UseExceptions()
ogr.UseExceptions()

def create_model_tpot(X_train, outModel, cv=6, cores=-1,
                      regress=False, params = None, scoring=None):
    
    """
    Create a model using the tpot library where genetic algorithms
    are used to optimise pipline and params. 
    
    This also supports xgboost incidentally
    
    Parameters
    ----------  
    
    X_train : np array
              numpy array of training data where the 1st column is labels
    
    outModel : string
               the output model path (which is a .py file)
               from which to run the pipeline
    
    cv : int
         no of folds
    
    cores : int or -1 (default)
            the no of parallel jobs
    
    strat : bool
            a stratified grid search
    
    regress : bool
              a regression model if True, a classifier if False
    
    params : a dict of model params (see tpot)
             enter your own params dict rather than the range provided
    
    scoring : string
              a suitable sklearn scoring type (see notes)
                           
    """
    #t0 = time()
    
    print('Preparing data')   
    
    """
    Prep of data for model fitting 
    """

    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    if params is None and regress is False:       
        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2,
                              n_jobs=cores, scoring = scoring,
                              warm_start=True)
        tpot.fit(X_train, y_train)
        
    elif params != None and regress is False:
        tpot = TPOTClassifier(config_dict=params, n_jobs=cores, scoring = scoring,
                              warm_start=True)
        tpot.fit(X_train, y_train)
        
    elif params is None and regress is True:       
        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2,
                              n_jobs=cores, scoring = scoring,
                              warm_start=True)
        tpot.fit(X_train, y_train)
        
    elif params != None and regress is True:
        tpot = TPOTRegressor(config_dict=params, n_jobs=cores, verbosity=2,
                             scoring = scoring,
                              warm_start=True)
        tpot.fit(X_train, y_train)

    tpot.export(outModel)    


def create_model(X_train, outModel, clf='svc', random=False, cv=6, cores=-1,
                 strat=True, regress=False, params = None, scoring=None, 
                 ply=False, save=True):
    
    """
    Brute force or random model creating using scikit learn. Either use the
    default params in this function or enter your own (recommended - see sklearn)
    
    Parameters
    ---------------   
    
    X_train : np array
              numpy array of training data where the 1st column is labels
    
    outModel : string
               the output model path which is a gz file, if using keras it is 
               h5 
    
    clf : string
          an sklearn or xgb classifier/regressor 
          logit, sgd, linsvc, svc, svm, nusvm, erf, rf, gb, xgb,
          
          keras nnt also available as a very limited option - the nnt is 
          currently a dense sequential of 32, 16, 8, 32 - please inspect the
          source. If using GPU, you will likely be limited to a sequential 
          grid search as multi-core overloads the GPUs quick!
          
    
    random : bool
             if True, a random param search
    
    cv : int
         no of folds
    
    cores : int or -1 (default)
            the no of parallel jobs
    
    strat : bool
            a stratified grid search
    
    regress : bool
              a regression model if True, a classifier if False
    
    params : a dict of model params (see scikit learn)
             enter your own params dict rather than the range provided
    
    scoring : string
              a suitable sklearn scoring type (see notes)
            
    
    General Note:
    --------------------    
        There are more sophisticated ways to tune a model, this greedily 
        searches everything but can be computationally costly. Fine tuning 
        in a more measured way is likely better. There are numerous books,
        guides etc...
        E.g. with gb- first tune no of trees for gb, then learning rate, then
        tree specific
        
    Notes on algorithms:
    ----------------------    
        From my own experience and reading around
        
    
        sklearn svms tend to be not great on large training sets and are
        slower with these (i have tried on HPCs and they time out on multi fits)
       
        sklearn 'gb' is very slow to train, though quick to predict 
        
        xgb is much faster, but rather different in algorithmic detail -
        ie won't produce same results as sklearn...
        
        xgb also uses the sklearn wrapper params which differ from those in
        xgb docs, hence they are commented next to the area of code
    
        Scoring types - there are a lot - some of which won't work for 
        multi-class, regression etc - see the sklearn docs!
        
        'accuracy', 'adjusted_rand_score', 'average_precision', 'f1',
        'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss',
        'neg_mean_absolute_error', 'neg_mean_squared_error',
        'neg_median_absolute_error', 'precision', 'precision_macro',
        'precision_micro', 'precision_samples', 'precision_weighted',
        'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
        'recall_weighted', 'roc_auc'
                    
    """
    #t0 = time()
    
    min_max_scaler = preprocessing.MinMaxScaler()
    print('Preparing data')   
    # TODO IMPORTANT add xgb boost functionality
    #inputImage = gdal.Open(inputIm)    
    
    """
    Prep of data for model fitting 
    """

    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    if ply == False:
        
        X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    if scoring is None and regress is False:
        scoring = 'accuracy'
    elif scoring is None and regress is True:    
        scoring = 'r2'
    # Choose the classifier type
    # TODO this has become rather messy (understatement)
    # and inefficient - need to make it more 
    # elegant
    no_classes = len(np.unique(y_train))
    if clf == 'keras':
        
        
        kf = StratifiedKFold(cv, shuffle=True)
        #Not currently workable
#        if gpu > 1:
#            
#
#
#            def _create_nnt(no_classes=no_classes):
#            	# create model - fixed at present
#                tf.compat.v1.disable_eager_execution()
#                with tf.device("/cpu:0"):
#                    model = Sequential()
#                    model.add(Dense(32, activation='relu', input_dim=bands))
#                    model.add(Dense(16, activation='relu'))
#                    model.add(Dense(8,  activation='relu'))
#                    model.add(Dense(32, activation='relu'))
#                    model.add(Dense(no_classes, activation='softmax'))
#                    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
#                                metrics=['accuracy'])
#                    model = multi_gpu_model(model, gpus=2)
#                    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
#                                metrics=['accuracy'])
#                    return model
       # else:
        def _create_nnt(no_classes=no_classes):
    	# create model - fixed at present 
        
            model = Sequential()
            model.add(Dense(32, activation='relu', input_dim=bands))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(8,  activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(no_classes, activation='softmax'))
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
                        metrics=['accuracy'])
            return model
        
        # the model
        model = KerasClassifier(build_fn=_create_nnt, verbose=1)
        # may require this to get both GPUs working
        
		# initialize the model

        # define the grid search parameters
        if params is None:
            batch_size = [10, 20, 40]#, 60, 80, 100]
            epochs = [10]#, 30]
            param_grid = dict(batch_size=batch_size, epochs=epochs)
        else:
            param_grid = params
        
        # It is of vital importance here that the estimator model is passed
        # like this otherwiae you get loky serialisation error
        # Also, at present it has to work sequentially, otherwise it overloads
        # the gpu
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, 
                            cv=kf, verbose=1)
        grid.fit(X_train, y_train)
        
        grid.best_estimator_.model.save(outModel)
        

        
        
    if clf == 'erf':
         RF_clf = ExtraTreesClassifier(n_jobs=cores)
         if random==True:
            param_grid = {"max_depth": [10, None],
                          "n_estimators": [500],
                          "min_samples_split": sp_randint(1, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}
                      
        # run randomized search
            grid = RandomizedSearchCV(RF_clf, param_distributions=param_grid,
                                       n_jobs=-1, n_iter=20,  verbose=2)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))
         else:
            if params is None: 
            #currently simplified for processing speed 
                param_grid ={"n_estimators": [500],
                             "max_features": ['sqrt', 'log2'],                                                
                             "max_depth": [10, None],
                             "min_samples_split": [2,3,5],
                             "min_samples_leaf": [5,10,20,50,100],
                             "bootstrap": [True, False]}
            else:
                param_grid = params
            if strat is True and regress is False:               
                grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
            elif strat is False and regress is False:  
                grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=2)
                
         grid.fit(X_train, y_train)
         joblib.dump(grid.best_estimator_, outModel) 
         
    if clf == 'xgb' and regress is False:
        xgb_clf = XGBClassifier()
        if params is None:
                # This is based on the Tianqi Chen author of xgb
                # tips for data science as a starter
                # he recommends fixing trees - they are 200 by default here
                # crunch this first then fine tune rest
                # 
                ntrees = 200
                param_grid={'n_estimators': [ntrees],
                            'learning_rate': [0.1], # fine tune last
                            'max_depth': [4, 6, 8, 10],
                            'colsample_bytree': [0.4,0.6,0.8,1.0]}
            #total available...
#            param_grid={['reg_lambda',
#                         'max_delta_step',
#                         'missing',
#                         'objective',
#                         'base_score',
#                         'max_depth':[6, 8, 10],
#                         'seed',
#                         'subsample',
#                         'gamma',
#                         'scale_pos_weight',
#                         'reg_alpha', 'learning_rate',
#                         'colsample_bylevel', 'silent',
#                         'colsample_bytree', 'nthread', 
#                         'n_estimators', 'min_child_weight']}
        else:
            param_grid = params
        grid = GridSearchCV(xgb_clf, param_grid=param_grid, 
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)
    if clf == 'gb' and regress is False:
        # Key parameter here is max depth
        gb_clf = GradientBoostingClassifier()
        if params is None:
            param_grid ={"n_estimators": [100], 
                         "learning_rate": [0.1, 0.75, 0.05, 0.025, 0.01],
                         "max_features": ['sqrt', 'log2'],                                                
                         "max_depth": [3,5],                    
                         "min_samples_leaf": [5,10,20,30]}
        else:
            param_grid = params
#                       cut due to time
        if strat is True and regress is False:               
            grid = GridSearchCV(gb_clf, param_grid=param_grid, 
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        elif strat is False and regress is False:
            grid = GridSearchCV(gb_clf, param_grid=param_grid, 
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
        
    if clf == 'gb'  and regress is True:
        gb_clf = GradientBoostingRegressor(n_jobs=cores)
        if params is None:
            param_grid = {"n_estimators": [500],
                          "loss": ['ls', 'lad', 'huber', 'quantile'],                      
                          "learning_rate": [0.1, 0.75, 0.05, 0.025, 0.01],
                          "max_features": ['sqrt', 'log2'],                                                
                          "max_depth": [3,5],                    
                          "min_samples_leaf": [5,10,20,30]}
        else:
            param_grid = params
        
        grid = GridSearchCV(gb_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
        
    #Find best params----------------------------------------------------------
    if clf == 'rf' and regress is False:
         RF_clf = RandomForestClassifier(n_jobs=cores, random_state = 123)
         if random==True:
            param_grid = {"max_depth": [10, None],
                          "n_estimators": [500],
                          "min_samples_split": sp_randint(1, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}
                      
        # run randomized search
            grid = RandomizedSearchCV(RF_clf, param_distributions=param_grid,
                                       n_jobs=-1, n_iter=20,  verbose=2)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))
         else:
            if params is None: 
            #currently simplified for processing speed 
                param_grid ={"n_estimators": [500],
                             "max_features": ['sqrt', 'log2'],                                                
                             "max_depth": [10, None],
                             "min_samples_split": [2,3,5],
                             "min_samples_leaf": [5,10,20,50,100,200,500],
                             "bootstrap": [True, False]}
            else:
                param_grid = params
            if strat is True and regress is False:               
                grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
            elif strat is False and regress is False:  
                grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=2)
                
         grid.fit(X_train, y_train)
         joblib.dump(grid.best_estimator_, outModel) 
         
    if clf == 'rf' and regress is True:
        RF_clf = RandomForestRegressor(n_jobs = cores, random_state = 123)
        if params is None:
            param_grid ={"n_estimators": [500],
                             "max_features": ['sqrt', 'log2'],                                                
                             "max_depth": [10, None],
                             "min_samples_split": [2,3,5],
                             "min_samples_leaf": [5,10,20,50,100,200,500],
                             "bootstrap": [True, False]}
        else:
            param_grid = params
        grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
                
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))
    
    # Random can be quicker and more often than not produces close to
    # exaustive results
    if clf == 'linsvc' and regress is False:
        X_train = min_max_scaler.fit_transform(X_train)
        svm_clf = svm.LinearSVC()
        if random == True:
            param_grid = [{'C': [expon(scale=100)], 'class_weight':['auto', None]}]
            #param_grid = [{'kernel':['rbf', 'linear']}]
            grid = GridSearchCV(svm_clf, param_grid=param_grid, cv=cv,
                                scoring=scoring, verbose=1, n_jobs=cores)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))
        else:
             param_grid = [{'C': [1, 10, 100, 1000], 'class_weight':['auto', None]}]
            #param_grid = [{'kernel':['rbf', 'linear']}]
             if strat is True:               
                grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
             elif strat is False and regress is False:  
                 grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
    if clf == 'linsvc' and regress is True:
        svm_clf = svm.LinearSVR()
        if params is None:
            param_grid = [{'C': [1, 10, 100, 1000]},
                           {'loss':['epsilon_insensitive',
                        'squared_epsilon_insensitive']}]
        else:
            param_grid = params
        grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
             #print("done in %0.3fs" % (time() - t0))
    if clf == 'svc': # Far too bloody slow
        X_train = min_max_scaler.fit_transform(X_train)
        svm_clf = svm.SVC(probability=False)
        if random == True:
            if params is None:
        
                param_grid = [{'C': [expon(scale=100)], 'gamma': [expon(scale=.1).astype(float)],
                  'kernel': ['rbf'], 'class_weight':['auto', None]}]
            else:
                param_grid = params
            #param_grid = [{'kernel':['rbf', 'linear']}]
            grid = GridSearchCV(svm_clf, param_grid=param_grid, cv=cv,
                                scoring=scoring, verbose=1, n_jobs=cores)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))

        if params is None:
    
             param_grid = [{'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4],
             'kernel': ['rbf'], 'class_weight':['auto', None]}]
        else:
            param_grid = params
            #param_grid = [{'kernel':['rbf', 'linear']}]
        if strat is True and regress is False:               
            grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        elif strat is False and regress is False: 
            grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
             #print("done in %0.3fs" % (time() - t0))
    
    if clf == 'nusvc' and regress is False:
        X_train = min_max_scaler.fit_transform(X_train)
        svm_clf = svm.NuSVC(probability=False)
        if random == True:
            if params is None:
                param_grid = [{'nu':[0.25, 0.5, 0.75, 1], 'gamma': [expon(scale=.1).astype(float)],
                                          'class_weight':['auto']}]
            else:
                param_grid = params
            #param_grid = [{'kernel':['rbf', 'linear']}]
            grid = GridSearchCV(svm_clf, param_grid=param_grid, cv=cv,
                                scoring=scoring, verbose=1, n_jobs=cores)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))
        else:
            if params is None:
                 param_grid = [{'nu':[0.25, 0.5, 0.75, 1],'gamma': [1e-3, 1e-4],
                                'class_weight':['auto']}]
            else:
                param_grid = params
            #param_grid = [{'kernel':['rbf', 'linear']}]
        if strat is True and regress is False:               
                grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
        elif strat is True and regress is False: 
             grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
    if clf == 'nusvc' and regress is True:
         svm_clf = svm.NuSVR()
         param_grid = [{'nu':[0.25, 0.5, 0.75, 1],'gamma': [1e-3, 1e-4]}]
         grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
         grid.fit(X_train, y_train)
         joblib.dump(grid.best_estimator_, outModel) 
             #print("done in %0.3fs" % (time() - t0))
    if clf == 'logit':
        logit_clf = LogisticRegression()
        if params is None:
            param_grid = [{'C': [1, 10, 100, 1000], 'penalty': ['l1', 'l2', ],
                           'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                           'multi_class':['ovr', 'multinomial']}]
        else:
            param_grid = params
        grid = GridSearchCV(logit_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
        
    if clf == 'sgd':
        logit_clf = SGDClassifier()
        if params is None:
            param_grid = [{'loss' : ['hinge, log', 'modified_huber',
                                     'squared_hinge', 'perceptron'], 
                           'penalty': ['l1', 'l2', 'elasticnet'],
                           'learning_rate':['constant', 'optimal', 'invscaling'],
                           'multi_class':['ovr', 'multinomial']}]
        else:
            param_grid = params
        grid = GridSearchCV(logit_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 

    return [grid.best_estimator_, grid.cv_results_, grid.best_score_, grid.best_params_]
#    print(grid.best_params_)
#    print(grid.best_estimator_)
#    print(grid.oob_score_)
#    
#    plt.plot(est_range, grid_mean_scores)
#    plt.xlabel('no of estimators')
#    plt.ylabel('Cross validated accuracy')    
    

    
    #Save the model
#    joblib.dump(grid.best_estimator_, outModel+'.pkl') 
#    print("done in %0.3fs" % (time() - t0))

def RF_oob_opt(model, X_train, min_est, max_est, step, regress=False):
    
    """ 
    This function uses the oob score to find the best parameters.
    
    This cannot be parallelized due to the warm start bootstrapping, so is
    potentially slower than the other cross val in the create_model function
        
    This function is based on an example from the sklearn site
        
    This function plots a graph diplaying the oob rate
        
    Parameters
    ---------------------
    
    model : string (.gz)
            path to model to be saved
    
    X_train : np array
              numpy array of training data where the 1st column is labels
    
    min_est : int
              min no of trees
    
    max_est : int
              max no of trees
    
    step : int
           the step at which no of trees is increased
    
    regress : bool
              boolean where if True it is a regressor
    
    Returns : tuple of np arrays
    -----------------------
        
    error rate, best estimator
        
    """
    # This is a bit slow at present, needs profiled
    #t0 = time()
    
    RANDOM_STATE = 123
    print('Preparing data')    
    
    """
    Prep of data for classification - getting bands one at a time to save memory
    """
    print('processing data for classification')

    
    bands = X_train.shape[1]-1
    
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    if regress is False:
        X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]
    
    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    print('iterating estimators')
    if regress is True:
        max_feat = X_train.shape[1]-1
        ensemble_clfs = [
        ("RandomForestClassifier, max_features='no_features'",
                RandomForestRegressor(warm_start=True, oob_score=True,
                                       max_features=max_feat,
                                       random_state=RANDOM_STATE))]
    else:    
        ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
                RandomForestClassifier(warm_start=True, oob_score=True,
                                       max_features="sqrt",
                                       random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
                RandomForestClassifier(warm_start=True, max_features='log2',
                                       oob_score=True,
                                       random_state=RANDOM_STATE)),
        ]
    
    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    
    # Range of `n_estimators` values to explore.
    min_estimators = min_est
    max_estimators = max_est
    
    for label, clf in ensemble_clfs:
        for i in tqdm(range(min_estimators, max_estimators + 1, step)):
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train)
    
            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)
    
    # suspect a slow down after here, likely the plot...
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="best")
    plt.show()
    
    # regression option
    if regress is True:
        max_features = max_feat
        er = np.array(error_rate["RandomForestClassifier, max_features='no_features'"][0:max_estimators])
        bestscr = er[:,1].min()
        data = er[np.where(er[:,1] == bestscr)]
        best_param = ((max_features, data[0]))
        best_model = RandomForestRegressor(warm_start=True, oob_score=True,
                                       max_features= max_features,
                                       n_estimators=best_param[1][0].astype(int),
                                       n_jobs=-1)
    else:    
        sqrt = np.array(error_rate["RandomForestClassifier, max_features='sqrt'"][0:max_estimators])
        log2 = np.array(error_rate["RandomForestClassifier, max_features='log2'"][0:max_estimators])
    #NoFeat = np.array(error["RandomForestClassifier, max_features='None'"][0:max_estimators])
        minsqrt = sqrt[:,1].min()
        minLog = log2[:,1].min()
        
        if minsqrt>minLog:
            minVal = minLog
            max_features = 'log2'
            data = log2[np.where(log2[:,1] == minVal)]
        else:
            minVal = minsqrt
            max_features = 'sqrt'
            data = sqrt[np.where(sqrt[:,1] == minVal)]
    
    
    
        best_param = ((max_features, data[0]))
    

        best_model = RandomForestClassifier(warm_start=True, oob_score=True,
                                       max_features= max_features,
                                       n_estimators=best_param[1][0].astype(int),
                                       n_jobs=-1)
    best_model.fit(X_train, y_train)                
    joblib.dump(best_model, model) 
    return error_rate, best_param







def plot_feature_importances(modelPth, featureNames):
    
    """
    Plot the feature importances of an ensemble classifier
    
    Parameters
    --------------------------
    
    modelPth : string
               A sklearn model path 
    
    featureNames : list of strings
                   a list of feature names
    
    """
    
    model = joblib.load(modelPth)
    
    n_features = model.n_features_
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), featureNames)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()

       
def classify_pixel(model, inputDir, bands, outMap, probMap):
    """ 
    A function to classify an image using a pre-saved model - assumes
    a folder of tiled rasters for memory management - classify_pixel_block is
    recommended instead of this function
    
    Parameters
    
    ---------------
        
    model : sklearn model
            a path to a scikit learn model that has been saved 
    
    inputDir : string
               a folder with images to be classified
    
    bands : int
            the no of image bands eg 8
    
    outMap : string
             path to output image excluding the file format 'pathto/mymap'
    
    probMap : string
              path to output prob image excluding the file format 'pathto/mymap'
    
    FMT : string 
          optional parameter - gdal readable fmt
           
        """ 

    imageList = glob.glob(inputDir+'*tile*')
    
    numBands = bands
    n = len(imageList)
    for im in range (0,n):
            
        inputImage = gdal.Open(imageList[im])
        shapeB = ((numBands, inputImage.RasterXSize* inputImage.RasterYSize))    
        
        
        X = np.zeros(shape = shapeB)    
        for band in tqdm(range(1,bands+1)):
            band1 = inputImage.GetRasterBand(band)
            im1 = band1.ReadAsArray()
            X[band-1] = im1.flatten()
        X = X.transpose() 
        X = np.where(np.isfinite(X),X,0)    
        
        #RF_clf = grid.best_estimator_
        
       #RF_clf.fit(X_train, y_train) 
        
        # We only want the band stats - must remove the training numbers
        #X = X[:,1:len(bands+1)] 
            
        model1 = joblib.load(model+'.gz')
        predictClass = model1.predict(X)
        #if inputImage.RasterXSize < inputImage.RasterYSize
        predictClass = np.reshape(predictClass, (im1.shape[0], im1.shape[1]))
#        else:
#            predictClass = np.reshape(predictClass, (inputImage.RasterXSize, inputImage.RasterYSize))
            
        print('Saving outputs')
        array2raster(predictClass, 1, inputImage, outMap+str(im)+'.tif', gdal.GDT_Byte, 'Gtiff')
        
        Probs=model1.predict_proba(X)
        print('classification '+str(im)+' done')
        #Class 1 = Deforestattion
        #if inputImage.RasterXSize < inputImage.RasterYSize:
        classArr = np.unique(predictClass)
        ProbsFinal = np.empty((im1.shape[0], im1.shape[1], inputImage.RasterCount))
        
       
        for band in tqdm(classArr):
            ProbsFinal[:,:,band-1] = np.reshape(Probs[:,band-1], (im1.shape[0], im1.shape[1])) 

        
        print('Prob_'+str(im)+' done')
        array2raster(ProbsFinal, 4, inputImage, probMap+str(im)+'.tif', gdal.GDT_Float64, 'Gtiff')

      
        
def classify_pixel_bloc(model, inputImage, bands, outMap, blocksize=None, 
                        FMT=None, ndvi = None, dtype = gdal.GDT_Int32):
    """
    A block processing classifier for large rasters, supports KEA, HFA, & Gtiff
    formats. KEA is recommended, Gtiff is the default
    
    Parameters
    ------------------
        
    model : sklearn model / keras model
            a path to a model that has been saved 
    
    inputImage : string
                 path to image including the file fmt 'Myimage.tif'
    
    bands : band
            the no of image bands eg 8
    
    outMap : string
             path to output image excluding the file format 'pathto/mymap'
    
    FMT : string
          optional parameter - gdal readable fmt
    
    blocksize : int (optional) 
                size of raster chunck in pixels 256 tends to be quickest
                if you put None it will read size from gdal (this doesn't always pay off!)
    
    dtype : int (optional - gdal syntax gdal.GDT_Int32) 
            a gdal dataype - default is int32


    Notes
    -------------------------------------------------
    
    Block processing is sequential, but quite a few sklearn models are parallel
    so that has been prioritised rather than raster IO
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
    
    inDataset = gdal.Open(inputImage)
    
    outDataset = _copy_dataset_config(inDataset, outMap = outMap,
                                     dtype = gdal.GDT_Byte, bands = 1)
    band = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    outBand = outDataset.GetRasterBand(1)
    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = band.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize

    # For either option below making a block index is FAR slower than the 
    # code used below - don't be tempted - likely cython is the solution to 
    # performance gain here or para processing (but the model is already multi-core)
    
    # TODO 1- find an efficient way of classifying only non-zero values
    # issue is extracting them, then writing back to the output array
    #e.g
    # [[02456812000002567], ]02456812000002567], ]02456812000002567]]
    # predict [[24568122567, 24568122567, 24568122567]]
    # then write back to original positions
    # scipy sparse doesn't seem to work....
    # TODO 2- thread or parallelise block/line processing
    # Pressumably writing to different parts of raster should be ok....
    
    if os.path.splitext[1] == ".h5":
        model1 = load_model(model)
    else:  
        model1 = joblib.load(model)
    if blocksizeY==1:
        rows = np.arange(cols, dtype=np.int)                
        for row in tqdm(rows):
            i = int(row)
            j = 0
            #X = np.zeros(shape = (bands, blocksizeX))
            #for band in range(1,bands+1):
            
            X = inDataset.ReadAsArray(j, i, xsize=blocksizeX, ysize=blocksizeY)
            X.shape = ((bands,blocksizeX))
            
            if X.max() == 0:
                predictClass = np.zeros_like(rows, dtype = np.int32)
            else: 
                X = np.where(np.isfinite(X),X,0) # this is a slower line                
                X = X.transpose() 
                #Xs = csr_matrix(X)
                predictClass = model1.predict(X)
                predictClass[X[:,0]==0]=0 
            
            outBand.WriteArray(predictClass.reshape(1, blocksizeX),j,i) 
                    #print(i,j)

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
                
                X = inDataset.ReadAsArray(j,i, xsize=numCols, ysize=numRows)
#                X = np.zeros(shape = (bands, numCols*numRows))
#                for band in range(1,bands+1):
#                    band1 = inDataset.GetRasterBand(band)
#                    data = band1.ReadAsArray(j, i, numCols, numRows)
                with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if X.max() == 0:
                                continue              
                            else:
                                X.shape = ((bands,numRows*numCols))
                                X = X.transpose() 
                                X = np.where(np.isfinite(X),X,0) 
                                # this is a slower line   
                                #Xs= csr_matrix(X)
                                
                                # YUCK!!!!!! This is a repulsive solution
                                if ndvi != None:
                                    ndvi1 = (X[:,3] - X[:,2]) / (X[:,3] + X[:,2]) 
                                    ndvi1.shape = (len(ndvi1), 1)
                                    ndvi1 = np.where(np.isfinite(ndvi1),ndvi1,0) 
                                    ndvi2 = (X[:,7] - X[:,6]) / (X[:,7] + X[:,6]) 
                                    ndvi2.shape = (len(ndvi2), 1)
                                    ndvi2 = np.where(np.isfinite(ndvi2),ndvi2,0) 
                                    
                                    X = np.hstack((X[:,0:4], ndvi1, X[:,4:8], ndvi2))
                                    
                                   
                                predictClass = model1.predict(X)
                                predictClass[X[:,0]==0]=0                    
                                predictClass = np.reshape(predictClass, (numRows, numCols))
                                outBand.WriteArray(predictClass,j,i)
                #print(i,j)
    outDataset.FlushCache()
    outDataset = None
    

def prob_pixel_bloc(model, inputImage, bands, outMap, classes, blocksize=None,
                    FMT=None, one_class = None):
    """
    A block processing classifier for large rasters that produces a probability,
    output.
    
    Supports KEA, HFA, & Gtiff formats -KEA is recommended, Gtiff is the default
    
    Parameters
    ----------
    model : string
            a path to a scikit learn model that has been saved 
        
    inputImage : string
                 path to image including the file fmt 'Myimage.tif'
    
    bands : int
            the no of image bands eg 8
    
    outMap : string
             path to output image excluding the file format 'pathto/mymap'
    
    classes : int
              no of classes
    
    blocksize : int (optional) 
                size of raster chunck 256 tends to be quickest if you put None it 
                will read size from gdal (this doesn't always pay off!)
               
    FMT : string
          optional parameter - gdal readable fmt eg 'Gtiff'
        
    one_class : int
                choose a single class to produce output prob raster


    -------------------------------------------------
    
    Block processing is sequential, but quite a few sklearn models are parallel
    so that has been prioritised rather than raster IO
    """
    if FMT == None:
        FMT = 'Gtiff'
        fmt = 'tif'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'
        
    # TODO - a list of classes would be better eliminating the need for the one
    # class param
    if one_class != None:
        classes = one_class
        
    inDataset = gdal.Open(inputImage)
    
    outDataset = _copy_dataset_config(inputImage, outMap = outMap,
                                     bands = bands)
    band = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    #outBand = outDataset.GetRasterBand(1)
    # So with most datasets blocksize is a row scanline
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
    # For either option below making a block index is FAR slower than the 
    # code used below - don't be tempted - likely cython is the solution to 
    # performance gain here
    model1 = joblib.load(model)
    if blocksizeY==1:
        rows = np.arange(cols, dtype=np.int)        
        for row in tqdm(rows):

            i = int(row)
            j = 0
            X = np.zeros(shape = (bands, blocksizeX))
            for band in range(1,bands+1):
                band1 = inDataset.GetRasterBand(band)
                data = band1.ReadAsArray(j, i, blocksizeX, 1)
                X[band-1] = data.flatten()
            X = X.transpose() 
            X = np.where(np.isfinite(X),X,0) # this is a slower line   
            Probs=model1.predict_proba(X)
            

            ProbsFinal = np.empty((data.shape[0], data.shape[1], classes))
            for band in range(1,classes+1):
                ProbsFinal[:,:,band-1] = np.reshape(Probs[:,band-1], (1, blocksizeX))
                outBand = outDataset.GetRasterBand(band)
                outBand.WriteArray(ProbsFinal[:,:,band-1],j,i)
                #print(i,j)

    # else it is a block            
    else:
        model1 = joblib.load(model)
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
                X = np.zeros(shape = (bands, numCols*numRows))
                for band in range(1,bands+1):
                    band1 = inDataset.GetRasterBand(band)
                    data = band1.ReadAsArray(j, i, numCols, numRows)
                    X[band-1] = data.flatten()
                X = X.transpose() 
                X = np.where(np.isfinite(X),X,0)  # this is a slower line  
                Probs=model1.predict_proba(X)
                #Class 1 = Deforestattion
                #if inputImage.RasterXSize < inputImage.RasterYSize:
                #classArr = np.arange(classes)
                
                if one_class != None:
                    ProbsFinal = np.reshape(Probs[:,one_class-1], 
                                            (data.shape[0], data.shape[1]))
                    outBand = outDataset.GetRasterBand(one_class)
                    outBand.WriteArray(ProbsFinal,j,i)
                else:               
                    ProbsFinal = np.empty((data.shape[0], data.shape[1], classes))
                    for band in range(1,classes+1):
                        ProbsFinal[:,:,band-1] = np.reshape(Probs[:,band-1], (data.shape[0], data.shape[1]))
                        outBand = outDataset.GetRasterBand(band)
                        outBand.WriteArray(ProbsFinal[:,:,band-1],j,i)
                    #print(i,j)
        

    #outBand.FlushCache()
    outDataset.FlushCache()
    outDataset = None
    #print("done in %0.3fs" % (time() - t0))

    
def classify_object(model, inShape, attributes, field_name=None):
    
    """
    Classify a polygon/point file attributes ('object based') using an sklearn
    model
    
    Parameters
    ------------------
    model : string
            path to input model
    
    inShape : string
              input shapefile path (must be .shp for now....)
    
    attributes : list of stings
                 list of attributes names
    
    field_name : string
                 name of classified label field (optional)
    """
    
    print('prepping data')
    dbf=simpledbf.Dbf5(inShape[:-4]+'.dbf')  
#    csv = inShape[:-4]+'.csv'
#    dbf.to_csv(inShape[:-4]+'.csv')
    
    df = dbf.to_dataframe()
    
    X = df[attributes].as_matrix()
    
    del df
    
    print('data ready')
    """
    Classification
    
    The data must be prepared for input and exit from scikit learn
    
    e.g we require cross tabulating training and input data
    
    The next three lines obviously depend on the state in which the training data
    comes into this process
    """
    #----------------------------------------------------------------------------------
    X[np.where(np.isnan(X))]=0
    X = X[np.isfinite(X).all(axis=1)]
     
    
    """
    Now the classification itself - see sklearn for details on params
    """
    print('Classifying')
#    RF_clf = RandomForestClassifier(n_estimators=500, \
#          oob_score=True, n_jobs=6, verbose=2)#R-forest
#    
#    RF_clf.fit(X_train, y_train) 
    model1 = joblib.load(model)
    predictClass = model1.predict(X)
#    predictClass = RF_clf.predict(X) 
    #----------------------------------------------------------------------------------
    """
    Finally! Now we sort the values to match the order of the vector attribute table
    """
    # clear redundant variables from memory

    del X
    
    predictClass = predictClass.transpose() 
    
    
    shp = ogr.Open(inShape, 1)
    lyr = shp.GetLayer()
    fldDef = ogr.FieldDefn(field_name, ogr.OFTInteger)
    lyr.CreateField(fldDef)
    
    labels = np.arange(lyr.GetFeatureCount())
    
    #a vector of label vals    
    #PropIM = np.zeros_like(segras.ReadAsArray(), dtype=np.double)

     
    # TODO - order is not quite right -sort this
    
    for label in tqdm(labels):
        val=predictClass[label]
        feat = lyr.GetFeature(label)
        feat.SetField(field_name, int(val))
        lyr.SetFeature(feat)

    lyr.SyncToDisk()
    shp.FlushCache()
    shp = None
    lyr = None

    
def get_training_shp(inShape, label_field, feat_fields,  outFile = None):
    """
    Collect training from a shapefile attribute table. Used for object-based 
    classification (typically). 
    
    Parameters
    --------------------    
    
    inShape : string
              the input shapefile - must be esri .shp at present
    
    label_field : string
                  the field name for the class labels
                  
    feat_fields : list
                  the field names of the feature data                

    outFile : string (optional)
              path to training data to be saved (.gz)
    
    Returns
    ---------------------
    training data as a dataframe, first column is labels, rest are features
    list of reject features
    
    """

    outData = list()
    
    feat_fields.insert(0,label_field)
    
    print('Loading & prepping data')    
    shp = ogr.Open(inShape)
    lyr = shp.GetLayer()
    labels = np.arange(lyr.GetFeatureCount())

    rejects = []     
  
    print('calculating stats')
    for label in tqdm(labels):
        #print(label)
        feat = lyr.GetFeature(label)
        if feat == None:
            print('no geometry for feature '+str(label))
            rejects.append(label)
            continue
        row = [feat.GetField(entry) for entry in feat_fields]
        outData.append(row)
        
    df=pd.DataFrame(outData, columns = feat_fields)    
    outArray = df.as_matrix()
    
    if outFile != None:
        jb.dump(outArray, outFile, compress=2)
    
    return df, rejects
             

    
    
def get_training(inShape, inRas, bands, field, outFile = None):
    """
    Collect training as an np array for use with create model function
    
    Parameters
    --------------
        
    inShape : string
              the input shapefile - must be esri .shp at present
        
    inRas : string
            the input raster from which the training is extracted
        
    bands : int
            no of bands
        
    field : string
            the attribute field containing the training labels
    
    outFile : string (optional)
              path to the training file saved as joblib format (eg - 'training.gz')
    
    Returns
    ---------------------
    
    A tuple containing:
    -np array of training data
    -list of polygons with invalid geometry that were not collected 
    
    """
    #t0 = time()
    outData = list()
    print('Loading & prepping data')    
    raster = gdal.Open(inRas)
    shp = ogr.Open(inShape)
    lyr = shp.GetLayer()
    labels = np.arange(lyr.GetFeatureCount())
    rb = raster.GetRasterBand(1)
    rgt = raster.GetGeoTransform()
    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')  
    rejects = []     
  
    print('calculating stats')
    for label in tqdm(labels):
        #print(label)
        feat = lyr.GetFeature(label)
        if feat == None:
            print('no geometry for feature '+str(label))
            continue
        iD = feat.GetField(field)
        geom = feat.GetGeometryRef()
        
        # Get raster georeference info
            
        src_offset = _bbox_to_pixel_offsets(rgt, geom)
        
        
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
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize it
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        
        # Mask the source data array with our current feature
        # Use the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
            
        rb = raster.GetRasterBand(1)
        src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                           src_offset[3])
        if np.shape(src_array) == ():
            rejects.append(label)
            continue
        # Read raster as arrays
        for band in range(1,bands+1): 
            
            rb = raster.GetRasterBand(band)
            src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                               src_offset[3])
            if src_array is None:
                src_array = rb.ReadAsArray(src_offset[0]-1, src_offset[1], src_offset[2],
                                           src_offset[3])

            masked = np.ma.MaskedArray(src_array, 
                                       mask=np.logical_or(src_array == 0,
                                                          np.logical_not(rv_array)))
            

            datafinal = masked.flatten()

            if band == 1:
                X = np.zeros(shape = (datafinal.shape[0], bands+1))
            X[:,0] = iD
            
            X[:,band] = datafinal


        outData.append(X)
    outData = np.asarray(outData)
    outData = np.concatenate(outData).astype(None)
    
    if outFile != None:
        jb.dump(outData, outFile, compress=2)
    
    return outData, rejects

def ply_features(incld, outcld=None, k=30):
    
    """ 
    Calculate point cloud features and write to file
    
    Currently memory intensive due to using pyntcloud....
    
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
        
    outcld: string
               the output point cloud
         

    """  
    
    pcd = PyntCloud.from_file(incld)
    #o3d.io.read_point_cloud(incld)

    #pcd.estimate_normals()
    
    #cloud = PyntCloud.from_instance("open3d", 


    pProps =['anisotropy', "curvature", "eigenentropy", "eigen_sum", "linearity",
             "omnivariance", "planarity", "sphericity"]#, "inclination_deg",
            # "inclination_rad", "orientation_deg", "orientation_rad"]
    #, "HueSaturationValue",#"RelativeLuminance"," RGBIntensity"]

    k_neighbors = pcd.get_neighbors(k=k)
    eigenvalues = pcd.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

    [pcd.add_scalar_field(p, ev=eigenvalues) for p in pProps]
    
    if outcld == None:
        pcd.to_file(incld)
    else:
        pcd.to_file(outcld)

def get_training_ply(incld, label_field="scalar_label", rgb=True, outFile=None):
    
    """ 
    Get training from a point cloud
    
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
    
    label_field: string
              the name of the field representing the training points which must
              be positive integers
    rgb: bool
              whether there is rgb data to be included
              
                
    outFile: string
               path to training array to be saved as .gz via joblib
    Returns
    -------
    
    np array of training where first column is labels

    """  
    # TODO Clean up lack of loops funcs to do stuff
    pf = PlyData.read(incld)
    
#    pProps =['anisotropy', 'curvature', "eigenentropy", "eigen_sum",
#             "linearity", "omnivariance", "planarity", "sphericity"]
   
    #TODO this is a mess - was being lazy must tidy this
    # doesn't matter if this is a float
    label = np.array(pf.elements[0].data[label_field], dtype='float64')
    z = np.array(pf.elements[0].data['z'], dtype='float64')
    if rgb == True:
        r = np.array(pf.elements[0].data['red'], dtype='float64')
        g = np.array(pf.elements[0].data['green'], dtype='float64')
        b = np.array(pf.elements[0].data['blue'], dtype='float64')
    else:
        print('No rgb selected for processing')
    nx = np.array(pf.elements[0].data['nx'], dtype='float64')
    ny = np.array(pf.elements[0].data['ny'], dtype='float64')
    nz = np.array(pf.elements[0].data['nz'], dtype='float64')
#    e1 = np.array(pf.elements[0].data["e1(31)"], dtype='float64')
#    e2 = np.array(pf.elements[0].data["e2(31)"], dtype='float64')
#    e3 = np.array(pf.elements[0].data["e3(31)"], dtype='float64')
    a = np.array(pf.elements[0].data['anisotropy(31)'], dtype='float64')
    c = np.array(pf.elements[0].data["curvature(31)"], dtype='float64')
    et = np.array(pf.elements[0].data["eigenentropy(31)"], dtype='float64')
    es = np.array(pf.elements[0].data["eigen_sum(31)"], dtype='float64')
    l = np.array(pf.elements[0].data["linearity(31)"], dtype='float64')
    pl = np.array(pf.elements[0].data["planarity(31)"], dtype='float64')
    om = np.array(pf.elements[0].data["omnivariance(31)"], dtype='float64')
    sp = np.array(pf.elements[0].data["sphericity(31)"], dtype='float64')

    label.shape = (label.shape[0], 1)
    z.shape=(label.shape[0], 1)
    a.shape=(label.shape[0], 1)
    if rgb == True:
        r.shape=(label.shape[0], 1)
        g.shape=(label.shape[0], 1)
        b.shape=(label.shape[0], 1)
    nx.shape=(label.shape[0], 1)
    ny.shape=(label.shape[0], 1)
    nz.shape=(label.shape[0], 1)
    c.shape=(label.shape[0], 1)
    et.shape=(label.shape[0], 1)
    es.shape=(label.shape[0], 1)
    l.shape=(label.shape[0], 1)
    om.shape=(label.shape[0], 1)
    pl.shape=(label.shape[0], 1)
    sp.shape=(label.shape[0], 1)
    
    if rgb == True:
        final = np.hstack((label,  r,g,b, z, nx,ny,nz, a, c, et, es, l, pl, om,  sp))
        del r,g,b, nx, ny, nz, a,  c, et, es, l, om, pl, sp, z
    else:
        final = np.hstack((label, z, nx,ny,nz, a, c, et, es, l, pl, om,  sp))
        del nx,ny,nz, pf,  a, c, et, es, l, om, pl, sp
        
    # all these could be dumped now
   
    
    # prep for sklearn
    X_train = final[final[:,0] >= 0]
        
    # Remove non-finite values
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    
    # y labels
    #y_train = X_train[:,0] 
    
    if outFile != None:
        jb.dump(X_train, outFile, compress=2)
    
    return X_train
    

def classify_ply(incld, inModel, class_field='scalar_class', rgb=True,
                 outcld=None):
    
    """ 
    Classify a point cloud (ply format)
    
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
    
                
    class_field: string
               the name of the field that the results will be written to
               this must already exist! Create in CldComp. or cgal
    rgb: bool
        whether there is rgb data to be included
                 
    outcld: string
               path to a new ply to write if not writing to the input one

    """  
    
    
    
    
    pf = PlyData.read(incld)
    
#    pProps =['anisotropy', 'curvature', "eigenentropy", "eigen_sum",
#             "linearity", "omnivariance", "planarity", "sphericity"]
   
    #TODO this is a mess - was being lazy must tidy this
    # doesn't matter if this is a float
    
#    data_type_relation = [
#    ('int8', 'i1'),('char', 'i1'),('uint8', 'u1'), ('uchar', 'b1'),('uchar', 'u1'),
#    ('int16', 'i2'),('short', 'i2'),('uint16', 'u2'),('ushort', 'u2'),('int32', 'i4'),
#    ('int', 'i4'), ('uint32', 'u4'),('uint', 'u4'),('float32', 'f4'),('float', 'f4'),
#    ('float64', 'f8'),('double', 'f8')]
#    
#    data_types = dict(data_type_relation)
#    _data_type_reverse = dict((b, a) for (a, b) in _data_type_relation)
        
        
#    x = np.array(pf.elements[0].data['x'], dtype='float64')
#    y = np.array(pf.elements[0].data['y'], dtype='float64')
    z = np.array(pf.elements[0].data['z'], dtype='float64')
    nx = np.array(pf.elements[0].data['nx'], dtype='float64')
    ny = np.array(pf.elements[0].data['ny'], dtype='float64')
    nz = np.array(pf.elements[0].data['nz'], dtype='float64')
#    e1 = np.array(pf.elements[0].data["e1(31)"], dtype='float64')
#    e2 = np.array(pf.elements[0].data["e2(31)"], dtype='float64')
#    e3 = np.array(pf.elements[0].data["e3(31)"], dtype='float64')
    if rgb == True:
        r = np.array(pf.elements[0].data['red'], dtype='float64')
        g = np.array(pf.elements[0].data['green'], dtype='float64')
        b = np.array(pf.elements[0].data['blue'], dtype='float64')
    else:
        print('No rgb selected for processing')
    a = np.array(pf.elements[0].data['anisotropy(31)'], dtype='float64')
    c = np.array(pf.elements[0].data["curvature(31)"], dtype='float64')
    et = np.array(pf.elements[0].data["eigenentropy(31)"], dtype='float64')
    es = np.array(pf.elements[0].data["eigen_sum(31)"], dtype='float64')
    l = np.array(pf.elements[0].data["linearity(31)"], dtype='float64')
    pl = np.array(pf.elements[0].data["planarity(31)"], dtype='float64')
    om = np.array(pf.elements[0].data["omnivariance(31)"], dtype='float64')
    sp = np.array(pf.elements[0].data["sphericity(31)"], dtype='float64')
  
#    # Now back to the task in hand
   
#    x.shape=(a.shape[0], 1)
#    y.shape=(a.shape[0], 1)
    z.shape=(a.shape[0], 1)
    nx.shape=(a.shape[0], 1)
    ny.shape=(a.shape[0], 1)
    nz.shape=(a.shape[0], 1)
    if rgb == True:
        r.shape=(a.shape[0], 1)
        g.shape=(a.shape[0], 1)
        b.shape=(a.shape[0], 1)
    a.shape=(a.shape[0], 1)
    c.shape=(a.shape[0], 1)
    et.shape=(a.shape[0], 1)
    es.shape=(a.shape[0], 1)
    l.shape=(a.shape[0], 1)
    om.shape=(a.shape[0], 1)
    pl.shape=(a.shape[0], 1)
    sp.shape=(a.shape[0], 1)
    
#    
    if rgb == True:
        X = np.hstack((r,g,b, z, nx,ny,nz, a, c, et, es, l, pl, om,  sp))
        del r,g,b, nx, ny, nz, a,  c, et, es, l, om, pl, sp, z
    else:
        X = np.hstack((z, nx,ny,nz, a, c, et, es, l, pl, om,  sp))
        del nx, ny, nz, a,  c, et, es, l, om, pl, sp, z
    
    # all these could be dumped now
    del r,g,b, nx, ny, nz, a,  c, et, es, l, om, pl, sp, z
    
    # keep a for the shape
    X[np.where(np.isnan(X))]=0
    X = X[np.isfinite(X).all(axis=1)]

    print('Classifying')
    
    if os.path.splitext(inModel)[1] == ".h5":
        model1 = load_model(inModel)
        predictClass = model1.predict(X)
        # get the class based on the location of highest prob
        predictClass = np.argmax(predictClass,axis=1)
    else:  
        model1 = joblib.load(inModel)
        predictClass = model1.predict(X)
    
    
    
#    if keras == True:
#        pf.elements[0].data[class_field] = np.argmax(predictClass, axis=1)
#    else:
    pf.elements[0].data[class_field]=predictClass
    
    if outcld != None:
    
        pf.write(outcld)
    else:
        pf.write(incld)



    
def rmse_vector_lyr(inShape, attributes):

    """ 
    Using sklearn get the rmse of 2 vector attributes 
    (the actual and predicted of course in the order ['actual', 'pred'])
    
    
    Parameters 
    ----------- 
    
    inShape : string
              the input vector of OGR type
        
    attributes : list
           a list of strings denoting the attributes
         

    """    
    
    #open the layer etc
    shp = ogr.Open(inShape)
    lyr = shp.GetLayer()
    labels = np.arange(lyr.GetFeatureCount())
    
    # empty arrays for att
    pred = np.zeros((1, lyr.GetFeatureCount()))
    true = np.zeros((1, lyr.GetFeatureCount()))
    
    for label in labels: 
        feat = lyr.GetFeature(label)
        true[:,label] = feat.GetField(attributes[0])
        pred[:,label] = feat.GetField(attributes[1])
    
    
    
    error = np.sqrt(metrics.mean_squared_error(true, pred))
    
    return error



# crap from classif point for ref##############################################
#label = np.array(pf.elements[0].data[train], dtype='float64')

# This ply lib is alright but writing I/O is extremely verbose
# Purgatorial stuff
# TODO Jeez tidy this too
    
    
    
#    pps = [PlyProperty('x', 'float'), PlyProperty('y', 'float'), 
#                                         PlyProperty('z', 'float'), 
#                                         PlyProperty('nx', 'float'), 
#                                         PlyProperty('ny', 'float'), 
#                                         PlyProperty('nz', 'float'), 
#                                         PlyProperty('training', 'int'),
#                                         PlyProperty('label', 'int'), 
#                                         PlyProperty('red', 'uchar'), 
#                                         PlyProperty('green', 'uchar'), 
#                                         PlyProperty('blue', 'uchar'), 
#                                         PlyProperty('e1(31)', 'float'), 
#                                         PlyProperty('e2(31)', 'float'), 
#                                         PlyProperty('e3(31)', 'float'), 
#                                         PlyProperty('anisotropy(31)', 'float'), 
#                                         PlyProperty('curvature(31)', 'float'), 
#                                         PlyProperty('eigenentropy(31)', 'float'), 
#                                         PlyProperty('eigen_sum(31)', 'float'), 
#                                         PlyProperty('linearity(31)', 'float'), 
#                                         PlyProperty('omnivariance(31)', 'float'), 
#                                         PlyProperty('planarity(31)', 'float'), 
#                                         PlyProperty('sphericity(31)', 'float'),
#                                         PlyProperty(train_label, 'int'),
#                                         PlyProperty(outclass, 'int')]
#    
#    pfn = PlyElement(['vertex'], pps, pf.elements[0].count)
#    

#    pfn = pf['vertex']
#    pfn.properties=()
#    
#    props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue','e1','e2', 'e3',
#             'anisotropy', 'curvature', "eigenentropy", "eigen_sum","linearity", 
#     "omnivariance", "planarity", "sphericity", train_label, outclass]
#    
#    pfn.data.dtype.names = props
#    
#    
#    
#    
#
#    
#    plist = PlyListProperty('float', 'uchar', 'int')
#    #PlyElement('vertex',
#    
#    pfn.properties = (plist, (PlyProperty('x', 'float'),
#                                 PlyProperty('y', 'float'),
#                                 PlyProperty('z', 'float'),
#                                 PlyProperty('nx', 'float'),
#                                 PlyProperty('ny', 'float'),
#                                 PlyProperty('nz', 'float'), 
#                                 PlyProperty('red', 'uchar'),
#                                 PlyProperty('green', 'uchar'),
#                                 PlyProperty('blue', 'uchar'),
#                                 PlyProperty('e1(31)', 'float'),
#                                 PlyProperty('e2(31)', 'float'),
#                                 PlyProperty('e3(31)', 'float'),
#                                 PlyProperty('anisotropy(31)', 'float'),
#                                 PlyProperty('curvature(31)', 'float'),
#                                 PlyProperty('eigen_sum(31)', 'float'),
#                                 PlyProperty('linearity(31)', 'float'),
#                                 PlyProperty('omnivariance(31)', 'float'),
#                                 PlyProperty('planarity(31)', 'float'),
#                                 PlyProperty('sphericity(31)', 'float'),
#                                 PlyProperty(train_label, 'int'),
#                                 PlyProperty(outclass, 'int')))
#    
#    
#    pfn.elements[0].data['x'] = x
#    pfn.elements[0].data['y'] = y
#    pfn.elements[0].data['z'] = z
#    pfn.elements[0].data['nx'] = nx
#    pfn.elements[0].data['ny'] = ny
#    pfn.elements[0].data['nz'] = nz
#    pfn.elements[0].data['red'] = r
#    pfn.elements[0].data['green'] = g
#    pfn.elements[0].data['blue'] = b
#    pfn.elements[0].data['anisotropy(31)'] = a
#    pfn.elements[0].data["curvature(31)"] = c
#    pfn.elements[0].data["eigenentropy(31)"] = et
#    pfn.elements[0].data["eigen_sum(31)"] = es
#    pfn.elements[0].data["linearity(31)"] = l
#    pfn.elements[0].data["planarity(31)"]  = pl
#    pfn.elements[0].data["omnivariance(31)"] = om
#    pfn.elements[0].data["sphericity(31)"] =sp    
