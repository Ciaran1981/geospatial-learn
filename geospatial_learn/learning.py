# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:28:13 2016

author: Ciaran Robb

Research Associate in Earth Observation
Centre for Landscape and Climate Research (CLCR)
Department of Geography, University of Leicester, University Road, Leicester, 
LE1 7RH, UK 
upport
If you use code to publish work cite/acknowledge me, Sciki-learn 

Description
-----------
The learning module set of functions provide a framework to optimise and classify
EO data for both per pixel or object properties


Performance notes

This was tested on an i7 intel with 16gb of ram, so with large images/arrays 
of stats this will inevitably be slower - especially with a more standard machine.
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
from geospatial_learn.geodata import copy_dataset_config
#from time import sleep
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import OrderedDict
#import os
import shapefile
import glob
from sklearn import svm
import gdal, ogr#,osr
import numpy as np
from sklearn.model_selection import StratifiedKFold
#from sklearn import cross_validation, metrics 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
import joblib as jb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from geospatial_learn.geodata import array2raster
from geospatial_learn.shape import _bbox_to_pixel_offsets#, zonal_stats
from scipy.stats import randint as sp_randint
from scipy.stats import expon
#from scipy.sparse import csr_matrix
import pandas as pd
import simpledbf
from tpot import TPOTClassifier, TPOTRegressor



gdal.UseExceptions()
ogr.UseExceptions()

def create_model_tpot(X_train, outModel, cv=6, cores=-1,
                      regress=False, params = None, scoring=None):
    
    """
    Create a model using the tpot library where genetic algorithms
    are used to optimise pipline and params. 
    
    Parameters
    ----------  
    X_train : np array
        numpy array of training data where the 1st column is labels
    
    outModel : string
        the output model path which is a .py file
    
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
                              n_jobs=cores, warm_start=True)
        tpot.fit(X_train, y_train)
        
    elif params != None and regress is False:
        tpot = TPOTClassifier(config_dict=params, n_jobs=cores, warm_start=True)
        tpot.fit(X_train, y_train)
        
    if params is None and regress is True:       
        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2,
                              n_jobs=cores, warm_start=True)
        tpot.fit(X_train, y_train)
        
    elif params != None and regress is True:
        tpot = TPOTRegressor(config_dict=params, n_jobs=cores, warm_start=True)
        tpot.fit(X_train, y_train)

    tpot.export(outModel)    
    
    

def create_model(X_train, outModel, clf='svc', random=False, cv=6, cores=-1,
                 strat=True, regress=False, params = None, scoring=None):
    
    """
    Brute force or random model creating using scikit learn. Either use the
    default params in this function or enter your own (recommended - see sklearn)
    
    Parameters
    ----------  
    X_train : np array
        numpy array of training data where the 1st column is labels
    
    outModel : string
        the output model path which is a gz file
    
    clf : string
        an sklearn or xgb classifier/regressor 
        logit, sgd, linsvc, svc, svm, nusvm, erf, rf, gb, xgb
    
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
            
    
    General Note
    ------------
    There are more sophisticated ways to tune a model, this greedily 
    searches everything but can be computationally costly. Fine tuning 
    in a more measured way is likely better. There are numerous books,
    guides etc...
    E.g. with gb- first tune no of trees for gb, then learning rate, then
    tree specific
        
    Notes on algorithms
    -------------------   
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
    # TODO this has become rather messy and inefficient - need to make it more 
    # elegant
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
         
    if clf is 'xgb' and regress is False:
        xgb_clf = XGBClassifier()
        if params is None:
                # This is based on the Tianqi Chen author of xgb
                # tips for data science as a starter
                # he recommends fixing trees - I haven't as default here...
                # crunch this first then fine tune rest
                # 
                ntrees = 500
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
    if clf is 'gb' and regress is False:
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
        
    if clf is 'gb'  and regress is True:
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
    if clf == 'rf':
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
         
    if clf is 'rf' and regress is True:
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
    if clf is 'linsvc' and regress is True:
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
        svm_clf = svm.SVC(probability=True)
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
        svm_clf = svm.NuSVC(probability=True)
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
    if clf is 'nusvc' and regress is True:
         svm_clf = svm.NuSVR()
         param_grid = [{'nu':[0.25, 0.5, 0.75, 1],'gamma': [1e-3, 1e-4]}]
         grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
         grid.fit(X_train, y_train)
         joblib.dump(grid.best_estimator_, outModel) 
             #print("done in %0.3fs" % (time() - t0))
    if clf is 'logit':
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
        
    if clf is 'sgd':
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

    return [grid.cv_results_, grid.best_score_, grid.best_params_, grid.best_estimator_]
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
    
    """ This function uses the oob score to find the best parameters.
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
        
        Returns
        -------
            
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
    
    for label, clf in tqdm(ensemble_clfs):
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
    Plot the 
    feature importances of an ensemble classifier
    
    Parameters
    ----------
    modelPth : string
        A sklearn model path 
    
    featureNames : list of strings
        a list of feature names

    Notes
    -----
    Adapted from the excellent ML with py book credit to the author
    
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
    """ A function to classify an image using a pre-saved model - assumes
    a folder of tiled rasters for memory management - classify_pixel_block is
    recommended instead of this function
    
    Where:
    
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
    
    Where:
    ------------------
        
    model : sklearn model
        a path to a scikit learn model that has been saved 
    
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
        o gdal dataype - default is int32

    Usage exampel:
    ---------------------    
    classify_pixel_block(model, inputImage, 8, outMap)
    
    
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
    
    outDataset = copy_dataset_config(inputImage, outMap = outMap,
                                     bands = bands)
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
    
    Where:
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

        
        
    Usage: 
    ---------------------------
    (typical - leaving defaults)
    prob_pixel_block(model, inputImage, 8, 8, outMap)
    
    Notes
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
    
    outDataset = copy_dataset_config(inputImage, outMap = outMap,
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
    
    Where:
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
    tempList = list()
    for name in attributes:
        tempList.append(df[name])
    X = pd.concat(tempList, axis=1)
    X = X.as_matrix()
    #X = np.delete(X, 0, axis=1)
    
    X = np.float32(X)
    print('data ready')
    """
    Classification
    
    The data must be prepared for input and exit from scikit learn
    
    e.g we require cross tabulating training and input data
    
    The next three lines obviously depend on the state in which the training data
    comes into this process
    """
    #----------------------------------------------------------------------------------
    X = X[X[:,0] != 0]
    X[np.where(np.isnan(X))]=0
    X = X[np.isfinite(X).all(axis=1)]
    
    # remove id/dn and labels from X_train

    #X = X[:,2:arrShp[1]+1]

    
    
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

    
def get_training_shp(inShape, train_col_number, outFile = None):
    """
    Collect training from a shapefile attribute table. Used for object-based 
    classification. 
    
    Where:
    --------------------    
    inShape : string
        the input shapefile - must be esri .shp at present
    
    train_col_number :
        the column number in the attribute table containing the training 
        labels 
    outFile : string
        path to training data to be saved (.gz)
    
    Returns:
    ---------------------
    training data as a numpy array, first column is labels, rest are features
    
    """
    
    r = shapefile.Reader(inShape)
    records = np.asarray(r.records(), dtype = np.float32())
    records = np.delete(records, 0,1)
    
    #TODO - come up with a field based solution to this

    records = records[records[:,train_col_number]>0]
    
    y = records[:,train_col_number]
    y.shape = (records.shape[0],1)
    records = np.delete(records, train_col_number, 1)
    records = np.hstack((y,records))
    
    if outFile != None:
        jb.dump(records, outFile, compress=2)
    

    return records
             

    
    
def get_training(inShape, inRas, bands, field, outFile = None):
    """
    Collect training as an np array for use with create model function
    Where:
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
    
    Returns:
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
            
        src_offset = bbox_to_pixel_offsets(rgt, geom)
        
        
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
        if np.shape(src_array) is ():
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

    
def get_training_point(inShape, inRas, bands, field):
    """ Collect training as a np array for use with create model function using 
          point data
    Where:
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
    
    Returns:
    ---------------------
        A tuple containing:
        -np array of training data
        -list of polygons with invalid geometry that were not collected 

    
    UNFINISHED DO NOT USE
    
    
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
    
    print('getting points')
    for label in tqdm(labels):
        #print(label)
        feat = lyr.GetFeature(label)
        if feat == None:
            print('no geometry for feature '+str(label))
            continue
        iD = feat.GetField(field)
        geom = feat.GetGeometryRef()
        mx,my=geom.GetX(), geom.GetY()  #coord in map units

        #Convert from map to pixel coordinates.
        #Only works for geotransforms with no rotation.
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel
    
        structval=rb.ReadRaster(px, py,1,1,buf_type=gdal.GDT_UInt16) #Assumes 16 bit int aka 'short'
        intval = struct.unpack('h' , structval) #use the 'short' format code (2 bytes) not int (4 bytes)
    
        print(intval[0])

            
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
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
            

        
        
        rb = raster.GetRasterBand(1)
        src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                           src_offset[3])
        if np.shape(src_array) is ():
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
        #print(label,fieldval,xcount, ycount)   
        outData.append(X)
    outData = np.asarray(outData)
    outData = np.concatenate(outData).astype(None)
    return outData, rejects
 
