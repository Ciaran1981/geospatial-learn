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

from xgboost.sklearn import XGBClassifier, XGBRegressor
import xgboost as xgb


from tqdm import tqdm

import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
from sklearn import svm
from osgeo import gdal, ogr#,osr
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, GroupKFold, KFold, 
                                     train_test_split,GroupShuffleSplit,
                                     StratifiedGroupKFold, 
                                     PredefinedSplit)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,RandomForestRegressor,
                              GradientBoostingRegressor, ExtraTreesRegressor,
                              VotingRegressor, VotingClassifier, StackingRegressor,
                              StackingClassifier,
                              HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import (LabelEncoder, MaxAbsScaler, MinMaxScaler,
                                   Normalizer, PowerTransformer,StandardScaler,
                                   QuantileTransformer)
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, LinearSVC, LinearSVR
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from catboost import (CatBoostClassifier, CatBoostRegressor,Pool )
import lightgbm as lgb
# from autosklearn.classification import AutoSklearnClassifier
# from autosklearn.regression import AutoSklearnRegressor
from skorch import NeuralNetClassifier
from optuna.integration import LightGBMPruningCallback
from optuna import create_study
import torch.nn as nn
import joblib
import joblib as jb
from geospatial_learn.raster import array2raster
from geospatial_learn.shape import _bbox_to_pixel_offsets#, zonal_stats
#from tpot import TPOTClassifier, TPOTRegressor
import warnings
from geospatial_learn.raster import _copy_dataset_config
import geospatial_learn.handyplots as hp
import geopandas as gpd
import pandas as pd
from psutil import virtual_memory
import os
gdal.UseExceptions()
ogr.UseExceptions()

# for current issue with numpy
np.warnings = warnings



# TODO -NOT WORKING of course stupid object structure making feeding params
# impossible
def create_model_optuna(X_train, outModel, clf='erf', group=None, random=False,
                 cv=5, params=None, pipe=None, cores=-1, strat=True, 
                 test_size=0.3, regress=False, return_test=True,
                 scoring=None, class_names=None, save=True):
    """
    Train a model using the Optuna framework
    
    Parameters
    ---------------   
    
    X_train: np array
              numpy array of training data where the 1st column is labels.
              If the groupkfold is used, the last column will be the group labels
    
    outModel: string
               the output model path which is a gz file, if using keras it is 
               h5
    
    params: dict
            a dict of model params (see scikit learn). If using a pipe(line)
            remember to prefix each param as follows with parent object 
            and two underscores.
            param_grid ={"selector__threshold": [0, 0.001, 0.01],
             "classifier__n_estimators": [1075]}
            
    clf: string
          an sklearn or xgb classifier/regressor 
          logit, sgd, linsvc, svc, svm, nusvm, erf, rf, gb, xgb,
    
    group: np.array
            array of group labels for train/test split and grid search
            useful to avoid autocorrelation          
          
    random: bool
             if True, a random param search
    
    cv: int
         no of folds
    
    cores: int or -1 (default)
            the no of parallel jobs
    
    strat: bool
            a stratified grid search
    
    test_size: float
            percentage to hold out to test
    
    regress: bool
              a regression model if True, a classifier if False
    
    return_test: bool
              return X_test and y_test along with results (last two entries
              in list)
    
    scoring: string
              a suitable sklearn scoring type (see notes)
    
    class_names: list of strings
                class names in order of their numercial equivalents
    
    Returns
    -------
    A list of:
        
    [grid.best_estimator_, grid.cv_results_, grid.best_score_, 
            grid.best_params_, classification_report, X_test, y_test]
    
    
    """
    
    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    if group is not None: #replace this later not good
        inds = np.where(np.isfinite(X_train).all(axis=1))
        group = group[inds]
    
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    #no_classes = len(np.unique(y_train))
    

    # this is not a good way to do this
    if group is not None:
        
        # maintain group based splitting from initial train/test split
        # to main train set
        # TODO - sep func?
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=0)
        split = splitter.split(X_train, y_train, group)
        train_inds, test_inds = next(split)
    
        X_test = X_train[test_inds]
        y_test = y_train[test_inds]
        X_train = X_train[train_inds]
        y_train = y_train[train_inds]
        group_trn = group[train_inds]
        
        group_kfold = GroupKFold(n_splits=cv) 
        # Create a nested list of train and test indices for each fold
        k_kfold = group_kfold.split(X_train, y_train, group_trn)  

        train_ind2, test_ind2 = [list(traintest) for traintest in zip(*k_kfold)]

        cv = [*zip(train_ind2, test_ind2)]
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)

        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    
    
    if regress is True:
        scr = metrics.mean_squared_error
    else:
        scr = metrics.log_loss
    
    # issue is how to pass the dict as an arg in this parent function
    def objective(trial, X, y, cv, group, score=scr):
        
        # how can one pass this as an arg
        param_grid = {
            #"device_type": trial.suggest_categorical("device_type", ['gpu']),
           #'metric': 'rmse', 
            'random_state': 42,
            'n_estimators': 20000,
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
            'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
            'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
            }
    

    
        cv_scores = np.empty(10)

        
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y, group)):
            
            # these next two need altered to fit my data structures above
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # had objective=binary but that is for class 
            model = lgb.LGBMRegressor( **param_grid)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                #eval_metric="l2",
                early_stopping_rounds=100,
                callbacks=[
                    LightGBMPruningCallback(trial, "rmse") 
                ],  # Add a pruning callback - suspect this may be a source of problems
            )
            preds = model.predict(X_test)
            cv_scores[idx] = score(y_test, preds)

        return np.mean(cv_scores)

    study = create_study(direction="minimize", study_name="Model training")
    func = lambda trial: objective(trial, X_train, y_train, group_kfold, group_trn, 
                                   score=metrics.mean_squared_error)
    study.optimize(func, n_trials=50)
    
    print(f"\tBest value (rmse or r2): {study.best_value:.5f}")
    print(f"\tBest params:")

def _group_cv(X_train, y_train, group, test_size=0.2, cv=10, strat=False):
    
    """
    Return the splits and and vars for a group grid search
    """
        
    # maintain group based splitting from initial train/test split
    # to main train set
    # TODO - sep func?
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=0)
    split = splitter.split(X_train, y_train, group)
    train_inds, test_inds = next(split)

    X_test = X_train[test_inds]
    y_test = y_train[test_inds]
    X_train = X_train[train_inds]
    y_train = y_train[train_inds]
    group_trn = group[train_inds]
    
    if strat == True:
        group_kfold = StratifiedGroupKFold(n_splits=cv).split(X_train,
                                                              y_train,
                                                              group_trn)
    else:
        group_kfold = GroupKFold(n_splits=cv).split(X_train,
                                                    y_train,
                                                    group_trn) 
    
    # all this not required produces same as above - keep for ref though
    # # Create a nested list of train and test indices for each fold
    # k_kfold = group_kfold.split(X_train, y_train, groups=group_trn)  

    # train_ind2, test_ind2 = [list(traintest) for traintest in zip(*k_kfold)]

    # cv = [*zip(train_ind2, test_ind2)]
    
    return X_train, y_train, X_test, y_test, group_kfold

def rec_feat_sel(X_train, featnames, preproc=('scaler', None),  clf='erf',  
                 group=None, 
                 cv=5, params=None, cores=-1, strat=True, 
                 test_size=0.3, regress=False, return_test=True,
                 scoring=None, class_names=None, save=True, cat_feat=None):
    
    """
    Recursive feature selection
    
    Parameters
    ---------------   
    
    X_train: np array
              numpy array of training data where the 1st column is labels.
              If the groupkfold is used, the last column will be the group labels
    
    
    params: dict
            a dict of model params (see scikit learn). If using a pipe(line)
            remember to prefix each param as follows with parent object 
            and two underscores.
            param_grid ={"selector__threshold": [0, 0.001, 0.01],
             "classifier__n_estimators": [1075]}
             
    clf: string
          an sklearn or xgb classifier/regressor 
          logit, sgd, linsvc, svc, svm, nusvm, erf, rf, gb, xgb,
    
    group: np.array
            array of group labels for train/test split and grid search
            useful to avoid autocorrelation          
    
    cv: int
         no of folds
    
    cores: int or -1 (default)
            the no of parallel jobs
    
    strat: bool
            a stratified grid search
    
    test_size: float
            percentage to hold out to test
    
    regress: bool
              a regression model if True, a classifier if False
    
    scoring: string
              a suitable sklearn scoring type (see notes)
    
    class_names: list of strings
                class names in order of their numercial equivalents
    
    Returns
    -------
    
    bool index of features, list of chosen feature names
    """
    #TODO need to make all this a func
    # Not woring with hgb....
    clfdict = {'rf': RandomForestClassifier(random_state=0),
               'erf': ExtraTreesClassifier(random_state=0),
               'gb': GradientBoostingClassifier(random_state=0),
               'xgb': XGBClassifier(random_state=0),
               'logit': LogisticRegression(),
               # 'catb': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42),
               # 'catbgpu': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42,
                                          # task_type="GPU",
                                          # devices='0:1'),
               'lgbm': lgb.LGBMClassifier(random_state=0),
 #use_best_model=True-needs non empty eval set
                'hgb': HistGradientBoostingClassifier(early_stopping=True,
                                                      random_state=0)}
    
    regdict = {'rf': RandomForestRegressor(random_state=0),
               'erf': ExtraTreesRegressor(random_state=0),
               'gb': GradientBoostingRegressor(early_stopping=True,
                                               random_state=0),
               'xgb': XGBRegressor(random_state=0),
               # 'catb': CatBoostRegressor(logging_level='Silent', 
               #                           random_seed=42),
               # 'catbgpu': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42,
               #                            task_type="GPU",
                                          # devices='0:1'),
               'lgbm': lgb.LGBMRegressor(random_state=0),

               'hgb': HistGradientBoostingRegressor(early_stopping=True,
                                                    random_state=0)}
    
    if regress is True:
        model = regdict[clf]
        if clf == 'hgb':
            # until this is fixed - enabling above does nothing
            model.do_early_stopping = True
        if scoring is None:
            scoring = 'r2'
    else:
        model = clfdict[clf]
        cv = StratifiedKFold(cv)
        if scoring is None:
            scoring = 'accuracy'
    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    if group is not None: #replace this later not good
        inds = np.where(np.isfinite(X_train).all(axis=1))
        group = group[inds]
    
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    #no_classes = len(np.unique(y_train))
    

    # this is not a good way to do this
    # Does this matter for feature selection??
    if group is not None:
        
        X_train, y_train, X_test, y_test, cv = _group_cv(X_train, y_train,
                                                         group, test_size,
                                                         cv)
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)
    

        
    rfecv = RFECV(estimator=model, 
                  step=1, 
                  cv=cv, 
                  scoring=scoring,
                  n_jobs=cores) # suspect this is no of folds
    
    pipeline  = Pipeline([preproc,
                          ('selector', rfecv)])


    # VERY slow - but I guess it is grid searching feats first
    #rfecv
    pipeline.fit(X_train, y_train)

    # First experiment is to add this as a fixed part of the process at the start
    # as it will slow it down otherwise

    # featind = pipeline[1].support_ # gains the feat indices(bool)
    # featnmarr = np.array(featnames)
    # featnames_sel = featnmarr[featind==True].tolist()

    # as X_train has changed we cant select from it within here
    
    return pipeline
    
    
def create_model(X_train, outModel, clf='erf', group=None, random=False,
                 cv=5, params=None, pipe='default', cores=-1, strat=True, 
                 test_size=0.3, regress=False, return_test=True,
                 scoring=None, class_names=None, save=True, cat_feat=None,
                 plot=True):
    
    """
    Brute force or random model creating using scikit learn.
    
    Parameters
    ---------------   
    
    X_train: np array
              numpy array of training data where the 1st column is labels.
              If the groupkfold is used, the last column will be the group labels
    
    outModel: string
               the output model path which is a gz file, if using keras it is 
               h5
    
    params: dict
            a dict of model params (see scikit learn). If using a pipe(line)
            remember to prefix each param as follows with parent object 
            and two underscores.
            param_grid ={"selector__threshold": [0, 0.001, 0.01],
             "classifier__n_estimators": [1075]}
            
    pipe: str,dict,None
            if 'default' will include a preprocessing pipeline consisting of
            StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler(),
            otherwise specify in this form 
            pipe = {'scaler': [StandardScaler(), MinMaxScaler(),
                  Normalizer()]}
            or None will not preprocess the data
             
    clf: string
          an sklearn or xgb classifier/regressor 
          logit, sgd, linsvc, svc, svm, nusvm, erf, rf, gb, xgb,
    
    group: np.array
            array of group labels for train/test split and grid search
            useful to avoid autocorrelation          
          
    random: bool
             if True, a random param search
    
    cv: int
         no of folds
    
    cores: int or -1 (default)
            the no of parallel jobs
    
    strat: bool
            a stratified grid search
    
    test_size: float
            percentage to hold out to test
    
    regress: bool
              a regression model if True, a classifier if False
    
    return_test: bool
              return X_test and y_test along with results (last two entries
              in list)
    
    scoring: string
              a suitable sklearn scoring type (see notes)
    
    class_names: list of strings
                class names in order of their numercial equivalents
    
    Returns
    -------
    A list of:
        
    [grid.best_estimator_, grid.cv_results_, grid.best_score_, 
            grid.best_params_, classification_report, X_test, y_test]
    
        
    Notes:
    ------      
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
    

    clfdict = {'rf': RandomForestClassifier(random_state=0),
               'erf': ExtraTreesClassifier(random_state=0),
               'gb': GradientBoostingClassifier(random_state=0),
               'xgb': XGBClassifier(random_state=0),
               'logit': LogisticRegression(),
               # 'catb': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42),
               # 'catbgpu': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42,
               #                            task_type="GPU",
               #                            devices='0:1'),
               'lgbm': lgb.LGBMClassifier(random_state=0),

                'hgb': HistGradientBoostingClassifier(random_state=0),
                'svm': SVC(),
                'nusvc': NuSVC(),
                'linsvc': LinearSVC()}
    
    regdict = {'rf': RandomForestRegressor(random_state=0),
               'erf': ExtraTreesRegressor(random_state=0),
               'gb': GradientBoostingRegressor(random_state=0),
               'xgb': XGBRegressor(random_state=0),
               # 'catb': CatBoostRegressor(logging_level='Silent', 
               #                           random_seed=42),
               # 'catbgpu': CatBoostRegressor(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42,
               #                            task_type="GPU",
               #                            devices='0:1'),
               'lgbm': lgb.LGBMRegressor(random_state=0),
               'hgb': HistGradientBoostingRegressor(random_state=0),
                'svm': SVR(),
                'nusvc': NuSVR(),
                'linsvc': LinearSVR()}
    
    if regress is True:
        model = regdict[clf]
        if scoring is None:
            scoring = 'r2'
    else:
        model = clfdict[clf]
        if group is None:
            cv = StratifiedKFold(cv)
        if scoring is None:
            scoring = 'accuracy'
    
    if cat_feat:
        model.categorical_features = cat_feat
    
    
    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    if group is not None: #replace this later not good
        inds = np.where(np.isfinite(X_train).all(axis=1))
        group = group[inds]
    
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    #no_classes = len(np.unique(y_train))
    

    # this is not a good way to do this
    if regress == True:
        strat = False # failsafe
        
    if group is not None: # becoming a mess

        X_train, y_train, X_test, y_test, cv = _group_cv(X_train, y_train,
                                                             group, test_size,
                                                             cv, strat=strat)        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)
        #cv = StratifiedKFold(cv)
        
    
    if pipe == 'default':
        # the dict must be in order of proc to work hence this
        # none is included to ensure
        # non prep data is considered
        #  should add selector?? seems to produce a load of nan errors (or warnings)
        sclr = {'scaler': [StandardScaler(), MinMaxScaler(),
                           Normalizer(), MaxAbsScaler(), 
                           QuantileTransformer(output_distribution='uniform'),
                           #PowerTransformer(),
                           None]} 
        sclr.update(params)
        
    else:
        sclr = pipe.copy() # to stop the var getting altered in script
        sclr.update(params)
    
    sk_pipe = Pipeline([("scaler", StandardScaler()),
                        #("selector", None), lord knows why this fails on var thresh
                        ("classifier", model)])
        
    
    if random is True:
        # recall the model is in the pipeline
        grid = RandomizedSearchCV(sk_pipe, param_distributions=sclr, 
                                  n_jobs=cores, n_iter=20,  verbose=2)
    else:
        grid = GridSearchCV(sk_pipe,  param_grid=sclr, 
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=1)


    grid.fit(X_train, y_train)
    
    joblib.dump(grid.best_estimator_, outModel) 
    
    testresult = grid.best_estimator_.predict(X_test)
    
    if regress == True:
        regrslt = regression_results(y_test, testresult, plot=plot)
        results = [grid]
        
    else:
        crDf = hp.plot_classif_report(y_test, testresult, target_names=class_names,
                                      save=outModel[:-3]+'._classif_report.png')
        
        confmat = metrics.confusion_matrix(testresult, y_test, labels=class_names)
        
        if plot == True:
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confmat,
                                          display_labels=class_names)
            disp.plot()
    
        # confmat = hp.plt_confmat(X_test, y_test, grid.best_estimator_, 
        #                          class_names=class_names, 
        #                          cmap=plt.cm.Blues, 
        #                          fmt="%d", 
        #                          save=outModel[:-3]+'_confmat.png')    
        results = [grid, crDf, confmat]
        
    if return_test == True:
        results.extend([X_test, y_test])
    return results

def combine_models(X_train, modelist, mtype='regress', method='voting', group=None, 
                   test_size=0.3, outmodel=None, class_names=None, params=None,
                   final_est='xgb', cv=5):#, cores=1):
    
    """
    Combine models using either the voting or stacking methods in scikit-learn
    
    Parameters
    ----------
    
    X_train: np array
              numpy array of training data where the 1st column is labels.
    
    outmodel: string
               the output model path which is a gz file, if using keras it is 
               h5
    
    modelist: dict
            a list of tuples of model type (str) and model (class) e.g.
            [('gb', gmdl), ('rf', rfmdl), ('lr', reg3)]
    
    mtype: string
            either clf or regress
    
    method: string
            either voting or k = clusterer.labels_stacking

    test_size: float
            percentage to hold out to test  
                
    group: np.array
            array of group labels for train/test split and grid search
            useful to avoid autocorrelation
              
    class_names: list of strings
                class names in order of their numercial equivalents
    
    params: dict
            a dict of model params (If using stacking method for final estimator)
    
    final_est: string
             the final estimator one of (rf, gb, erf, xgb, logit)

    """
    bands = X_train.shape[1]-1
    
    X_train = X_train[X_train[:,0] != 0]
    
    if group is not None: #replace this later not good
        inds = np.where(np.isfinite(X_train).all(axis=1))
        group = group[inds]
    
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
        # Remove non-finite values
 
    
    # this is not a good way to do this
    if group is not None:
        # this in theory should not make any difference at this stage...
        # ...included for now
        # maintain group based splitting from initial train/test split
        # to main train set
        # TODO - sep func?
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=0)
        split = splitter.split(X_train, y_train, group)
        train_inds, test_inds = next(split)
    
        X_test = X_train[test_inds]
        y_test = y_train[test_inds]
        X_train = X_train[train_inds]
        y_train = y_train[train_inds]
        group_trn = group[train_inds]
        
        group_kfold = GroupKFold(n_splits=cv) 
        # Create a nested list of train and test indices for each fold
        k_kfold = group_kfold.split(X_train, y_train, group_trn)  

        train_ind2, test_ind2 = [list(traintest) for traintest in zip(*k_kfold)]

        cv = [*zip(train_ind2, test_ind2)]
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)
    
    if method == 'voting':
        comb = VotingRegressor(estimators=modelist)#, n_jobs=cores)
        # we only wish to predict really - but  necessary 
        # for sklearn model construct
    else:
        clfdict = {'rf': RandomForestClassifier(random_state=0),
                   'erf': ExtraTreesClassifier(random_state=0),
                   'gb': GradientBoostingClassifier(random_state=0),
                   'xgb': XGBClassifier(random_state=0),
                   'logit': LogisticRegression(),
                   'lgbm': lgb.LGBMClassifier(random_state=0),
                    'hgb': HistGradientBoostingClassifier(random_state=0),
                    'svm': SVC(),
                    'nusvc': NuSVC(),
                    'linsvc': LinearSVC()}
        
        regdict = {'rf': RandomForestRegressor(random_state=0),
                   'erf': ExtraTreesRegressor(random_state=0),
                   'gb': GradientBoostingRegressor(random_state=0),
                   'xgb': XGBRegressor(random_state=0),
                   'lgbm': lgb.LGBMRegressor(random_state=0),
                   'hgb': HistGradientBoostingRegressor(random_state=0),
                    'svm': SVR(),
                    'nusvc': NuSVR(),
                    'linsvc': LinearSVR()}
        
        if mtype == 'regress':
            # won't accept the dict even with the ** to unpack it
            fe = regdict[final_est]()
            fe.set_params(**params)
            
            comb = StackingRegressor(estimators=modelist,
                                final_estimator=fe)
        else:
            fe = clfdict[final_est]()
            fe.set_params(**params)
            cv = StratifiedKFold(cv)            
            comb = StackingClassifier(estimators=modelist,
                                final_estimator=fe)
            
    comb.fit(X_train, y_train)
    
    # Since there is no train/test this is misleading...
    train_pred = comb.predict(X_train)
    test_pred = comb.predict(X_test)  
    if mtype == 'regress':
        print('On the train split (not actually trained on this)')
        regression_results(y_train, train_pred)
        
        print('On the test split')
        regression_results(y_test, test_pred)
        
    else:
        crDf = hp.plot_classif_report(y_test, test_pred, target_names=class_names,
                              save=outmodel[:-3]+'._classif_report.png')
    
        confmat = hp.plt_confmat(X_test, y_test, comb, 
                                 class_names=class_names, 
                                 cmap=plt.cm.Blues, 
                                 fmt="%d", 
                                 save=outmodel[:-3]+'_confmat.png')
    
    if outmodel is not None:
        joblib.dump(comb, outmodel)
    return comb, X_test, y_test 


def regression_results(y_true, y_pred, plot=True):

    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    #mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    #print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('MedianAE', round(median_absolute_error, 4))
    print('RMSE: ', round(np.sqrt(mse),4))   
    #TODO add when sklearn updated
    if plot == True:    
        display = metrics.PredictionErrorDisplay.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            kind="actual_vs_predicted",
            #ax=ax,
            scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
            line_kwargs={"color": "tab:red"},
        )



def RF_oob_opt(model, X_train, min_est, max_est, step, group=None,
               regress=False):
    
    """ 
    This function uses the oob score to find the best parameters.
    
    This cannot be parallelized due to the warm start bootstrapping, so is
    potentially slower than the other cross val in the create_model function
        
    This function is based on an example from the sklearn site
        
    This function plots a graph diplaying the oob rate
        
    Parameters
    ---------------------
    
    model: string (.gz)
            path to model to be saved
    
    X_train: np array
              numpy array of training data where the 1st column is labels
    
    min_est: int
              min no of trees
    
    max_est: int
              max no of trees
    
    step: int
           the step at which no of trees is increased
    
    regress: bool
              boolean where if True it is a regressor
    
    Returns: tuple of np arrays
    -----------------------
        
    error rate, best estimator
        
    """
    # This is a bit slow at present, needs profiled
    #t0 = time()
    
    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    if group is not None: #replace this later not good
        inds = np.where(np.isfinite(X_train).all(axis=1))
        group = group[inds]
    
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    RANDOM_STATE = 123
    
    if group is not None:
        
        # maintain group based splitting from initial train/test split
        # to main train set
        # TODO - sep func?
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=0)
        split = splitter.split(X_train, y_train, group)
        train_inds, test_inds = next(split)
    
        X_test = X_train[test_inds]
        y_test = y_train[test_inds]
        X_train = X_train[train_inds]
        y_train = y_train[train_inds]
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0)
    
    
    
    print('iterating estimators')
    if regress is True:
        max_feat = X_train.shape[1]-1
        ensemble_clfs = [
        ("RandomForestRegressor, max_features='no_features'",
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
        er = np.array(error_rate["RandomForestRegressor, max_features='no_features'"][0:max_estimators])
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
    
    if regress is True:
        pred = best_model.predict(X_test)
        print('Test results are:')
        regression_results(np.expand_dims(y_test,1), pred)
    
    return error_rate, best_param


def plot_feature_importances(modelPth, featureNames, 
                             rank=True):
    
    """
    Plot the feature importances of an ensemble classifier
    
    Parameters
    --------------------------
    
    modelPth : string
               A sklearn model path or the estimator itself
    
    featureNames : list of strings
                   a list of feature names
    
    """
    if modelPth is not str:
        model = modelPth
    else:
        model = joblib.load(modelPth)
    
    # if model_type=='scikit':
    #     n_features = model.n_features_
    # if model_type=='xgb':
    #     n_features = model.n_features_in_
    # plt.barh(range(n_features), model.feature_importances_, align='center')
    # plt.yticks(np.arange(n_features), featureNames)
    # plt.xlabel("Feature importance")
    # plt.ylabel("Feature")
    # plt.ylim(-1, n_features)
    # plt.show()
    
    # fimp = pd.Series(model.feature_importances_, index=featureNames)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    fimp = pd.Series(importances, index=featureNames)
    
    if rank == True:
        fimp = fimp.sort_values(ascending=False)
    
    fig, ax = plt.subplots()
    fimp.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    
    
    return fimp



def plot_feat_importance_permutation(modelPth, featureNames,  X_test, y_test,
                                     rank=True):
    
    """
    Plot the feature importances of an ensemble classifier
    
    Parameters
    --------------------------
    
    modelPth : string
               A sklearn model path or the estimator itself
    
    featureNames : list of strings
                   a list of feature names
    
    Returns
    -------
    
    pandas df of importances
    
    """
    
    if modelPth is not str:
        model = modelPth
    else:
        model = joblib.load(modelPth)
    
    # if model_type=='scikit':
    #     n_features = model.n_features_
    # if model_type=='xgb':
    #     n_features = model.n_features_in_

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1
    )
    
    
    fimp = pd.Series(result.importances_mean, index=featureNames)
    if rank == True:
        fimp = fimp.sort_values(ascending=False)
    
    fig, ax = plt.subplots()
    fimp.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
    
    
    return fimp

       
def classify_pixel(model, inputDir, bands, outMap, probMap):
    """ 
    A function to classify an image using a pre-saved model - assumes
    a folder of tiled rasters for memory management - classify_pixel_block is
    recommended instead of this function
    
    Parameters
    
    ---------------
        
    model: sklearn model
            a path to a scikit learn model that has been saved 
    
    inputDir: string
               a folder with images to be classified
    
    bands: int
            the no of image bands eg 8
    
    outMap: string
             path to output image excluding the file format 'pathto/mymap'
    
    probMap: string
              path to output prob image excluding the file format 'pathto/mymap'
    
    FMT: string 
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

      
        
def classify_pixel_bloc(model, inputImage,  outMap, bands=[1,2,3], blocksize=None, 
                        FMT=None, ndvi = None, dtype = gdal.GDT_Int32):
    """
    A block processing classifier for large rasters, supports KEA, HFA, & Gtiff
    formats. KEA is recommended, Gtiff is the default
    
    Parameters
    ------------------
        
    model: sklearn model / keras model
            a path to a model that has been saved 
    
    inputImage: string
                 path to image including the file fmt 'Myimage.tif'
    
    bands: band
            list of band indices to be used eg [1,2,3]
    
    outMap: string
             path to output image excluding the file format 'pathto/mymap'
    
    FMT: string
          optional parameter - gdal readable fmt
    
    blocksize: int (optional) 
                size of raster chunck in pixels 256 tends to be quickest
                if you put None it will read size from gdal (this doesn't always pay off!)
    
    dtype: int (optional - gdal syntax gdal.GDT_Int32) 
            a gdal dataype - default is int32


    Notes
    -----
    
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
    
    outDataset = _copy_dataset_config(inDataset, outMap=outMap,
                                     dtype=dtype, bands=1)
    band = inDataset.GetRasterBand(1)
    cols = int(inDataset.RasterXSize)
    rows = int(inDataset.RasterYSize)
    outBand = outDataset.GetRasterBand(1)
    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = band.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize


    if os.path.splitext(model)[1] == ".h5":
        model1 = load_model(model)
    else:  
        model1 = joblib.load(model)
    
    noBands = inDataset.RasterCount
    
    # minus the bands to zero start
    bands = list(np.array(bands)-1)
    
    if blocksizeY==1:
        rows = np.arange(cols, dtype=np.int)                
        for row in tqdm(rows):
            i = int(row)
            j = 0
            #X = np.zeros(shape = (bands, blocksizeX))
            #for band in range(1,bands+1):
            
            X = inDataset.ReadAsArray(j, i, xsize=blocksizeX, ysize=blocksizeY)
            X.shape = ((noBands,blocksizeX))
            
            X = X[bands, :]
            
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
                                
                                X.shape = ((noBands,numRows*numCols))
                                X = X[bands, :]
                                X = X.transpose() 
                                X = np.where(np.isfinite(X),X,0) 

                                predictClass = model1.predict(X)
                                # this deletes vals betweem 0 and 1 so it is scrubbed 
                                # not sure why it was here
                                #predictClass[X[:,0]==0]=0                   
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
    model: string
            a path to a scikit learn model that has been saved 
        
    inputImage : string
                 path to image including the file fmt 'Myimage.tif'
    
    bands: int
            the no of image bands eg 8
    
    outMap: string
             path to output image excluding the file format 'pathto/mymap'
    
    classes: int
              no of classes
    
    blocksize: int (optional) 
                size of raster chunck 256 tends to be quickest if you put None it 
                will read size from gdal (this doesn't always pay off!)
               
    FMT: string
          optional parameter - gdal readable fmt eg 'Gtiff'
        
    one_class: int
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

    
def classify_object(model, inShape, attributes, field_name=None, write='gpd'):
    
    """
    Classify a polygon/point file attributes ('object based') using an sklearn
    model
    
    Parameters
    ------------------
    model: string
            path to input model
    
    inShape: string
              input shapefile path (must be .shp for now....)
    
    attributes: list of stings
                 list of attributes names
    
    field_name: string
                 name of classified label field (optional)
    
    write: string
                either gpd(geopandas) or ogr
    """
    
#    old method for ref - limited to .shp
#    dbf=simpledbf.Dbf5(inShape[:-4]+'.dbf')  
#    csv = inShape[:-4]+'.csv'
#    dbf.to_csv(inShape[:-4]+'.csv')
#    df = dbf.to_dataframe()

    # it seems rather ugly/inefficient to read in every row via ogr
    model1 = joblib.load(model)
    
    df = gpd.read_file(inShape)
    
    X = df[attributes].to_numpy()
    
    if write == 'ogr':
        del df

    X[np.where(np.isnan(X))]=0
    X = X[np.isfinite(X).all(axis=1)]

    #Now the classification itself - see sklearn for details on params

    predictClass = model1.predict(X)

    # clear redundant variables from memory

    del X
    
    predictClass = predictClass.transpose() 
    
    if write == 'ogr':
        shp = ogr.Open(inShape, 1)
        lyr = shp.GetLayer()
        fldDef = ogr.FieldDefn(field_name, ogr.OFTInteger)
        lyr.CreateField(fldDef)
        
        labels = np.arange(lyr.GetFeatureCount())
        
        for label in tqdm(labels):
            val=predictClass[label]
            feat = lyr.GetFeature(label)
            feat.SetField(field_name, int(val))
            lyr.SetFeature(feat)
    
        lyr.SyncToDisk()
        shp.FlushCache()
        shp = None
        lyr = None
    else:
        df[field_name] = predictClass
        df.to_file(inShape)
        
def get_polars(inShp, polars=["VV", "VH"]):
    
    """
    Get list of fields containing polarisations from a polygon/point file
    
    Parameters
    ----------
    
    inShp: string
          the input polygon
          
    polars: list of strings
            the attributes headed with polarisations eg 'VV'
    
    """
    
    shp = ogr.Open(inShp)
    lyr = shp.GetLayer()
    lyrdefn = lyr.GetLayerDefn()
    
    ootlist = []
    
    for f in range(lyrdefn.GetFieldCount()):
        defn = lyrdefn.GetFieldDefn(f)
        ootlist.append(defn.name)
    
    final = [o for o in ootlist if "VV" in o or "VH" in o]
    
    return final
    
def get_training_shp(inShape, label_field, feat_fields,  outFile = None):
    """
    Collect training from a shapefile attribute table. Used for object-based 
    classification (typically). 
    
    Parameters
    --------------------    
    
    inShape: string
              the input polygon
    
    label_field: string
                  the field name for the class labels
                  
    feat_fields: list
                  the field names of the feature data                

    outFile: string (optional)
              path to training data to be saved (.gz)
    
    Returns
    ---------------------
    training data as a dataframe, first column is labels, rest are features
    list of reject features
    
    """
    # TODO Could be done in a less verbose way by gpd
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
    outArray = df.to_numpy()
    
    if outFile != None:
        jb.dump(outArray, outFile, compress=2)
    
    return df, rejects
             

    
    
def get_training(inShape, inRas, bands, field, outFile = None):
    """
    Collect training as an np array for use with create model function
    
    Parameters
    --------------
        
    inShape: string
              the input shapefile - must be esri .shp at present
        
    inRas: string
            the input raster from which the training is extracted
        
    bands: int
            no of bands
        
    field: string
            the attribute field containing the training labels
    
    outFile: string (optional)
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

        # Rasterize 
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


def rmse_vector_lyr(inShape, attributes):

    """ 
    Using sklearn get the rmse of 2 vector attributes 
    (the actual and predicted of course in the order ['actual', 'pred'])
    
    
    Parameters 
    ----------- 
    
    inShape: string
              the input vector of OGR type
        
    attributes: list
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




