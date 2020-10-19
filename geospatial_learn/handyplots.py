# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:21:38 2015
@author: Ciaran Robb
author: Ciaran Robb
Research Associate in Earth Observation
Centre for Landscape and Climate Research (CLCR)
Department of Geography, University of Leicester, University Road, Leicester, 
LE1 7RH, UK 

If you use code to publish work cite/acknowledge me and authors of libs as 
appropriate 



"""

import numpy as np



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,jaccard_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as prf
import scikitplot as skplt

import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
def plot_classif_report(trueVals, predVals, labels=None, target_names=None,
                        colmap=plt.cm.Spectral_r, save=None):
    
    """
    Plot a classification report
    
     Parameters
    -------------------
    
    trueVals : nparray
        array of reference/training/validation values
    
    predVals : nparray
        the predicted values
    
    cmap : matplot lib object (optional)
        eg plt.cm.gray
        
    Returns:
        
    The confusion matrix and a plot
    """
    
    clf_report = classification_report(trueVals, predVals, labels=labels,
                                       target_names=target_names, output_dict=True)

    dF = pd.DataFrame(clf_report).iloc[:-1, :].T
    
    cbVl = dF.values
    mn = np.round(cbVl.min(), decimals=2)
    mx= np.round(cbVl.max(), decimals=2)
    del cbVl
    
    
    
    f, ax = plt.subplots(figsize=(10, 10))
    
    splot = sns.heatmap(dF, annot=True, linewidths=.5, fmt='.2f', cmap=colmap,
                        ax=ax, vmin=mn, 
                        vmax=mx, annot_kws={"size": 20})
    
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([mn, mx])
    cbar.set_ticklabels([str(mn), str(mx)])
    
    

    
    if save != None:
    
        fig = splot.get_figure()
        fig.savefig(save) 
    
    return dF


def _plt_heatmap(values, xlabel, ylabel, xticklabels, yticklabels, 
                cmap=plt.cm.gray_r,vmin=None, vmax=None, ax=None, fmt="%d"):
    
    """
    Plot a heamap for something like a confusion matrix
    
  
    """
    
    
    if ax is None:
        ax = plt.gca()
    # to be used for confusion matrix
    #ax.set_anchor('NW')
    #ax.invert_yaxis()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=None, vmax=None)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(), img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img

def plt_confmat(trueVals, predVals, cmap = plt.cm.gray, fmt="%d"):
    
    
    """
    Plot a confusion matrix
    
    Parameters
    -------------------
    
    trueVals : nparray
        array of reference/training/validation values
    
    predVals : nparray
        the predicted values
    
    cmap : matplot lib object (optional)
        eg plt.cm.gray
        
    Returns:
        
    The confusion matrix and a plot
    """
    labels = np.unique(trueVals)
    # the above heatmap function is used to create the plot
    
    skplt.metrics.plot_confusion_matrix(trueVals, predVals, normalize=True)
    
    conf = confusion_matrix(trueVals, predVals)
                      
    
#    ax = plt.gca()
#    # to be used for confusion matrix
#
#    ax.invert_yaxis()
#    #ax.set_anchor('NW')
#    #ax.invert_yaxis()
#    # plot the mean cross-validation scores
#    img = ax.pcolor(conf, cmap=cmap, vmin=None, vmax=None)
#    img.update_scalarmappable()
#    ax.set_xlabel('True')
#    ax.set_ylabel('Predicted')
#    ax.set_xticks(np.arange(len(labels)) + .5)
#    ax.set_yticks(np.arange(len(labels)) + .5)
#    ax.set_xticklabels(labels)
#    ax.set_yticklabels(labels)
#
#    ax.set_aspect(1)
#
#    for p, color, value in zip(img.get_paths(), img.get_facecolors(), img.get_array()):
#        x, y = p.vertices[:-2, :].mean(0)
#        if np.mean(color[:3]) > 0.5:
#            c = 'k'
#        else:
#            c = 'w'
#        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    
    return
    conf


    

def plot2d(data, features, feature_names, point_color = 0):
    
    """ plot 3d feature space (for example the bands of an image or the fields
    of a shapefile/database)
    This assumes the features are columns in an np array as would be fed to scikit 
    learn
    
    Parameters
    ------------------
    
    data : np array
        the aformentioned array of features
    
    features : list 
        a list of feature indices, eg [1,2,3] or [4,3,1] 
    
    feature_names : list of strings
        a list of feature names ef ['red', 'green', 'blue']
    
    Notes:
    ----------------
    the colors available are:
   
    ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'] 
    
    So if there are more than 8 labels it won't work
    """
    fig = plt.figure()
    # 111 means 1x1 grid first subplot (kinda like matlab)
    ax = fig.add_subplot(111)
    
    labels = np.unique(data[:,point_color])
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'] 
    
    for label in labels:
        cls = data[data[:, 0]==label]
        ax.scatter(cls[:,features[0]], cls[:,features[1]],
                   c=colors[int(label-1)], alpha=0.3, label = int(label))
    
    ax.set_xlabel(str(feature_names[0]))
    ax.set_ylabel(str(feature_names[1]))
    ax.legend(title = 'Classes', loc=0,  scatterpoints=1)
    
    plt.show()    

def plot3d(data, features, feature_names, point_color = 0):
    
    """ plot 3d feature space (for example the bands of an image or the fields
    of a shapefile/database)
    This assumes the features are columns in an np array as would be fed to scikit 
    learn

    Parameters
    ------------------
    
    data : np array
        the aformentioned array of features
    
    features : list 
        a list of feature indices, eg [1,2,3] or [4,3,1] 
    
    feature_names : list of strings
        a list of feature names ef ['red', 'green', 'blue']
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    labels = np.unique(data[:,point_color])
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'] 
    
    for label in labels:
        cls = data[data[:, 0]==label]
        ax.scatter(cls[:,features[0]], cls[:,features[1]], cls[:,features[2]],
                   c=colors[int(label-1)], alpha=0.3, label = int(label))

    
    ax.set_xlabel(str(feature_names[0]))
    ax.set_ylabel(str(feature_names[1]))
    ax.set_zlabel(str(feature_names[2]))
    

    ax.legend(title = 'Classes', loc=0,  scatterpoints=1)

    plt.show()   
    
    

def plot_change(inArray):
    """ This assumes that features are column wise and rows are samples
    This will kill computer with too much data"""
    
    shape = inArray.shape
    
    for i in range(0,shape[0]):
        label = inArray[i, 0] 
        plt.plot(inArray[i,1:10], color=plt.cm.RdYlBu(label))










    
    
