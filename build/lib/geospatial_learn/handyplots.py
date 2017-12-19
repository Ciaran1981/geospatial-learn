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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from geospatial_learn import data





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
    
    conf = confusion_matrix(trueVals, predVals)
                      
    
    ax = plt.gca()
    # to be used for confusion matrix

    ax.invert_yaxis()
    #ax.set_anchor('NW')
    #ax.invert_yaxis()
    # plot the mean cross-validation scores
    img = ax.pcolor(conf, cmap=cmap, vmin=None, vmax=None)
    img.update_scalarmappable()
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_xticks(np.arange(len(labels)) + .5)
    ax.set_yticks(np.arange(len(labels)) + .5)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(), img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    
    return
    conf

def plot_classif_report(trueVals, predVals,
                               title='Classification report ', cmap='RdBu'):
    
    
    '''
    Plot scikit-learn classification report.
    
    Parameters
    -------------------
    
    trueVals : nparray
        array of reference/training/validation values
    
    predVals : nparray
        the predicted values
    
    title : string (optional)
        plot title 
    
    cmap : string 
        matplotlib colormap string e.g. 'RdBu'
        
    Notes:
    ----------------------
    Extension based on https://stackoverflow.com/a/31689645/395857 
    
    
    
    
    '''
    def _show_values(pc, fmt="%.2f", **kw):
        '''
        Heatmap with text in each cell with matplotlib's pyplot
        Source: https://stackoverflow.com/a/25074150/395857 
        By HYRY
        '''
        #from itertools import izip
        pc.update_scalarmappable()
        ax = pc.get_axes()
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(),
                                   pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color,
                    **kw)

    
    def _cm2inch(*tupl):
        '''
        Specify figure size in centimeter in matplotlib
        Source: https://stackoverflow.com/a/22787457/395857
        By gns-ank
        '''
        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)


    def _heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels,
                figure_width=40, figure_height=20, correct_orientation=False,
                cmap='RdBu'):
        '''
        Inspired by:
        - https://stackoverflow.com/a/16124677/395857 
        - https://stackoverflow.com/a/25074150/395857
        '''
    
        # Plot it out
        fig, ax = plt.subplots()    
        #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
        c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2,
                      cmap=cmap)
    
        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    
        # set tick labels
        #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)
    
        # set title and x/y labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)      
    
        # Remove last blank column
        plt.xlim( (0, AUC.shape[1]) )
    
        # Turn off all the ticks
        ax = plt.gca()    
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
    
        # Add color bar
        plt.colorbar(c)
    
        # Add text in each cell 
        _show_values(c)
    
        # Proper orientation (origin at the top left instead of bottom left)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()       
    
        # resize 
        fig = plt.gcf()
        #fig.set_size_inches(cm2inch(40, 20))
        #fig.set_size_inches(cm2inch(40*4, 20*4))
        fig.set_size_inches(_cm2inch(figure_width, figure_height))
    
    rep = classification_report(trueVals, predVals)
    
    lines = rep.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx],
                   sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    _heatmap(np.array(plotMat), title, xlabel, ylabel,
            xticklabels, yticklabels, figure_width, figure_height,
            correct_orientation, cmap=cmap)
    
    
    #plt.savefig('test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
    #plt.close()
    
         
#    
#def polarplot(Data, *args, **kwargs): 
#    Data = np.genfromtxt(Data, delimiter=',')
#    ax = plt.subplot(111, polar=True)
#    c = plt.scatter(Data[:,0], Data[:,1], *args, **kwargs)
#    plt.show()
#    return c
    

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
    
    
def plot_S2_cloud(user, paswd, start_date, end_date, aoiJson, print_date=True):
    """
    
    Plot Sentinel 2 cloud percentage for a given time period 
    
    Parameters
    ------------------
    
    data : np array
        the aformentioned array of features
    
    features : list 
        a list of feature indices, eg [1,2,3] or [4,3,1] 
    
    feature_names : list of strings
        a list of feature names ef ['red', 'green', 'blue']
        
    """
    dF, products = data.sent2_query(user, paswd, aoiJson, start_date, end_date)
    
    dF.sort_values('ingestiondate', inplace=True)

    dF.plot.line('ingestiondate', 'cloudcoverpercentage')

def plot_change(inArray):
    """ This assumes that features are column wise and rows are samples
    This will kill computer with too much data"""
    
    shape = inArray.shape
    
    for i in range(0,shape[0]):
        label = inArray[i, 0] 
        plt.plot(inArray[i,1:10], color=plt.cm.RdYlBu(label))










    
    