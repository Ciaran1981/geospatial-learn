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
import geopandas as gpd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,jaccard_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import plot_confusion_matrix
import scikitplot as skplt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pprint
from scipy import stats
from IPython.display import display

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

def plt_confmat(X_test, y_test, model, class_names=None, cmap=plt.cm.Blues, 
                fmt="%d", save=None):
    
    
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
    #labels = np.unique(trueVals)
    # the above heatmap function is used to create the plot
        
    
    #skplt.metrics.plot_confusion_matrix(trueVals, predVals, normalize=True)
    
    
    
    #conf = confusion_matrix(trueVals, predVals)
    
#    titles_options = [("Confusion matrix, without normalization", None),
#                  ("Normalized confusion matrix", 'true')]
    
    confs = []
    disps = []
    #first is gonna be several digits, second a decimal of max 2
    
    font_sizes = [{'font.size': 5}, {'font.size': 5}] 
    # one is a count the other a decimal
    vformats = ['.0f', '.2f']
    
    titles = ["Confusion matrix, without normalization", 
               "Normalized confusion matrix"]
    normalise = [None, 'true']
    
    #fig, ax = plt.subplots(figsize=(10, 10))
    
    for t, n,f,v in zip(titles, normalise, font_sizes, vformats):
        disp = plot_confusion_matrix(model, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=cmap,
                                     normalize=n,
                                     xticks_rotation='vertical',
                                     values_format=v)
        plt.rcParams.update(f)
        disp.ax_.set_title(t)
        disps.append(disp)
        confs.append(disp.confusion_matrix)
    
    plt.show()
    
    types = ['cnts','nrm']
    for t, d in zip(types, disps):
        # must set a size
#        d.figure_(figsize=(18, 9))
        d.figure_.savefig(save[:-3]+t+'.png', bbox_inches='tight')
    
    return confs


    

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



def plot_group(df, group, index, name,  year=None, title=None, fill=False,
               min_max=False, 
               freq='M', plotstat='mean'):
    
    """
    Plot time series per poly or point eg for S2 ndvi, met var, S1
    If the the no of entries > 20, then the average/median with min/max
    shading will be plotted
    
    Parameters
    ----------
    
    df: byte
        input pandas/geopandas df
    
    group: string
          the attribute to group by
          
    index: list
            the index of interest
            
    name: string
            the name of interest
            
    legend_col: string
            the column that legend lines will be labeled by
    
    year: string
            the year to summarise e.g. '16' for 2016 (optional)
            
    title: string
            plot title

    fill: bool
            fill the area between plot lines 
            (this a bit imprecise in some areas)
    
    plotstat: string
            the line to plot when the dataset is too big (> 20 entries)
            either 'mean' or 'median'
    
    """

    #TODO potentially drop this as could be done outside func
    sqr = df.loc[df[group].isin(index)]
    
    yrcols = [y for y in sqr.columns if name in y]
    yrcols.sort()
    if year != None:
        if len(year) > 2:
            raise ValueError('year must be last 2 digits eg 16 for 2016') 
        yrcols = [y for y in yrcols if year in y]
        dtrange = pd.date_range(start='20'+year+'-01-01',
                                end='20'+year+'-12-31',
                                freq=freq)
    else:
        
        # set the dtrange......
        # this code is dire  due to way i wrote other stuff from gee plus
        # ad hoc changes as I go
        #TODO alter this crap
        if freq == 'M':
            startd = yrcols[0][-5:]
            startd = '20'+startd+'-01'
            # this seems stupid
            endd = yrcols[-1:][0][-5:]
            endd = '20'+endd[0:5]+'-31'
            dtrange = pd.date_range(start=startd, end=endd, freq=freq)
            
#        else:
#            startd = yrcols[0][-8:]
#            startd = '20'+startd
#            endd = yrcols[-1:][0][-5:]
#            endd = '20'+endd

            
    # TODO - this is crap really needs replaced....
    ndplotvals = sqr[yrcols]
    
    new = ndplotvals.transpose()
    
    if freq != 'M':
        new['Date'] = new.index
        new['Date'] = new['Date'].str.replace(name+'-','20')
        new['Date'] = new['Date'].str.replace('_','0')
        new['Date'] = pd.to_datetime(new['Date'])
    else:
        new['Date'] = dtrange
    
    
    new = new.set_index('Date')
    # what is this doing....I have forgotten
    #new.columns=[name]
    # to add the index as the legend
    if plotstat == 'mean':
        mn = new.mean(axis=1)
    elif plotstat == 'median':
        mn = new.median(axis=1)
    
    if len(new.columns) < 20:
        ax = new.plot.line(title=title)
    else:
        ax = mn.plot.line(title=title, label=plotstat)
    
    if fill == True:
         minr = new.min(axis=1, numeric_only=True)
         maxr = new.max(axis=1, numeric_only=True)
         stdr = new.std(axis=1, numeric_only=True)
         if min_max == True:
             ax.fill_between(new.index, minr, maxr, alpha=0.3, label='Min/Max')
         # plot std around each point (think this is right.....)
         ax.fill_between(new.index, (mn-2*stdr), 
                         (mn+2*stdr), color='r', alpha=0.1, label='Stdev')
#    if year is None:
#        # does not appear to work....
#        # TODO this could be adapted to highlight the phenostages with their
#        # labels
#        xpos = [pd.to_datetime(startd), pd.to_datetime(endd)]
#        for xc in xpos:
#            ax.axvline(x=xc, color='k', linestyle='-')
            
    

    if len(new.columns) < 20:
        ax.legend(labels=sqr[group].to_list())
    else:
        ax.legend()
    plt.show()

def extract_by_date(df, start, end, freq='M', band='VV'):
    
    """
    Extract a date range from multi year pandas df
    aimed at getting dates relevant to certain crops
    
    Paramaters
    ----------
    
    df: byte
        (geo)pandas dataframe with date like columns derived from
        time series function
    
    start: string
            start date in yyyy-mm-dd format
    
    end: string
            end date in yyyy-mm-dd format
            
    freq: string
            pandas date frequency short hand eg 'M' = month, 'D' = day
    
    band: string
            the band (will be the first to letters of each date column)
            e.g. 'VV' for S1 vertical emit receipt. 
        
    
    """
    
    # TODO would it be simpler to convert the columns to date range, select
    # the range, then convert back to list of strings, THEN use to select 
    # columns from df
    # This would save the transposes etc
    
    yrcols = [y for y in df.columns if band in y]
    yrcols.sort()
    
   
    if freq == 'M':
        startd = yrcols[0][-5:]
        startd = '20'+startd+'-01'
        # this seems stupid
        endd = yrcols[-1:][0][-5:]
        endd = '20'+endd[0:2]+'-12-31'
        dtrange = pd.date_range(start=startd, end=endd, freq=freq)

    ndplotvals = df[yrcols]
    
    # reckon the pandas.pivot function may be more elegant
    # than what I have done
    
    new = ndplotvals.transpose()
    
    if freq != 'M':
        new['Date'] = new.index
        new['Date'] = new['Date'].str.replace(band+'-','20')
        new['Date'] = new['Date'].str.replace('_','0')
        new['Date'] = pd.to_datetime(new['Date'])
    else:
        new['Date'] = dtrange
    
    
    new = new.set_index('Date')
    
    new = new.loc[start : end]
    
    # reverse the procedure, to be improved this is ugly
    new = new.transpose()
    
    
    new.columns = new.columns.astype(str)
    
    chlist = new.columns.tolist()

    newcols = [band + '-' + c[2:7] for c in chlist]
    
    new.columns = newcols
    
    return new
    
    

def plot_crop(df, group, index, name, crop="SP BA", year=None, title=None,
              fill=False, freq='M'):
    
    """
    Plot time series per poly or point eg for S2 ndvi, S1, or met var
    
    Parameters
    ----------
    
    df: byte
        input pandas/geopandas df
    
    group: string
          the attribute to group by
          
    index: list
            the index of interest
            
    name: string
            the name of interest
            
    crop: string
            the crop of interest
            
    legend_col: string
            the column that legend lines will be labeled by
    
    year: string
            the year to summarise e.g. '16' for 2016 (optional)
            
    title: string
            plot title

    fill: bool
            fill the area between plot lines 
            (this a bit imprecise in some areas)
    
    """
    
    # Quick dirty time series plotting of crops with corresponding stages
    

    sqr = df.loc[df[group].isin(index)]
    
    #fcover
    yrcols = [y for y in sqr.columns if 'F-' in y]
    # soil
    soilcols = [y for y in sqr.columns if 'G-' in y]
    
    if year != None:
        if len(year) > 2:
            raise ValueError('year must be last 2 digits eg 16 for 2016') 
        yrcols = [y for y in yrcols if year in y]
        dtrange = pd.date_range(start='20'+year+'-01-01',
                                end='20'+year+'-12-31',
                                freq='M')
    else:
        # set the dtrange......
        # this code is dire but due to way i wrote other stuff from gee
        
        if freq == 'M':
            startd = yrcols[0][-5:]
            startd = '20'+startd+'-01'
            # this seems stupid
            endd = yrcols[-1:][0][-8:]
            endd = '20'+endd[0:2]+'-12-31'
            dtrange = pd.date_range(start=startd, end=endd, freq=freq)

            
    # TODO - this is crap really needs replaced....
    ndplotvals = sqr[yrcols]
    soilplotvals = sqr[soilcols]
    
    slnew = soilplotvals.transpose()
    new = ndplotvals.transpose()
    
    if freq != 'M':
        new['Date'] = new.index
        new['Date'] = new['Date'].str.replace(name+'-','20')
        new['Date'] = new['Date'].str.replace('_','0')
        new['Date'] = pd.to_datetime(new['Date'])
    else:
        new['Date'] = dtrange
    
    new = new.set_index('Date')
    # forgot what this is for useless
    #new.columns=[name]
    
    # it is not scaled to put lines between months....
    # but the line doesn't plot so sod this
#    days = pd.date_range(start='20'+year+'-01-01',
#                                end='20'+year+'-12-31',
#                                freq='D')
#    
#    new = new.reindex(days)
#   and before you try it - dropna just returns the axis to monthly.
    
#    # to add the index as the legend
    ax = new.plot.line(title=title)
    

    

    if fill == True:
         minr = new.min(axis=1, numeric_only=True)
         maxr = new.max(axis=1, numeric_only=True)
         ax.fill_between(new.index, minr, maxr, alpha=0.1, label='min')
    
    if year is None:
        # TODO this could be adapted to highlight the phenostages with their
        # labels
        xpos = [pd.to_datetime(startd), pd.to_datetime(endd)]
        for xc in xpos:
            ax.axvline(x=xc, color='k', linestyle='--', label='pish')
        ax.axvspan(xpos[0], xpos[1], facecolor='gray', alpha=0.2)

    ax.legend(labels=sqr[group].to_list(), loc='center left', 
              bbox_to_anchor=(1.0, 0.5))
   
    
    yr = "20"+year
    # The crop dicts -approx patterns of crop growth....
    # this is a hacky mess for now
    # Problem is axis is monthly from previous
    # so changing to days or weeks doesn't work, including half months
    crop_dict = {"SP BA": [pd.to_datetime(yr+"-04-01"), pd.to_datetime(yr+"-05-01"),
                           pd.to_datetime(yr+"-06-01"), pd.to_datetime(yr+"-08-01"),
                           pd.to_datetime(yr+"-09-01")],
                 "SP BE": [],
                 "SP Oats": [],
                 "SP WH": [pd.to_datetime(yr+"-04-01"), pd.to_datetime(yr+"-05-01"),
                           pd.to_datetime(yr+"-06-01"), pd.to_datetime(yr+"-08-01"),
                           pd.to_datetime(yr+"-09-01")],
                 "W Rye": [],
                 "W Spelt": [],
                 "W WH": []}
    
    # these are placeholders and NOT correct!!!
    # mind the list much be same length otherwise zip will stop at end of shortest
    # also the last entry is of course not between lines
    crop_seq = {"SP BA": [' Em', ' Stem' ,' Infl',
                           ' Flr>\n Gr>\n Sen',
                           ""],
                 "SP BE": [],
                 "SP Oats": [],
                 "SP WH": [' Em', ' Stem' ,' Infl',
                           ' Flr>\n Gr> \nSen',
                           ""],
                 "W Rye": [],
                 "W Spelt": [],
                 "W WH": []}
    
    # rainbow
    clrs = cm.rainbow(np.linspace(0, 1, len(crop_dict[crop])))
    
    #for the text 
    style = dict(size=10, color='black')
    
    
    #useful stuff here
    #https://jakevdp.github.io/PythonDataScienceHandbook/04.09-text-and-annotation.html
    
    # this is also a mess
    # Gen the min of mins for the position of text on y axis
    minr = new.min(axis=1, numeric_only=True)
    btm = minr.min() - (minr.min() / 2)
    for xc, nm in zip(crop_dict[crop], crop_seq[crop]):
        ax.axvline(x=xc, color='k', linestyle='--')
        ax.text(xc, minr.min(), " "+nm, ha='left', **style)
        # can't get this to work as yet
#        ax.annotate(nm, xy=(xc, 0.1),
#                    xycoords='data', bbox=dict(boxstyle="round", fc="none", ec="gray"),
#                    xytext=(10, -40), textcoords='offset points', ha='center',
#                    arrowprops=dict(arrowstyle="->"), 
#                    annotation_clip=False)
   # theend = len(crop_dict[crop])-1
    
    for idx, c in enumerate(clrs):
        if idx == len(clrs)-1:
            break
        ax.axvspan(crop_dict[crop][idx], crop_dict[crop][idx+1], 
                   facecolor=c, alpha=0.1)
        
    
    
    # so if we take the mid date point we can put the label in the middle
 
    # hmm this does not do what I had hoped, it's not in the middle....
#    midtime = xpos[0] + (xpos[1] - xpos[0])/2
    # can cheat by putting a space at the start!!!!
    #ax.text(midtime, 0.5, " crop", ha='left', **style)
    plt.show()


# __author__ = "Juanma Hernández"
# __copyright__ = "Copyright 2019"
# __credits__ = ["Juanma Hernández", "George Fisher"]
# __license__ = "GPL"
# __maintainer__ = "Juanma Hernández"
# __email__ = "https://twitter.com/juanmah"
# __status__ = "Utility script"


def plot_grid_search(clf):
    """Plot as many graphs as parameters are in the grid search results.

    Each graph has the values of each parameter in the X axis and the Score in the Y axis.

    Parameters
    ----------
    clf: estimator object result of a GridSearchCV
        This object contains all the information of the cross validated results for all the parameters combinations.
    """
    # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
    # As it is frequent to have more than one combination with the same max score,
    # the one with the least mean fit time SHALL appear first.
    cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])

    # Get parameters
    parameters=cv_results['params'][0].keys()

    # Calculate the number of rows and columns necessary
    rows = -(-len(parameters) // 2)
    columns = min(len(parameters), 2)
    # Create the subplot
    fig = make_subplots(rows=rows, cols=columns)
    # Initialize row and column indexes
    row = 1
    column = 1

    # For each of the parameters
    for parameter in parameters:

        # As all the graphs have the same traces, and by default all traces are shown in the legend,
        # the description appears multiple times. Then, only show legend of the first graph.
        if row == 1 and column == 1:
            show_legend = True
        else:
            show_legend = False

        # Mean test score
        mean_test_score = cv_results[cv_results['rank_test_score'] != 1]
        fig.add_trace(go.Scatter(
            name='Mean test score',
            x=mean_test_score['param_' + parameter],
            y=mean_test_score['mean_test_score'],
            mode='markers',
            marker=dict(size=mean_test_score['mean_fit_time'],
                        color='SteelBlue',
                        sizeref=2. * cv_results['mean_fit_time'].max() / (40. ** 2),
                        sizemin=4,
                        sizemode='area'),
            text=mean_test_score['params'].apply(
                lambda x: pprint.pformat(x, width=-1).replace('{', '').replace('}', '').replace('\n', '<br />')),
            showlegend=show_legend),
            row=row,
            col=column)

        # Best estimators
        rank_1 = cv_results[cv_results['rank_test_score'] == 1]
        fig.add_trace(go.Scatter(
            name='Best estimators',
            x=rank_1['param_' + parameter],
            y=rank_1['mean_test_score'],
            mode='markers',
            marker=dict(size=rank_1['mean_fit_time'],
                        color='Crimson',
                        sizeref=2. * cv_results['mean_fit_time'].max() / (40. ** 2),
                        sizemin=4,
                        sizemode='area'),
            text=rank_1['params'].apply(str),
            showlegend=show_legend),
            row=row,
            col=column)

        fig.update_xaxes(title_text=parameter, row=row, col=column)
        fig.update_yaxes(title_text='Score', row=row, col=column)

        # Check the linearity of the series
        # Only for numeric series
        if pd.to_numeric(cv_results['param_' + parameter], errors='coerce').notnull().all():
            x_values = cv_results['param_' + parameter].sort_values().unique().tolist()
            r = stats.linregress(x_values, range(0, len(x_values))).rvalue
            # If not so linear, then represent the data as logarithmic
            if r < 0.86:
                fig.update_xaxes(type='log', row=row, col=column)

        # Increment the row and column indexes
        column += 1
        if column > columns:
            column = 1
            row += 1

            # Show first the best estimators
    fig.update_layout(legend=dict(traceorder='reversed'),
                      width=columns * 360 + 100,
                      height=rows * 360,
                      title='Best score: {:.6f} with {}'.format(cv_results['mean_test_score'].iloc[0],
                                                                str(cv_results['params'].iloc[0]).replace('{',
                                                                                                          '').replace(
                                                                    '}', '')),
                      hovermode='closest',
                      template='none')
    fig.show()

#TODO pinched from online first one doesn't work with my grid
# def table_grid_search(clf, all_columns=False, all_ranks=False, save=True):
#     """Show tables with the grid search results.

#     Parameters
#     ----------
#     clf: estimator object result of a GridSearchCV
#         This object contains all the information of the cross validated results for all the parameters combinations.

#     all_columns: boolean, default: False
#         If true all columns are returned. If false, the following columns are dropped:

#         - params. As each parameter has a column with the value.
#         - std_*. Standard deviations.
#         - split*. Split scores.

#     all_ranks: boolean, default: False
#         If true all ranks are returned. If false, only the rows with rank equal to 1 are returned.

#     save: boolean, default: True
#         If true, results are saved to a CSV file.
#     """
#     # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
#     # As it is frequent to have more than one combination with the same max score,
#     # the one with the least mean fit time SHALL appear first.
#     cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])

#     # Reorder
#     columns = cv_results.columns.tolist()
#     # rank_test_score first, mean_test_score second and std_test_score third
#     columns = columns[-1:] + columns[-3:-1] + columns[:-3]
#     cv_results = cv_results[columns]

#     if save:
#         cv_results.to_csv('--'.join(cv_results['params'][0].keys()) + '.csv', index=True, index_label='Id')

#     # Unless all_columns are True, drop not wanted columns: params, std_* split*
#     if not all_columns:
#         cv_results.drop('params', axis='columns', inplace=True)
#         cv_results.drop(list(cv_results.filter(regex='^std_.*')), axis='columns', inplace=True)
#         cv_results.drop(list(cv_results.filter(regex='^split.*')), axis='columns', inplace=True)

#     # Unless all_ranks are True, filter out those rows which have rank equal to one
#     if not all_ranks:
#         cv_results = cv_results[cv_results['rank_test_score'] == 1]
#         cv_results.drop('rank_test_score', axis = 'columns', inplace = True)        
#         cv_results = cv_results.style.hide_index()

#     display(cv_results)


# def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
#     # Get Test Scores Mean and std for each grid search
#     scores_mean = cv_results['mean_test_score']
#     scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

#     scores_sd = cv_results['std_test_score']
#     scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

#     # Plot Grid search scores
#     _, ax = plt.subplots(1,1)

#     # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
#     for idx, val in enumerate(grid_param_2):
#         ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

#     ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
#     ax.set_xlabel(name_param_1, fontsize=16)
#     ax.set_ylabel('CV Average Score', fontsize=16)
#     ax.legend(loc="best", fontsize=15)
#     ax.grid('on')


    
