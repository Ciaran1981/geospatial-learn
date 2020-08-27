.. -*- mode: rst -*-

.. |Python38| image:: https://img.shields.io/badge/python-3.8-blue.svg

.. image:: https://zenodo.org/badge/101751668.svg
   :target: https://zenodo.org/badge/latestdoi/101751668

geospatial-learn
============

Geospatial-learn is a Python module for raster and vector manipulation and using scikit-learn and xgb models with  the aformentioned geo-spatial data. The aim is to produce convenient, relatively minimal commands for putting together geo-spatial processing chains and using machine learning (ML) libs. The name is a play-on of scikit-learn, though I really ought to think of something better. The functions are mainly a collection resulting from my own research area of remote sensing and image processing;   hence some of it may be esoteric but there are some fairly typical processing tasks too. 

- There is a 'raster' module for (you guessed it) raster/image processing. This is not exaustive set of functions of course, just things that have been convenient and repeated such as I/O, masking, some filtering 

- There is a 'shape' module for vector processing which is mainly based around extracting image properties and writing them to a vector format. Functions include zonal stats, glcm-based texture etc as well further manipulation of lines and polygons using things like active contours.

- The 'learning' module is for applying creating ML models and applying them to raster, vector and point cloud data. This is all based around sklearn, xgboost, keras and t-pot. 

- The 'handyplots' module contains a few simple functions that may be useful e.g. plot a classifcation report, confusion matrix etc.    


- The 'utils' module is full of stuff which is yet to be given a home that makes sense and may not all be doc'd - take a look...


Dependencies
~~~~~~~~~~~~

geospatial-learn requires:

- Python 3

User installation
~~~~~~~~~~~~~~~~~

Installation use the anaconda/miniconda system please install this first

.. code-block:: bash
   
conda env create -f geolearn_env.yml


Quickstart
----------

A summary of some functions can be found here:

https://github.com/Ciaran1981/geospatial-learn/blob/master/docs/quickstart.rst

This is currently a work in progress of course! 

Docs
----

Documentation can be found here:

https://ciaran1981.github.io/geospatial-learn/docs/html/index.html 

These are a work in progress!


Development
-----------

New contributors of all experience levels are welcome

Useful links
~~~~~~~~~~~~~~~
Here are some links to the principal libs used in geospatial-learn.

https://github.com/scikit-learn/

http://xgboost.readthedocs.io/en/latest/

http://scikit-learn.org/stable/

http://www.gdal.org/

http://www.numpy.org/

https://www.scipy.org/

http://scikit-image.org/



Project History
---------------

Geospatial-learn is written and maintained by Ciaran Robb. The functionality was written as part of various research projects involving Earth observation & geo-spatial data. 


Citation
~~~~~~~~

If you use geospatial-learn in a scientific publication, citations would be appreciated - click on the blue zenodo link at the top. 

Robb, C. (2017). Ciaran1981/geospatial-learn: Geospatial-learn 0.3 release. Zenodo. https://doi.org/10.5281/ZENODO.3968431

A .bib file is available in the repo (geolearn.bib)
