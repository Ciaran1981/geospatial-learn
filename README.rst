.. -*- mode: rst -*-

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg

geospatial-learn
============

Geospatial-learn is a Python module for raster and vector manipulation and using scikit-learn and xgb models with  the aformentioned geo-spatial data. The aim is to produce convenient, minimal commands for putting together geo-spatial processing chains and using machine learning (ML) libs. Development will aim to expand the variety of libs/algorithms available for machine learning beyond the current complement. The name is a play-on of scikit-learn, though I really ought to think of something better. The functions are mainly a result of my own research area of remote sensing and image processing but cover some fairly typical processing tasks.

- There is a 'raster' module for (you guessed it) raster/image processing. 

- There is a 'shape' module for vector processing which is mainly based around extracting image properties and writing them to a vector format. Functions include zonal stats, glcm-based texture etc as well further manipulation of lines and polygons using things like active contours.

- The 'learning' module is for applying creating ML models and applying them to raster and vector data. This is all based around Sklearn, xgboost and t-pot.    


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

Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~
available soon

Project History
---------------

Geospatial-learn is written and maintained by Ciaran Robb. The functionality was written as part of various research projects involving Earth observation & geo-spatial data. 


Citation
~~~~~~~~

If you use geospatial-learn in a scientific publication, citations would be appreciated 
