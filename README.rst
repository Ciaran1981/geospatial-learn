.. -*- mode: rst -*-

.. |Python38| image:: https://img.shields.io/badge/python-3.8-blue.svg

.. image:: https://zenodo.org/badge/101751668.svg
   :target: https://zenodo.org/badge/latestdoi/101751668

geospatial-learn
============

Geospatial-learn is a Python lib for using scikit-learn, xgb and keras models with geo-spatial data. Some raster and vector manipulation is also included. The aim is to produce convenient, relatively minimal commands for putting together geo-spatial processing chains and using machine learning (ML) libs. The name is a play-on of scikit-learn, though I really ought to think of something better. The functions are mainly a collection resulting from my own research area of remote sensing and image processing; hence some of it may be esoteric but there are some fairly typical processing tasks too. 

- There is a 'raster' module for (you guessed it) raster/image processing. This is not exaustive set of functions of course, just things that have been convenient and repeated such as I/O, masking, some filtering. 

- There is a 'shape' module for vector processing which is mainly based around extracting image properties and writing them to a vector format. Functions include zonal stats, glcm-based texture etc as well as further manipulation of lines and polygons using things like active contours.

- The 'learning' module is for applying creating ML models and applying them to raster, vector and point cloud data. This is all based around sklearn, xgboost, keras and t-pot. 

- The 'handyplots' module contains a few simple functions that may be useful e.g. plot a classifcation report, confusion matrix etc.    


- The 'utils' module is full of stuff which is yet to be given a home that makes sense and may not all be doc'd - take a look...


Dependencies
~~~~~~~~~~~~

geospatial-learn requires:

- Python 3

- Anaconda 

User installation
~~~~~~~~~~~~~~~~~

1. Installation uses the anaconda/miniconda system - please install this first if you don't have it already

2. Clone the repository or download and unzip the tar

3. cd into the folder and type the following

.. code-block:: bash
   
conda env create -f geolearn_env.yml

Alternatively, for a shorter wait (conda is quite slow these days), the mamba system is recommended, install this in your base conda then:

.. code-block:: bash

mamba env create -f geolearn_env.yml


4. To activate - type

.. code-block:: bash

conda activate geospatial_learn

Quickstart
----------

A summary of some functions can be found here:

https://github.com/Ciaran1981/geospatial-learn/blob/master/docs/quickstart.rst

There are some jupyter-based workflows to be found here:

https://github.com/Ciaran1981/geospatial-learn/tree/master/example_notebooks

These are currently a work in progress of course! 

Docs
----

Documentation can be found here:

https://ciaran1981.github.io/geospatial-learn/docs/html/index.html 

These are a work in progress!


Development
-----------

Any contributors of all experience levels are welcome


Project History
---------------

The functionality was written as part of various research projects involving Earth observation & geo-spatial data. 


Citation
~~~~~~~~

If you use geospatial-learn in a scientific publication, citations would be appreciated - click on the blue zenodo link at the top. 

Robb, C. (2017). Ciaran1981/geospatial-learn: Geospatial-learn 0.3 release. Zenodo. https://doi.org/10.5281/ZENODO.3968431

A .bib file is available in the repo (geolearn.bib)
