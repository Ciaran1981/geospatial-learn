.. -*- mode: rst -*-

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg

geospatial-learn
============

geospatial-learn is a Python module for using scikit-learn and xgb models with geo-spatial data, chiefly raster and vector formats. 

The module also contains various fuctionality for manipulating raster and vector data as well as some utilities aimed at processing Sentinel 2 data.

The aim is to produce convenient, minimal commands for putting together geo-spatial processing chains using machine learning libs. Development will aim to expand the variety of libs/algorithms available for machine learning beyond the current complement.  


Dependencies
~~~~~~~~~~~~

geospatial-learn requires:

- Python 3

User installation
~~~~~~~~~~~~~~~~~

Installation use the anaconda/miniconda system please install this first

If you wish to isolate the lib in its own environment simply create one using the conda cmd line with your own env name in quotes:

E.g.

.. code-block:: bash
   
   conda create -n "pygeolearn"

Linux/Unix - based
~~~~~~~~~~~~~~~~~

Library install

download using the git clone cmd or download as a zip

- cd into the folder

- open a terminal and type:

.. code-block:: bash
    
   python setup.py install

This will install the library and dependencies

Windows - based
~~~~~~~~~~~~~~~~~   

Commiserations, you are using Windows (hehe)

Same procedure as Unix based system

If you run into problems here, such as certain packages unavailable with Python 3.5/6, I suggest creating a conda environment with python 3.4, then following the above procedure. At the time of writing for example (31/08/17), gdal is not available in py3.5+ on windows anaconda and py3.6 on linux platforms.

I have not provided xgboost instructions here, there are some on the native website along with ensuring the lib points to your python environment of choice. 


Quickstart
----------

A summary of some functions can be found here:

https://github.com/Ciaran1981/geospatial-learn/blob/master/docs/quickstart.rst

This is currently a work in progress of course! 

Docs
----

Documentation can be found here:

http://geospatial-learn.readthedocs.io/en/latest/

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

Geospatial-learn was originally written by Dr Ciaran Robb whilst at the University of Leicester. The functionality was written as part of various research projects involving Earth observation & geo-spatial data. 

Geospatial-learn is currently written and maintained by Ciaran Robb and John Roberts. The module is at a very early stage at present and there is more material wrtten that has yet to be added.     

Help and Support
----------------

available soon

Citation
~~~~~~~~

If you use geospatial-learn in a scientific publication, citations would be appreciated 
