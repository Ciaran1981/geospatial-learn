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

At present the setup.py only installs some of the dependencies. An anaconda package is in the works, but until that is done please do the following. This assumes you have an anaconda installation with a python 3  root OR env.


Step 1.

- download the zip from here (see clone or download)

- cd into the folder

- open a terminal and type:

.. code-block:: bash
    
   python setup.py install

This will install the library and packages unavailable on anaconda.

Step 2.

Conda is very handy at managing packages, hence this stage, as some of these are external to python or themselves have multiple depends.

Next, type the following (in the same terminal).

.. code-block:: bash
   
   install_conda_packages.sh

All the appropriate anaconda packages will then install
   

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

It is currently maintained voluntaliry by Dr Ciaran Robb, University of Leicester. The functionality was originally written as part of various research projects involving Earth observation & geo-spatial data. The module is at a very early stage at present and there is more material wrtten that has yet to be added (including docs!).     

Help and Support
----------------

available soon

Citation
~~~~~~~~

If you use geospatial-learn in a scientific publication, citations would be appreciated 
