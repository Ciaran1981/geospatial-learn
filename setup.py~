# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from io import open

descript = "geospatial-learn is a Python module for using scikit-learn and xgb models with geo-spatial data, chiefly raster and vector formats."

with open('README.rst', encoding='utf-8') as f:
long_description = f.read()

setup(
    name="geospatial-learn",
    version="0.1",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['docutils>=0.3'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
    },
    classifiers=[
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: GIS',
	  'Topic :: Utilities',
    # metadata for upload to PyPI
    author="Ciaran Robb",
    description=descript,
    long_description=long_description,
    license='GPLv3+',
    url="https://github.com/Ciaran1981/geospatial-learn",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
