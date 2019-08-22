# -*- coding: utf-8 -*-
"""Alternations to setup.py based on Brandon Rhodes' conda setup.py:
https://github.com/brandon-rhodes/conda-install"""
from setuptools import setup
from setuptools.command.install import install
from io import open
import subprocess

descript = ('geospatial-learn is a Python module for using scikit-learn and'
            'xgb models with geo-spatial data, chiefly raster and vector'
            'formats.')

with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()


class CondaInstall(install):
    def run(self):
        try:
            command = ['conda', 'install', '-y']
            packages = open('conda_modules.txt').read().splitlines()
            command.extend(packages)
            subprocess.check_call(command)
            install.do_egg_install(self)
        except subprocess.CalledProcessError:
            print("Conda install failed: do you have Anaconda/miniconda installed and on your PATH?")


setup(
    cmdclass={'install': CondaInstall},
    name="geospatial-learn",
    version="0.3",
    packages=['geospatial_learn'],
    install_requires=open('requirements.txt').read().splitlines(),
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    include_package_data=True,# {
        # If any package contains *.txt or *.rst files, include them:
        # And include any *.msg files found in the 'hello' package, too:
    #},
    classifiers=[
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: GIS',
          'Topic :: Utilities'],
    # metadata for upload to PyPI
    # zip_safe = True,
    author="Ciaran Robb",
    description=descript,
    long_description=long_description,
    license='GPLv3+',
    url="https://github.com/Ciaran1981/geospatial-learn",   # project home page, if any
    download_url="https://github.com/Ciaran1981/geospatial-learn"
    # could also include long_description, download_url, classifiers, etc.
)
