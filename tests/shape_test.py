import pytest
from geospatial_learn import shape
import numpy as np
import gdal
import tempdir
import pyshp

@pytest.fixture
def shape_test_dir():
    """
    Returns a temporary directory containing a raster and a shapefile

    """
    shape_dir

def test_zonal_stats()