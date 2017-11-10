import pytest
from geospatial_learn import shape
import numpy as np
import gdal
import tempfile
from osgeo import ogr, osr
import os


def test_managed_geotiff_dir(managed_geotiff_shapefile_dir):
    target = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1],
                       [1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1],
                       [1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1],
                       [1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1],
                       [1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]], dtype=np.ubyte)
    result = gdal.Open(managed_geotiff_shapefile_dir.image_paths[0]).ReadAsArray()
    print(result)
    assert np.array_equal(result, target)     # It's upside-down; need to learn how geocoords work


def test_zonal_stats(managed_geotiff_shapefile_dir):
    result = shape.zonal_stats(
        vector_path=managed_geotiff_shapefile_dir.vector_paths[0],
        raster_path=managed_geotiff_shapefile_dir.image_paths[0],
        band=1,
        bandname=1)
    assert pytest.approx(result, 0.5, 0.1)
