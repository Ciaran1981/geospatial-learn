import pytest
from geospatial_learn import shape
import numpy as np
import gdal
import tempfile
from osgeo import ogr, osr
import os


def test_zonal_stats(managed_geotiff_shapefile_dir):
    result = shape.zonal_stats(
        vector_path=managed_geotiff_shapefile_dir.vector_paths[0],
        raster_path=managed_geotiff_shapefile_dir.image_paths[0],
        band=1,
        bandname=1)
    assert result
