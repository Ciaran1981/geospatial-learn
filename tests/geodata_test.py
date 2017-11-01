import pytest
import os
import gdal
import numpy as np
from context import geospatial_learn
from geospatial_learn.geodata import array2raster


def test_array2raster(geotiff_dir):
    array = np.zeros([3, 3, 3])
    bands = 3
    inRaster = os.path.join(geotiff_dir, "tempTiff.tif")
    outRaster = os.path.join(geotiff_dir, "fromArray")
    dtype = gdal.GDT_Int32
    array2raster(array, bands, inRaster, outRaster, dtype)
    result = gdal.Open(outRaster + ".tif").ReadAsArray()
    assert result[0, 1, 2] == 0