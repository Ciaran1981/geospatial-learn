import pytest
import os
import gdal
from context import geospatial_learn
from geospatial_learn.geodata import array2raster

def array2raster_test(geotiff_dir):
    array = np.zeros([3,3,3])
    bands = 3
    inRaster = os.path.join(geotiff_dir, "tempTiff.tif")
    outRaster = os.path.join(geotiff_dir, "fromArray.tif")
    dtype = gdal.GDT_Int32
    array2raster(array, bands, inRaster, outRaster, dtype)

