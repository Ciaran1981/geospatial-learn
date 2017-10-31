import numpy as np
import os
import gdal
from .context import geospatial_learn
from geospatial_learn import geodata as gd



def test_array2raster(geotiff):
    testArray = np.zeros([3, 3, 3])
    gd.array2raster(testArray,
                    os.path.join(geotiff, 'tempTiff.tif'),
                    os.path.join(geotiff, 'outTiff.tif'),
                    gdal.gdalconst.GDT_CFloat32)
    assert gdal.Open(os.path.join(geotiff, 'outTiff.tif'))




