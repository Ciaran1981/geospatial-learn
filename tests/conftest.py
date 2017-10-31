import pytest
import tempfile
#import gdal
import os
import numpy as np


@pytest.fixture
def geotiff():
    """

    Returns
    -------
    A pointer to a temporary folder that contains a 3-band geotiff
    of 3x3, with all values being 1.

    """
    tempDir = tempfile.TemporaryDirectory()
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)
    metadata = driver.GetMetadata()
    tempPath = os.path.join(tempDir.name, "tempTiff.tif")
    testDataset = driver.Create(os.path.join(tempDir.name, "tempTiff.tif"),
        xsize=3, ysize=3, bands=3, eType=gdal.GDT_CFloat32)
    for i in range(3):
        testDataset.GetRasterBand(i+1).WriteArray(np.ones([3, 3]))
    testDataset = None
    yield tempPath
    tempDir.cleanup()