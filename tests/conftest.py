import pytest
import tempfile
import gdal
import os
import numpy as np


class TestGeotiffManager:
    """
    An object that provides methods to create and manipulate a temporary
    directory that contains a 3-band geotiff, tempTiff.tif

    Attributes
    ----------
    path : str
        The path to the temporary folder
    image : list[str]
        A list of paths to images in the temporary folder



    """

    def __enter__(self):
        """Creates a randomly named temp folder
        """
        tempDir = tempfile.TemporaryDirectory()
        fileformat = "GTiff"
        driver = gdal.GetDriverByName(fileformat)
        tempPath = os.path.join(tempDir.name)
        testDataset = driver.Create(os.path.join(tempDir.name, "tempTiff.tif"),
                                    xsize=3, ysize=3, bands=3, eType=gdal.GDT_CFloat32)
        for i in range(3):
            testDataset.GetRasterBand(i + 1).WriteArray(np.ones([3, 3]))

        self.path = tempPath
        self.imagePaths = [os.path.join(tempPath, "tempTiff.tif")]
        self.images = [testDataset]
        self._tempDir = tempDir

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tempDir.cleanup()

    def create_temp_tiff(self, name, content = None):
        """Creates a temporary geotiff in self.path
        """
        if not content:
            content = np.zeros([3, 3, 3])
        path = os.path.join(self.path, name)
        driver = gdal.GetDriverByName('.tif')
        driver.Create(
            path,
            xsize=content.shape[1],
            ysize=content.shape[2],
            bands=content.shape[0],
            eType=gdal.GDT_CFloat32
        )
        self.image.append(path)


@pytest.fixture
def managed_geotiff_dir():
    """Holds context for the TestGeotiffManager class defined above"""
    with TestGeotiffManager() as tgm:
        yield tgm


@pytest.fixture
def geotiff_dir():
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
    tempPath = os.path.join(tempDir.name)
    testDataset = driver.Create(os.path.join(tempDir.name, "tempTiff.tif"),
        xsize=3, ysize=3, bands=3, eType=gdal.GDT_CFloat32)
    for i in range(3):
        testDataset.GetRasterBand(i+1).WriteArray(np.ones([3, 3]))
    testDataset = None
    yield tempPath
    tempDir.cleanup()