import pytest
import tempfile
import gdal
import ogr
import osr
import os
import numpy as np


class TestGeodataManager:
    """
    An object that provides methods to create and manipulate a temporary
    directory that contains geotiffs, shapefiles, ect
    All are created with the same projection, corner locations all
    default to 100,100

    Usage: with TestGeodataMangager() as tgm:
        tgm.create_temp_tiff(....)

    Attributes
    ----------
    path : str
        The path to the temporary folder
    images : list[str]
        A list of images in the temporary folder



    """

    def create_temp_tiff(self, name, content=np.zeros([3, 3, 3]), geotransform=(100, 1, 0, 100, 0, 1)):
        """Creates a temporary geotiff in self.path
        """
        path = os.path.join(self.path, name)
        driver = gdal.GetDriverByName('Gtiff')
        if len(content.shape) < 3:
            content = content.reshape((content.shape[0], content.shape[1], 1))
        new_image = driver.Create(
            path,
            xsize=content.shape[0],
            ysize=content.shape[1],
            bands=content.shape[2],
            eType=gdal.GDT_CFloat32
        )
        new_image.SetProjection(self.srs.ExportToWkt())
        new_image.SetGeoTransform(geotransform)
        for band in range(content.shape[2]):
            new_image.GetRasterBand(band+1).WriteArray = content[..., band]
        new_image.FlushCache()
        self.image_paths.append(path)
        self.images.append(new_image)

    def create_temp_shp(self, name):
        vector_file = os.path.join(self.temp_dir.name, name)
        shape_driver = ogr.GetDriverByName("ESRI Shapefile")  # Depreciated; replace at some point
        vector_data_source = shape_driver.CreateDataSource(vector_file)
        vector_layer = vector_data_source.CreateLayer("geometry", self.srs, geom_type=ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(100.0, 100.0)
        ring.AddPoint(100.0, 110.0)
        ring.AddPoint(110.0, 110.0)
        ring.AddPoint(110.0, 100.0)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        vector_feature_definition = vector_layer.GetLayerDefn()
        vector_feature = ogr.Feature(vector_feature_definition)
        vector_feature.SetGeometry(poly)
        vector_layer.CreateFeature(vector_feature)
        vector_data_source.FlushCache()
        self.vectors.append(vector_data_source)  # Check this is the right thing to be saving here
        self.vector_paths.append(vector_file)

    def __init__(self, srs=4326):
        self.srs = osr.SpatialReference()
        self.srs.ImportFromEPSG(srs)

    def __enter__(self):
        """Creates a randomly named temp folder
        """
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name)
        self.path = temp_path
        self.temp_dir = temp_dir
        self.images = []
        self.image_paths = []
        self.vectors = []
        self.vector_paths = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.temp_dir.cleanup()


@pytest.fixture
def managed_geotiff_dir():
    """Holds context for the TestGeotiffManager class defined above"""
    with TestGeodataManager() as tgm:
        tgm.create_temp_tiff("temp.tif")
        yield tgm

@pytest.fixture
def managed_geotiff_shapefile_dir():
    """Creates a temp dir with a globally contiguous shapefile and geotiff"""
    with TestGeodataManager() as tgm:
        array = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                          [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                          [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                          [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                          [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        tgm.create_temp_tiff("temp.tif", array)
        tgm.create_temp_shp("temp.shp")
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
