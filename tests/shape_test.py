import pytest
from geospatial_learn import shape
import numpy as np
import gdal
import tempfile
from osgeo import ogr, osr
import os

@pytest.fixture
def shape_test_dir():
    """
    Returns a temporary directory containing a raster and a shapefile

    """
    temp_dir = tempfile.TemporaryDirectory()

    vector_file = os.path.join(temp_dir.name, "test.shp")

    shape_driver = ogr.GetDriverByName("ESRI Shapefile")
    vector_data_source = shape_driver.CreateDataSource(vector_file)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    vector_layer = vector_data_source.CreateLayer("geometry", srs, geom_type=ogr.wkbLinearRing)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(100.0, 100.0)
    ring.AddPoint(100.0, 110.0)
    ring.AddPoint(110.0, 110.0)
    ring.AddPoint(110.0, 100.0)

    vector_feature_definition = vector_layer.GetLayerDefn()
    vector_feature = ogr.Feature(vector_feature_definition)
    vector_feature.SetGeometry(ring)
    vector_layer.CreateFeature(vector_feature)

    vector_data_source = None   # Discard vector_data_source to force write



    yield os.path.join(temp_dir.name)
    temp_dir.cleanup()


def test_zonal_stats(shape_test_dir):
    pass