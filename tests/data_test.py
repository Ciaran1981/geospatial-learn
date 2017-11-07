import pytest
import os
import gdal
import numpy as np
from context import geospatial_learn
from geospatial_learn import data
import json


# Test frame
@pytest.mark.download
def test_planet_query(managed_geotiff_dir):
    with open("/home/jfr10/maps/aois/brazil/window_areas.json") as brazil_json:
        area = json.load(brazil_json)
    simple_area = area["features"][0]["geometry"]
    data.planet_query(simple_area, '2017-01-01T00:00:00Z', '2017-02-01T00:00:00Z', 'test_outputs/brazil/planetImage.tif')
    assert gdal.Open("test_outputs/brazil/planetImage.tif")
