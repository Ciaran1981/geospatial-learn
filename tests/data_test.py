import pytest
import os
import gdal
import numpy as np
from geospatial_learn import data
import json


# Test frame
@pytest.mark.download
def test_planet_query():
    with open("/home/jfr10/maps/aois/brazil/window_areas.json") as brazil_json:
        area = json.load(brazil_json)
    simple_area = area["features"][0]["geometry"]
    data.planet_query(simple_area,
                      '2017-11-01T00:00:00Z',
                      '2017-11-02T00:00:00Z',
                      'tests/test_outputs/brazil/')
    assert gdal.Open("test_outputs/brazil/1.tif")
