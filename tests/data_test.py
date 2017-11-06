import pytest
import os
import gdal
import numpy as np
from context import geospatial_learn
from geospatial_learn import data

#Test frame
def test_planet_query(managed_geotiff_dir):
    area = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.54, 37.81],
                [-122.38, 37.84],
                [-122.35, 37.71],
                [-122.53, 37.70],
                [-122.54, 37.81]
            ]
        ]
    }

    data.planet_query(area, '2017-01-01', '2017-02-01', 'test_outputs/planetImage.tif')
    assert gdal.Open("test_outputs/planetImage.tif")
