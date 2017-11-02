import pytest
import os
import gdal
import numpy as np
from context import geospatial_learn
from geospatial_learn import data

#Test stub
def test_planet_query(managed_geotiff_dir):
    data.planet_query(
        target="some target",
        out="some output"
    )
    assert gdal.Open("some_output")
