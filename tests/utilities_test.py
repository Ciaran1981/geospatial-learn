import pytest
from .context import geospatial_learn
from geospatial_learn.utilities import min_bound_rectangle


def test_min_bound_rectangle():
    assert len(min_bound_rectangle([[1, -1], [1, 1], [-1, 1], [-1, -1]])) == 4
