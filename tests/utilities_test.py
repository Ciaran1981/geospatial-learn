import pytest

from geospatial_learn.utilities import min_bound_rectangle

def test_min_bound_rectangle():
    assert min_bound_rectangle([[1, -1], [1, 1], [-1, 1], [-1, -1]]) == [(1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]

