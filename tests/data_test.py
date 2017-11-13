import pytest
import os
import gdal
import numpy as np
from geospatial_learn import data
import json
import requests


def test_planet_query(monkeypatch):
    # Actual download function monkeypatched out
    def mock_download(session, item, asset_type, file_path):
        if item["id"] and item["item_types"]:
            return
        raise Exception("bad augments")
    monkeypatch.setattr(data, 'activate_and_dl_planet_item', mock_download)

    with open("/home/jfr10/maps/tests/brazil_small.json") as brazil_json:
        simple_area = json.load(brazil_json)
    data.planet_query(simple_area,
                      '2017-11-01T00:00:00Z',
                      '2017-11-02T00:00:00Z',
                      'tests/test_outputs/brazil/')


@pytest.mark.download
def test_activate_and_dl_planet_item():
    session = requests.Session()
    session.auth = (os.environ['PL_API_KEY'], '')
    test_item = {"id": "20160707_195147_1057916_RapidEye-1",
                 "item_types": "REOrthoTile"}
    test_fp = "test_outputs/"
    asset_type = "visual"
    data.activate_and_dl_planet_item(session, test_item, asset_type, test_fp)
