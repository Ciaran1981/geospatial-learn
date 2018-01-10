import pytest
import os
import gdal
import numpy as np
from geospatial_learn import geodata


def test_array2raster(geotiff_dir):
    array = np.zeros([3, 3, 3])
    bands = 3
    inRaster = os.path.join(geotiff_dir, "tempTiff.tif")
    outRaster = os.path.join(geotiff_dir, "fromArray")
    dtype = gdal.GDT_Int32
    geodata.array2raster(array, bands, inRaster, outRaster, dtype)
    result = gdal.Open(outRaster + ".tif").ReadAsArray()
    assert result[0, 1, 2] == 0


def test_copy_dataset_config(managed_geotiff_dir):
    result = geodata._copy_dataset_config(managed_geotiff_dir.images[0])
    assert result.ReadAsArray().shape == (3, 3)
    result = None


def test_multi_temp_filter(managed_geotiff_dir):
    managed_geotiff_dir.create_temp_tiff("result.tif")
    geodata.multi_temp_filter(
        inRas=managed_geotiff_dir.image_paths[0],
        outRas=managed_geotiff_dir.image_paths[1]
    )
    # TODO: improve this assert statement to make it more that just a check
    assert gdal.Open(managed_geotiff_dir.image_paths[1])


