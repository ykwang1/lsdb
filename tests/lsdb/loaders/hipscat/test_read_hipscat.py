import hipscat as hc
import pandas as pd
import pytest

import lsdb
from lsdb import Catalog


def assert_divisions_are_correct(catalog: Catalog):
    # Get partitions in correct order
    partitions = [
        catalog.get_partition(hp.order, hp.pixel)
        for hp in catalog.get_ordered_healpix_pixels()
    ]
    # Check that divisions are the ones expected
    divisions = []
    for index, partition in enumerate(partitions):
        indices = partition.compute().index.values
        divisions.append(min(indices))
        if index == len(partitions) - 1:
            divisions.append(max(indices))
    assert catalog._ddf.divisions == tuple(divisions)


def test_read_hipscat(small_sky_order1_dir, small_sky_order1_hipscat_catalog):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == small_sky_order1_hipscat_catalog.catalog_base_dir
    assert catalog.get_healpix_pixels() == small_sky_order1_hipscat_catalog.get_healpix_pixels()
    assert_divisions_are_correct(catalog)


def test_pixels_in_map_equal_catalog_pixels(small_sky_order1_dir, small_sky_order1_hipscat_catalog):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    for healpix_pixel in small_sky_order1_hipscat_catalog.get_healpix_pixels():
        catalog.get_partition(healpix_pixel.order, healpix_pixel.pixel)


def test_wrong_pixel_raises_value_error(small_sky_order1_dir):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    with pytest.raises(ValueError):
        catalog.get_partition(-1, -1)


def test_parquet_data_in_partitions_match_files(small_sky_order1_dir, small_sky_order1_hipscat_catalog):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    for healpix_pixel in small_sky_order1_hipscat_catalog.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        partition = catalog.get_partition(hp_order, hp_pixel)
        partition_df = partition.compute()
        parquet_path = hc.io.paths.pixel_catalog_file(
            small_sky_order1_hipscat_catalog.catalog_base_dir, hp_order, hp_pixel
        )
        loaded_df = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(partition_df, loaded_df)


def test_read_hipscat_specify_catalog_type(small_sky_catalog, small_sky_dir):
    catalog = lsdb.read_hipscat(small_sky_dir, catalog_type=lsdb.Catalog)
    assert isinstance(catalog, lsdb.Catalog)
    pd.testing.assert_frame_equal(catalog.compute(), small_sky_catalog.compute())
    assert catalog.get_healpix_pixels() == small_sky_catalog.get_healpix_pixels()
    assert catalog.hc_structure.catalog_info == small_sky_catalog.hc_structure.catalog_info


def test_read_hipscat_specify_wrong_catalog_type(small_sky_dir):
    with pytest.raises(ValueError):
        lsdb.read_hipscat(small_sky_dir, catalog_type=int)
