import hipscat as hc
import pandas as pd
import pytest

import lsdb


def test_read_hipscat(small_sky_order1_dir):
    cat = lsdb.read_hipscat(small_sky_order1_dir)
    print("")
    print(cat._ddf.compute())


def test_hc_catalog_equal(small_sky_order1_dir, small_sky_order1_hipscat_catalog):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    assert (
        catalog.hc_structure.catalog_base_dir
        == small_sky_order1_hipscat_catalog.catalog_base_dir
    )
    pd.testing.assert_frame_equal(
        catalog.hc_structure.get_pixels(), small_sky_order1_hipscat_catalog.get_pixels()
    )


def test_pixels_in_map_equal_catalog_pixels(
    small_sky_order1_dir, small_sky_order1_hipscat_catalog
):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    for _, row in small_sky_order1_hipscat_catalog.get_pixels().iterrows():
        hp_order = row["Norder"]
        hp_pixel = row["Npix"]
        catalog.get_partition(hp_order, hp_pixel)


def test_wrong_pixel_raises_value_error(small_sky_order1_dir):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    with pytest.raises(ValueError):
        catalog.get_partition(-1, -1)


def test_parquet_data_in_partitions_match_files(
    small_sky_order1_dir, small_sky_order1_hipscat_catalog
):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    for _, row in small_sky_order1_hipscat_catalog.get_pixels().iterrows():
        hp_order = row["Norder"]
        hp_pixel = row["Npix"]
        partition = catalog.get_partition(hp_order, hp_pixel)
        partition_df = partition.compute()
        parquet_path = hc.io.paths.pixel_catalog_file(
            small_sky_order1_hipscat_catalog.catalog_base_dir, hp_order, hp_pixel
        )
        loaded_df = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(partition_df, loaded_df)
