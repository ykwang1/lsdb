import os

import hipscat as hc
import hipscat.io.file_io.file_io as file_io
import pandas as pd
import pytest

import lsdb

DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_XMATCH_NAME = "small_sky_xmatch"
SMALL_SKY_NO_METADATA_DIR_NAME = "small_sky_no_metadata"
SMALL_SKY_ORDER1_DIR_NAME = "small_sky_order1"
XMATCH_CORRECT_FILE = "xmatch_correct.csv"
XMATCH_CORRECT_005_FILE = "xmatch_correct_0_005.csv"
XMATCH_CORRECT_3n_2t_NO_MARGIN_FILE = "xmatch_correct_3n_2t_no_margin.csv"
XMATCH_MOCK_FILE = "xmatch_mock.csv"
TEST_DIR = os.path.dirname(__file__)


@pytest.fixture
def example_abfs_path():
    return "abfs:///hipscat/pytests"


@pytest.fixture
def example_abfs_storage_options():
    storage_options = {
        "account_key" : os.environ.get("ABFS_LINCCDATA_ACCOUNT_KEY"),
        "account_name" : os.environ.get("ABFS_LINCCDATA_ACCOUNT_NAME")
    }
    return storage_options


@pytest.fixture
def test_data_dir_cloud(example_abfs_path):
    return os.path.join(example_abfs_path, "lsdb", "data")


@pytest.fixture
def small_sky_dir_cloud(test_data_dir_cloud):
    return os.path.join(test_data_dir_cloud, SMALL_SKY_DIR_NAME)


@pytest.fixture
def small_sky_xmatch_dir_cloud(test_data_dir_cloud):
    return os.path.join(test_data_dir_cloud, SMALL_SKY_XMATCH_NAME)


@pytest.fixture
def small_sky_no_metadata_dir_cloud(test_data_dir_cloud):
    return os.path.join(test_data_dir_cloud, SMALL_SKY_NO_METADATA_DIR_NAME)


@pytest.fixture
def small_sky_order1_dir_cloud(test_data_dir_cloud):
    return os.path.join(test_data_dir_cloud, SMALL_SKY_ORDER1_DIR_NAME)


@pytest.fixture
def small_sky_hipscat_catalog_cloud(small_sky_dir_cloud, example_abfs_storage_options):
    return hc.catalog.Catalog.read_from_hipscat(
        small_sky_dir_cloud, storage_options=example_abfs_storage_options
    )

@pytest.fixture
def small_sky_catalog_cloud(small_sky_dir_cloud, example_abfs_storage_options):
    return lsdb.read_hipscat(small_sky_dir_cloud, storage_options=example_abfs_storage_options)


@pytest.fixture
def small_sky_xmatch_catalog_cloud(small_sky_xmatch_dir_cloud, example_abfs_storage_options):
    return lsdb.read_hipscat(small_sky_xmatch_dir_cloud, storage_options=example_abfs_storage_options)


@pytest.fixture
def small_sky_order1_hipscat_catalog_cloud(small_sky_order1_dir_cloud, example_abfs_storage_options):
    return hc.catalog.Catalog.read_from_hipscat(small_sky_order1_dir_cloud, storage_options=example_abfs_storage_options)


@pytest.fixture
def small_sky_order1_catalog_cloud(small_sky_order1_dir_cloud, example_abfs_storage_options):
    return lsdb.read_hipscat(small_sky_order1_dir_cloud, storage_options=example_abfs_storage_options)


@pytest.fixture
def xmatch_correct_cloud(small_sky_xmatch_dir_cloud, example_abfs_storage_options):
    pathway = os.path.join(small_sky_xmatch_dir_cloud, XMATCH_CORRECT_FILE)
    return file_io.load_csv_to_pandas(pathway, storage_options=example_abfs_storage_options)
    #return pd.read_csv(os.path.join(small_sky_xmatch_dir_cloud, XMATCH_CORRECT_FILE), storage_options=example_abfs_storage_options)


@pytest.fixture
def xmatch_correct_005_cloud(small_sky_xmatch_dir_cloud, example_abfs_storage_options):
    pathway = os.path.join(small_sky_xmatch_dir_cloud, XMATCH_CORRECT_005_FILE)
    return file_io.load_csv_to_pandas(pathway, storage_options=example_abfs_storage_options)
    #return pd.read_csv(os.path.join(small_sky_xmatch_dir, XMATCH_CORRECT_005_FILE))


@pytest.fixture
def xmatch_correct_3n_2t_no_margin_cloud(small_sky_xmatch_dir_cloud, example_abfs_storage_options):
    pathway = os.path.join(small_sky_xmatch_dir_cloud, XMATCH_CORRECT_3n_2t_NO_MARGIN_FILE)
    return file_io.load_csv_to_pandas(pathway, storage_options=example_abfs_storage_options)


@pytest.fixture
def xmatch_mock_cloud(small_sky_xmatch_dir_cloud, example_abfs_storage_options):
    pathway = os.path.join(small_sky_xmatch_dir_cloud, XMATCH_MOCK_FILE)
    return file_io.load_csv_to_pandas(pathway, storage_options=example_abfs_storage_options)


####
# @pytest.fixture
# def test_data_dir():
#     return os.path.join(TEST_DIR, DATA_DIR_NAME)


# @pytest.fixture
# def small_sky_dir(test_data_dir):
#     return os.path.join(test_data_dir, SMALL_SKY_DIR_NAME)


# @pytest.fixture
# def small_sky_xmatch_dir(test_data_dir):
#     return os.path.join(test_data_dir, SMALL_SKY_XMATCH_NAME)


# @pytest.fixture
# def small_sky_no_metadata_dir(test_data_dir):
#     return os.path.join(test_data_dir, SMALL_SKY_NO_METADATA_DIR_NAME)


# @pytest.fixture
# def small_sky_order1_dir(test_data_dir):
#     return os.path.join(test_data_dir, SMALL_SKY_ORDER1_DIR_NAME)


# @pytest.fixture
# def small_sky_hipscat_catalog(small_sky_dir):
#     return hc.catalog.Catalog.read_from_hipscat(small_sky_dir)


# @pytest.fixture
# def small_sky_catalog(small_sky_dir):
#     return lsdb.read_hipscat(small_sky_dir)


# @pytest.fixture
# def small_sky_xmatch_catalog(small_sky_xmatch_dir):
#     return lsdb.read_hipscat(small_sky_xmatch_dir)


# @pytest.fixture
# def small_sky_order1_hipscat_catalog(small_sky_order1_dir):
#     return hc.catalog.Catalog.read_from_hipscat(small_sky_order1_dir)


# @pytest.fixture
# def small_sky_order1_catalog(small_sky_order1_dir):
#     return lsdb.read_hipscat(small_sky_order1_dir)


# @pytest.fixture
# def xmatch_correct(small_sky_xmatch_dir):
#     return pd.read_csv(os.path.join(small_sky_xmatch_dir, XMATCH_CORRECT_FILE))


# @pytest.fixture
# def xmatch_correct_005(small_sky_xmatch_dir):
#     return pd.read_csv(os.path.join(small_sky_xmatch_dir, XMATCH_CORRECT_005_FILE))


# @pytest.fixture
# def xmatch_correct_3n_2t_no_margin(small_sky_xmatch_dir):
#     return pd.read_csv(os.path.join(small_sky_xmatch_dir, XMATCH_CORRECT_3n_2t_NO_MARGIN_FILE))


# @pytest.fixture
# def xmatch_mock(small_sky_xmatch_dir):
#     return pd.read_csv(os.path.join(small_sky_xmatch_dir, XMATCH_MOCK_FILE))
