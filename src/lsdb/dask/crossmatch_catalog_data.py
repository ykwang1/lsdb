from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Type, cast

import dask
import dask.dataframe as dd
import pandas as pd
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.dask.merge_catalog_functions import (
    align_catalogs_to_alignment_mapping,
    filter_by_hipscat_index_to_pixel,
    generate_meta_df_for_joined_tables,
    get_partition_map_from_alignment_pixels, get_healpix_pixels_from_alignment,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog

builtin_crossmatch_algorithms = {BuiltInCrossmatchAlgorithm.KD_TREE: KdTreeCrossmatch}


# pylint: disable=too-many-arguments
@dask.delayed
def perform_crossmatch(
    algorithm,
    left_df,
    right_df,
    left_order,
    left_pixel,
    right_order,
    right_pixel,
    left_hc_structure,
    right_hc_structure,
    suffixes,
    **kwargs,
):
    """Performs a crossmatch on data from a HEALPix pixel in each catalog

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.
    """
    if right_order > left_order:
        left_df = filter_by_hipscat_index_to_pixel(left_df, right_order, right_pixel)
    return algorithm(
        left_df,
        right_df,
        left_order,
        left_pixel,
        right_order,
        right_pixel,
        left_hc_structure,
        right_hc_structure,
        suffixes,
    ).crossmatch(**kwargs)


# pylint: disable=too-many-locals
def crossmatch_catalog_data(
    left: Catalog,
    right: Catalog,
    suffixes: Tuple[str, str],
    algorithm: Type[AbstractCrossmatchAlgorithm]
    | BuiltInCrossmatchAlgorithm = BuiltInCrossmatchAlgorithm.KD_TREE,
    **kwargs,
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    """Cross-matches the data from two catalogs

    Args:
        left (Catalog): the left catalog to perform the cross-match on
        right (Catalog): the right catalog to perform the cross-match on
        suffixes (Tuple[str,str]): the suffixes to append to the column names from the left and
            right catalogs respectively
        algorithm (BuiltInCrossmatchAlgorithm | Callable): The algorithm to use to perform the
            crossmatch. Can be specified using a string for a built-in algorithm, or a custom
            method. For more details, see `crossmatch` method in the `Catalog` class.
        **kwargs: Additional arguments to pass to the cross-match algorithm

    Returns:
        A tuple of the dask dataframe with the result of the cross-match, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """
    crossmatch_algorithm = get_crossmatch_algorithm(algorithm)

    # perform alignment on the two catalogs
    alignment = align_trees(
        left.hc_structure.pixel_tree,
        right.hc_structure.pixel_tree,
        alignment_type=PixelAlignmentType.INNER,
    )
    join_pixels = alignment.pixel_mapping

    # align partitions from the catalogs to match the pixel alignment
    left_aligned_partitions, right_aligned_partitions = align_catalogs_to_alignment_mapping(
        join_pixels, left, right
    )

    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(join_pixels)

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    apply_crossmatch = np.vectorize(lambda left_df, right_df, left_pix, right_pix: perform_crossmatch(
            crossmatch_algorithm,
            left_df,
            right_df,
            left_pix.order,
            left_pix.pixel,
            right_pix.order,
            right_pix.pixel,
            left.hc_structure,
            right.hc_structure,
            suffixes,
            **kwargs,
        ))

    joined_partitions = apply_crossmatch(
        left_aligned_partitions,
        right_aligned_partitions,
        left_pixels,
        right_pixels,
    )

    # generate dask df partition map from alignment
    partition_map = get_partition_map_from_alignment_pixels(join_pixels)

    # generate meta table structure for dask df
    meta_df = generate_meta_df_for_joined_tables(
        [left, right], suffixes, extra_columns=crossmatch_algorithm.extra_columns
    )

    # create dask df from delayed partitions
    ddf = dd.from_delayed(joined_partitions, meta=meta_df)
    ddf = cast(dd.DataFrame, ddf)

    return ddf, partition_map, alignment


def generate_meta_df_for_joined_tables(
    catalogs: Sequence[Catalog],
    suffixes: Sequence[str],
    extra_columns: pd.DataFrame | None = None,
    index_name: str = HIPSCAT_ID_COLUMN,
) -> pd.DataFrame:
    """Generates a Dask meta DataFrame that would result from joining two catalogs

    Creates an empty dataframe with the columns of each catalog appended with a suffix. Allows specifying
    extra columns that should also be added, and the name of the index of the resulting dataframe.

    Args:
        catalogs (Sequence[Catalog]): The catalogs to merge together
        suffixes (Sequence[Str]): The column suffixes to apply each catalog
        extra_columns (pd.Dataframe): Any additional columns to the merged catalogs
        index_name (str): The name of the index in the resulting DataFrame

    Returns:
    An empty dataframe with the columns of each catalog with their respective suffix, and any extra columns
    specified, with the index name set.
    """
    meta = {}
    # Construct meta for crossmatched catalog columns
    for table, suffix in zip(catalogs, suffixes):
        for name, col_type in table.dtypes.items():
            meta[name + suffix] = pd.Series(dtype=col_type)
    # Construct meta for crossmatch result columns
    if extra_columns is not None:
        meta.update(extra_columns)
    meta_df = pd.DataFrame(meta)
    meta_df.index.name = index_name
    return meta_df


def get_partition_map_from_alignment_pixels(join_pixels: pd.DataFrame) -> DaskDFPixelMap:
    """Gets a dictionary mapping HEALPix pixel to index of pixel in the pixel_mapping of a `PixelAlignment`

    Args:
        join_pixels (pd.DataFrame): The pixel_mapping from a `PixelAlignment` object

    Returns:
        A dictionary mapping HEALPix pixel to the index that the pixel occurs in the pixel_mapping table
    """
    partition_map = {}
    for i, (_, row) in enumerate(join_pixels.iterrows()):
        pixel = HealpixPixel(
            order=row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME],
            pixel=row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME],
        )
        partition_map[pixel] = i
    return partition_map


def get_crossmatch_algorithm(
    algorithm: Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm,
) -> Type[AbstractCrossmatchAlgorithm]:
    """Gets the function to perform a cross-match algorithm

    Args:
        algorithm: The algorithm to use to perform the cross-match. Can be specified using a string
        for a built-in algorithm, or a custom method.

    Returns:
        The function to perform the specified crossmatch. Either by looking up the method for a
        built-in algorithm, or returning the custom function.
    """
    if isinstance(algorithm, BuiltInCrossmatchAlgorithm):
        return builtin_crossmatch_algorithms[algorithm]
    if issubclass(algorithm, AbstractCrossmatchAlgorithm):
        return algorithm
    raise TypeError("algorithm must be either callable or a string for a builtin algorithm")
