from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Dict, Tuple

import dask.dataframe as dd
import hipscat as hc
import pandas as pd
from hipscat.pixel_math import HealpixPixel
import numpy as np
import healpy as hp
from hipscat.pixel_tree.pixel_tree import PixelTree
from hipscat.pixel_tree.pixel_tree_builder import PixelTreeBuilder

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.cone_search import cone_filter
from lsdb.dask.crossmatch_catalog_data import crossmatch_catalog_data
from lsdb.dask.join_catalog_data import join_catalog_data
from lsdb.dask.skymap_catalog_data import skymap_catalog_data


if TYPE_CHECKING:
    from lsdb.catalog.association_catalog.association_catalog import \
        AssociationCatalog

DaskDFPixelMap = Dict[HealpixPixel, int]


# pylint: disable=R0903, W0212
class Catalog(Dataset):
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        name: Name of the catalog
        hc_structure: `hipscat.Catalog` object representing the structure
                      and metadata of the HiPSCat catalog
    """

    def __init__(
        self,
        ddf: dd.core.DataFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.Catalog,
    ):
        """Initialise a Catalog object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.load_...` methods

        Args:
            ddf: Dask DataFrame with the source data of the catalog
            ddf_pixel_map: Dictionary mapping HEALPix order and pixel to partition index of ddf
            hc_structure: `hipscat.Catalog` object with hipscat metadata of the catalog
        """
        super().__init__(ddf, hc_structure)
        self._ddf_pixel_map = ddf_pixel_map

    def get_partition(self, order: int, pixel: int) -> dd.core.DataFrame:
        """Get the dask partition for a given HEALPix pixel

        Args:
            order: Order of HEALPix pixel
            pixel: HEALPix pixel number in NESTED ordering scheme
        Returns:
            Dask Dataframe with a single partition with data at that pixel
        Raises:
            Value error if no data exists for the specified pixel
        """
        partition_index = self.get_partition_index(order, pixel)
        return self._ddf.partitions[partition_index]

    def get_partition_index(self, order: int, pixel: int) -> int:
        """Get the dask partition for a given HEALPix pixel

        Args:
            order: Order of HEALPix pixel
            pixel: HEALPix pixel number in NESTED ordering scheme
        Returns:
            Dask Dataframe with a single partition with data at that pixel
        Raises:
            Value error if no data exists for the specified pixel
        """
        hp_pixel = HealpixPixel(order, pixel)
        if not hp_pixel in self._ddf_pixel_map:
            raise ValueError(f"Pixel at order {order} pixel {pixel} not in Catalog")
        partition_index = self._ddf_pixel_map[hp_pixel]
        return partition_index

    def join(self, other: Catalog, through: AssociationCatalog=None, suffixes: Tuple[str, str] | None = None) -> Catalog:
        if through is None:
            raise NotImplementedError("must specify through association catalog")
        if suffixes is None:
            suffixes = (f"_{self.hc_structure.catalog_info.catalog_name}", f"_{other.hc_structure.catalog_info.catalog_name}")
        ddf, ddf_map, alignment = join_catalog_data(self, other, through, suffixes=suffixes)
        new_catalog_info = dataclasses.replace(
            self.hc_structure.catalog_info,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )
        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)

    def crossmatch(self, other: Catalog, suffixes: Tuple[str, str] | None = None) -> Catalog:
        if suffixes is None:
            suffixes = (f"_{self.hc_structure.catalog_info.catalog_name}", f"_{other.hc_structure.catalog_info.catalog_name}")
        ddf, ddf_map, alignment = crossmatch_catalog_data(self, other, suffixes)
        new_catalog_info = dataclasses.replace(
            self.hc_structure.catalog_info,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )
        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)

    def query(self, qarg: str=None) -> Catalog:
        if qarg is None:
            raise Exception("Must pass a string query argument like: 'column_name1 > 0'")
        ddf = self._ddf.query(qarg)
        return Catalog(ddf, self._ddf_pixel_map, self.hc_structure)

    def cone_search(self, ra, dec, radius):
        max_order = max(self.hc_structure.pixel_tree.pixels.keys())
        n_side = hp.order2nside(max_order)
        center_vec = hp.ang2vec(ra, dec, lonlat=True)
        radius_radians = np.radians(radius)
        cone_pixels = hp.query_disc(n_side, center_vec, radius_radians, inclusive=True, nest=True)
        cone_pixel_info_dict = {
            hc.catalog.PartitionInfo.METADATA_ORDER_COLUMN_NAME: [max_order for _ in range(len(cone_pixels))],
            hc.catalog.PartitionInfo.METADATA_PIXEL_COLUMN_NAME: cone_pixels,
        }
        cone_partition_info_df = pd.DataFrame.from_dict(cone_pixel_info_dict)
        cone_tree = PixelTreeBuilder.from_partition_info_df(cone_partition_info_df)
        pixels_in_cone = []
        for pixel in self._ddf_pixel_map.keys():
            if len(cone_tree.get_leaf_nodes_at_healpix_pixel(pixel)) > 0:
                pixels_in_cone.append(pixel)
        dfs = self._ddf.to_delayed()
        partitions_in_cone = [dfs[self._ddf_pixel_map[pixel]] for pixel in pixels_in_cone]
        filtered_partitions = [cone_filter(partition, ra, dec, radius, self.hc_structure) for partition in partitions_in_cone]
        cone_search_ddf = dd.from_delayed(filtered_partitions, meta=self._ddf._meta)
        filtered_pixel_info_dict = {
            hc.catalog.PartitionInfo.METADATA_ORDER_COLUMN_NAME: [pixel.order for pixel in pixels_in_cone],
            hc.catalog.PartitionInfo.METADATA_PIXEL_COLUMN_NAME: [pixel.pixel for pixel in pixels_in_cone],
        }
        partition_info_df = pd.DataFrame.from_dict(filtered_pixel_info_dict)
        hc_catalog = hc.catalog.Catalog(catalog_info=self.hc_structure.catalog_info, pixels=partition_info_df)
        ddf_partition_map = {pixel: i for i, pixel in enumerate(pixels_in_cone)}
        return Catalog(cone_search_ddf, ddf_partition_map, hc_catalog)


    def where(self, qarg: str) -> Catalog:
        return self.query(qarg=qarg)
    
    def assign(self, **kwargs) -> Catalog:
        if len(kwargs) == 0 or len(kwargs) > 1:
            raise Exception("Invalid assigning of column. Must be a single lambda function")
        ddf = self._ddf.assign(**kwargs)
        return Catalog(ddf, self._ddf_pixel_map, self.hc_structure)
    
    def for_each(self, ufunc, **kwargs) -> Catalog:
        if "cat_info" not in kwargs.keys():
            kwargs["cat_info"] = self.hc_structure.catalog_info
        ddf = self._ddf.groupby("_hipscat_index").apply(ufunc, **kwargs)
        return Catalog(ddf, self._ddf_pixel_map, self.hc_structure)
    
    def compute_skymap(self, col, f=np.mean, k=6) -> np.ndarray:
        img = skymap_catalog_data(self, col=col, order=k, func=f)
        return img

