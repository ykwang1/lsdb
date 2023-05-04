
from typing import Callable, TYPE_CHECKING, Dict
import healpy as hp
import numpy as np
import pandas as pd
import hipscat as hc
import dask
from hipscat.catalog import PartitionInfo
from hipscat.pixel_math import HealpixPixel

from lsdb.dask.join_catalog_data import filter_index_to_range

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


def skymap_agg(df, col, f=np.mean):
    if len(df) == 0:
        return 0
    return f(df[col].values)


@dask.delayed
def get_img(results: Dict[int, int], order: int):
    npix = hp.order2npix(order)
    img = np.zeros(npix)
    for index, result in results.items():
        img[index] = result
    return img


def skymap_catalog_data(cat, col: str=None, order: int=6, func: Callable=np.mean):
    concat_pixels = {}
    filter_pixels = {}
    for _, row in cat.hc_structure.get_pixels().iterrows():
        hp_pixel = HealpixPixel(
            order=row[PartitionInfo.METADATA_ORDER_COLUMN_NAME],
            pixel=row[PartitionInfo.METADATA_PIXEL_COLUMN_NAME]
        )
        if hp_pixel.order < order:
            for order_pixel in hp_pixel.convert_to_higher_order(order-hp_pixel.order):
                partition_index = cat.get_partition_index(hp_pixel.order, hp_pixel.pixel)
                filter_pixels[order_pixel.pixel] = partition_index
        elif hp_pixel.order == order:
            partition_index = cat.get_partition_index(hp_pixel.order, hp_pixel.pixel)
            filter_pixels[hp_pixel.pixel] = partition_index
        else:
            order_pixel = hp_pixel.convert_to_lower_order(hp_pixel.order - order).pixel
            if order_pixel not in concat_pixels:
                concat_pixels[order_pixel] = []
            partition_index = cat.get_partition_index(hp_pixel.order, hp_pixel.pixel)
            concat_pixels[order_pixel].append(partition_index)
    partitions = cat._ddf.to_delayed()
    results = {}
    for order_pixel, partition_index in filter_pixels.items():
        lower_bound = hc.pixel_math.hipscat_id.healpix_to_hipscat_id(order, order_pixel)
        upper_bound = hc.pixel_math.hipscat_id.healpix_to_hipscat_id(order, order_pixel+1)
        partition = partitions[partition_index]
        filtered_pixel = filter_index_to_range(partition, lower_bound, upper_bound)
        pixel_result = dask.delayed(skymap_agg)(filtered_pixel, col, f=func)
        results[order_pixel] = pixel_result
    for order_pixel, partition_indexes in concat_pixels.items():
        pixel_partitions = [partitions[partition_index] for partition_index in partition_indexes]
        concat_partitions = dask.delayed(pd.concat)(pixel_partitions)
        pixel_result = dask.delayed(skymap_agg)(concat_partitions, col, f=func)
        results[order_pixel] = pixel_result
    img = get_img(results, order)
    return img
