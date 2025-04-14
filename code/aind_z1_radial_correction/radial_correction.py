"""
Computes radial correction in microscopic data
"""

import logging
import os
import time
from math import ceil
from pathlib import Path
from threading import Thread
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

import dask
import dask.array as da
import numpy as np
import s3fs
import yaml
import zarr
from aind_data_transfer.transformations.ome_zarr import (
    _get_bytes,
    downsample_and_store,
    store_array,
    write_ome_ngff_metadata,
)
from aind_data_transfer.util.chunk_utils import (
    ensure_array_5d,
    ensure_shape_5d,
)
from aind_data_transfer.util.io_utils import BlockedArrayWriter
from dask.distributed import Client, LocalCluster, performance_report
from numpy.typing import ArrayLike
from scipy.ndimage import map_coordinates

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def calculate_corner_shift_from_pixel_size(XY_pixel_size: float) -> float:
    """
    Compute the corner shift value based on pixel size.

    Parameters
    ----------
    XY_pixel_size : float
        The size of a pixel in microns (not currently used).

    Returns
    -------
    float
        A constant value of 4.3.
    """
    return 4.3


def calculate_frac_cutoff_from_pixel_size(XY_pixel_size: float) -> float:
    """
    Compute the fractional cutoff for radial correction.

    Parameters
    ----------
    XY_pixel_size : float
        The size of a pixel in microns (not currently used).

    Returns
    -------
    float
        A constant value of 0.5.
    """
    return 0.5


def radial_correction(
    tile_data: np.ndarray,
    corner_shift: Optional[float] = 5.5,
    frac_cutoff: Optional[float] = 0.5,
) -> np.ndarray:
    """
    Apply 3D radial correction to a tile.

    Parameters
    ----------
    tile_data : np.ndarray
        The 3D tile data (Z, Y, X) to be corrected.
    corner_shift : Optional[float]
        The amount of radial shift to apply (default is 5.5).

    frac_cutoff : Optional[float]
        Fraction of the radius to begin applying correction (default is 0.5).

    Returns
    -------
    np.ndarray
        The corrected tile.
    """
    edge = ceil(corner_shift / np.sqrt(2)) + 1
    shape = tile_data.shape
    pixels = shape[1]  # Assume square XY plane
    cutoff = pixels * frac_cutoff
    img = np.zeros(shape, np.uint16)

    def read(src):
        img[:] = np.array(src, np.uint16)

    t = Thread(target=read, args=(tile_data,))
    t.start()

    grid = np.meshgrid(np.arange(pixels), np.arange(pixels), indexing="ij")
    coords = (np.array(grid) - pixels // 2).astype(np.float32)

    r = np.sqrt((coords**2).sum(0))
    angle = np.arctan2(coords[0], coords[1])
    rmax = r.max()
    r_piece = r + (r > cutoff) * (r - cutoff) * corner_shift / (rmax - cutoff)

    coords[0], coords[1] = r_piece * np.sin(angle), r_piece * np.cos(angle)
    coords = np.array(
        coords[:, edge:-edge, edge:-edge] + pixels // 2, np.float32
    )

    new_shape = np.array(img.shape) - [0, edge * 2, edge * 2]
    warp_coords = np.meshgrid(
        *[np.arange(x).astype(np.int32) for x in new_shape], indexing="ij"
    )
    warp_coords = np.array(warp_coords, dtype=np.float32)
    warp_coords[1] = coords[0][None, ...]
    warp_coords[2] = coords[1][None, ...]

    t.join()
    return map_coordinates(img, warp_coords, order=1, mode="constant")


def radial_correction_2d(
    tile_data: np.ndarray,
    corner_shift: Optional[float] = 5.5,
    frac_cutoff: Optional[float] = 0.5,
) -> np.ndarray:
    """
    Apply 2D radial correction plane-wise to a tile.

    Parameters
    ----------
    tile_data : np.ndarray
        The 3D tile data (Z, Y, X) to be corrected.

    corner_shift : Optional[float]
        The amount of radial shift to apply (default is 5.5).

    frac_cutoff : Optional[float]
        Fraction of the radius to begin applying correction (default is 0.5).

    Returns
    -------
    np.ndarray
        The corrected tile.
    """
    edge = ceil(corner_shift / np.sqrt(2)) + 1
    shape = tile_data.shape
    pixels = shape[1]
    cutoff = pixels * frac_cutoff

    grid = np.meshgrid(np.arange(pixels), np.arange(pixels), indexing="ij")
    coords = (np.array(grid) - pixels // 2).astype(np.float32)

    r = np.sqrt((coords**2).sum(0))
    angle = np.arctan2(coords[0], coords[1])
    rmax = r.max()
    r_piece = r + (r > cutoff) * (r - cutoff) * corner_shift / (rmax - cutoff)

    coords[0], coords[1] = r_piece * np.sin(angle), r_piece * np.cos(angle)
    coords = np.array(
        coords[:, edge:-edge, edge:-edge] + pixels // 2, np.float32
    )

    new_shape = np.array(shape) - [0, edge * 2, edge * 2]
    img = np.zeros(new_shape, np.uint16)

    warp_coords = np.meshgrid(
        np.arange(new_shape[1]), np.arange(new_shape[2]), indexing="ij"
    )
    warp_coords = np.array(warp_coords, dtype=np.float32)
    warp_coords[0] = coords[0][None]
    warp_coords[1] = coords[1][None]

    for z in range(tile_data.shape[0]):
        img[z] = map_coordinates(
            tile_data[z], warp_coords, order=1, mode="constant"
        )

    return img


def apply_corr_to_zarr_tile(
    zarr_file_loc: str,
    corner_shift: Optional[float] = 5.5,
    frac_cutoff: Optional[float] = 0.5,
) -> np.ndarray:
    """
    Load a Zarr tile, apply radial correction, and return corrected tile.

    Parameters
    ----------
    zarr_file_loc : str
        Path to the Zarr file containing the tile.

    corner_shift : Optional[float]
        The amount of shift to apply to corners (default is 5.5).

    frac_cutoff : Optional[float]
        The fractional radius where correction starts (default is 0.5).

    Returns
    -------
    np.ndarray
        The corrected tile.
    """
    tile = da.from_zarr(zarr_file_loc)
    z_size = tile.shape[2]

    if z_size < 400:
        return radial_correction(
            tile[0, 0].compute(), corner_shift, frac_cutoff
        )
    else:
        return radial_correction_2d(
            tile[0, 0].compute(), corner_shift, frac_cutoff
        )


def run_multiscale(
    full_res_arr: dask.array,
    out_group: zarr.group,
    voxel_sizes_zyx: Tuple[int],
    scale_factors: Optional[Tuple[int]] = (2, 2, 2),
    n_levels: Optional[int] = 5,
):
    """
    Creates a multiscale representation of the
    full resolution data.

    Parameters
    ----------
    full_res_arr: dask.array
        Lazy array with the full resolution data.

    out_group: zarr.group
        Output zarr group

    voxel_sizes_zyx: Tuple[int]
        Voxel sizes in zyx order

    scale_factors: Optional[Tuple[int]]
        Scale factors for each of the axis.
        Default: (2, 2, 2)

    n_levels: Optional[int]
        Number of levels in the multiscale
        Default: 5
    """

    if len(voxel_sizes_zyx) != 3:
        raise ValueError(f"Please, provide the voxel sizes in ZYX order.")

    arr = ensure_array_5d(full_res_arr)

    LOGGER.info(f"input array: {arr}")

    LOGGER.info(f"input array size: {arr.nbytes / 2 ** 20} MiB")

    block_shape = ensure_shape_5d(BlockedArrayWriter.get_block_shape(arr))
    LOGGER.info(f"block shape: {block_shape}")

    scale_factors = ensure_shape_5d(scale_factors)
    compressor = None  # blosc.Blosc("zstd", 1, shuffle=blosc.SHUFFLE)

    # Actual Processing
    t0 = time.time()

    write_ome_ngff_metadata(
        out_group,
        arr,
        out_group.path,
        n_levels,
        scale_factors[-3:],
        voxel_sizes_zyx[-3:],
        origin=None,
    )

    store_array(arr, out_group, "0", block_shape, compressor)
    # out_group.create_dataset("0", data = arr, compressor=compressor, overwrite = True, chunks = (1, 1, 128, 256, 256))

    pyramid = downsample_and_store(
        arr, out_group, n_levels, scale_factors, block_shape, compressor
    )
    write_time = time.time() - t0

    LOGGER.info(
        f"Finished writing tile.\n"
        f"Took {write_time}s. {_get_bytes(pyramid) / write_time / (1024 ** 2)} MiB/s"
    )


def main():
    pass


if __name__ == "__main__":
    main()
