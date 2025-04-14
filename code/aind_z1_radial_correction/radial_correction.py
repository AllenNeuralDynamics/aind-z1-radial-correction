"""
Computes radial correction in microscopic data
"""

import logging
import os
import time
from math import ceil
from pathlib import Path
from threading import Thread
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
    tile_data: np.ndarray, corner_shift: float = 5.5, frac_cutoff: float = 0.5
) -> np.ndarray:
    """
    Apply 3D radial correction to a tile.

    Parameters
    ----------
    tile_data : np.ndarray
        The 3D tile data (Z, Y, X) to be corrected.
    corner_shift : float, optional
        The amount of radial shift to apply (default is 5.5).
    frac_cutoff : float, optional
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
    tile_data: np.ndarray, corner_shift: float = 5.5, frac_cutoff: float = 0.5
) -> np.ndarray:
    """
    Apply 2D radial correction plane-wise to a tile.

    Parameters
    ----------
    tile_data : np.ndarray
        The 3D tile data (Z, Y, X) to be corrected.
    corner_shift : float, optional
        The amount of radial shift to apply (default is 5.5).
    frac_cutoff : float, optional
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
    zarr_file_loc: str, corner_shift: float = 5.5, frac_cutoff: float = 0.5
) -> np.ndarray:
    """
    Load a Zarr tile, apply radial correction, and return corrected tile.

    Parameters
    ----------
    zarr_file_loc : str
        Path to the Zarr file containing the tile.
    corner_shift : float, optional
        The amount of shift to apply to corners (default is 5.5).
    frac_cutoff : float, optional
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


def main():
    pass


if __name__ == "__main__":
    main()
