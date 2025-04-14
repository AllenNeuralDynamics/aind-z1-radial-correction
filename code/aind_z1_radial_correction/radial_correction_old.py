import glob
import logging
import os
import time
from math import ceil
from pathlib import Path
from threading import Thread
from typing import Union
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
from numcodecs import blosc
from ome_zarr.io import parse_url
from scipy.ndimage import map_coordinates
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def calculate_corner_shift_from_pixel_size(XY_pixel_size):

    # Discovered it is independent of pixel size
    return 4.3


def calculate_frac_cutoff_from_pixel_size(XY_pixel_size):

    # Should always be 0.5 if we want to only Radially correct the corners.

    return 0.5


def radial_correction(tile_data, corner_shift=5.5, frac_cutoff=0.5):
    """This code is adapted from Tim Wang

    It is intended to perform radial correction on tiles (to remove lens artifacts) for tiff files.
    It is being further modified to perform this same function on zarr files.

    Parameters:
    -------------------
    tile_data: np.ndarray
        The tile data to be corrected.
    corner_shift: float
        The amount of shift to be applied to the corners of the image.
    frac_cutoff: float
        The fraction of the image to be cut off.

    Returns:
    -------------------
    img.get(): np.ndarray
        The corrected image of tile_data


    """

    edge = ceil(corner_shift / (2**0.5)) + 1

    # fn = str(np.load(sys.argv[2]))
    # fn = str(tile_location)
    shape = tile_data.shape
    # print(f'tile shape {shape}')
    pixels = shape[1]  # assume X = Y
    cutoff = pixels * frac_cutoff
    img = np.zeros(shape, np.uint16)

    def read(tile_data):
        # img[:] = cupy.array(tf.imread(fn))
        img[:] = np.array(tile_data, np.uint16)
        # print('read', fn, flush=True)

    t = Thread(target=read, args=(tile_data,))
    t.start()

    grid = np.array(
        np.meshgrid(np.arange(pixels), np.arange(pixels), indexing="ij")
    )
    coords = (grid - pixels // 2).astype(np.float32)

    r = np.sqrt((coords**2).sum(0))
    angle = np.arctan2(coords[0], coords[1])
    rmax = r.max()
    r_piece = r + (r > cutoff) * (r - cutoff) * corner_shift / (
        rmax - cutoff
    )  # piecewise linear

    coords[0], coords[1] = r_piece * np.sin(angle), r_piece * np.cos(angle)
    coords = np.array(
        coords[:, edge:-edge, edge:-edge] + pixels // 2, np.float32
    )

    new_s = np.array(img.shape) - [0, edge * 2, edge * 2]

    arange_list = [np.arange(x).astype(np.int32) for x in new_s]
    # warp_coords = np.meshgrid(*arange_list, indexing='ij')
    warp_coords = np.array(
        np.meshgrid(*arange_list, indexing="ij")
    )  # this line is problematic - using too much memory
    # chunksize = 256
    # warp_coords = generate_warp_coords_chunked(new_s, chunksize)
    # cupy.get_default_memory_pool().free_all_blocks()

    warp_coords[1] = coords[0][None, ...]
    warp_coords[2] = coords[1][None, ...]

    t.join()
    img = map_coordinates(img, warp_coords, order=1, mode="constant")
    # tf.imsave(out_name(fn), img.get())
    return img


def radial_correction_2d(tile_data, corner_shift=5.5, frac_cutoff=0.5):
    """This code is adapted from Tim Wang

    It is intended to perform radial correction on tiles (to remove lens artifacts) for tiff files.
    It is being further modified to perform this same function on zarr files.

    Parameters:
    -------------------
    tile_data: np.ndarray
        The tile data to be corrected.
    corner_shift: float
        The amount of shift to be applied to the corners of the image.
    frac_cutoff: float
        The fraction of the image to be cut off.

    Returns:
    -------------------
    img.get(): np.ndarray
        The corrected image of tile_data


    """

    edge = ceil(corner_shift / (2**0.5)) + 1

    # fn = str(np.load(sys.argv[2]))
    # fn = str(tile_location)
    shape = tile_data.shape
    # print(f'tile shape {shape}')
    pixels = shape[1]  # assume X = Y
    cutoff = pixels * frac_cutoff
    img = np.zeros(shape, np.uint16)

    grid = np.array(
        np.meshgrid(np.arange(pixels), np.arange(pixels), indexing="ij")
    )
    coords = (grid - pixels // 2).astype(np.float32)

    r = np.sqrt((coords**2).sum(0))
    angle = np.arctan2(coords[0], coords[1])
    rmax = r.max()
    r_piece = r + (r > cutoff) * (r - cutoff) * corner_shift / (
        rmax - cutoff
    )  # piecewise linear

    coords[0], coords[1] = r_piece * np.sin(angle), r_piece * np.cos(angle)
    coords = np.array(
        coords[:, edge:-edge, edge:-edge] + pixels // 2, np.float32
    )

    new_s = np.array(img.shape) - [0, edge * 2, edge * 2]

    arange_list = [np.arange(x).astype(np.int32) for x in new_s]
    warp_coords = np.array(np.meshgrid(*arange_list[1:], indexing="ij"))

    warp_coords[0] = coords[0][None, ...]
    warp_coords[1] = coords[1][None, ...]

    img = np.zeros(new_s, np.uint16)
    for z in range(tile_data.shape[0]):
        img[z] = map_coordinates(
            tile_data[z], warp_coords, order=1, mode="constant"
        )
    return img


def apply_corr_to_zarr_tile(zarr_file_loc, corner_shift=5.5, frac_cutoff=0.5):
    """Apply radial correction to a zarr"""

    # print(f'loading tile')
    # Load the zarr file
    tile = da.from_zarr(zarr_file_loc)

    # print(f'loaded tile')

    # Apply the radial correction
    z_size = tile.shape[2]

    # split it up like this to be faster for small tiles, and slower, but less memory footprint for large itles
    if z_size < 400:
        corrected_tile = radial_correction(
            tile[0, 0, ...].compute(), corner_shift, frac_cutoff
        )
    else:
        corrected_tile = radial_correction_2d(
            tile[0, 0, ...].compute(), corner_shift, frac_cutoff
        )

    # print(f'applied correction')
    # Return the corrected tile
    return corrected_tile


def write_s3_path_as_yml(xml_file_loc: str, output_yml_file_loc: str):
    """Gets the s3 location of the dataset and saves it in a yml file"""
    tree = ET.parse(xml_file_loc)

    root = tree.getroot()

    for elem in root.iter("zarr"):
        loc_path = elem.text

    # check if this s3_path exists

    # separate on '/', appending s3 prefix
    s3_folder = loc_path.split("/")[2:]  # ignore the first two '/'
    s3_path = "s3://aind-open-data/" + s3_folder[0] + "/" + s3_folder[1] + "/"

    out_dict = {"s3_path": s3_path}

    # write to YML file
    # with open(output_yml_file_loc, 'w') as file:
    #    yaml.dump(out_dict, file)

    # make radial_correction folder
    os.makedirs("/scratch/radial_correction/", exist_ok=True)

    # pass on dataset name for pipeline
    with open("/scratch/radial_correction/s3_path.yml", "w") as file:
        yaml.dump(out_dict, file)


##################################################################
#                       In PROGRESS
##################################################################

from typing import Any, Generator, List, Tuple

from numcodecs.abc import Codec
from numpy.typing import ArrayLike


def run_multiscale(
    full_res_arr: dask.array, out_group: zarr.group, voxel_sizes_zyx: tuple
):

    arr = ensure_array_5d(full_res_arr)
    # arr = arr.rechunk((1, 1, 128, 256, 256))
    LOGGER.info(f"input array: {arr}")

    LOGGER.info(f"input array size: {arr.nbytes / 2 ** 20} MiB")
    block_shape = ensure_shape_5d(BlockedArrayWriter.get_block_shape(arr))
    # block_shape = arr.chunksize
    LOGGER.info(f"block shape: {block_shape}")

    scale_factors = (2, 2, 2)
    scale_factors = ensure_shape_5d(scale_factors)
    n_levels = 5
    compressor = None  # blosc.Blosc("zstd", 1, shuffle=blosc.SHUFFLE)  #None

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


def read_to_do_yml(yml_loc):
    with open(yml_loc, "r") as file:
        yml_file = yaml.safe_load(file)

    input_path = yml_file["input_path"]
    output_path = yml_file["output_path"]
    temp = yml_file["resolution_zyx"]

    resolution_zyx_um = []
    for res in temp:
        resolution_zyx_um.append(float(res))

    return input_path, output_path, resolution_zyx_um


def correct_and_save_tile(dataset_loc, output_path, resolution_zyx):
    """correct and save a single tile"""
    # resolution_zyx = (1,0.256, 0.256)
    # print(f'{dataset_loc}')

    # get number of cpus
    num_cpus = 8

    client = Client(
        LocalCluster(n_workers=num_cpus, threads_per_worker=1, processes=True)
    )

    s3 = s3fs.S3FileSystem(
        config_kwargs={
            "max_pool_connections": num_cpus,
            "retries": {
                "total_max_attempts": 1000,
                "mode": "adaptive",
            },
        },
        use_ssl=True,
    )

    split_out = output_path.split("/")
    # ome_path = 's3://'+split_out[2]+'/' + split_out[3] + '/'+split_out[4]+'.ome.zarr'

    # store = s3fs.S3Map(root=ome_path, s3=s3, check=False)
    # root_group = zarr.group(store=store, overwrite=False)

    tilename = str(Path(output_path).name)
    # out_group = root_group.create_group(tilename, overwrite=True)

    corner_shift = calculate_corner_shift_from_pixel_size(resolution_zyx[1])
    frac_cutoff = calculate_frac_cutoff_from_pixel_size(resolution_zyx[1])

    LOGGER.info(f"Corner Shift: {corner_shift} pixels")

    corrected_tile = apply_corr_to_zarr_tile(
        dataset_loc + "/0", corner_shift, frac_cutoff
    )
    # make dask array

    # start dask client()

    # downsample and save
    with performance_report(filename=f"/results/{tilename}-dask-report.html"):

        # save as local zarr

        zarr_loc = f"/results/{tilename}"
        zarr.save_array(
            zarr_loc, corrected_tile
        )  # slow... consider doing with dask
        # temp_zarr = zarr.load(zarr_loc)
        # corr_arr = da.from_array(corrected_tile, name=tilename)
        # corr_arr = da.from_zarr(zarr_loc, chunks = (128,256,256))
        # run_multiscale(corr_arr, out_group, resolution_zyx)

    return


# -------------new main-----------------------


def do_radial_correction_on_tile_in_parallel(yml_loc):

    # read yml file (get inputs/outputs/resolution)

    in_path, output_path, resolution_zyx = read_to_do_yml(yml_loc)

    # do radial correction,

    correct_and_save_tile(in_path, output_path, resolution_zyx)


############################################################################
#                           Run Radial Correction
############################################################################


if __name__ == "__main__":

    # if run_from_capsule ==True:
    #     debug = True
    # else:
    #     debug = False #TEMP SWITCH BACK

    # get yml_list
    yml_list = glob.glob("/data/*to_do_radial_correction.yml")

    for yml_loc in yml_list:
        do_radial_correction_on_tile_in_parallel(yml_loc)
