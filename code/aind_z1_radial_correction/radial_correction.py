"""
Computes radial correction in microscopic data
"""

import logging
import os
import time
from math import ceil
from pathlib import Path
from threading import Thread
from typing import Optional, Tuple, List

import dask
import dask.array as da
import numpy as np
import s3fs
import yaml
import zarr
from aind_data_schema.core.processing import (
    DataProcess,
    ProcessName,
)

from scipy.ndimage import map_coordinates
from xml.etree import ElementTree as ET
from natsort import natsorted

from . import __maintainers__, __pipeline_version__, __url__, __version__
from .utils import utils

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


import numpy as np
from scipy.ndimage import map_coordinates
from math import ceil
from typing import Optional, Literal, Union
import numba as nb
from concurrent.futures import ThreadPoolExecutor


@nb.njit(parallel=True)
def _compute_coordinates(
    pixels: int, cutoff: float, corner_shift: float, edge: int
) -> tuple:
    """
    Compute radial correction coordinates with Numba acceleration.

    Returns:
        Tuple of transformed coordinates and new shape dimensions
    """
    # Create coordinate grid relative to center
    x = np.arange(pixels) - pixels // 2
    y = np.arange(pixels) - pixels // 2

    coords = np.zeros((2, pixels, pixels), dtype=np.float32)

    # Compute r and angle for each point
    rmax = np.sqrt(2) * (pixels // 2)

    for i in nb.prange(pixels):
        for j in range(pixels):
            # Calculate radius and angle
            r = np.sqrt(x[i] ** 2 + y[j] ** 2)
            angle = np.arctan2(x[i], y[j])

            # Apply radial correction
            r_piece = r
            if r > cutoff:
                r_piece += (r - cutoff) * corner_shift / (rmax - cutoff)

            # Store transformed coordinates
            coords[0, i, j] = r_piece * np.sin(angle)
            coords[1, i, j] = r_piece * np.cos(angle)

    # Apply shift and crop edges
    return coords[:, edge:-edge, edge:-edge] + pixels // 2


def _process_plane(args):
    """Helper function to process a single z-plane for parallel execution"""
    z, plane, coords, order = args
    warp_coords = np.zeros((2, *coords[0].shape), dtype=np.float32)
    warp_coords[0] = coords[0]
    warp_coords[1] = coords[1]
    return z, map_coordinates(plane, warp_coords, order=order, mode="constant")


def radial_correction_new(
    tile_data: np.ndarray,
    corner_shift: Optional[float] = 5.5,
    frac_cutoff: Optional[float] = 0.5,
    mode: Union[Literal["2d"], Literal["3d"]] = "3d",
    order: int = 1,
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Apply radial correction to a tile with optimized performance.

    Parameters
    ----------
    tile_data : np.ndarray
        The 3D tile data (Z, Y, X) to be corrected.
    corner_shift : Optional[float]
        The amount of radial shift to apply (default is 5.5).
    frac_cutoff : Optional[float]
        Fraction of the radius to begin applying correction (default is 0.5).
    mode : Union[Literal["2d"], Literal["3d"]]
        Processing mode - "2d" for plane-wise processing or "3d" for full volume (default is "3d").
    order : int
        Interpolation order for map_coordinates (default is 1).
    max_workers : Optional[int]
        Maximum number of worker threads for parallel processing (default is None, which uses CPU count).

    Returns
    -------
    np.ndarray
        The corrected tile.
    """
    # Convert to uint16 once at the beginning to avoid repeated conversions
    tile_data = np.asarray(tile_data, dtype=np.uint16)

    edge = ceil(corner_shift / np.sqrt(2)) + 1
    shape = tile_data.shape
    pixels = shape[1]  # Assume square XY plane
    cutoff = pixels * frac_cutoff

    # Pre-compute transformed coordinates using numba
    coords = _compute_coordinates(pixels, cutoff, corner_shift, edge)

    # Calculate new shape after edge cropping
    new_shape = np.array(shape) - [0, edge * 2, edge * 2]

    # Different processing methods based on mode
    if mode == "2d":
        # Process each z-plane separately in parallel
        result = np.zeros(new_shape, dtype=np.uint16)

        # Use ThreadPoolExecutor for parallel processing of z-planes
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(z, tile_data[z], coords, order) for z in range(shape[0])]
            for z, processed_plane in executor.map(
                lambda args: _process_plane(args), tasks
            ):
                result[z] = processed_plane

        return result

    else:  # 3D mode
        # Create full 3D warping coordinates array
        warp_coords = np.zeros((3, *new_shape), dtype=np.float32)

        # Z coordinates remain unchanged
        for z in range(new_shape[0]):
            warp_coords[0, z] = z

        # Apply pre-computed X-Y coordinate transformations to each z-plane
        warp_coords[1] = np.repeat(
            coords[0][np.newaxis, :, :], new_shape[0], axis=0
        )
        warp_coords[2] = np.repeat(
            coords[1][np.newaxis, :, :], new_shape[0], axis=0
        )

        # Process the entire volume at once
        return map_coordinates(
            tile_data, warp_coords, order=order, mode="constant"
        )


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
    tile = da.squeeze(da.from_zarr(zarr_file_loc))
    z_size = tile.shape[2]

    output_radial = None
    data_in_memory = tile.compute()

    print("Z size: ", z_size, " data shape: ", data_in_memory.shape)

    if z_size < 400:
        output_radial = radial_correction(
            data_in_memory, corner_shift, frac_cutoff  # , mode="3d"
        )
    else:
        # output_radial = radial_correction_2d(
        #     data_in_memory, corner_shift, frac_cutoff#, mode="2d"
        # )
        print("2d mode!")
        output_radial = radial_correction_new(
            tile_data=data_in_memory,
            corner_shift=5.5,
            frac_cutoff=0.5,
            mode="2d",
            order=1,
            max_workers=None,
        )

    return output_radial


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


def store_zarrv2(
    image_data,
    stack_name,
    output_path,
    final_chunksize,
    scale_factor=[2, 2, 2],
    n_lvls=5,
    channel_colors=[None],
):

    # Rechunking dask array
    image_data = image_data.rechunk()
    image_data = pad_array_n_d(arr=image_data)

    image_name = stack_name

    print(f"Writing {image_data} from {stack_name} to {output_path}")

    # Creating Zarr dataset
    store = parse_url(path=output_path, mode="w").store
    root_group = zarr.group(store=store)

    # Using 1 thread since is in single machine.
    # Avoiding the use of multithreaded due to GIL

    if np.issubdtype(image_data.dtype, np.integer):
        np_info_func = np.iinfo

    else:
        # Floating point
        np_info_func = np.finfo

    # Getting min max metadata for the dtype
    channel_minmax = [
        (
            np_info_func(image_data.dtype).min,
            np_info_func(image_data.dtype).max,
        )
        for _ in range(image_data.shape[1])
    ]

    # Setting values for SmartSPIM
    # Ideally we would use da.percentile(image_data, (0.1, 95))
    # However, it would take so much time and resources and it is
    # not used that much on neuroglancer
    channel_startend = [(0.0, 350.0) for _ in range(image_data.shape[1])]

    new_channel_group = root_group.create_group(
        name=image_name, overwrite=True
    )

    # Writing OME-NGFF metadata
    write_ome_ngff_metadata(
        group=new_channel_group,
        arr=image_data,
        image_name=image_name,
        n_lvls=n_lvls,
        scale_factors=scale_factor,
        voxel_size=voxel_size,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
        metadata=_get_pyramid_metadata(),
    )

    # performance_report_path = f"{output_path}/report_{stack_name}.html"

    start_time = time.time()
    # Writing zarr and performance report
    # with performance_report(filename=performance_report_path):

    # Writing zarr
    block_shape = list(
        BlockedArrayWriter.get_block_shape(
            arr=image_data, target_size_mb=12800  # 51200,
        )
    )

    # Formatting to 5D block shape
    block_shape = ([1] * (5 - len(block_shape))) + block_shape
    written_pyramid = []
    pyramid_group = None

    for level in range(n_lvls):
        if not level:
            array_to_write = image_data

        else:
            # It's faster to write the scale and then read it back
            # to compute the next scale
            previous_scale = da.from_zarr(pyramid_group, pyramid_group.chunks)
            new_scale_factor = (
                [1] * (len(previous_scale.shape) - len(scale_factor))
            ) + scale_factor

            previous_scale_pyramid, _ = compute_pyramid(
                data=previous_scale,
                scale_axis=new_scale_factor,
                chunks=image_data.chunksize,
                n_lvls=2,
            )
            array_to_write = previous_scale_pyramid[-1]

        logger.info(f"[level {level}]: pyramid level: {array_to_write}")

        # Create the scale dataset
        pyramid_group = new_channel_group.create_dataset(
            name=level,
            shape=array_to_write.shape,
            chunks=array_to_write.chunksize,
            dtype=array_to_write.dtype,
            compressor=writing_options,
            dimension_separator="/",
            overwrite=True,
        )

        # Block Zarr Writer
        BlockedArrayWriter.store(array_to_write, pyramid_group, block_shape)
        written_pyramid.append(array_to_write)


# TODO: Improve performance and zarr writing
def correct_and_save_tile(dataset_loc, output_path, resolution_zyx, scale="0"):
    """
    correct and save a single tile
    """

    tilename = str(Path(output_path).name)

    corner_shift = calculate_corner_shift_from_pixel_size(resolution_zyx[1])
    frac_cutoff = calculate_frac_cutoff_from_pixel_size(resolution_zyx[1])

    LOGGER.info(f"Corner Shift: {corner_shift} pixels")

    start_time = time.time()
    corrected_tile = apply_corr_to_zarr_tile(
        dataset_loc.joinpath(scale), corner_shift, frac_cutoff
    )
    end_time = time.time()
    print(
        f"Time to correct a single tile {end_time - start_time} - New shape {corrected_tile.shape}"
    )

    zarr_loc = f"/results/{tilename}_new"
    zarr.save_array(
        zarr_loc, corrected_tile
    )  # slow... consider doing with dask
    # temp_zarr = zarr.load(zarr_loc)
    # corr_arr = da.from_array(corrected_tile, name=tilename)
    # corr_arr = da.from_zarr(zarr_loc, chunks = (128,256,256))
    # run_multiscale(corr_arr, out_group, resolution_zyx)

    data_process = None
    # DataProcess(
    #     name=ProcessName.IMAGE_RADIAL_CORRECTION,
    #     software_version=__version__,
    #     start_date_time=start_time,
    #     end_date_time=end_time,
    #     input_location=dataset_loc,
    #     output_location=output_path,
    #     code_version=__version__,
    #     code_url=__url__,
    #     parameters={
    #         'corner_shift': corner_shift,
    #         'frac_cutoff': frac_cutoff
    #     },
    # )

    return data_process


def get_voxelsize_from_xml(stitching_xml_path: str) -> List[str]:
    """
    Extract zarr path and voxel size information from a stitching XML file.

    Parameters
    ----------
    stitching_xml_path: str
        Path to the stitching XML file

    Returns
    -------
    List
        list of voxel sizes in ZYX order

    Raises
    ------
    FileNotFoundError: If the XML file doesn't exist
    ValueError: If required elements are not found in the XML
    """
    if not Path(stitching_xml_path).exists():
        raise FileNotFoundError(f"{stitching_xml_path} path does not exist.")

    tree = ET.parse(stitching_xml_path)
    root = tree.getroot()

    # Find voxel size
    xyz_voxelsize = None
    for elem in root.iter("voxelSize"):
        xyz_voxelsize = elem.findtext("size")
        if xyz_voxelsize is not None:
            break

    if xyz_voxelsize is None:
        raise ValueError("No 'voxelSize/size' element found in the XML file.")

    # Convert from XYZ to ZYX order
    zyx_voxelsize_list = xyz_voxelsize.split(" ")
    zyx_voxelsize_list.reverse()

    return zyx_voxelsize_list


def main():
    """
    Radial correction to multiple tiles
    based on provided YMLs.
    """
    data_folder = Path(
        os.path.abspath("../data/HCR_785830_2025-03-19_17-00-00/SPIM")
    )
    results_folder = Path(os.path.abspath("../results"))

    stitching_xml_path = data_folder.joinpath(
        "derivatives/stitching_single_channel.xml"
    )

    zyx_voxel_size = get_voxelsize_from_xml(
        stitching_xml_path=stitching_xml_path
    )

    data_processes = []
    zarr_paths = natsorted(list(data_folder.glob("*.zarr")))

    for zarr_path in zarr_paths:
        output_path = results_folder.joinpath(zarr_path.stem)
        data_process = correct_and_save_tile(
            dataset_loc=zarr_path,
            output_path=output_path,
            resolution_zyx=zyx_voxel_size,
        )
        s

    # utils.generate_processing(
    #     data_processes=data_processes,
    #     dest_processing=results_folder,
    #     processor_full_name=__maintainers__[0],
    #     pipeline_version=__pipeline_version__,
    #     prefix='radial_correction'
    # )


if __name__ == "__main__":
    main()
