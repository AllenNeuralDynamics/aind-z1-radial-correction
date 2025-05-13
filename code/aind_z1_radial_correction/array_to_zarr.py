"""
Writes a multiscale zarrv3 dataset from an array
"""

from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import dask.array as da
import numpy as np
import zarr
from aind_hcr_data_transformation.compress.czi_to_zarr import (
    _get_pyramid_metadata,
    compute_pyramid,
    write_ome_ngff_metadata,
)
from aind_hcr_data_transformation.compress.zarr_writer import (
    BlockedArrayWriter,
)
from aind_hcr_data_transformation.utils.utils import pad_array_n_d
from numcodecs.blosc import Blosc
from numpy.typing import ArrayLike
from ome_zarr.io import parse_url
from zarr.storage import FSStore


def is_s3_path(path: str) -> bool:
    """
    Checks if a path is an s3 path

    Parameters
    ----------
    path: str
        Provided path

    Returns
    -------
    bool
        True if it is a S3 path,
        False if not.
    """
    parsed = urlparse(str(path))
    return parsed.scheme == "s3"


def get_parent_path(path: str) -> str:
    """
    Gets parent path

    Parameters
    ----------
    path: str
        Provided path

    Returns
    -------
    str
        Parent path
    """
    parsed = urlparse(path)
    if parsed.scheme == "s3":
        # Remove the last part of the S3 key
        parts = parsed.path.strip("/").split("/")
        parent_key = "/".join(parts[:-1])
        return f"s3://{parsed.netloc}/{parent_key}"
    else:
        # Local path fallback
        return str(Path(path).parent)


def convert_array_to_zarr(
    array: ArrayLike,
    chunk_size: List[int],
    output_path: str,
    voxel_size: List[float],
    n_lvls: Optional[int] = 6,
    scale_factor: Optional[List[int]] = [2, 2, 2],
    compressor_kwargs: Optional[Dict] = {
        "cname": "zstd",
        "clevel": 3,
        "shuffle": Blosc.SHUFFLE,
    },
    target_size_mb: Optional[int] = 24000,
):
    """
    Converts an array to zarr format

    Parameters
    ----------
    array: ArrayLike
        Array to convert to zarr v3

    chunk_size: List[int]
        Chunksize in each shard

    output_path: str
        Output path. It must contain the ome.zarr
        extension attached.

    voxel_size: List[float]
        Voxel size

    n_lvls: Optional[int]
        Number of downsampled levels to write.
        Default: 6

    scale_factor: Optional[List[int]]
        Scaling factor per axis. Default: [2, 2, 2]

    compressor_kwargs: Optional[Dict]
        Compressor parameters
        Default: {"cname": "zstd", "clevel": 3, "shuffle": "shuffle"}
    """
    array = pad_array_n_d(array)
    dataset_shape = tuple(i for i in array.shape if i != 1)
    extra_axes = (1,) * (5 - len(dataset_shape))
    dataset_shape = extra_axes + dataset_shape
    chunk_size = ([1] * (5 - len(chunk_size))) + chunk_size

    compressor = Blosc(
        cname=compressor_kwargs["cname"],
        clevel=compressor_kwargs["clevel"],
        shuffle=compressor_kwargs["shuffle"],
        blocksize=0,
    )

    # Getting channel color
    channel_colors = None
    stack_name = Path(output_path).name
    parent_path = get_parent_path(output_path)
    # Creating Zarr dataset in s3 or local
    if is_s3_path(output_path):
        store = FSStore(parent_path, mode="w", dimension_separator="/")
    else:
        store = parse_url(path=parent_path, mode="w").store

    root_group = zarr.group(store=store)

    # Using 1 thread since is in single machine.
    # Avoiding the use of multithreaded due to GIL
    if np.issubdtype(array.dtype, np.integer):
        np_info_func = np.iinfo

    else:
        # Floating point
        np_info_func = np.finfo

    # Getting min max metadata for the dtype
    channel_minmax = [
        (
            np_info_func(array.dtype).min,
            np_info_func(array.dtype).max,
        )
        for _ in range(dataset_shape[1])
    ]

    # Setting values for CZI
    # Ideally we would use da.percentile(image_data, (0.1, 95))
    # However, it would take so much time and resources and it is
    # not used that much on neuroglancer
    channel_startend = [(0.0, 550.0) for _ in range(dataset_shape[1])]

    # Writing OME-NGFF metadata
    scale_factor = [int(s) for s in scale_factor]
    voxel_size = [float(v) for v in voxel_size]

    new_channel_group = root_group.create_group(
        name=stack_name, overwrite=True
    )

    # Writing OME-NGFF metadata
    write_ome_ngff_metadata(
        group=new_channel_group,
        arr_shape=dataset_shape,
        image_name=stack_name,
        n_lvls=n_lvls,
        scale_factors=scale_factor,
        voxel_size=voxel_size,
        channel_names=None,
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
        metadata=_get_pyramid_metadata(),
        final_chunksize=chunk_size,
    )

    # Writing first multiscale by default
    pyramid_group = new_channel_group.create_dataset(
        name="0",
        shape=dataset_shape,
        chunks=chunk_size,
        dtype=array.dtype,
        compressor=compressor,
        dimension_separator="/",
        overwrite=True,
    )

    # Writing multiscales
    previous_scale = da.from_array(array, pyramid_group.chunks)

    block_shape = list(
        BlockedArrayWriter.get_block_shape(
            arr=previous_scale,
            target_size_mb=target_size_mb,
            chunks=chunk_size,
        )
    )
    block_shape = extra_axes + tuple(block_shape)

    for level in range(0, n_lvls):
        if not level:
            array_to_write = previous_scale

        else:
            previous_scale = da.from_zarr(pyramid_group, pyramid_group.chunks)
            new_scale_factor = (
                [1] * (len(previous_scale.shape) - len(scale_factor))
            ) + scale_factor

            previous_scale_pyramid, _ = compute_pyramid(
                data=previous_scale,
                scale_axis=new_scale_factor,
                chunks=chunk_size,
                n_lvls=2,
            )
            array_to_write = previous_scale_pyramid[-1]

            print(f"[level {level}]: pyramid level: {array_to_write.shape}")

            pyramid_group = new_channel_group.create_dataset(
                name=str(level),
                shape=array_to_write.shape,
                chunks=chunk_size,
                dtype=array_to_write.dtype,
                compressor=compressor,
                dimension_separator="/",
                overwrite=True,
            )

        BlockedArrayWriter.store(array_to_write, pyramid_group, block_shape)


if __name__ == "__main__":
    BASE_PATH = "/data"
    tilename = "Tile_X_0000_Y_0011_Z_0000_ch_488.ome.zarr"
    test_dataset = f"HCR_785830_2025-03-19_17-00-00/SPIM/{tilename}"
    scale = "0"

    dataset = da.from_zarr(f"{BASE_PATH}/{test_dataset}/{scale}").compute()
    convert_array_to_zarr(
        array=dataset,
        voxel_size=[1.0] * 3,
        shard_size=[512] * 3,
        chunk_size=[128] * 3,
        output_path="/results/test.ome.zarr",
    )
