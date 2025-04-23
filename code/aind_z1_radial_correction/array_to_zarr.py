"""
Writes a multiscale zarrv3 dataset from an array
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorstore as ts
from aind_hcr_data_transformation.compress.czi_to_zarr import (
    create_downsample_dataset,
    create_spec,
)
from aind_hcr_data_transformation.compress.omezarr_metadata import (
    _get_pyramid_metadata,
    write_ome_ngff_metadata,
)
from aind_hcr_data_transformation.utils.utils import (
    pad_array_n_d,
    write_json,
)
from numpy.typing import ArrayLike


def convert_array_to_zarr(
    array: ArrayLike,
    shard_size: List[int],
    chunk_size: List[int],
    output_path: str,
    voxel_size: List[float],
    n_lvls: Optional[int] = 6,
    scale_factor: Optional[List[int]] = [2, 2, 2],
    bucket_name: Optional[str] = None,
    compressor_kwargs: Optional[Dict] = {
        "cname": "zstd",
        "clevel": 3,
        "shuffle": "shuffle",
    },
):
    """
    Converts an array to zarr format

    Parameters
    ----------
    array: ArrayLike
        Array to convert to zarr v3

    shard_size: List[int]
        Shard size

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

    bucket_name: Optional[str]
        Bucket name
        Default: None

    compressor_kwargs: Optional[Dict]
        Compressor parameters
        Default: {"cname": "zstd", "clevel": 3, "shuffle": "shuffle"}
    """
    dataset_shape = tuple(i for i in array.shape if i != 1)
    extra_axes = (1,) * (5 - len(dataset_shape))
    dataset_shape = extra_axes + dataset_shape

    shard_size = ([1] * (5 - len(shard_size))) + shard_size
    chunk_size = ([1] * (5 - len(chunk_size))) + chunk_size

    # Getting channel color
    channel_colors = None
    stack_name = Path(output_path).stem

    print(f"Writing from {stack_name} to {output_path} bucket {bucket_name}")

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

    # Setting values for array
    # Ideally we would use da.percentile(image_data, (0.1, 95))
    # However, it would take so much time and resources and it is
    # not used that much on neuroglancer
    channel_startend = [(0.0, 550.0) for _ in range(dataset_shape[1])]

    # Writing OME-NGFF metadata
    scale_factor = [float(s) for s in scale_factor]
    voxel_size = [float(v) for v in voxel_size]

    multiscale_zarr_json = write_ome_ngff_metadata(
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
        chunk_size=chunk_size,
    )

    # Full resolution spec
    spec = create_spec(
        output_path=output_path,
        bucket_name=bucket_name,
        data_shape=dataset_shape,
        data_dtype=array.dtype.name,
        shard_shape=shard_size,
        chunk_shape=chunk_size,
        zyx_resolution=voxel_size,
        compressor_kwargs=compressor_kwargs,
    )

    dataset = ts.open(spec).result()

    dataset.write(pad_array_n_d(array)).result()

    for level in range(n_lvls):
        asyncio.run(
            create_downsample_dataset(
                dataset_path=output_path,
                start_scale=level,
                downsample_factor=scale_factor,
                compressor_kwargs=compressor_kwargs,
                bucket_name=bucket_name,
            )
        )

    # Writes top level json
    write_json(
        bucket_name=bucket_name,
        output_path=output_path,
        json_data=multiscale_zarr_json,
    )


if __name__ == "__main__":
    import dask.array as da

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
