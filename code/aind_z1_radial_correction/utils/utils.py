"""
Utility functions for the radial correction step
"""

import json
import os
from pathlib import Path
from typing import List

from aind_data_schema.core.processing import (
    DataProcess,
    PipelineProcess,
    Processing,
)


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: str,
    prefix: str,
    processor_full_name: str,
    pipeline_version: str,
):
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    processor_full_name: str
        Person in charged of running the pipeline
        for this data asset

    pipeline_version: str
        Terastitcher pipeline version

    """
    # flake8: noqa: E501
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="",
        note="Metadata for radial correction",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing only contains metadata about radial correction \
            and needs to be compiled with other steps at the end",
    )

    processing.write_standard_file(
        output_directory=dest_processing, prefix=prefix
    )


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def get_voxel_resolution(acquisition_path: Path) -> List[float]:
    """
    Get the voxel resolution from an acquisition.json file.

    Parameters
    ----------
    acquisition_path: Path
        Path to the acquisition.json file.
    Returns
    -------
    List[float]
        Voxel resolution in the format [z, y, x].
    """

    if not Path(acquisition_path).is_file():
        raise FileNotFoundError(
            f"acquisition.json file not found at: {acquisition_path}"
        )

    acquisition_config = read_json_as_dict(acquisition_path)

    if not acquisition_config:
        raise ValueError(
            f"acquisition.json file is empty or invalid: {acquisition_path}"
        )

    # Grabbing a tile with metadata from acquisition - we assume all
    # dataset was acquired with the same resolution
    tile_coord_transforms = acquisition_config["tiles"][0][
        "coordinate_transformations"
    ]

    scale_transform = [
        x["scale"] for x in tile_coord_transforms if x["type"] == "scale"
    ][0]

    x = float(scale_transform[0])
    y = float(scale_transform[1])
    z = float(scale_transform[2])

    return [z, y, x]


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs
