"""
Utility functions for the radial correction step
"""

import os
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml


def write_s3_path_as_yml(xml_file_loc: str, output_folder: str):
    """
    Gets the s3 location of the dataset and saves it in a yml file

    Parameters
    ----------
    xml_file_loc: str
        XML file location

    output_folder: str
        Output XML location

    """
    tree = ET.parse(xml_file_loc)

    root = tree.getroot()

    for elem in root.iter("zarr"):
        loc_path = elem.text

    # check if this s3_path exists

    # separate on '/', appending s3 prefix
    s3_folder = loc_path.split("/")[2:]  # ignore the first two '/'
    s3_path = "s3://aind-open-data/" + s3_folder[0] + "/" + s3_folder[1] + "/"

    out_dict = {"s3_path": s3_path}

    os.makedirs(f"{output_folder}/radial_correction/", exist_ok=True)

    # pass on dataset name for pipeline
    with open(f"{output_folder}/radial_correction/s3_path.yml", "w") as file:
        yaml.dump(out_dict, file)


def read_to_do_yml(yml_loc: str):
    """
    Reads a yaml and returns input path,
    output path and resolution in zyx.

    Parameters
    ----------
    yml_loc: str
        Location of the yaml file

    Raises
    ------
    FileNotFoundError:
        If the yaml file does not exist.

    Returns
    -------
    Tuple[str, str, list]
        str:
            Input path of the tile to correct

        str:
            Output path of the tile to correct

        List[float]:
            ZYX resolution
    """
    yml_loc = Path(yml_loc)

    if not yml_loc.exists():
        raise FileNotFoundError(
            f"Please, provide a path that exists. Given: {yml_loc}"
        )

    with open(yml_loc, "r") as file:
        yml_file = yaml.safe_load(file)

    input_path = yml_file["input_path"]
    output_path = yml_file["output_path"]
    temp = yml_file["resolution_zyx"]

    resolution_zyx_um = []
    for res in temp:
        resolution_zyx_um.append(float(res))

    return input_path, output_path, resolution_zyx_um
