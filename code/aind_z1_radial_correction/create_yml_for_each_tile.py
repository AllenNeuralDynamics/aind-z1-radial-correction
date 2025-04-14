import glob
import json
import os
from pathlib import Path
from typing import Union
from xml.etree import ElementTree as ET

import yaml
from tqdm import tqdm

# Find new processing location in s3 from processing_manifest.json


def _load_manifest(DATA_FOLDER):
    """Load the processing manifest JSON file"""
    manifest_path = list(
        Path(DATA_FOLDER).glob("derived/processing_manifest.json")
    )

    if not len(manifest_path):
        print("Didn't find pipeline processing manifest")
        manifest_path = list(
            Path(DATA_FOLDER).glob("*/derived/processing_manifest.json")
        )
        if not len(manifest_path):
            raise FileNotFoundError(
                "No capsule processing_manifest.json was found!"
            )

    print(f"Manifest_path {manifest_path}")

    try:
        with open(manifest_path[0], "r") as f:
            manifest = json.load(f)
            print(
                f"Loaded manifest with channels: {manifest.get('spot_channels', [])}"
            )
    except FileNotFoundError:
        manifest = None
        raise FileNotFoundError(
            f"Processing manifest not found at {manifest_path}"
        )

    return manifest


def get_output_path(manifest):
    output_path = manifest["s3_location"]
    return output_path


#                           Original Scheduler
#################################################################################################


def write_output_yml(
    dataset_loc, yml_file_name, input_path, resolution_zyx=(1, 0.256, 0.256)
):
    """Writes output yaml files that act as inputs to diSPIM_multiscale_scheduler

    Parameters:
    -----------
    Dataset_loc: str
    The s3 path to the dataset asset

    resolution: tuple TODO - should this be a list?
    ZYX resolution in um/pixel

    Returns:
    yaml_loc

    """
    # test dataset_loc exists

    # test resolution_zyx is a tuple of size 3

    yml_dict = {
        "input_path": str(input_path),
        "output_path": str(dataset_loc),
        "resolution_zyx": list(resolution_zyx),
    }

    # yml_file_name = f"/results/"

    # write yml file out
    os.makedirs(Path(yml_file_name).parent, exist_ok=True)
    with open(yml_file_name, "w") as file:
        yaml.dump(yml_dict, file)


def write_yml_for_worker(dataset_loc, output_path):
    """correct and save a single tile"""

    # print(f'corrected_tile_shape {corrected_tile.shape}')
    # Save the corrected tile

    output_path = str(output_path).rstrip("/")

    s3_path_yml_loc = "/scratch/radial_correction/s3_path.yml"

    with open(s3_path_yml_loc, "r") as file:
        yml_file = yaml.safe_load(file)

    s3_path = yml_file["s3_path"]
    zyx_voxel_size_um = yml_file["zyx_voxel_size_um"]

    dataset_pre_name = s3_path.split("/")[3:][0]  # ignore the first three '/'

    dataset_name = dataset_pre_name

    if "processed" in dataset_name:
        dataset_name = dataset_name.split("_processed")[0]

    to_read_loc = f"s3://aind-open-data/{dataset_name}/SPIM.ome.zarr/{dataset_loc.parent.stem}.zarr"
    to_save_loc = (
        f"{output_path}/radial_correction/{dataset_loc.parent.stem}.zarr"
    )

    yml_file_name = (
        f"/results/{dataset_loc.parent.stem}_to_do_radial_correction.yml"
    )

    write_output_yml(
        to_save_loc, yml_file_name, to_read_loc, zyx_voxel_size_um
    )

    return


def prepare_to_correct_all_tiles(dataset_loc, debug=False):
    """Correct all tiles in a zarr file"""

    # Load the zarr file
    # dataset_loc = '/data/HCR_BL6-000_2023-06-07_00-00-00/SPIM.ome.zarr'

    print(f"dataset_loc {Path(dataset_loc).parent}")
    manifest = _load_manifest(Path(dataset_loc).parent)
    print(f"manifest {manifest}")

    output_path = get_output_path(manifest)

    list_of_tiles = Path(dataset_loc).glob("*.zarr")
    # print(f'list of tiles {list(list_of_tiles)}')

    # Correct each tile
    for tile_loc in tqdm(list_of_tiles):
        tile_loc = Path(tile_loc).joinpath("0")
        try:
            # print(f'correcting tile{tile_loc}')
            write_yml_for_worker(tile_loc, output_path)
        except:
            print(f"Error correcting {tile_loc}")
            continue
        if debug:
            break

    print(f"Processed all tiles")

    return


def write_s3_path_as_yml(xml_file_loc: str, output_yml_file_loc: str):
    """Gets the s3 location of the dataset and saves it in a yml file"""
    tree = ET.parse(xml_file_loc)
    root = tree.getroot()

    for elem in root.iter("zarr"):
        loc_path = elem.text

    for elem in root.iter("voxelSize"):
        xyz_voxelsize = elem.findtext("size")
        break

    zyx_voxelsize_list = xyz_voxelsize.split(" ")
    zyx_voxelsize_list.reverse()  # does in place

    # check if this s3_path exists

    # separate on '/', appending s3 prefix
    s3_folder = loc_path.split("/")[2:]  # ignore the first two '/'
    s3_path = "s3://aind-open-data/" + s3_folder[0] + "/" + s3_folder[1] + "/"

    out_dict = {"s3_path": s3_path, "zyx_voxel_size_um": zyx_voxelsize_list}

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


############################################################################
#                           Run Radial Correction
############################################################################


if __name__ == "__main__":
    # test_radial_correction()

    # dataset_loc = '/data/HCR_BL6-000_2023-06-07_00-00-00/SPIM.ome.zarr'

    print(f" cwd: {os.getcwd()}")
    data_folder = os.path.abspath("../data")
    print(f"data folder  {data_folder}")

    try:  # run from pipeline
        run_from_capsule = False  # run from pipeline
        dataset_loc = glob.glob("../data/**/SPIM.ome.zarr/")
        print(f"dataset_loc {dataset_loc}")
        xml_loc = glob.glob("../data/**/*.xml")
        print(f"xml_loc {xml_loc}")
        if len(xml_loc) > 0:
            local_xml_loc = [
                xml_loc_str
                for xml_loc_str in xml_loc
                if "remote" not in xml_loc_str
            ]
            xml_loc = str(local_xml_loc[0])

        if len(dataset_loc) > 0:
            dataset_loc = Path(dataset_loc[0]).as_posix()
            # xml_loc = str(glob.glob(Path(dataset_loc).parent.as_posix()+'*.xml')[0])

        else:
            dataset_loc = glob.glob("../data/SPIM.ome.zarr")
            print(f"dataset_loc {dataset_loc}")
            if len(dataset_loc) > 0:
                dataset_loc = Path(dataset_loc[0]).as_posix()

            else:
                dataset_loc = "../data/SPIM.ome.zarr"
                print(f"dataset_loc exists {Path(dataset_loc).exists()}")

            xml_loc = glob.glob("../data/*.xml")
            print(f"xml_loc {xml_loc}")
            if len(xml_loc) > 0:
                local_xml_loc = [
                    xml_loc_str
                    for xml_loc_str in xml_loc
                    if "remote" not in xml_loc_str
                ]
                xml_loc = str(local_xml_loc[0])

    except:  # run from capsule

        dataset_loc = glob.glob("/data/**/SPIM.ome.zarr")
        print(f"dataset_loc {dataset_loc}")
        if len(dataset_loc) > 0:
            dataset_loc = Path(dataset_loc[0]).as_posix()

        run_from_capsule = True

        xml_loc = glob.glob("/data/**/*.xml")
        print(f"xml path {xml_loc}")
        if len(xml_loc) == 0:
            xml_loc = (
                Path(dataset_loc)
                .parent.joinpath("stitching_single_channel.xml")
                .as_posix()
            )
        else:
            local_xml_loc = [
                xml_loc_str
                for xml_loc_str in xml_loc
                if "remote" not in xml_loc_str
            ]
            xml_loc = str(local_xml_loc[0])

    yml_s3_path_loc = "/results/s3_path.yml"
    write_s3_path_as_yml(xml_loc, yml_s3_path_loc)

    if run_from_capsule == True:
        debug = True
    else:
        debug = False  # TEMP SWITCH BACK
    # add a debug switch if run from capsule, false if run from pipeline

    results_folder = os.path.abspath("../results")

    prepare_to_correct_all_tiles(dataset_loc, debug=debug)
