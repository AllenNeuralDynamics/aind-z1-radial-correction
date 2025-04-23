"""
Runs radial correction in a set of tiles provided
to the data directory
"""

import os

from aind_z1_radial_correction import radial_correction


def run():
    """
    Main run file in Code Ocean
    """
    data_folder = os.path.abspath(
        "../data/HCR_785830_2025-03-19_17-00-00/SPIM"
    )
    results_folder = os.path.abspath("../results")
    stitching_xml_path = (
        f"{data_folder}/derivatives/stitching_single_channel.xml"
    )

    tilenames = [
        "Tile_X_0000_Y_0001_Z_0000_ch_488.ome.zarr",
        # "Tile_X_0000_Y_0002_Z_0000_ch_488.ome.zarr",
    ]
    radial_correction.main(
        data_folder=data_folder,
        results_folder=results_folder,
        stitching_xml_path=stitching_xml_path,
        tilenames=tilenames,
    )


if __name__ == "__main__":
    run()
