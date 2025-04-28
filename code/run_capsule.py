"""
Runs radial correction in a set of tiles provided
to the data directory
"""

import os

from aind_z1_radial_correction import radial_correction, utils


def run():
    """
    Main run file in Code Ocean
    """
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")
    acquisition_path = f"{data_folder}/acquisition.json"
    radial_correction_parameters_path = (
        f"{data_folder}/radial_correction_parameters.json"
    )

    required_input_elements = [
        acquisition_path,
        radial_correction_parameters_path,
    ]

    missing_files = utils.validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    radial_correction_parameters = utils.read_json_as_dict(
        radial_correction_parameters_path
    )

    tilenames = radial_correction_parameters.get("tilenames", [])
    worker_id = radial_correction_parameters.get("worker_id", None)

    print(f"Worker ID: {worker_id} processing {len(tilenames)} tiles!")

    if len(tilenames):
        radial_correction.main(
            data_folder=data_folder,
            results_folder=results_folder,
            acquisition_path=acquisition_path,
            tilenames=tilenames,
        )

    else:
        print(f"Nothing to do! Tilenames: {tilenames}")


if __name__ == "__main__":
    run()
