"""Tests for the darkfield handling and evaluation of radial correction."""

import json
import tempfile
import unittest
from math import ceil
from pathlib import Path

import numpy as np
import tensorstore as ts
from aind_z1_radial_correction.evaluation import (
    evaluate_tile,
    plot_evaluation,
    radial_profile,
    ring_artifact,
)
from aind_z1_radial_correction.radial_correction import (
    apply_corr_to_zarr_tile,
    radial_correction,
)
from aind_z1_radial_correction.utils import utils

CORNER_SHIFT = 4.3
FRAC_CUTOFF = 0.5
TILE = "Tile_X_0000_Y_0000_Z_0000_ch_405.ome.zarr"


def write_tile(folder, raw, flatfield, darkfield, tile_name=TILE):
    """Write a uint16 tile and its calibration as the pipeline expects."""
    tile_path = Path(folder) / tile_name
    array = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(tile_path)},
            "path": "0",
            "metadata": {
                "dtype": "<u2",
                "shape": [1, 1, *raw.shape],
                "chunks": [1, 1, *raw.shape],
            },
        },
        create=True,
        delete_existing=True,
    ).result()
    array.write(raw[None, None]).result()

    flats_dir = Path(folder) / "flats"
    flats_dir.mkdir()
    flat_path = flats_dir / "flatfield_ch_405_level0.npy"
    dark_path = flats_dir / "darkfield_ch_405_level0.npy"
    np.save(flat_path, flatfield.astype(np.float32))
    np.save(dark_path, darkfield.astype(np.float32))
    metadata = {
        "guard": {"ok": True},
        "upsampled": {
            "level": 0,
            "shape": list(flatfield.shape),
            "flatfield_path": str(flat_path),
            "darkfield_path": str(dark_path),
        },
    }
    (flats_dir / "flatfield_ch_405.json").write_text(json.dumps(metadata))
    return str(tile_path), str(flats_dir)


def dim_tile(n=128, z=8, sigma=3.5, seed=0):
    """Dim tile: per-pixel dark map plus zero-mean noise, no signal."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    darkfield = 88.0 + 4.0 * (yy + xx) / (2 * n)
    flatfield = 0.8 + 0.4 * np.exp(
        -((yy - n / 2) ** 2 + (xx - n / 2) ** 2) / n**2
    )
    raw = np.rint(darkfield + rng.normal(0, sigma, (z, n, n)))
    return raw.astype(np.uint16), flatfield, darkfield


class TestDarkfieldAddedBack(unittest.TestCase):
    """The darkfield is added back before the uint16 conversion."""

    def setUp(self):
        """Write a dim synthetic tile to a temporary folder."""
        self.tmp = tempfile.TemporaryDirectory()
        self.raw, self.flat, self.dark = dim_tile()
        self.tile, self.flats = write_tile(
            self.tmp.name, self.raw, self.flat, self.dark
        )

    def tearDown(self):
        """Remove the temporary folder."""
        self.tmp.cleanup()

    def run_tile(self, **kwargs):
        """Run the package entry point on the synthetic tile."""
        return apply_corr_to_zarr_tile(
            self.tile,
            scale="0",
            corner_shift=CORNER_SHIFT,
            frac_cutoff=FRAC_CUTOFF,
            driver="zarr",
            flats_dir=self.flats,
            **kwargs,
        )

    def test_output_is_uint16_and_cropped(self):
        """Output keeps the input dtype and the warp's edge crop."""
        out = self.run_tile()
        edge = ceil(CORNER_SHIFT / np.sqrt(2)) + 1
        n = self.raw.shape[-1] - 2 * edge
        self.assertEqual(out.dtype, np.uint16)
        self.assertEqual(out.shape, (self.raw.shape[0], n, n))

    def test_no_voxel_clipped_at_zero(self):
        """Dim data stays near the camera offset instead of piling at 0."""
        out = self.run_tile()
        self.assertEqual(int((out == 0).sum()), 0)
        self.assertGreater(float(out.mean()), 80.0)

    def test_matches_float_pipeline(self):
        """Output equals rint(warp((raw - dark) / flat) + dark), clipped."""
        out = self.run_tile()
        corrected = (self.raw.astype(np.float32) - self.dark) / self.flat
        warped = radial_correction(
            corrected.astype(np.float32),
            corner_shift=CORNER_SHIFT,
            frac_cutoff=FRAC_CUTOFF,
            mode="2d",
            order=1,
        )
        edge = (self.raw.shape[-1] - out.shape[-1]) // 2
        dark = self.dark[edge:-edge, edge:-edge].astype(np.float32)
        expected = np.clip(np.rint(warped + dark), 0, 65535)
        np.testing.assert_array_equal(out, expected.astype(np.uint16))

    def test_darkfield_aligned_with_output(self):
        """A tile equal to its dark map comes back as the cropped map."""
        raw = np.repeat(np.rint(self.dark)[None], 2, axis=0).astype(np.uint16)
        dark = np.rint(self.dark)
        with tempfile.TemporaryDirectory() as folder:
            tile, flats = write_tile(folder, raw, self.flat, dark)
            out = apply_corr_to_zarr_tile(
                tile,
                scale="0",
                corner_shift=CORNER_SHIFT,
                frac_cutoff=FRAC_CUTOFF,
                driver="zarr",
                flats_dir=flats,
            )
        edge = (raw.shape[-1] - out.shape[-1]) // 2
        np.testing.assert_array_equal(
            out[0], dark[edge:-edge, edge:-edge].astype(np.uint16)
        )

    def test_qc_figure_written(self):
        """A QC figure is saved when a path is given."""
        qc_path = Path(self.tmp.name) / "qc.png"
        self.run_tile(qc_path=str(qc_path))
        self.assertTrue(qc_path.exists())
        self.assertGreater(qc_path.stat().st_size, 0)

    def test_saturation_is_reported(self):
        """Values pushed above uint16 by the flatfield are logged."""
        n = 64
        raw = np.full((2, n, n), 40000, dtype=np.uint16)
        flat = np.full((n, n), 0.5)
        dark = np.full((n, n), 90.0)
        with tempfile.TemporaryDirectory() as folder:
            tile, flats = write_tile(folder, raw, flat, dark)
            with self.assertLogs(level="WARNING") as logs:
                out = apply_corr_to_zarr_tile(
                    tile,
                    scale="0",
                    corner_shift=CORNER_SHIFT,
                    frac_cutoff=FRAC_CUTOFF,
                    driver="zarr",
                    flats_dir=flats,
                )
        self.assertTrue(any("exceed 65535" in m for m in logs.output))
        self.assertEqual(int(out.max()), 65535)

    def test_partial_calibration_raises(self):
        """Only some calibration files present is an error."""
        (Path(self.flats) / "darkfield_ch_405_level0.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_tile()


class TestWithoutCalibration(unittest.TestCase):
    """Without flatfield and darkfield only the radial correction runs."""

    def setUp(self):
        """Write a dim tile with an empty calibration folder."""
        self.tmp = tempfile.TemporaryDirectory()
        self.raw, flat, dark = dim_tile()
        self.tile, flats = write_tile(self.tmp.name, self.raw, flat, dark)
        for path in Path(flats).iterdir():
            path.unlink()
        self.flats = flats

    def tearDown(self):
        """Remove the temporary folder."""
        self.tmp.cleanup()

    def expected(self):
        """Radial correction of the raw tile, rounded and clipped."""
        warped = radial_correction(
            self.raw.astype(np.float32),
            corner_shift=CORNER_SHIFT,
            frac_cutoff=FRAC_CUTOFF,
            mode="2d",
            order=1,
        )
        return np.clip(np.rint(warped), 0, 65535).astype(np.uint16)

    def run_tile(self, flats_dir, **kwargs):
        """Run the package entry point with the given calibration folder."""
        return apply_corr_to_zarr_tile(
            self.tile,
            scale="0",
            corner_shift=CORNER_SHIFT,
            frac_cutoff=FRAC_CUTOFF,
            driver="zarr",
            flats_dir=flats_dir,
            **kwargs,
        )

    def test_empty_calibration_folder(self):
        """No files for the channel: radial correction of the raw data."""
        out = self.run_tile(self.flats)
        np.testing.assert_array_equal(out, self.expected())
        self.assertEqual(int((out == 0).sum()), 0)

    def test_no_calibration_folder(self):
        """flats_dir=None: radial correction of the raw data."""
        np.testing.assert_array_equal(self.run_tile(None), self.expected())

    def test_tile_name_without_channel(self):
        """A name without a channel cannot match calibration: radial only."""
        with tempfile.TemporaryDirectory() as folder:
            tile, flats = write_tile(
                folder,
                self.raw,
                np.ones(self.raw.shape[1:]),
                np.zeros(self.raw.shape[1:]),
                tile_name="Tile_X_0000_Y_0000_Z_0000.ome.zarr",
            )
            out = apply_corr_to_zarr_tile(
                tile,
                scale="0",
                corner_shift=CORNER_SHIFT,
                frac_cutoff=FRAC_CUTOFF,
                driver="zarr",
                flats_dir=flats,
            )
        np.testing.assert_array_equal(out, self.expected())

    def test_qc_figure_without_calibration(self):
        """The QC figure is written with only raw and radial stages."""
        qc_path = Path(self.tmp.name) / "qc.png"
        self.run_tile(None, qc_path=str(qc_path))
        self.assertTrue(qc_path.exists())


class TestInputsAndMetadata(unittest.TestCase):
    """Input checks and processing metadata."""

    def test_non_square_tile_raises(self):
        """Radial correction needs a square XY plane."""
        with self.assertRaises(ValueError):
            radial_correction(np.zeros((2, 64, 80), np.float32))

    def test_saturation_metric(self):
        """The saturated fraction is reported in the QC metrics."""
        raw, flat, dark = dim_tile(n=64, z=2)
        edge = ceil(CORNER_SHIFT / np.sqrt(2)) + 1
        corrected = raw[:, edge:-edge, edge:-edge]
        evaluation = evaluate_tile(
            raw, corrected, flat, dark, 0.5, saturated_fraction=0.25
        )
        self.assertEqual(
            evaluation["metrics"]["voxels saturated at max (%)"], 25.0
        )

    def test_processing_written_locally_without_s3(self):
        """Without an S3 destination processing.json is written locally."""
        from datetime import datetime, timezone

        from aind_data_schema.components.identifiers import Code
        from aind_data_schema.core.processing import (
            DataProcess,
            ProcessStage,
        )
        from aind_data_schema_models.process_names import ProcessName

        now = datetime.now(timezone.utc)
        process = DataProcess(
            process_type=ProcessName.IMAGE_RADIAL_CORRECTION,
            stage=ProcessStage.PROCESSING,
            code=Code(url="https://github.com/AllenNeuralDynamics"),
            experimenters=["test"],
            start_date_time=now,
            end_date_time=now,
        )
        with tempfile.TemporaryDirectory() as folder:
            utils.generate_processing([process], folder, None)
            written = json.loads(
                (Path(folder) / "processing.json").read_text()
            )
        self.assertEqual(len(written["data_processes"]), 1)

    def test_s3_parameters_need_a_bucket(self):
        """S3 input or output without a bucket fails before processing."""
        from run_capsule import _validate_s3_parameters

        with self.assertRaises(ValueError):
            _validate_s3_parameters(None, True, None)
        with self.assertRaises(ValueError):
            _validate_s3_parameters(None, False, "dataset/SPIM")
        _validate_s3_parameters(None, False, None)
        _validate_s3_parameters("bucket", True, "dataset/SPIM")


class TestEvaluation(unittest.TestCase):
    """Radial profile and ring artifact metrics."""

    def test_radial_profile_of_constant_image(self):
        """A constant image has a constant radial profile."""
        radii, profile = radial_profile(np.full((100, 100), 3.0), edge=5)
        self.assertTrue(np.all(np.diff(radii) > 0))
        np.testing.assert_allclose(profile[~np.isnan(profile)], 3.0)

    def test_ring_artifact_detects_rings(self):
        """Rings beyond the cutoff score higher than a flat profile."""
        radii = np.arange(4, 1360, 8.0)
        flat = np.ones_like(radii)
        rings = flat + 0.3 * np.sin(2 * np.pi * radii / 92) * (radii > 960)
        self.assertLess(ring_artifact(radii, flat, 960), 1e-9)
        self.assertGreater(ring_artifact(radii, rings, 960), 0.1)

    def test_evaluate_tile_metrics(self):
        """Dark add-back gives no clipping and no rings on dim noise."""
        raw, flat, dark = dim_tile(n=256, z=16)
        corrected = (raw.astype(np.float32) - dark) / flat
        warped = radial_correction(
            corrected,
            corner_shift=CORNER_SHIFT,
            frac_cutoff=FRAC_CUTOFF,
            mode="2d",
            order=1,
        )
        edge = (raw.shape[-1] - warped.shape[-1]) // 2
        crop = dark[edge:-edge, edge:-edge]
        with_dark = np.clip(np.rint(warped + crop), 0, 65535)
        clipped = np.clip(np.rint(warped), 0, 65535)

        good = evaluate_tile(raw, with_dark.astype(np.uint16), flat, dark, 0.5)
        bad = evaluate_tile(raw, clipped.astype(np.uint16), flat, dark, 0.5)
        self.assertEqual(good["metrics"]["voxels clipped to 0 (%)"], 0.0)
        self.assertGreater(bad["metrics"]["voxels clipped to 0 (%)"], 30.0)
        self.assertAlmostEqual(
            good["metrics"]["median signal, after flatfield + radial"],
            0.0,
            delta=0.5,
        )

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qc.png"
            plot_evaluation(good, str(path), title="synthetic")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
