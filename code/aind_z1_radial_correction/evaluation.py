"""
Quality-control metrics and plots for the radial correction.

The figure shows, for one tile, the mean over planes of the raw data,
the flatfield-corrected data and the final radially corrected output
(top row), and under each one its ring map (bottom row): every pixel is
coloured by how much brighter or darker its ring is than the smooth
radial trend. Two numbers summarise each stage:

- median signal: typical brightness above the camera dark (counts).
- ring artifact: how much the tile's average brightness rises and
  falls in rings beyond the warp cutoff (counts). It is the standard
  deviation of the radial profile around its smooth trend; about the
  same value as the raw data means no rings were added.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter, uniform_filter1d

LOGGER = logging.getLogger(__name__)


def ring_index(
    shape: Tuple[int, int], edge: int, bin_width: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ring number of every pixel of an edge-cropped tile.

    Parameters
    ----------
    shape: Tuple[int, int]
        (Y, X) shape of the edge-cropped image.
    edge: int
        Pixels cropped from each side of the original tile.
    bin_width: int
        Ring width in pixels. Default: 8

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Ring number per pixel (0-based, same shape as the image) and the
        ring centre radii in pixels.
    """
    full = shape[0] + 2 * edge
    y = np.arange(shape[0]) + edge - full // 2
    x = np.arange(shape[1]) + edge - full // 2
    radius = np.hypot(y[:, None], x[None, :])
    bins = np.arange(0, radius.max() + bin_width, bin_width)
    return np.digitize(radius, bins) - 1, (bins[:-1] + bins[1:]) / 2


def radial_profile(
    image: np.ndarray, edge: int, bin_width: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average an edge-cropped image in rings around the warp centre.

    Parameters
    ----------
    image: np.ndarray
        2D (Y, X) image, already cropped by ``edge`` on each side, as
        returned by the radial correction.
    edge: int
        Pixels cropped from each side of the original tile.
    bin_width: int
        Ring width in pixels. Default: 8

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Ring centre radii (pixels) and the mean value per ring. Rings
        without pixels are NaN.
    """
    rings, radii = ring_index(image.shape, edge, bin_width)
    n_rings = len(radii)
    counts = np.bincount(rings.ravel(), minlength=n_rings)[:n_rings]
    totals = np.bincount(
        rings.ravel(), image.ravel().astype(np.float64), minlength=n_rings
    )[:n_rings]
    profile = np.full(n_rings, np.nan)
    np.divide(totals, counts, out=profile, where=counts > 0)
    return radii, profile


def smooth_trend(profile: np.ndarray, window: int = 25) -> np.ndarray:
    """
    Moving average of a radial profile over its non-empty rings.

    Parameters
    ----------
    profile: np.ndarray
        Radial profile, NaN where a ring has no pixels.
    window: int
        Number of rings averaged. Default: 25 (200 px with 8 px rings),
        wider than the ring period so the rings are not followed.

    Returns
    -------
    np.ndarray
        Smooth trend, NaN where the profile is NaN.
    """
    valid = ~np.isnan(profile)
    trend = np.full_like(profile, np.nan)
    trend[valid] = uniform_filter1d(profile[valid], window, mode="nearest")
    return trend


def ring_window(
    radii: np.ndarray, cutoff: float, outer_fraction: float = 0.96
) -> np.ndarray:
    """
    Select the rings where the warp acts and ring artifacts can appear.

    Parameters
    ----------
    radii: np.ndarray
        Ring centre radii in pixels.
    cutoff: float
        Radius in pixels beyond which the warp moves pixels.
    outer_fraction: float
        Outer limit as a fraction of the largest radius, avoiding the
        sparsely sampled corner rings. Default: 0.96

    Returns
    -------
    np.ndarray
        Boolean mask over ``radii``. On small tiles where that window
        is empty, every ring beyond the cutoff is used.
    """
    window = (radii > cutoff + 20) & (radii < outer_fraction * radii.max())
    if not window.any():
        window = radii > cutoff
    return window


def ring_artifact(
    radii: np.ndarray, profile: np.ndarray, cutoff: float
) -> float:
    """
    Strength of radial rings in a profile, in the profile's units.

    Parameters
    ----------
    radii: np.ndarray
        Ring centre radii in pixels.
    profile: np.ndarray
        Radial profile from ``radial_profile``.
    cutoff: float
        Radius in pixels beyond which the warp moves pixels.

    Returns
    -------
    float
        Standard deviation of the profile around its smooth trend,
        within the ring window. NaN if no ring lies beyond the cutoff.
    """
    residual = (profile - smooth_trend(profile))[ring_window(radii, cutoff)]
    if np.isnan(residual).all():
        return float("nan")
    return float(np.nanstd(residual))


def ring_map(
    shape: Tuple[int, int], edge: int, profile: np.ndarray
) -> np.ndarray:
    """
    Paint each pixel with its ring's deviation from the smooth trend.

    Parameters
    ----------
    shape: Tuple[int, int]
        (Y, X) shape of the edge-cropped image.
    edge: int
        Pixels cropped from each side of the original tile.
    profile: np.ndarray
        Radial profile from ``radial_profile`` (8 px rings).

    Returns
    -------
    np.ndarray
        Image of profile minus smooth trend, ring by ring (counts).
    """
    rings, _ = ring_index(shape, edge)
    return (profile - smooth_trend(profile))[rings]


def evaluate_tile(
    raw: np.ndarray,
    corrected: np.ndarray,
    flatfield: Optional[np.ndarray],
    darkfield: Optional[np.ndarray],
    frac_cutoff: float,
    saturated_fraction: float = 0.0,
) -> Dict:
    """
    Compute the quality-control images and metrics for one tile.

    Parameters
    ----------
    raw: np.ndarray
        Raw tile (Z, Y, X).
    corrected: np.ndarray
        Radially corrected output (Z, Y', X'), with the darkfield added
        back, as returned by ``apply_corr_to_zarr_tile``.
    flatfield: Optional[np.ndarray]
        Flatfield (Y, X), or None when no calibration was applied.
    darkfield: Optional[np.ndarray]
        Darkfield (Y, X), or None when no calibration was applied. Then
        the stages are only "raw" and "after radial", and values include
        the camera offset.
    frac_cutoff: float
        Fraction of the tile width where the warp starts.
    saturated_fraction: float
        Fraction of voxels that exceeded the dtype maximum before the
        integer conversion and were clipped to it. Default: 0.0

    Returns
    -------
    Dict
        ``images`` (signal above dark per stage, Y' x X'), ``profiles``
        (radii and one radial profile per stage) and ``metrics``.
    """
    edge = (raw.shape[-1] - corrected.shape[-1]) // 2
    crop = (
        slice(edge, raw.shape[-2] - edge),
        slice(edge, raw.shape[-1] - edge),
    )
    calibrated = flatfield is not None and darkfield is not None
    dark = darkfield[crop].astype(np.float32) if calibrated else 0.0
    raw_mean = raw.mean(axis=0, dtype=np.float64)[crop].astype(np.float32)
    out_mean = corrected.mean(axis=0, dtype=np.float64).astype(np.float32)

    images = {"raw": raw_mean - dark}
    if calibrated:
        images["after flatfield"] = (raw_mean - dark) / flatfield[crop]
        images["after flatfield + radial"] = out_mean - dark
    else:
        images["after radial"] = out_mean
    cutoff = raw.shape[-1] * frac_cutoff
    profiles, metrics = {}, {}
    for stage, image in images.items():
        radii, profile = radial_profile(image, edge)
        profiles[stage] = profile
        metrics[f"median signal, {stage}"] = float(np.median(image))
        metrics[f"ring artifact, {stage}"] = ring_artifact(
            radii, profile, cutoff
        )
    metrics["voxels clipped to 0 (%)"] = float(100 * np.mean(corrected == 0))
    metrics["voxels saturated at max (%)"] = float(100 * saturated_fraction)
    profiles["radii"] = radii
    return {
        "images": images,
        "profiles": profiles,
        "metrics": metrics,
        "cutoff": cutoff,
        "n_planes": raw.shape[0],
        "edge": edge,
        "calibrated": calibrated,
    }


def plot_evaluation(
    evaluation: Dict,
    output_path: str,
    title: Optional[str] = None,
    smooth_px: float = 4,
) -> None:
    """
    Save the quality-control figure for one tile.

    Parameters
    ----------
    evaluation: Dict
        Output of ``evaluate_tile``.
    output_path: str
        Local path of the PNG to write.
    title: Optional[str]
        Figure title, e.g. the tile name. Default: None
    smooth_px: float
        Gaussian smoothing of the images for display only, so faint
        rings are visible above the noise. Default: 4
    """
    images, profiles = evaluation["images"], evaluation["profiles"]
    metrics, cutoff = evaluation["metrics"], evaluation["cutoff"]
    radii, edge = profiles["radii"], evaluation["edge"]
    shown = {k: gaussian_filter(v, smooth_px) for k, v in images.items()}
    maps = {k: ring_map(images[k].shape, edge, profiles[k]) for k in images}

    # Same contrast width for every image, each centred on its own median
    ref = shown.get("after flatfield", shown["raw"])
    half = 0.5 * np.subtract(*np.percentile(ref, [99, 1]))
    # One colour scale for all ring maps: at least 10x the raw data's
    # ring artifact, so ordinary tissue and noise look pale and real
    # rings stand out
    window = ring_window(radii, cutoff)
    in_band = window[ring_index(ref.shape, edge)[0]]
    peak = max(np.nanmax(np.abs(m[in_band]), initial=0) for m in maps.values())
    limit = max(float(peak), 10 * metrics["ring artifact, raw"], 0.05)
    # Show the ring maps only where the ring artifact is measured; any
    # radial tissue feature elsewhere would also look like a circle
    maps = {k: np.where(in_band, m, np.nan) for k, m in maps.items()}
    ring_cmap = colormaps["RdBu_r"].with_extremes(bad="0.85")

    # Tile centre and the measurement band, in edge-cropped pixels
    centre_px = (ref.shape[1] + 2 * edge) // 2 - edge
    band = (radii[window].min(), radii[window].max()) if window.any() else ()

    # Figure API, not pyplot: no global backend change, safe in threads
    n_stages = len(shown)
    fig = Figure(figsize=(6 * n_stages, 12.5))
    axs = fig.subplots(2, n_stages)
    # Room for the multi-line title; set before the colour bar is placed
    fig.subplots_adjust(top=0.82)
    value_label = (
        "counts above dark"
        if evaluation["calibrated"]
        else "counts (offset included)"
    )
    for j, (stage, image) in enumerate(shown.items()):
        centre = np.median(image)
        axs[0, j].imshow(
            image, cmap="gray", vmin=centre - half, vmax=centre + half
        )
        axs[0, j].set_title(
            f"{stage}\nmedian signal "
            f"{metrics[f'median signal, {stage}']:.2f} {value_label}"
        )
        ring_im = axs[1, j].imshow(
            maps[stage], cmap=ring_cmap, vmin=-limit, vmax=limit
        )
        axs[1, j].set_title(
            f"ring artifact "
            f"{metrics[f'ring artifact, {stage}']:.3f} counts"
        )
        for ax in axs[:, j]:
            for r in band:
                ax.add_patch(
                    Circle(
                        (centre_px, centre_px),
                        r,
                        fill=False,
                        ec="#e67e22",
                        ls="--",
                        lw=1.5,
                    )
                )
            ax.set_xticks([])
            ax.set_yticks([])
    axs[0, 0].set_ylabel("average image", fontsize=13)
    axs[1, 0].set_ylabel(
        "ring map\nred: brighter ring\nblue: darker ring\ngrey: not measured",
        fontsize=13,
    )
    cbar = fig.colorbar(ring_im, ax=axs[1, :], shrink=0.8, pad=0.01)
    cbar.set_label("counts above or below the trend")

    fig.suptitle(
        (f"{title}\n" if title else "")
        + (
            ""
            if evaluation["calibrated"]
            else "No flatfield/darkfield provided: radial correction only; "
            "values include the camera offset.\n"
        )
        + "Good result: no red/blue circles appear after radial correction, "
        "and its ring artifact stays close to raw.\n"
        "Ring artifact = spread of the ring map between the orange dashed "
        "circles, where the warp moves pixels. "
        f"Voxels clipped to 0: {metrics['voxels clipped to 0 (%)']:.2f}%, "
        "saturated at max: "
        f"{metrics['voxels saturated at max (%)']:.2f}%.\n"
        f"Images are the average of {evaluation['n_planes']} planes: rings "
        "are hidden by the noise in a single plane, and averaging makes "
        "them visible.",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=90, bbox_inches="tight")
    LOGGER.info("Saved radial correction QC figure to %s", output_path)
