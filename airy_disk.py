#!/usr/bin/env python3
"""
Telescope Airy Disk Simulator

Simulates the Airy diffraction pattern produced by a circular-aperture
telescope given focal length and aperture diameter.

Physics
-------
The intensity of the Fraunhofer diffraction pattern of a circular aperture is:

    I(r) = I₀ · [2·J₁(x) / x]²

where  x = π · D · r / (λ · f)
    D  = aperture diameter (m)
    f  = focal length (m)
    λ  = wavelength (m)
    r  = radial distance from optical axis in the focal plane (m)
    J₁ = Bessel function of the first kind, order 1

The first dark ring (Airy disk boundary) occurs at x ≈ 3.8317, giving:

    r_airy = 1.22 · λ · f / D

The Rayleigh angular resolution criterion is:

    θ = 1.22 · λ / D   (radians)

Usage
-----
Interactive:
    python airy_disk.py

Command-line:
    python airy_disk.py --focal-length 1000 --aperture 100
    python airy_disk.py -f 500 -a 80 -w 656 -o halpha_airy.png
    python airy_disk.py -f 2000 -a 200 --size 1024 --rings 12
"""

import argparse
import sys

import numpy as np
from scipy.special import j1
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Wavelength → approximate RGB colour  (used for false-colouring the image)
# ---------------------------------------------------------------------------

def wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    """
    Convert a visible-light wavelength (380–780 nm) to an approximate sRGB
    triple in [0, 1].  Based on the piecewise model by Dan Bruton.
    """
    wl = wavelength_nm
    if 380 <= wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif 440 <= wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif 490 <= wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif 510 <= wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif 580 <= wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    elif 645 <= wl <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        # outside visible range: white
        r, g, b = 1.0, 1.0, 1.0

    # Intensity rolloff at the edges of the visible spectrum
    if 380 <= wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / 40
    elif 700 < wl <= 780:
        factor = 0.3 + 0.7 * (780 - wl) / 80
    else:
        factor = 1.0

    return (r * factor, g * factor, b * factor)


def make_wavelength_cmap(wavelength_nm: float):
    """
    Build a matplotlib colormap that goes from black → star colour.
    This gives the Airy disk image a tint matching the simulated wavelength.
    """
    from matplotlib.colors import LinearSegmentedColormap
    star_color = wavelength_to_rgb(wavelength_nm)
    return LinearSegmentedColormap.from_list(
        "airy_cmap",
        [(0, 0, 0), star_color],
        N=512,
    )


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------

def compute_airy_pattern(
    focal_length_mm: float,
    aperture_mm: float,
    wavelength_nm: float = 550.0,
    image_size: int = 512,
    n_airy_radii: int = 8,
) -> tuple[np.ndarray, float, float]:
    """
    Compute a 2-D Airy disk intensity pattern.

    Parameters
    ----------
    focal_length_mm : float
        Focal length in millimetres.
    aperture_mm : float
        Aperture (entrance pupil) diameter in millimetres.
    wavelength_nm : float
        Wavelength of light in nanometres (default 550 nm, green).
    image_size : int
        Side length of the output array in pixels (square).
    n_airy_radii : int
        Number of Airy-disk radii spanned by the half-image.  Controls how
        many diffraction rings are visible.

    Returns
    -------
    intensity : ndarray, shape (image_size, image_size)
        Normalised intensity values in [0, 1].
    airy_radius_um : float
        Radius of the first dark ring (Airy disk) in micrometres.
    um_per_pixel : float
        Pixel scale in micrometres per pixel.
    """
    wl_mm   = wavelength_nm * 1e-6          # nm → mm
    f_mm    = focal_length_mm
    D_mm    = aperture_mm

    # Airy disk radius in the focal plane
    airy_radius_mm = 1.22 * wl_mm * f_mm / D_mm
    airy_radius_um = airy_radius_mm * 1e3   # mm → μm

    # Pixel scale: n_airy_radii across each half of the image
    half_extent_mm = n_airy_radii * airy_radius_mm
    um_per_pixel   = (half_extent_mm * 2e3) / image_size   # μm / pixel

    # Spatial grid in mm
    half = image_size // 2
    coords_mm = (np.arange(image_size) - half) * (um_per_pixel * 1e-3)
    xx, yy = np.meshgrid(coords_mm, coords_mm)
    r_mm = np.sqrt(xx**2 + yy**2)

    # Dimensionless argument  x = π D r / (λ f)
    x = np.pi * D_mm * r_mm / (wl_mm * f_mm)

    # Airy pattern  I = [2 J₁(x) / x]²   (Jinc²)
    # At x = 0 the limit is 1 (L'Hôpital / series expansion)
    with np.errstate(invalid="ignore", divide="ignore"):
        jinc = np.where(x == 0.0, 1.0, 2.0 * j1(x) / x)
    intensity = jinc**2                     # already normalised to 1 at centre

    return intensity, airy_radius_um, um_per_pixel


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_airy_image(
    intensity: np.ndarray,
    focal_length_mm: float,
    aperture_mm: float,
    wavelength_nm: float,
    airy_radius_um: float,
    um_per_pixel: float,
    output_path: str,
) -> None:
    """
    Render the Airy disk image (log-scale 2-D view + linear cross-section)
    and save it as a PNG file.
    """
    image_size     = intensity.shape[0]
    fratio         = focal_length_mm / aperture_mm
    extent_half_mm = (image_size / 2) * um_per_pixel * 1e-3  # mm

    # Angular resolution (Rayleigh criterion)
    theta_rad    = 1.22 * wavelength_nm * 1e-9 / (aperture_mm * 1e-3)
    theta_arcsec = np.degrees(theta_rad) * 3600.0

    # ---- figure layout ------------------------------------------------
    BG = "#09090e"
    fig = plt.figure(figsize=(16, 7), facecolor=BG)
    gs  = gridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[1, 0.03, 1],
        wspace=0.35,
        left=0.06, right=0.96, top=0.84, bottom=0.18,
    )
    ax_img = fig.add_subplot(gs[0])
    ax_cb  = fig.add_subplot(gs[1])
    ax_pro = fig.add_subplot(gs[2])

    for ax in (ax_img, ax_pro):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#aaa", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # ------------------------------------------------------------------ #
    # Left panel – 2-D log-scale image
    # ------------------------------------------------------------------ #
    cmap  = make_wavelength_cmap(wavelength_nm)
    vmin  = intensity.max() * 1e-5          # dynamic range: 5 decades
    vmax  = intensity.max()
    extent = [-extent_half_mm, extent_half_mm, -extent_half_mm, extent_half_mm]

    im = ax_img.imshow(
        intensity,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="bilinear",
    )

    # Airy disk radius circle
    airy_r_mm = airy_radius_um * 1e-3
    circle = Circle(
        (0, 0), airy_r_mm,
        fill=False, color="cyan", linewidth=1.2, linestyle="--", alpha=0.75,
        label=f"Airy disk  r = {airy_radius_um:.2f} μm",
    )
    ax_img.add_patch(circle)

    # Second ring (second dark ring ≈ 2.233 × first)
    circle2 = Circle(
        (0, 0), airy_r_mm * 2.233,
        fill=False, color="#66aaff", linewidth=0.8, linestyle=":", alpha=0.5,
    )
    ax_img.add_patch(circle2)

    ax_img.set_xlabel("Position (mm)", color="#ccc", fontsize=10)
    ax_img.set_ylabel("Position (mm)", color="#ccc", fontsize=10)
    ax_img.set_title("Focal-Plane Airy Pattern  (log scale)", color="white", fontsize=11, pad=8)
    ax_img.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", loc="upper right")

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Relative intensity (log₁₀)", color="#aaa", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="#aaa", labelcolor="#aaa", labelsize=7)

    # ------------------------------------------------------------------ #
    # Right panel – horizontal cross-section (linear scale)
    # ------------------------------------------------------------------ #
    cx      = image_size // 2
    profile = intensity[cx, :]                        # central row
    x_um    = (np.arange(image_size) - cx) * um_per_pixel

    star_rgb = wavelength_to_rgb(wavelength_nm)
    ax_pro.plot(x_um, profile, color=star_rgb, linewidth=1.6, zorder=3)
    ax_pro.fill_between(x_um, profile, alpha=0.12, color=star_rgb, zorder=2)

    # Mark Airy disk edges
    for sign in (-1, 1):
        ax_pro.axvline(
            sign * airy_radius_um,
            color="cyan", linestyle="--", linewidth=1.0, alpha=0.7,
            label="Airy disk boundary" if sign == 1 else None,
        )

    ax_pro.set_xlabel("Position (μm)", color="#ccc", fontsize=10)
    ax_pro.set_ylabel("Relative Intensity", color="#ccc", fontsize=10)
    ax_pro.set_title("Cross-Section Through Centre  (linear scale)", color="white", fontsize=11, pad=8)
    ax_pro.set_ylim(-0.02, 1.08)
    ax_pro.set_xlim(x_um[0], x_um[-1])
    ax_pro.grid(True, color="#1c1c2e", linewidth=0.8)
    ax_pro.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    # ------------------------------------------------------------------ #
    # Header / info block
    # ------------------------------------------------------------------ #
    header = (
        f"Telescope Airy Disk Simulator  —  "
        f"f = {focal_length_mm:.0f} mm   D = {aperture_mm:.0f} mm   "
        f"f/{fratio:.1f}   λ = {wavelength_nm:.0f} nm"
    )
    fig.suptitle(header, color="white", fontsize=13, y=0.97, fontweight="bold")

    info = (
        f"Airy disk radius: {airy_radius_um:.3f} μm   |   "
        f"Rayleigh resolution: {theta_arcsec:.3f} arcsec   |   "
        f"Pixel scale: {um_per_pixel:.3f} μm/px"
    )
    fig.text(
        0.5, 0.01, info,
        ha="center", va="bottom", color="#888", fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#111", edgecolor="#333", alpha=0.9),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="airy_disk.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-f", "--focal-length", type=float, metavar="MM",
                   help="Focal length in millimetres")
    p.add_argument("-a", "--aperture", type=float, metavar="MM",
                   help="Aperture diameter in millimetres")
    p.add_argument("-w", "--wavelength", type=float, default=550.0, metavar="NM",
                   help="Wavelength in nanometres (default: 550 nm = green)")
    p.add_argument("-o", "--output", type=str, default="airy_disk.png",
                   help="Output PNG file (default: airy_disk.png)")
    p.add_argument("--size", type=int, default=512, metavar="PX",
                   help="Image side length in pixels (default: 512)")
    p.add_argument("--rings", type=int, default=8, metavar="N",
                   help="Airy-disk radii shown across each half-image (default: 8)")
    return p.parse_args(argv)


def prompt_float(prompt: str, lo: float = 1e-6, hi: float = 1e9) -> float:
    while True:
        try:
            val = float(input(prompt))
            if lo < val < hi:
                return val
            print(f"  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Invalid input – please enter a number.")
        except EOFError:
            print("\nNo input provided. Use -f / -a flags to pass values non-interactively.")
            sys.exit(1)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Interactive prompts for missing required parameters
    if args.focal_length is None:
        print("=== Telescope Airy Disk Simulator ===")
        args.focal_length = prompt_float("  Focal length (mm): ", lo=1.0)
    if args.aperture is None:
        args.aperture = prompt_float("  Aperture diameter (mm): ", lo=1.0)

    f  = args.focal_length
    D  = args.aperture
    wl = args.wavelength

    if D <= 0 or f <= 0:
        print("Error: focal length and aperture must be positive.", file=sys.stderr)
        return 1

    fratio = f / D
    print()
    print("  Focal length  :", f, "mm")
    print("  Aperture      :", D, "mm")
    print(f"  f-ratio       : f/{fratio:.1f}")
    print("  Wavelength    :", wl, "nm")
    print("  Image size    :", args.size, "px")
    print()

    intensity, airy_radius_um, um_per_pixel = compute_airy_pattern(
        focal_length_mm=f,
        aperture_mm=D,
        wavelength_nm=wl,
        image_size=args.size,
        n_airy_radii=args.rings,
    )

    theta_rad    = 1.22 * wl * 1e-9 / (D * 1e-3)
    theta_arcsec = np.degrees(theta_rad) * 3600.0
    print(f"  Airy disk radius          : {airy_radius_um:.4f} μm in the focal plane")
    print(f"  Rayleigh resolution limit : {theta_arcsec:.4f} arcsec")
    print(f"  Pixel scale               : {um_per_pixel:.4f} μm/pixel")
    print()

    render_airy_image(
        intensity=intensity,
        focal_length_mm=f,
        aperture_mm=D,
        wavelength_nm=wl,
        airy_radius_um=airy_radius_um,
        um_per_pixel=um_per_pixel,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
