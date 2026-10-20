#!/usr/bin/env python3

# scan_focus.py: Calculate PSF containment diameter vs focal plane offset.

# Author: Stephen Fegan <sfegan@llr.in2p3.fr> (2026-10-17)
# Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris

import sys
import os
import argparse
import datetime
import json
import socket
import numpy as np
import concurrent.futures

# Add directory containing 'pypanodecoder' to sys.path
package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_parent = os.path.dirname(package_parent)
for p in (repo_parent, package_parent):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from pypanodecoder import optical_model
except ImportError:
    import optical_model


_worker_datapack = None


def _worker_init(datapack_file=None):
    """
    Initialize worker process: reseed random number generator and load datapack.
    """
    global _worker_datapack
    np.random.seed()
    _worker_datapack = optical_model.load_datapack(datapack_file)


def _worker_task(task_args):
    """
    Compute PSF diameter for a single focal plane distance.
    """
    (focal_distance, x, y, num_rays, thick_lens, zn_rad, quantile, b_minus_v, source_distance, energy_eV, focal_offset) = task_args
    global _worker_datapack
    if _worker_datapack is None:
        _worker_datapack = optical_model.load_datapack()

    res = optical_model.generate_psf_image(
        x=x,
        y=y,
        num_rays=num_rays,
        datapack=_worker_datapack,
        thick_lens=thick_lens,
        zn=zn_rad,
        focal_offset=focal_offset,
        energy_eV=energy_eV,
        calc_diameter=True,
        diameter_quantile=quantile,
        b_minus_v=b_minus_v,
        source_distance=source_distance,
    )
    # res is (nvalid, image, center_mean, r_mean, center_median, r_median, center_Q, diameter_Q)
    nvalid = int(res[0])
    center_Q = res[-2]
    diameter_Q = float(res[-1])
    return focal_distance, diameter_Q, float(center_Q[0]), float(center_Q[1]), nvalid


def main():
    parser = argparse.ArgumentParser(
        description="Calculate telescope PSF containment diameter as a function of focal plane distance."
    )
    parser.add_argument(
        "--focal-distance-min",
        "--f-min",
        type=float,
        default=None,
        help="Minimum focal distance in cm (default: 58.5 cm).",
    )
    parser.add_argument(
        "--focal-distance-max",
        "--f-max",
        type=float,
        default=None,
        help="Maximum focal distance in cm (default: 61.5 cm).",
    )
    parser.add_argument(
        "--focal-distance-step",
        "--f-step",
        "--step",
        type=float,
        default=0.5,
        help="Step size for focal distance in cm (default: 0.5).",
    )
    parser.add_argument(
        "--focal-distance-range",
        "--f-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Convenience shortcut to specify min and max focal distance in cm.",
    )
    parser.add_argument(
        "--offset-min",
        type=float,
        default=None,
        help="Minimum focal plane offset from nominal F in cm (alternative to --f-min).",
    )
    parser.add_argument(
        "--offset-max",
        type=float,
        default=None,
        help="Maximum focal plane offset from nominal F in cm (alternative to --f-max).",
    )
    parser.add_argument(
        "--offset-step",
        type=float,
        default=None,
        help="Step size for focal plane offset in cm (alternative to --f-step).",
    )
    parser.add_argument(
        "--offset-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Convenience shortcut to specify min and max offset from nominal F in cm.",
    )
    parser.add_argument(
        "--field-angle",
        "-a",
        type=float,
        default=0.0,
        help="Field angle in degrees off-axis (default: 0.0).",
    )
    parser.add_argument(
        "--field-azimuth",
        type=float,
        default=0.0,
        help="Azimuthal angle for off-axis field angle in degrees (default: 0.0, along +X).",
    )
    parser.add_argument(
        "--field-x",
        type=float,
        default=None,
        help="Explicit target X position in pixels on nominal focal plane (overrides field-angle).",
    )
    parser.add_argument(
        "--field-y",
        type=float,
        default=None,
        help="Explicit target Y position in pixels on nominal focal plane (overrides field-angle).",
    )
    parser.add_argument(
        "--containment-fraction",
        "-q",
        "--fraction",
        type=float,
        default=0.80,
        help="Containment fraction for diameter calculation (e.g. 0.80 for D80, 0.50 for D50; default: 0.80).",
    )
    parser.add_argument(
        "--b-minus-v",
        "--bv",
        type=float,
        default=None,
        help="B-V color index of the source for a thermal (stellar) spectrum. If omitted, flat (Cherenkov) spectrum is used.",
    )
    parser.add_argument(
        "--ev",
        "--energy-ev",
        type=float,
        default=None,
        help="Fixed photon energy in eV (default: None, drawn from standard spectrum).",
    )
    parser.add_argument(
        "--zenith-angle",
        "--zn",
        type=float,
        default=0.0,
        help="Zenith angle in degrees for atmospheric absorption (default: 0.0).",
    )
    parser.add_argument(
        "-n",
        "--num-rays",
        "--nrays",
        type=int,
        default=1000000,
        help="Number of rays to trace per offset (default: 1000000).",
    )
    parser.add_argument(
        "--thick-lens",
        action="store_true",
        help="Use thick lens optical model instead of thin lens approximation.",
    )
    parser.add_argument(
        "-j",
        "--nthread",
        "--threads",
        type=int,
        default=0,
        help="Number of worker processes (default: 0, assumes number of CPU cores). If 1, multiprocessing is skipped and runs in the main process.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output text filename. If omitted, a name based on options is generated.",
    )
    parser.add_argument(
        "--datapack",
        type=str,
        default=None,
        help="Path to optical model JSON datapack file (default: bundled datapack).",
    )
    parser.add_argument(
        "--source-distance",
        type=float,
        default=None,
        help="Finite distance from aperture to point source in cm (default: infinite / parallel beam).",
    )
    parser.add_argument(
        "--units",
        type=str,
        default="pix",
        choices=["cm", "mm", "pix", "deg"],
        help="Units for PSF containment diameter (choices: 'cm', 'mm', 'pix', 'deg'; default: 'pix').",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="%.4f %.4f %.4f %.4f %d",
        help="Format string for np.savetxt columns (default: '%%.4f %%.4f %%.4f %%.4f %%d').",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )

    args = parser.parse_args()

    # Validate containment fraction
    fraction = args.containment_fraction
    if fraction > 1.0:
        fraction = fraction / 100.0
    if not (0.0 < fraction < 1.0):
        parser.error(f"Containment fraction must be strictly between 0 and 1, got {args.containment_fraction}.")

    pct = fraction * 100.0
    if abs(pct - round(pct)) < 1e-5:
        containment_name = f"D{int(round(pct))}"
    else:
        containment_name = f"D{pct:g}"

    # Determine concurrency
    if args.nthread < 0:
        parser.error("--nthread must be non-negative.")
    elif args.nthread == 0:
        num_workers = os.cpu_count() or 1
    else:
        num_workers = args.nthread

    # Load datapack in main process to obtain optical parameters
    datapack = optical_model.load_datapack(args.datapack)
    optics = datapack["optical_model"]
    focal_length_nominal = optics["F"]
    pixel_spacing = optics["pixel_spacing"]

    # Determine focal distance step
    if args.offset_step is not None:
        dist_step = args.offset_step
    else:
        dist_step = args.focal_distance_step
    if dist_step <= 0:
        parser.error("Step size must be positive.")

    # Determine focal distance min and max
    if args.focal_distance_range is not None:
        dist_min, dist_max = args.focal_distance_range
    elif args.offset_range is not None:
        dist_min = focal_length_nominal + args.offset_range[0]
        dist_max = focal_length_nominal + args.offset_range[1]
    else:
        if args.focal_distance_min is not None:
            dist_min = args.focal_distance_min
        elif args.offset_min is not None:
            dist_min = focal_length_nominal + args.offset_min
        else:
            dist_min = 58.5

        if args.focal_distance_max is not None:
            dist_max = args.focal_distance_max
        elif args.offset_max is not None:
            dist_max = focal_length_nominal + args.offset_max
        else:
            dist_max = 61.5

    if dist_max < dist_min:
        parser.error(f"Maximum focal distance ({dist_max}) must be >= minimum ({dist_min}).")

    n_steps = int(round((dist_max - dist_min) / dist_step)) + 1
    distances = np.linspace(dist_min, dist_max, n_steps)

    zn_rad = np.deg2rad(args.zenith_angle)
    theta_rad = np.deg2rad(args.field_angle)
    phi_rad = np.deg2rad(args.field_azimuth)
    tan_theta = np.tan(theta_rad)

    # Prepare tasks for each focal distance
    tasks = []
    for dist in distances:
        focal_offset = dist - focal_length_nominal
        if args.field_x is not None or args.field_y is not None:
            x = args.field_x if args.field_x is not None else 0.0
            y = args.field_y if args.field_y is not None else 0.0
        else:
            # Physical position of principal ray on focal plane at focal distance dist
            x = (dist * tan_theta * np.cos(phi_rad)) / pixel_spacing
            y = (dist * tan_theta * np.sin(phi_rad)) / pixel_spacing

        tasks.append((
            dist,
            x,
            y,
            args.num_rays,
            args.thick_lens,
            zn_rad,
            fraction,
            args.b_minus_v,
            args.source_distance,
            args.ev,
            focal_offset,
        ))

    # Validate units
    unit = args.units.strip().lower()
    if unit in ("pixel", "pixels"):
        unit = "pix"
    elif unit in ("degree", "degrees", "deg"):
        unit = "deg"
    if unit not in ("cm", "mm", "pix", "deg"):
        parser.error(f"Invalid --units '{args.units}'. Choices are: cm, mm, pix, deg.")
    args.units = unit

    unit_scale = 1.0
    if args.units == "cm":
        unit_scale = pixel_spacing
    elif args.units == "mm":
        unit_scale = pixel_spacing * 10.0
    elif args.units == "deg":
        unit_scale = (pixel_spacing / focal_length_nominal) * (180.0 / np.pi)

    # Output filename
    if args.output is not None:
        output_filename = args.output
    else:
        if args.ev is not None:
            spec_str = f"{args.ev:g}eV"
        elif args.b_minus_v is not None:
            spec_str = f"bv{args.b_minus_v:g}"
        else:
            spec_str = "cherenkov"
        output_filename = f"scan_focus_{containment_name.lower()}_{args.field_angle:g}deg_{spec_str}_{args.units}.dat"

    if args.verbose:
        print(f"Scanning {len(distances)} focal distances from {dist_min:.2f} to {dist_max:.2f} cm (step {dist_step:.2f} cm)")
        print(f"Nominal focal length F = {focal_length_nominal:.2f} cm")
        print(f"Field angle: {args.field_angle:g} deg, Containment: {containment_name} ({fraction:.2f})")
        print(f"Diameter units: {args.units} (pixel_spacing = {pixel_spacing} cm)")
        if args.ev is not None:
            print(f"Fixed photon energy: {args.ev:g} eV")
        print(f"Rays per point: {args.num_rays:,}, Workers: {num_workers} (nthread arg: {args.nthread})")
        print(f"Output file: {output_filename}")

    # Run calculations
    results = []
    if num_workers == 1 or args.nthread == 1:
        if args.verbose:
            print("Running in main process (multiprocessing skipped)...")
        global _worker_datapack
        _worker_datapack = datapack
        for i, task in enumerate(tasks):
            focal_dist, d_pix, cx_pix, cy_pix, nvalid = _worker_task(task)
            d_val = d_pix * unit_scale
            cx_val = cx_pix * unit_scale
            cy_val = cy_pix * unit_scale
            results.append((focal_dist, d_val, cx_val, cy_val, nvalid))
            if args.verbose:
                print(
                    f"  [{i+1}/{len(tasks)}] focal_dist = {focal_dist:7.4f} cm -> "
                    f"{containment_name} = {d_val:.4f} {args.units}, "
                    f"center = ({cx_val:+.4f}, {cy_val:+.4f}) {args.units}, "
                    f"nvalid = {nvalid:,}"
                )
    else:
        if args.verbose:
            print(f"Dispatching tasks across {num_workers} worker processes...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
            initargs=(args.datapack,),
        ) as executor:
            for i, (focal_dist, d_pix, cx_pix, cy_pix, nvalid) in enumerate(executor.map(_worker_task, tasks)):
                d_val = d_pix * unit_scale
                cx_val = cx_pix * unit_scale
                cy_val = cy_pix * unit_scale
                results.append((focal_dist, d_val, cx_val, cy_val, nvalid))
                if args.verbose:
                    print(
                        f"  [{i+1}/{len(tasks)}] focal_dist = {focal_dist:7.4f} cm -> "
                        f"{containment_name} = {d_val:.4f} {args.units}, "
                        f"center = ({cx_val:+.4f}, {cy_val:+.4f}) {args.units}, "
                        f"nvalid = {nvalid:,}"
                    )

    # Build Configuration JSON for file header
    config = {
        "n": args.num_rays,
        "nthread": num_workers,
        "focal_length_nominal": focal_length_nominal,
        "focal_distance_min": dist_min,
        "focal_distance_max": dist_max,
        "focal_distance_step": dist_step,
        "field_angle": args.field_angle,
        "field_azimuth": args.field_azimuth,
        "containment_fraction": fraction,
        "containment_name": containment_name,
        "units": args.units,
        "columns": [
            "focal_distance_cm",
            f"{containment_name}_{args.units}",
            f"center_x_{args.units}",
            f"center_y_{args.units}",
            "n_valid",
        ],
        "energy_ev": args.ev,
        "b_minus_v": args.b_minus_v,
        "zenith_angle": args.zenith_angle,
        "thick_lens": args.thick_lens,
        "output": output_filename,
        "_generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "_host": socket.gethostname(),
    }
    if args.field_x is not None or args.field_y is not None:
        config["field_x"] = args.field_x
        config["field_y"] = args.field_y
    if args.source_distance is not None:
        config["source_distance"] = args.source_distance

    header = "Configuration (JSON):\n" + json.dumps(config, indent=2)

    data = np.array(results)
    np.savetxt(output_filename, data, header=header, comments="# ", fmt=args.format)

    if args.verbose:
        print(f"Results successfully saved to {output_filename}")


if __name__ == "__main__":
    main()
