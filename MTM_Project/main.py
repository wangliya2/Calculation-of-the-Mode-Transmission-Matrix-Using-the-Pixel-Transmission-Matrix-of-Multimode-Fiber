from __future__ import annotations

import argparse
import os

from config import FiberParams, MTMConfig, Paths, PreprocessConfig
from src.mtm_calculator import run_mtm_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PTM(TIFF) -> MTM end-to-end pipeline (Plöschner 2015)"
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=Paths.input_tiff_dir,
        help="Folder containing PTM TIFF files (default: data/input_tiff)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=Paths.output_dir,
        help="Folder to write outputs (default: data/output_mtm)",
    )
    p.add_argument(
        "--num-modes",
        type=int,
        default=MTMConfig.num_modes,
        help="Number of LP modes to use (default: 8)",
    )
    p.add_argument(
        "--denoise-method",
        type=str,
        default=PreprocessConfig.denoise_method,
        choices=["gaussian", "median"],
        help="Denoising method for amplitude (default: gaussian)",
    )
    p.add_argument(
        "--input-basis",
        type=str,
        default="hadamard",
        choices=["hadamard", "identity", "file"],
        help="Input basis for M_in: hadamard/identity/or load from file (default: hadamard)",
    )
    p.add_argument(
        "--m-in-path",
        type=str,
        default="data/standard_lp_modes/M_in_hadamard_8.npy",
        help="When --input-basis=file, load M_in from this path (default points to generated file)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_root, args.input_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    fiber_params = {
        "core_radius": FiberParams.core_radius_um,
        "wavelength": FiberParams.wavelength_um,
        "na": FiberParams.na,
    }

    preprocess_params = (
        {"ksize": PreprocessConfig.median_ksize}
        if args.denoise_method == "median"
        else {"ksize": PreprocessConfig.gaussian_ksize, "sigma": PreprocessConfig.gaussian_sigma}
    )

    run_mtm_pipeline(
        input_dir=input_dir,
        output_root=output_dir,
        fiber_params=fiber_params,
        num_modes=int(args.num_modes),
        denoise_method=args.denoise_method,
        denoise_params=preprocess_params,
        input_basis=args.input_basis,
        m_in_path=os.path.join(project_root, args.m_in_path),
        mtm_thresholds={
            "offdiag_max": MTMConfig.offdiag_max_threshold,
            "diag_mean_target": MTMConfig.diag_mean_target,
            "diag_mean_tol": MTMConfig.diag_mean_tol,
        },
    )


if __name__ == "__main__":
    main()

