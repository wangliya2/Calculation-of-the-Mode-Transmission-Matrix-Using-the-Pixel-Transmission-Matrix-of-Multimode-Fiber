from __future__ import annotations

import argparse
import os

from config import (
    BasisCorrectionConfig,
    FiberParams,
    MTMConfig,
    Paths,
    PreprocessConfig,
)
from src.lp_theory import FiberModel, recommended_num_modes
from src.mtm_calculator import run_mtm_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end PTM(TIFF) -> MTM pipeline (Plöschner 2015)"
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=Paths.input_tiff_dir,
        help="Directory containing PTM TIFF files (default: data/input_tiff)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=Paths.output_dir,
        help="Output directory (default: data/output_mtm)",
    )
    p.add_argument(
        "--num-modes",
        type=int,
        default=MTMConfig.num_modes,
        help="Number of LP modes to use (default: 8)",
    )
    p.add_argument(
        "--disable-auto-num-modes",
        action="store_true",
        help="Disable auto limit of supported mode count based on fiber parameters",
    )
    p.add_argument(
        "--denoise-method",
        type=str,
        default=PreprocessConfig.denoise_method,
        choices=["gaussian", "median", "bilateral"],
        help="Amplitude denoising method (default: gaussian; bilateral preserves edges)",
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
        help="When --input-basis=file, load M_in from this path",
    )
    p.add_argument(
        "--basis-correction",
        action="store_true",
        help="Enable basis correction (scale/shift/rotation optimization)",
    )
    p.add_argument(
        "--basis-correction-no-opt",
        action="store_true",
        help="Enable basis correction without optimizing parameters (use defaults)",
    )
    p.add_argument(
        "--joint-basis-correction",
        action="store_true",
        help="During basis correction, if M_in is spatial (H*W,N_modes), jointly optimize with M_out using the same geometry",
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
        "sim_extent_um": FiberParams.sim_extent_um,
    }

    if args.denoise_method == "median":
        preprocess_params = {"ksize": PreprocessConfig.median_ksize}
    elif args.denoise_method == "bilateral":
        preprocess_params = {
            "d": PreprocessConfig.bilateral_d,
            "sigma_color": PreprocessConfig.bilateral_sigma_color,
            "sigma_space": PreprocessConfig.bilateral_sigma_space,
        }
    else:
        preprocess_params = {
            "ksize": PreprocessConfig.gaussian_ksize,
            "sigma": PreprocessConfig.gaussian_sigma,
        }

    req_modes = int(args.num_modes)
    if not args.disable_auto_num_modes:
        auto_modes = recommended_num_modes(
            fiber=FiberModel(
                core_radius_um=FiberParams.core_radius_um,
                wavelength_um=FiberParams.wavelength_um,
                na=FiberParams.na,
            ),
            preferred_max=req_modes,
        )
        use_modes = min(req_modes, auto_modes)
        if use_modes < req_modes:
            print(
                f"[AutoMode] requested={req_modes}, supported≈{auto_modes}, using={use_modes}"
            )
    else:
        use_modes = req_modes

    run_mtm_pipeline(
        input_dir=input_dir,
        output_root=output_dir,
        fiber_params=fiber_params,
        num_modes=use_modes,
        denoise_method=args.denoise_method,
        denoise_params=preprocess_params,
        input_basis=args.input_basis,
        m_in_path=os.path.join(project_root, args.m_in_path),
        mtm_thresholds={
            "offdiag_max": MTMConfig.offdiag_max_threshold,
            "diag_mean_target": MTMConfig.diag_mean_target,
            "diag_mean_tol": MTMConfig.diag_mean_tol,
        },
        basis_correction={
            "enabled": bool(args.basis_correction),
            "optimize": bool(args.basis_correction and (not args.basis_correction_no_opt)),
            "joint": bool(args.joint_basis_correction),
            "scale_min": BasisCorrectionConfig.scale_min,
            "scale_max": BasisCorrectionConfig.scale_max,
            "shift_max_px": BasisCorrectionConfig.shift_max_px,
            "rotation_max_deg": BasisCorrectionConfig.rotation_max_deg,
            "max_iter": BasisCorrectionConfig.max_iter,
        },
    )


if __name__ == "__main__":
    main()
