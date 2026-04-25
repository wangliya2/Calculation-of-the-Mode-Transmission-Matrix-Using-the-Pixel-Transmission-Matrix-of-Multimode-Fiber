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
        description="PTM(TIFF) -> MTM 端到端流程（Plöschner 2015）"
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=Paths.input_tiff_dir,
        help="包含PTM TIFF文件的文件夹（默认：data/input_tiff）",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=Paths.output_dir,
        help="输出结果写入的文件夹（默认：data/output_mtm）",
    )
    p.add_argument(
        "--num-modes",
        type=int,
        default=MTMConfig.num_modes,
        help="使用的LP模式（LP mode）数量（默认：8）",
    )
    p.add_argument(
        "--disable-auto-num-modes",
        action="store_true",
        help="关闭基于当前光纤参数的自动模式数限制",
    )
    p.add_argument(
        "--denoise-method",
        type=str,
        default=PreprocessConfig.denoise_method,
        choices=["gaussian", "median", "bilateral"],
        help="振幅的去噪方法（默认：gaussian；bilateral 保边，适合 LP 环状场）",
    )
    p.add_argument(
        "--input-basis",
        type=str,
        default="hadamard",
        choices=["hadamard", "identity", "file"],
        help="M_in的输入基：hadamard/identity/或从文件加载（默认：hadamard）",
    )
    p.add_argument(
        "--m-in-path",
        type=str,
        default="data/standard_lp_modes/M_in_hadamard_8.npy",
        help="当--input-basis=file时，从此路径加载M_in（默认指向生成的文件）",
    )
    p.add_argument(
        "--basis-correction",
        action="store_true",
        help="启用模式基校正（缩放/平移/旋转优化）",
    )
    p.add_argument(
        "--basis-correction-no-opt",
        action="store_true",
        help="启用模式基校正但不做参数优化（使用默认参数）",
    )
    p.add_argument(
        "--joint-basis-correction",
        action="store_true",
        help="在 basis-correction 优化时，若 M_in 为空间模式矩阵 (H*W,N_modes)，则与 M_out 使用同一组几何参数联合优化",
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
            print(f"[AutoMode] 请求模式数={req_modes}，可支持模式数≈{auto_modes}，自动降为 {use_modes}")
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
