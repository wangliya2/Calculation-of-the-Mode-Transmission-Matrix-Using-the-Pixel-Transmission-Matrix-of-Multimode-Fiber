"""
从 article_MMF_disorder 仓库 Data/ 目录加载像素域 TM 与模式矩阵，运行完整 MTM 重建。

典型用法（与仓库 README 一致：拼接 TM 分片）：
  python run_article_mtm.py ^
    --tm-part0 path/TM25_0.npy --tm-part1 path/TM25_1.npy ^
    --conversion path/conversion_matrices.npz ^
    --output-dir data/output_mtm

可选：启用与 main.py 相同的 basis-correction / joint-correction（joint 需要空间 M_in）。
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np

from config import BasisCorrectionConfig, FiberParams, MTMConfig, Paths
from src.mtm_calculator import _save_complex_csv, _setup_logging
from src.mtm_reconstruction import MTMReconstructor


def _infer_grid_size(n_out_pix: int) -> Tuple[int, int]:
    candidates = [
        (320, 256),
        (256, 320),
        (64, 64),
        (512, 512),
        (1024, 1024),
    ]
    for h, w in candidates:
        if h * w == n_out_pix:
            return h, w
    side = int(np.sqrt(n_out_pix))
    h, w = side, n_out_pix // max(side, 1)
    if h * w != n_out_pix:
        return n_out_pix, 1
    return h, w


def _maybe_transpose_to_match(
    M: np.ndarray, target_rows: int, name: str
) -> np.ndarray:
    M = np.asarray(M, dtype=np.complex128)
    if M.shape[0] == target_rows:
        return M
    if M.shape[1] == target_rows:
        return M.T
    raise ValueError(
        f"{name} 形状 {M.shape} 无法对齐到行数 {target_rows}（尝试转置仍不匹配）"
    )


def _load_conversion_modes(path: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    if "modes_in" not in z.files or "modes_out" not in z.files:
        raise KeyError(f"{path} 中缺少 modes_in / modes_out 键: {z.files}")
    return z["modes_in"], z["modes_out"]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="使用 article_MMF_disorder Data/ 数据跑 MTM 重建")
    p.add_argument("--tm-part0", type=str, required=True, help="TMxx_0.npy 路径")
    p.add_argument("--tm-part1", type=str, required=True, help="TMxx_1.npy 路径")
    p.add_argument(
        "--conversion",
        type=str,
        required=True,
        help="conversion_matrices.npz 路径（需含 modes_in, modes_out）",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=Paths.output_dir,
        help=f"输出根目录（默认 {Paths.output_dir}）",
    )
    p.add_argument("--tag", type=str, default="", help="可选 run 标签后缀")
    p.add_argument(
        "--param-json",
        type=str,
        default="",
        help="可选 param.json（仅用于记录到日志/统计文件，不参与矩阵维度推断）",
    )
    p.add_argument(
        "--num-modes",
        type=int,
        default=MTMConfig.num_modes,
        help="模式数量上限（会与矩阵列数取最小值）",
    )
    p.add_argument("--basis-correction", action="store_true")
    p.add_argument("--basis-correction-no-opt", action="store_true")
    p.add_argument("--joint-basis-correction", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))

    p0 = args.tm_part0 if os.path.isabs(args.tm_part0) else os.path.join(project_root, args.tm_part0)
    p1 = args.tm_part1 if os.path.isabs(args.tm_part1) else os.path.join(project_root, args.tm_part1)
    part0 = np.load(p0)
    part1 = np.load(p1)
    H_pixel = np.concatenate([part0, part1], axis=0).astype(np.complex128)

    cpath = args.conversion if os.path.isabs(args.conversion) else os.path.join(project_root, args.conversion)
    modes_in0, modes_out0 = _load_conversion_modes(cpath)
    n_out, n_in = H_pixel.shape
    M_in = _maybe_transpose_to_match(modes_in0, n_in, "modes_in")
    M_out = _maybe_transpose_to_match(modes_out0, n_out, "modes_out")

    k0 = min(M_in.shape[1], M_out.shape[1], int(args.num_modes))
    M_in = M_in[:, :k0]
    M_out = M_out[:, :k0]

    grid_size = _infer_grid_size(n_out)

    fiber_params = {
        "core_radius": FiberParams.core_radius_um,
        "wavelength": FiberParams.wavelength_um,
        "na": FiberParams.na,
        "sim_extent_um": FiberParams.sim_extent_um,
    }

    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.tag.strip():
        run_tag = f"{run_tag}_{args.tag.strip()}"
    out_root = os.path.join(project_root, args.output_dir)
    run_dir = os.path.join(out_root, run_tag)
    log_path = _setup_logging(run_dir)

    basis_correction = {
        "enabled": bool(args.basis_correction),
        "optimize": bool(args.basis_correction and (not args.basis_correction_no_opt)),
        "joint": bool(args.joint_basis_correction),
        "scale_min": BasisCorrectionConfig.scale_min,
        "scale_max": BasisCorrectionConfig.scale_max,
        "shift_max_px": BasisCorrectionConfig.shift_max_px,
        "rotation_max_deg": BasisCorrectionConfig.rotation_max_deg,
        "max_iter": BasisCorrectionConfig.max_iter,
    }

    print(f"[article_mtm] H_pixel={H_pixel.shape}, M_in={M_in.shape}, M_out={M_out.shape}")
    print(f"[article_mtm] grid_size={grid_size}, log={log_path}")

    recon = MTMReconstructor(num_modes=k0)
    M_in_use = M_in
    M_out_use = M_out
    orth_err = 0.0
    correction_stats: Dict[str, float] = {}

    if basis_correction["enabled"]:
        
        ident = {
            "scale": 1.0,
            "shift_x_px": 0.0,
            "shift_y_px": 0.0,
            "rotation_deg": 0.0,
        }
        if basis_correction["optimize"]:
            if basis_correction["joint"]:
                M_out_use, M_in_use, orth_err, correction_stats = recon.optimize_flat_mode_bases(
                    H_pixel=H_pixel,
                    M_in_flat=M_in,
                    M_out_flat=M_out,
                    grid_size=grid_size,
                    correction_cfg=basis_correction,
                )
            else:
                M_out_use, orth_err, correction_stats = recon.optimize_flat_output_only(
                    H_pixel=H_pixel,
                    M_in_flat=M_in,
                    M_out_flat=M_out,
                    grid_size=grid_size,
                    correction_cfg=basis_correction,
                )
        else:
            if basis_correction["joint"]:
                M_out_use, orth_out = recon.reorthonormalize_flat_spatial_modes(
                    M_out, grid_size, ident
                )
                M_in_use, orth_in = recon.reorthonormalize_flat_spatial_modes(
                    M_in, grid_size, ident
                )
                orth_err = float(max(orth_out, orth_in))
            else:
                M_out_use, orth_err = recon.reorthonormalize_flat_spatial_modes(
                    M_out, grid_size, ident
                )
                M_in_use = M_in
            correction_stats = {
                **ident,
                "objective": float("nan"),
                "opt_success": 1.0,
                "joint_mode": 14.0 if basis_correction["joint"] else 15.0,
            }

    if M_out_use.shape[1] < 1 or M_in_use.shape[1] < 1:
        print(
            "[article_mtm] 警告: 模式基校正后有效列数为 0，回退为未校正 modes_in/modes_out。"
        )
        M_in_use, M_out_use = M_in, M_out
        orth_err = 0.0
        correction_stats = {
            **ident,
            "objective": float("nan"),
            "opt_success": 0.0,
            "joint_mode": -1.0,
        }

    H_modes = recon.compute_mtm(H_pixel=H_pixel, M_in=M_in_use, M_out=M_out_use)
    stats = recon.evaluate_mtm(H_modes)
    stats["orth_err"] = float(orth_err)
    for ck, cv in correction_stats.items():
        stats[f"basis_correction_{ck}"] = float(cv)

    base = "article_TM"
    out_prefix = os.path.join(run_dir, base)
    npy_path = f"{out_prefix}_mtm.npy"
    csv_path = f"{out_prefix}_mtm.csv"
    stats_path = f"{out_prefix}_mtm_stats.txt"
    heatmap_path = f"{out_prefix}_mtm_heatmap.png"

    np.save(npy_path, H_modes)
    _save_complex_csv(csv_path, H_modes)

    param_note = ""
    if args.param_json.strip():
        pj = (
            args.param_json
            if os.path.isabs(args.param_json)
            else os.path.join(project_root, args.param_json)
        )
        if os.path.exists(pj):
            with open(pj, "r", encoding="utf-8") as f:
                param_note = f.read()

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("MTM统计（article_MMF_disorder 数据）\n")
        f.write("=================\n")
        f.write(f"tm_part0: {args.tm_part0}\n")
        f.write(f"tm_part1: {args.tm_part1}\n")
        f.write(f"conversion: {args.conversion}\n")
        f.write(f"H_pixel形状: {H_pixel.shape}\n")
        f.write(f"M_in形状: {M_in_use.shape}\n")
        f.write(f"M_out形状: {M_out_use.shape}\n")
        f.write(f"grid_size: {grid_size}\n")
        if param_note:
            f.write("\nparam.json:\n")
            f.write(param_note)
            f.write("\n")
        f.write("\n指标:\n")
        for k2, v2 in stats.items():
            f.write(f"  {k2}: {v2}\n")

    recon.plot_mtm_heatmap(H_modes, save_path=heatmap_path, title="MTM |article_TM|")
    print(
        "OK:",
        os.path.basename(npy_path),
        os.path.basename(csv_path),
        os.path.basename(stats_path),
        os.path.basename(heatmap_path),
    )


if __name__ == "__main__":
    main()
