"""
主MTM计算流水线模块

端到端闭环流水线：
  单个TIFF / 批量TIFF文件夹 -> 预处理（H_pixel） -> MTM重建（H_modes）
  -> 输出NPY/CSV + 热图 + 独立TXT日志（每次运行一个）
"""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .data_preprocessing import PTMPreprocessor
from .mtm_reconstruction import MTMReconstructor


def _setup_logging(run_dir: str) -> str:
    """为每次运行创建独立的TXT日志文件，并输出到控制台。"""
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("日志系统已初始化")
    return log_path


def _find_ptm_tiffs(input_dir: str) -> List[str]:
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))
    return sorted(set(files))


def _save_complex_csv(path: str, mat: np.ndarray) -> None:
    """以'a+bj'字符串格式保存复数矩阵，便于检查。"""
    m = np.asarray(mat, dtype=np.complex128)
    with open(path, "w", encoding="utf-8") as f:
        for row in m:
            f.write(",".join([f"{c.real:.10g}{c.imag:+.10g}j" for c in row]))
            f.write("\n")


def _build_m_in(n_input: int, n_modes: int) -> np.ndarray:
    """
    使用单位矩阵假设构建输入侧模式基矩阵。
    - 如果 N_input == N_modes：使用单位矩阵 I
    - 如果 N_input >  N_modes：取 I(N_input) 的前 N_modes 列
    - 如果 N_input <  N_modes：降级为 N_input 模式
    """
    if n_input >= n_modes:
        return np.eye(n_input, n_modes, dtype=np.complex128)
    return np.eye(n_input, n_input, dtype=np.complex128)


def _build_m_in_hadamard(n_input: int, n_modes: int) -> np.ndarray:
    """
    使用Hadamard正交基（Hadamard orthogonal basis）构建输入侧M_in。

    约定：
      - 当 n_input >= n_modes：返回形状为 (n_input, n_modes)
      - 当 n_input <  n_modes：返回形状为 (n_input, n_input)，模式数降级
    """
    def next_pow2(x: int) -> int:
        p = 1
        while p < x:
            p *= 2
        return p

    if n_input <= 0:
        raise ValueError("n_input 必须为正")
    if n_modes <= 0:
        raise ValueError("n_modes 必须为正")

    m = next_pow2(n_input)
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])
    H = H / np.sqrt(float(m))
    H = H[:n_input, :n_input]

    
    H = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-12)

    if n_input >= n_modes:
        return H[:, :n_modes].astype(np.complex128)
    return H.astype(np.complex128)


def run_mtm_pipeline(
    input_dir: str,
    output_root: str,
    fiber_params: Dict[str, float],
    num_modes: int,
    denoise_method: str = "gaussian",
    denoise_params: Optional[Dict[str, float]] = None,
    input_basis: str = "hadamard",
    m_in_path: Optional[str] = None,
    mtm_thresholds: Optional[Dict[str, float]] = None,
    basis_correction: Optional[Dict[str, float]] = None,
) -> None:
    """
    运行端到端MTM计算流水线。

    参数:
        input_dir:   PTM TIFF文件目录
        output_root: 输出根目录（创建 run_YYYYmmdd_HHMMSS 子目录）
        fiber_params: 光纤参数，用于LP模式生成（单位：微米）
        num_modes:   LP模式数量
        denoise_method/denoise_params: 预处理去噪配置
        input_basis: 输入基类型（'identity'，'hadamard'，或 'file'）
        m_in_path:   M_in文件路径（当input_basis='file'时必需）
        mtm_thresholds: 直纤对角线接受阈值
        basis_correction: 模式基校正配置（缩放/平移/旋转）
    """
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_tag)
    log_path = _setup_logging(run_dir)

    logger = logging.getLogger("mtm_pipeline")
    logger.info("流水线启动")
    logger.info("输入目录: %s", input_dir)
    logger.info("输出目录: %s", run_dir)
    logger.info("日志文件: %s", log_path)
    logger.info("模式数量: num_modes=%d", num_modes)
    logger.info(
        "预处理配置: denoise_method=%s, denoise_params=%s",
        denoise_method, denoise_params,
    )
    logger.info("输入基配置: input_basis=%s, m_in_path=%s", input_basis, m_in_path)
    logger.info("模式基校正配置: %s", basis_correction)

    thresholds = mtm_thresholds or {}
    offdiag_thr = float(thresholds.get("offdiag_max", 0.05))
    diag_target = float(thresholds.get("diag_mean_target", 1.0))
    diag_tol = float(thresholds.get("diag_mean_tol", 0.20))

    tiff_files = _find_ptm_tiffs(input_dir)
    if not tiff_files:
        logger.error("在目录中未找到TIFF文件: %s", input_dir)
        return
    logger.info("找到 %d 个PTM TIFF文件", len(tiff_files))

    pre = PTMPreprocessor(
        denoise_method=denoise_method, denoise_params=denoise_params
    )
    recon = MTMReconstructor(num_modes=num_modes)

    for tif_path in tiff_files:
        base = os.path.splitext(os.path.basename(tif_path))[0]
        logger.info("处理文件: %s", tif_path)

        try:
            H_pixel, pre_stats = pre.preprocess_to_h_pixel(tif_path)
            n_out_pix, n_input = H_pixel.shape
            logger.info(
                "构建H_pixel: 形状=%s (N_out_pix x N_input)",
                H_pixel.shape,
            )
            logger.info("预处理统计: %s", pre_stats)

            
            candidates = [
                (320, 256), (256, 320),  
                (64, 64),                
                (512, 512), (1024, 1024),
            ]
            grid_size = None
            for h, w in candidates:
                if h * w == n_out_pix:
                    grid_size = (h, w)
                    break
            if grid_size is None:
                side = int(np.sqrt(n_out_pix))
                grid_size = (side, n_out_pix // max(side, 1))
                if grid_size[0] * grid_size[1] != n_out_pix:
                    grid_size = (n_out_pix, 1)
                logger.warning(
                    "无法从像素数推断CCD尺寸，使用grid_size=%s",
                    grid_size,
                )

            
            
            M_out_init, _ = recon.build_output_mode_matrix(
                grid_size=grid_size, fiber_params=fiber_params
            )
            if input_basis == "identity":
                M_in = _build_m_in(n_input=n_input, n_modes=M_out_init.shape[1])
            elif input_basis == "hadamard":
                M_in = _build_m_in_hadamard(n_input=n_input, n_modes=M_out_init.shape[1])
            elif input_basis == "file":
                if not m_in_path:
                    raise ValueError("input_basis='file'但未提供m_in_path")
                if not os.path.exists(m_in_path):
                    raise FileNotFoundError(f"M_in文件未找到: {m_in_path}")
                M_in = np.load(m_in_path).astype(np.complex128)
            else:
                raise ValueError(f"不支持的input_basis: {input_basis}")

            correction_stats = {}
            correction_enabled = bool((basis_correction or {}).get("enabled", False))
            correction_opt = bool((basis_correction or {}).get("optimize", True))
            joint = bool((basis_correction or {}).get("joint", False))
            if correction_enabled:
                if correction_opt:
                    if joint:
                        M_out, M_in, orth_err, correction_stats = recon.optimize_joint_mode_bases(
                            H_pixel=H_pixel,
                            M_in=M_in,
                            grid_size=grid_size,
                            fiber_params=fiber_params,
                            correction_cfg=(basis_correction or {}),
                        )
                    else:
                        M_out, orth_err, correction_stats = recon.optimize_output_mode_matrix(
                            H_pixel=H_pixel,
                            M_in=M_in,
                            grid_size=grid_size,
                            fiber_params=fiber_params,
                            correction_cfg=(basis_correction or {}),
                        )
                else:
                    M_out, orth_err = recon.build_output_mode_matrix(
                        grid_size=grid_size,
                        fiber_params=fiber_params,
                        correction_params={
                            "scale": 1.0,
                            "shift_x_px": 0.0,
                            "shift_y_px": 0.0,
                            "rotation_deg": 0.0,
                        },
                    )
                    correction_stats = {
                        "scale": 1.0,
                        "shift_x_px": 0.0,
                        "shift_y_px": 0.0,
                        "rotation_deg": 0.0,
                        "objective": float("nan"),
                        "opt_success": 1.0,
                    }
            else:
                M_out, orth_err = recon.build_output_mode_matrix(
                    grid_size=grid_size, fiber_params=fiber_params
                )

            if M_in.shape[1] != M_out.shape[1]:
                logger.warning(
                    "输入/输出模式数量不匹配，使用最小值: M_in=%s, M_out=%s",
                    M_in.shape, M_out.shape,
                )

            
            k = min(M_in.shape[1], M_out.shape[1])
            M_in_use = M_in[:, :k]
            M_out_use = M_out[:, :k]

            H_modes = recon.compute_mtm(H_pixel=H_pixel, M_in=M_in_use, M_out=M_out_use)

            stats = recon.evaluate_mtm(H_modes)
            stats_norm = recon.evaluate_mtm_gain_normalized(H_modes)
            stats["norm_diag_mean"] = stats_norm["diag_mean"]
            stats["norm_offdiag_max"] = stats_norm["offdiag_max"]
            stats["orth_err"] = float(orth_err)
            for ck, cv in correction_stats.items():
                stats[f"basis_correction_{ck}"] = float(cv)
            logger.info("MTM指标: %s", stats)

            
            if stats_norm["offdiag_max"] > offdiag_thr:
                logger.warning(
                    "MTM（增益归一化后）非对角幅值过大: norm_offdiag_max=%.3g > %.3g",
                    stats_norm["offdiag_max"], offdiag_thr,
                )
            if abs(stats_norm["diag_mean"] - diag_target) > diag_tol:
                logger.warning(
                    "MTM（增益归一化后）对角均值偏离: norm_diag_mean=%.3g, target=%.3g+/-%.3g",
                    stats_norm["diag_mean"], diag_target, diag_tol,
                )

            
            out_prefix = os.path.join(run_dir, base)
            npy_path = f"{out_prefix}_mtm.npy"
            csv_path = f"{out_prefix}_mtm.csv"
            stats_path = f"{out_prefix}_mtm_stats.txt"
            heatmap_path = f"{out_prefix}_mtm_heatmap.png"

            np.save(npy_path, H_modes)
            _save_complex_csv(csv_path, H_modes)

            with open(stats_path, "w", encoding="utf-8") as f:
                f.write("MTM统计\n")
                f.write("=================\n")
                f.write(f"文件: {tif_path}\n")
                f.write(f"H_pixel形状: {H_pixel.shape}\n")
                f.write(f"M_in形状: {M_in_use.shape}\n")
                f.write(f"M_out形状: {M_out_use.shape}\n")
                f.write("\n指标:\n")
                for k2, v2 in stats.items():
                    f.write(f"  {k2}: {v2}\n")

            recon.plot_mtm_heatmap(
                H_modes, save_path=heatmap_path, title=f"MTM |{base}|"
            )

            logger.info(
                "输出已保存: %s, %s, %s, %s",
                os.path.basename(npy_path),
                os.path.basename(csv_path),
                os.path.basename(stats_path),
                os.path.basename(heatmap_path),
            )

        except Exception as exc:
            logger.exception("处理失败: %s, 错误: %s", tif_path, exc)

    logger.info("所有文件处理完成")
