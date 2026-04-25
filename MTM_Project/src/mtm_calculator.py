"""
主 MTM 计算流程模块

端到端闭环（中期验收口试可演示）：
  单 TIFF / 文件夹批量 TIFF -> 预处理(H_pixel) -> MTM 重构(H_modes)
  -> 输出 NPY/CSV + 热力图 + 独立 TXT 日志（每次运行一份）
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
    """
    每次运行生成独立 TXT 日志文件，并同时输出到控制台。
    """
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
    logging.info("日志系统初始化完成")
    return log_path


def _find_ptm_tiffs(input_dir: str) -> List[str]:
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))
    return sorted(set(files))


def _save_complex_csv(path: str, mat: np.ndarray) -> None:
    """
    以 'a+bj' 的字符串形式保存复矩阵，便于 Excel/人工检查。
    """
    m = np.asarray(mat, dtype=np.complex128)
    with open(path, "w", encoding="utf-8") as f:
        for row in m:
            f.write(",".join([f"{c.real:.10g}{c.imag:+.10g}j" for c in row]))
            f.write("\n")


def _build_m_in(n_input: int, n_modes: int) -> np.ndarray:
    """
    在缺少“输入侧模式基矢标定”的情况下，
    采用激励基 = 模式基 的默认假设：
      - 若 N_input == N_modes: 使用单位阵 I
      - 若 N_input >  N_modes: 取前 N_modes 列的单位阵 (N_input, N_modes)
      - 若 N_input <  N_modes: 仅能计算前 N_input 个模式（降级）
    """
    if n_input >= n_modes:
        return np.eye(n_input, n_modes, dtype=np.complex128)
    
    return np.eye(n_input, n_input, dtype=np.complex128)


def _build_m_in_hadamard(n_input: int, n_modes: int) -> np.ndarray:
    """
    使用 Hadamard 正交基构造输入侧 M_in（在输入 pattern 空间中）。

    约定：
      - 当 n_input >= n_modes 时，返回 shape (n_input, n_modes)
      - 当 n_input <  n_modes 时，只能返回 (n_input, n_input) 并降级模式数
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
) -> None:
    """
    一键运行端到端 MTM 管线。

    Args:
        input_dir:   PTM TIFF 文件夹
        output_root: 输出根目录；会在其下创建 run_YYYYmmdd_HHMMSS 子目录
        fiber_params: generate_lp_modes 所需光纤参数（um）
        num_modes:   LP 模式数量
        denoise_method/denoise_params: 预处理去噪配置
        mtm_thresholds: 直光纤对角特性验收阈值：
            {'offdiag_max': 0.05, 'diag_mean_target': 1.0, 'diag_mean_tol': 0.2}
    """
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_tag)
    log_path = _setup_logging(run_dir)

    logger = logging.getLogger("mtm_pipeline")
    logger.info("程序启动")
    logger.info("输入目录: %s", input_dir)
    logger.info("输出目录: %s", run_dir)
    logger.info("日志文件: %s", log_path)
    logger.info("模式数量配置: num_modes=%d", num_modes)
    logger.info(
        "预处理配置: denoise_method=%s, denoise_params=%s",
        denoise_method,
        denoise_params,
    )
    logger.info("输入基矢配置: input_basis=%s, m_in_path=%s", input_basis, m_in_path)

    thresholds = mtm_thresholds or {}
    offdiag_thr = float(thresholds.get("offdiag_max", 0.05))
    diag_target = float(thresholds.get("diag_mean_target", 1.0))
    diag_tol = float(thresholds.get("diag_mean_tol", 0.20))

    tiff_files = _find_ptm_tiffs(input_dir)
    if not tiff_files:
        logger.error("未找到任何 TIFF 文件: %s", input_dir)
        return
    logger.info("发现 %d 个 PTM TIFF 文件", len(tiff_files))

    pre = PTMPreprocessor(
        denoise_method=denoise_method, denoise_params=denoise_params
    )
    recon = MTMReconstructor(num_modes=num_modes)

    for tif_path in tiff_files:
        base = os.path.splitext(os.path.basename(tif_path))[0]
        logger.info("开始处理: %s", tif_path)

        try:
            H_pixel, pre_stats = pre.preprocess_to_h_pixel(tif_path)
            n_out_pix, n_input = H_pixel.shape
            logger.info(
                "H_pixel 构建完成: shape=%s (N_out_pix x N_input)",
                H_pixel.shape,
            )
            logger.info("预处理统计: %s", pre_stats)

            
            
            
            
            
            
            candidates = [(320, 256), (256, 320), (512, 512), (1024, 1024)]
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
                    "无法从像素数精确推断 CCD 尺寸，使用 grid_size=%s",
                    grid_size,
                )

            M_out, orth_err = recon.build_output_mode_matrix(
                grid_size=grid_size, fiber_params=fiber_params
            )

            
            if input_basis == "identity":
                M_in = _build_m_in(n_input=n_input, n_modes=M_out.shape[1])
            elif input_basis == "hadamard":
                M_in = _build_m_in_hadamard(n_input=n_input, n_modes=M_out.shape[1])
            elif input_basis == "file":
                if not m_in_path:
                    raise ValueError("input_basis=file 但未提供 m_in_path")
                if not os.path.exists(m_in_path):
                    raise FileNotFoundError(f"M_in 文件不存在: {m_in_path}")
                M_in = np.load(m_in_path).astype(np.complex128)
            else:
                raise ValueError(f"不支持的 input_basis: {input_basis}")

            if M_in.shape[1] != M_out.shape[1]:
                logger.warning(
                    "输入侧模式数与输出侧不一致，将以较小值为准: M_in=%s, M_out=%s",
                    M_in.shape,
                    M_out.shape,
                )

            
            k = min(M_in.shape[1], M_out.shape[1])
            M_in_use = M_in[:, :k]
            M_out_use = M_out[:, :k]

            H_modes = recon.compute_mtm(H_pixel=H_pixel, M_in=M_in_use, M_out=M_out_use)

            stats = recon.evaluate_mtm(H_modes)
            stats["orth_err"] = float(orth_err)
            logger.info("MTM 指标: %s", stats)

            
            if stats["offdiag_max"] > offdiag_thr:
                logger.warning(
                    "MTM 非对角元素过大: offdiag_max=%.3g > %.3g",
                    stats["offdiag_max"],
                    offdiag_thr,
                )
            if abs(stats["diag_mean"] - diag_target) > diag_tol:
                logger.warning(
                    "MTM 主对角均值偏离目标: diag_mean=%.3g, target=%.3g±%.3g",
                    stats["diag_mean"],
                    diag_target,
                    diag_tol,
                )

            
            out_prefix = os.path.join(run_dir, base)
            npy_path = f"{out_prefix}_mtm.npy"
            csv_path = f"{out_prefix}_mtm.csv"
            stats_path = f"{out_prefix}_mtm_stats.txt"
            heatmap_path = f"{out_prefix}_mtm_heatmap.png"

            np.save(npy_path, H_modes)
            _save_complex_csv(csv_path, H_modes)

            with open(stats_path, "w", encoding="utf-8") as f:
                f.write("MTM statistics\n")
                f.write("=================\n")
                f.write(f"file: {tif_path}\n")
                f.write(f"H_pixel shape: {H_pixel.shape}\n")
                f.write(f"M_in shape: {M_in_use.shape}\n")
                f.write(f"M_out shape: {M_out_use.shape}\n")
                f.write("\nmetrics:\n")
                for k2, v2 in stats.items():
                    f.write(f"  {k2}: {v2}\n")

            recon.plot_mtm_heatmap(
                H_modes, save_path=heatmap_path, title=f"MTM |{base}|"
            )

            logger.info(
                "输出完成: %s, %s, %s, %s",
                os.path.basename(npy_path),
                os.path.basename(csv_path),
                os.path.basename(stats_path),
                os.path.basename(heatmap_path),
            )

        except Exception as exc:
            logger.exception("处理失败: %s, 错误: %s", tif_path, exc)

    logger.info("全部文件处理完成")

