"""
主MTM计算流程模块
从标准PTM输入(TIFF)到MTM结果与日志输出的端到端管线
"""

import os
import glob
import logging
from typing import Dict, List

import numpy as np
import tifffile

from .data_preprocessing import PTMPreprocessor
from .mtm_reconstruction import MTMReconstructor
from .error_metrics import mse, relative_error, save_error_row_csv


def get_project_root() -> str:
    """根据当前文件位置推断项目根目录（此项目位于桌面根目录下）"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "mtm_calculation.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("MTM计算日志系统初始化完成")
    return log_path


def find_ptm_tiffs(input_dir: str) -> List[str]:
    """在输入目录中查找所有PTM TIFF文件"""
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))
    return sorted(files)


def run_mtm_pipeline(
    fiber_params: Dict[str, float] | None = None,
    num_modes: int = 10,
    denoise_method: str = "gaussian",
) -> None:
    """
    执行端到端MTM计算流程

    步骤：
      1. 扫描 data/input_tiff 下的所有PTM TIFF文件
      2. 对每个文件：加载 -> 去噪 -> 相位提取
      3. 基于LP模式基和相位数据估计MTM
      4. 将去噪结果、相位结果和MTM保存到 data/output_mtm
      5. 全过程写入 logs/mtm_calculation.log
    """

    project_root = get_project_root()
    input_dir = os.path.join(project_root, "data", "input_tiff")
    output_dir = os.path.join(project_root, "data", "output_mtm")
    log_dir = os.path.join(project_root, "logs")

    os.makedirs(output_dir, exist_ok=True)
    log_path = setup_logging(log_dir)
    logger = logging.getLogger("mtm_pipeline")

    if fiber_params is None:
        
        fiber_params = {
            "core_radius": 25.0,   
            "wavelength": 0.532,   
            "na": 0.22,
        }

    logger.info(f"项目根目录: {project_root}")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"日志文件: {log_path}")

    tiff_files = find_ptm_tiffs(input_dir)
    if not tiff_files:
        logger.warning("未在 data/input_tiff 中找到任何PTM TIFF文件，流程结束")
        return

    logger.info(f"发现 {len(tiff_files)} 个PTM输入文件")

    preprocessor = PTMPreprocessor(denoise_method=denoise_method)
    reconstructor = MTMReconstructor(num_modes=num_modes)

    
    ideal_mtm = np.eye(num_modes, dtype=np.complex128)
    baseline_csv = os.path.join(output_dir, "mtm_baseline_errors.csv")

    for tif_path in tiff_files:
        base_name = os.path.splitext(os.path.basename(tif_path))[0]
        logger.info(f"开始处理: {tif_path}")

        try:
            
            denoised, phase = preprocessor.preprocess(
                tif_path, extract_phase_method="hilbert"
            )
            logger.info(
                f"预处理完成: 形状 denoised={denoised.shape}, phase={phase.shape}"
            )

            
            if phase.ndim == 3:
                num_meas, h, w = phase.shape
            elif phase.ndim == 2:
                num_meas, (h, w) = 1, phase.shape
                phase = phase[None, ...]
            else:
                raise ValueError(f"不支持的相位数据维度: {phase.shape}")

            mode_basis = reconstructor.generate_lp_modes((h, w), fiber_params)

            
            T = reconstructor.compute_mtm_from_phase(phase, mode_basis)
            logger.info(f"MTM估计完成，矩阵形状: {T.shape}")

            
            
            try:
                T_square = np.asarray(T)
                if T_square.shape[0] != T_square.shape[1]:
                    raise ValueError("MTM 非方阵，无法直接与理想对角阵比较")
                
                min_dim = min(num_modes, T_square.shape[0])
                T_use = T_square[:min_dim, :min_dim]
                ideal_use = ideal_mtm[:min_dim, :min_dim]
                mse_val = mse(T_use, ideal_use)
                re_val = relative_error(T_use, ideal_use)
                save_error_row_csv(
                    baseline_csv,
                    {
                        "file": base_name,
                        "num_modes_used": min_dim,
                        "mse": mse_val,
                        "relative_error": re_val,
                    },
                )
                logger.info(
                    f"与理想直光纤MTM比较: mse={mse_val:.3e}, RE={re_val:.3e}"
                )
            except Exception as exc_err:
                logger.warning(f"计算基线误差失败: {exc_err}")

            
            out_prefix = os.path.join(output_dir, base_name)

            
            denoised_tiff = f"{out_prefix}_denoised.tiff"
            phase_tiff = f"{out_prefix}_phase.tiff"
            denoised_npy = f"{out_prefix}_denoised.npy"
            phase_npy = f"{out_prefix}_phase.npy"
            mtm_npy = f"{out_prefix}_mtm.npy"

            tifffile.imwrite(denoised_tiff, denoised.astype(np.float32))
            tifffile.imwrite(phase_tiff, phase.astype(np.float32))
            np.save(denoised_npy, denoised)
            np.save(phase_npy, phase)
            np.save(mtm_npy, T)

            logger.info(
                f"结果已保存: {denoised_tiff}, {phase_tiff}, {mtm_npy}"
            )

        except Exception as exc:
            logger.exception(f"处理文件 {tif_path} 时出错: {exc}")

    logger.info("全部PTM文件处理完成")


if __name__ == "__main__":
    run_mtm_pipeline()
