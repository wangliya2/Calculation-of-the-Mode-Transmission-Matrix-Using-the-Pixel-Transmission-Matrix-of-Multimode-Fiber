from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FiberParams:
    """
    光纤参数（用于 LP 模式基矢构造/仿真）
    单位约定：微米（um）
    """

    core_radius_um: float = 50.0
    na: float = 0.22
    n_cladding: float = 1.444
    wavelength_um: float = 1.064  # 1064 nm


@dataclass(frozen=True)
class PreprocessConfig:
    """
    预处理参数（对应中期验收要求）
    """

    denoise_method: str = "gaussian"  # 'gaussian' or 'median'
    gaussian_sigma: float = 1.2
    gaussian_ksize: int = 5
    median_ksize: int = 3


@dataclass(frozen=True)
class MTMConfig:
    """
    MTM 计算与验收阈值配置
    """

    num_modes: int = 8
    offdiag_max_threshold: float = 0.05
    diag_mean_target: float = 1.0
    diag_mean_tol: float = 0.20  


@dataclass(frozen=True)
class Paths:
    """
    默认目录结构
    - input_tiff_dir: 放 PTM TIFF（可批量）
    - output_dir:     输出 MTM/指标/热力图/日志（每次运行独立子目录）
    """

    input_tiff_dir: str = "data/input_tiff"
    output_dir: str = "data/output_mtm"

