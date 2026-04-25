from __future__ import annotations

"""
项目配置文件

包含光纤参数、预处理设置、MTM计算阈值和默认目录路径。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FiberParams:
    """
    光纤参数（用于LP模式基矢构造和仿真）
    单位：微米（um），除非另有说明
    """

    core_diameter_um: float = 50.0      
    core_radius_um: float = 25.0        
    na: float = 0.22                    
    n_cladding: float = 1.444           
    n_core: float = 1.461               
    wavelength_um: float = 1.55         
    sim_extent_um: float = 35.0         


@dataclass(frozen=True)
class PreprocessConfig:
    """
    PTM数据预处理参数
    """

    denoise_method: str = "gaussian"    
    gaussian_sigma: float = 1.2         
    gaussian_ksize: int = 5             
    median_ksize: int = 3               
    bilateral_d: int = 7                
    bilateral_sigma_color: float = 0.055
    bilateral_sigma_space: float = 6.0


@dataclass(frozen=True)
class MTMConfig:
    """
    MTM计算与验收阈值配置
    """

    num_modes: int = 8                  
    offdiag_max_threshold: float = 0.05 
    diag_mean_target: float = 1.0       
    diag_mean_tol: float = 0.20         


@dataclass(frozen=True)
class BasisCorrectionConfig:
    """
    模式基校正配置（参考文献中“基底误差导致伪耦合”的思路）
    """
    enabled: bool = False
    optimize: bool = True
    scale_min: float = 0.90
    scale_max: float = 1.10
    shift_max_px: float = 12.0
    rotation_max_deg: float = 10.0
    max_iter: int = 30


@dataclass(frozen=True)
class Paths:
    """
    默认目录结构
    - input_tiff_dir: 存放PTM TIFF文件（支持批量处理）
    - output_dir:     输出MTM结果（指标、热力图、日志，每次运行独立子目录）
    """

    input_tiff_dir: str = "data/input_tiff"
    output_dir: str = "data/output_mtm"
