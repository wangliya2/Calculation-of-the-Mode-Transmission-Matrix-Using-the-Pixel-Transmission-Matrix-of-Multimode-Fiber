from __future__ import annotations

"""
Project configuration.

Defines fiber parameters, preprocessing settings, MTM thresholds, and default paths.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FiberParams:
    """
    Fiber parameters used for LP-mode simulation and basis construction.
    Units are micrometers (um) unless noted otherwise.
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
    PTM preprocessing parameters.
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
    MTM computation and acceptance thresholds.
    """

    num_modes: int = 8                  
    offdiag_max_threshold: float = 0.05 
    diag_mean_target: float = 1.0       
    diag_mean_tol: float = 0.20         


@dataclass(frozen=True)
class BasisCorrectionConfig:
    """
    Basis-correction configuration (to reduce pseudo-coupling caused by geometric mismatch).
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
    Default directory structure.
    - input_tiff_dir: PTM TIFF inputs (batch supported)
    - output_dir:     MTM outputs (metrics, heatmaps, logs; timestamped runs)
    """

    input_tiff_dir: str = "data/input_tiff"
    output_dir: str = "data/output_mtm"
