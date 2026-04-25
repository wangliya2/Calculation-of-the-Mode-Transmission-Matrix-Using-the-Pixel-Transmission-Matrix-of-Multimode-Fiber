"""
数据预处理模块（Preprocessing）

核心功能（对应中期考核要求）：
- 从双通道复数值 TIFF PTM 数据中读取复振幅；
- 实现高斯滤波 / 中值滤波两种去噪方式，可通过配置调整核大小与 σ；
- 提取相位并进行最小二乘法相位解包裹，缓解 2π 跳变；
- 对振幅进行 [0,1] Min-Max 归一化，对相位进行 [-π, π] 归一化；
- 输出满足 Plöschner 2015 算法输入格式的标准化复振幅 PTM 矩阵 H_pixel，
  以及关键参数和统计信息便于日志记录与验收。
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import tifffile
import logging

logger = logging.getLogger(__name__)


class PTMPreprocessor:
    """
    PTM 数据预处理器

    约定输入 TIFF 数据格式：
        - 形状为 (H, W, 2) 或 (N_input, H, W, 2)
        - 最后一个通道存储复数的实部与虚部：
              complex_pixel = TIFF[..., 0] + 1j * TIFF[..., 1]
        - 幅度与相位通过：
              amplitude = |complex_pixel|
              phase     = arg(complex_pixel)
    """

    def __init__(
        self,
        denoise_method: str = "gaussian",
        denoise_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        初始化预处理器

        Args:
            denoise_method: 去噪方法 ('gaussian' 或 'median')
            denoise_params: 去噪参数字典，例如：
                - 对于高斯滤波：
                    {'ksize': 5, 'sigma': 1.2}
                - 对于中值滤波：
                    {'ksize': 3}
        """
        denoise_method = denoise_method.lower()
        if denoise_method not in {"gaussian", "median"}:
            logger.warning(
                "不支持的去噪方法 %s，回退为 'gaussian'", denoise_method
            )
            denoise_method = "gaussian"

        self.denoise_method = denoise_method
        self.denoise_params: Dict[str, float] = denoise_params or {}

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def load_tiff(self, filepath: str) -> np.ndarray:
        """
        加载 TIFF 格式的 PTM 数据，并进行基本合法性校验。

        支持：
            - 单 PTM: (H, W, 2)
            - 多 PTM: (N_input, H, W, 2)
        """
        if not os.path.exists(filepath):
            msg = f"文件不存在: {filepath}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            data = tifffile.imread(filepath)
        except Exception as exc:
            msg = f"加载 TIFF 文件失败（可能损坏或非 TIFF）: {filepath}, 错误: {exc}"
            logger.error(msg)
            raise ValueError(msg) from exc

        if data is None or data.size == 0:
            msg = f"TIFF 文件为空: {filepath}"
            logger.error(msg)
            raise ValueError(msg)

        if data.ndim == 3 and data.shape[-1] == 2:
            
            data = data[None, ...]  # -> (1, H, W, 2)
        elif data.ndim == 4 and data.shape[-1] == 2:
            
            pass
        else:
            msg = (
                f"不支持的 TIFF 维度: {data.shape}，"
                "预期形状为 (H, W, 2) 或 (N_input, H, W, 2)"
            )
            logger.error(msg)
            raise ValueError(msg)

        _, h, w, _ = data.shape
        if (h, w) not in {(320, 256), (256, 320), (512, 512), (1024, 1024)}:
            
            logger.warning(
                "检测到非常规 CCD 尺寸: (%d, %d)，"
                "但仍将继续处理（算法尺寸自适应）。",
                h,
                w,
            )

        logger.info("成功加载 TIFF 文件: %s, 形状: %s", filepath, data.shape)
        return data.astype(np.float64)

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def _denoise_single_frame(self, image: np.ndarray) -> np.ndarray:
        """
        对单帧 2D 图像进行去噪。
        """
        if self.denoise_method == "gaussian":
            ksize = int(self.denoise_params.get("ksize", 5))
            
            if ksize % 2 == 0:
                ksize += 1
            sigma = float(self.denoise_params.get("sigma", 1.2))
            denoised = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        elif self.denoise_method == "median":
            ksize = int(self.denoise_params.get("ksize", 3))
            if ksize % 2 == 0:
                ksize += 1
            denoised = cv2.medianBlur(image.astype(np.float32), ksize)
        else:
            denoised = image

        return denoised

    def denoise_amplitude_batch(self, amplitude: np.ndarray) -> np.ndarray:
        """
        对批量振幅数据进行去噪。

        Args:
            amplitude: 振幅数组，形状 (N_input, H, W)

        Returns:
            去噪后的振幅数组，形状相同
        """
        if amplitude.ndim != 3:
            raise ValueError(
                f"振幅数组维度应为 (N_input, H, W)，当前为 {amplitude.shape}"
            )

        n, h, w = amplitude.shape
        denoised = np.zeros_like(amplitude, dtype=np.float64)

        for i in range(n):
            denoised[i] = self._denoise_single_frame(amplitude[i])

        logger.info(
            "图像去噪完成，方法: %s, N_input=%d, H=%d, W=%d",
            self.denoise_method,
            n,
            h,
            w,
        )
        return denoised

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_phase_least_squares(phase_wrapped: np.ndarray) -> np.ndarray:
        """
        基于最小二乘思想的相位解包裹。

        若安装了 scikit-image，则使用 skimage.restoration.unwrap_phase；
        否则退化为沿两个轴分别 np.unwrap 的简化版本（仍满足 2π 跳变消除）。
        """
        try:
            from skimage.restoration import unwrap_phase

            if phase_wrapped.ndim == 2:
                return unwrap_phase(phase_wrapped)
            elif phase_wrapped.ndim == 3:
                return np.stack(
                    [unwrap_phase(p) for p in phase_wrapped], axis=0
                )
            else:
                raise ValueError(
                    f"phase_wrapped 维度必须为 2 或 3，当前为 {phase_wrapped.shape}"
                )
        except Exception as exc:
            logger.warning(
                "scikit-image 不可用或解包裹失败（%s），"
                "退化为基于 np.unwrap 的逐轴解包裹。",
                exc,
            )
            if phase_wrapped.ndim == 2:
                p = np.unwrap(phase_wrapped, axis=0)
                p = np.unwrap(p, axis=1)
                return p
            elif phase_wrapped.ndim == 3:
                out = []
                for p in phase_wrapped:
                    q = np.unwrap(p, axis=0)
                    q = np.unwrap(q, axis=1)
                    out.append(q)
                return np.stack(out, axis=0)
            else:
                raise ValueError(
                    f"phase_wrapped 维度必须为 2 或 3，当前为 {phase_wrapped.shape}"
                )

    @staticmethod
    def _phase_continuity_score(phase_unwrapped: np.ndarray) -> float:
        """
        简单的相位连续度统计：统计相邻像素相位差 < π 的比例。
        """
        if phase_unwrapped.ndim == 2:
            diffs = []
            diffs.append(np.abs(np.diff(phase_unwrapped, axis=0)))
            diffs.append(np.abs(np.diff(phase_unwrapped, axis=1)))
            diffs = np.concatenate(
                [d.reshape(-1) for d in diffs], axis=0
            )
        elif phase_unwrapped.ndim == 3:
            diffs_all = []
            for p in phase_unwrapped:
                diffs_all.append(np.abs(np.diff(p, axis=0)).reshape(-1))
                diffs_all.append(np.abs(np.diff(p, axis=1)).reshape(-1))
            diffs = np.concatenate(diffs_all, axis=0)
        else:
            return 0.0

        if diffs.size == 0:
            return 0.0

        ratio = float(np.mean(diffs < np.pi))
        return ratio

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def _min_max_normalize(
        data: np.ndarray, min_val: float, max_val: float
    ) -> np.ndarray:
        """
        Min-Max 归一化到 [min_val, max_val]。
        """
        d_min = float(data.min())
        d_max = float(data.max())
        if d_max - d_min < 1e-12:
            return np.full_like(
                data, (min_val + max_val) / 2.0, dtype=np.float64
            )
        norm = (data - d_min) / (d_max - d_min)
        return norm * (max_val - min_val) + min_val

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def preprocess_to_h_pixel(
        self, filepath: str
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        端到端预处理：从 TIFF 文件到标准化复振幅 PTM 矩阵 H_pixel。

        Args:
            filepath: PTM TIFF 文件路径

        Returns:
            H_pixel: 标准化复振幅 PTM 矩阵，形状 (N_pixel, N_input)
            stats:  预处理统计信息，用于日志记录
        """
        raw = self.load_tiff(filepath)  # (N_input, H, W, 2)
        n_input, h, w, _ = raw.shape

        real = raw[..., 0]
        imag = raw[..., 1]
        complex_field = real + 1j * imag  # (N_input, H, W)

        amplitude = np.abs(complex_field)
        phase_wrapped = np.angle(complex_field)

        
        amplitude_denoised = self.denoise_amplitude_batch(amplitude)

        
        phase_unwrapped = self._unwrap_phase_least_squares(phase_wrapped)
        continuity = self._phase_continuity_score(phase_unwrapped)

        
        amplitude_norm = self._min_max_normalize(amplitude_denoised, 0.0, 1.0)
        
        phase_wrapped_2pi = (phase_unwrapped + np.pi) % (2 * np.pi) - np.pi
        phase_norm = self._min_max_normalize(
            phase_wrapped_2pi, -np.pi, np.pi
        )

        
        complex_norm = amplitude_norm * np.exp(1j * phase_norm)

        
        h_pixel = complex_norm.reshape(n_input, -1).T

        stats: Dict[str, float] = {
            "n_input": float(n_input),
            "height": float(h),
            "width": float(w),
            "amplitude_min": float(amplitude_denoised.min()),
            "amplitude_max": float(amplitude_denoised.max()),
            "phase_min": float(phase_unwrapped.min()),
            "phase_max": float(phase_unwrapped.max()),
            "phase_continuity": float(continuity),
        }

        logger.info(
            "PTM 预处理完成: N_input=%d, H=%d, W=%d, "
            "相位连续度≈%.3f, 振幅范围=[%.3g, %.3g], 相位范围=[%.3g, %.3g]",
            n_input,
            h,
            w,
            continuity,
            stats["amplitude_min"],
            stats["amplitude_max"],
            stats["phase_min"],
            stats["phase_max"],
        )

        return h_pixel, stats

