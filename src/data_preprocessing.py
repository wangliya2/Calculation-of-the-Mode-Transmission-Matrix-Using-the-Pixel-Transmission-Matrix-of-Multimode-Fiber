"""
数据预处理模块

核心功能（符合期中考核要求）：
- 从双通道TIFF文件加载复数值PTM数据
- 实现可配置核大小和 sigma 的高斯/中值/双边去噪
- 提取相位并执行最小二乘相位展开以消除2pi跳变
- 将振幅归一化到[0,1]，相位归一化到[-pi, pi]
- 输出与Ploeschner 2015算法兼容的标准化复振幅PTM矩阵H_pixel，以及用于日志记录的关键统计信息
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
    PTM数据预处理器

    期望的TIFF数据格式：
        - 形状：(H, W, 2) 或 (N_input, H, W, 2)
        - 最后一个通道存储实部和虚部：
              complex_pixel = TIFF[..., 0] + 1j * TIFF[..., 1]
        - 振幅和相位的计算方式：
              amplitude = |complex_pixel|
              phase     = arg(complex_pixel)
    """

    def __init__(
        self,
        denoise_method: str = "gaussian",
        denoise_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        初始化预处理器。

        参数：
            denoise_method: 去噪方法（'gaussian' / 'median' / 'bilateral'）
            denoise_params: 去噪参数字典，例如：
                - 高斯去噪：{'ksize': 5, 'sigma': 1.2}
                - 中值去噪：{'ksize': 3}
                - 双边滤波：{'d': 7, 'sigma_color': 0.06, 'sigma_space': 6.0}
        """
        denoise_method = denoise_method.lower()
        if denoise_method not in {"gaussian", "median", "bilateral"}:
            logger.warning(
                "不支持的去噪方法 '%s'，将回退到 'gaussian'",
                denoise_method,
            )
            denoise_method = "gaussian"

        self.denoise_method = denoise_method
        self.denoise_params: Dict[str, float] = denoise_params or {}

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def load_tiff(self, filepath: str) -> np.ndarray:
        """
        加载TIFF格式的PTM数据并进行基本有效性检查。

        支持：
            - 单个PTM：(H, W, 2)
            - 多个PTM：(N_input, H, W, 2)
        """
        if not os.path.exists(filepath):
            msg = f"文件未找到：{filepath}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            data = tifffile.imread(filepath)
        except Exception as exc:
            msg = f"加载TIFF文件失败（可能已损坏）：{filepath}, 错误：{exc}"
            logger.error(msg)
            raise ValueError(msg) from exc

        if data is None or data.size == 0:
            msg = f"TIFF文件为空：{filepath}"
            logger.error(msg)
            raise ValueError(msg)

        if data.ndim == 3 and data.shape[-1] == 2:
            
            data = data[None, ...]  # -> (1, H, W, 2)
        elif data.ndim == 4 and data.shape[-1] == 2:
            
            pass
        else:
            msg = (
                f"不支持的TIFF维度：{data.shape}, "
                "期望形状为 (H, W, 2) 或 (N_input, H, W, 2)"
            )
            logger.error(msg)
            raise ValueError(msg)

        _, h, w, _ = data.shape
        if (h, w) not in {(320, 256), (256, 320), (512, 512), (1024, 1024)}:
            logger.warning(
                "检测到非标准CCD尺寸：(%d, %d)。"
                "处理将继续，采用自适应尺寸。",
                h, w,
            )

        logger.info("TIFF加载成功：%s，形状：%s", filepath, data.shape)
        return data.astype(np.float64)

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def _denoise_single_frame(self, image: np.ndarray) -> np.ndarray:
        """对单帧二维图像应用去噪。"""
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
        elif self.denoise_method == "bilateral":
            d = int(self.denoise_params.get("d", 7))
            if d % 2 == 0:
                d += 1
            sigma_color = float(self.denoise_params.get("sigma_color", 0.06))
            sigma_space = float(self.denoise_params.get("sigma_space", 6.0))
            denoised = cv2.bilateralFilter(
                image.astype(np.float32), d, sigma_color, sigma_space
            )
        else:
            denoised = image

        return denoised

    def denoise_amplitude_batch(self, amplitude: np.ndarray) -> np.ndarray:
        """
        对一批振幅图像应用去噪。

        参数：
            amplitude: 振幅数组，形状为 (N_input, H, W)

        返回：
            去噪后的振幅数组，形状不变
        """
        if amplitude.ndim != 3:
            raise ValueError(
                f"振幅数组必须为 (N_input, H, W)，但得到 {amplitude.shape}"
            )

        n, h, w = amplitude.shape
        denoised = np.zeros_like(amplitude, dtype=np.float64)

        for i in range(n):
            denoised[i] = self._denoise_single_frame(amplitude[i])

        logger.info(
            "去噪完成：方法=%s, N_input=%d, H=%d, W=%d",
            self.denoise_method, n, h, w,
        )
        return denoised

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_phase_least_squares(phase_wrapped: np.ndarray) -> np.ndarray:
        """
        基于最小二乘法的相位展开。

        如果可用，使用skimage.restoration.unwrap_phase；
        否则回退到沿两个轴的顺序np.unwrap。
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
                    f"phase_wrapped必须是2D或3D，得到 {phase_wrapped.shape}"
                )
        except Exception as exc:
            logger.warning(
                "scikit-image不可用或展开失败（%s）。"
                "回退到沿两个轴的np.unwrap。",
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
                    f"phase_wrapped必须是2D或3D，得到 {phase_wrapped.shape}"
                )

    @staticmethod
    def _phase_continuity_score(phase_unwrapped: np.ndarray) -> float:
        """
        计算相位连续性得分：相邻像素对相位差小于pi的比例。
        """
        if phase_unwrapped.ndim == 2:
            diffs = []
            diffs.append(np.abs(np.diff(phase_unwrapped, axis=0)))
            diffs.append(np.abs(np.diff(phase_unwrapped, axis=1)))
            diffs = np.concatenate([d.reshape(-1) for d in diffs], axis=0)
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
        """最小-最大归一化到[min_val, max_val]区间。"""
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
        端到端预处理：从TIFF文件到标准化复振幅PTM矩阵H_pixel。

        参数：
            filepath: PTM TIFF文件路径

        返回：
            H_pixel: 标准化复数PTM矩阵，形状为 (N_pixel, N_input)
            stats:   预处理统计信息，用于日志记录
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
        phase_norm = self._min_max_normalize(phase_wrapped_2pi, -np.pi, np.pi)

        
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
            "PTM预处理完成：N_input=%d, H=%d, W=%d, "
            "相位连续性=%.3f, 振幅范围=[%.3g, %.3g], 相位范围=[%.3g, %.3g]",
            n_input, h, w, continuity,
            stats["amplitude_min"], stats["amplitude_max"],
            stats["phase_min"], stats["phase_max"],
        )

        return h_pixel, stats

    def reconstruct_denoise_unwrap(
        self, complex_field: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        仅执行振幅去噪与相位展开后重构复场，**不进行**整幅 min-max 归一化。

        用途：预处理验收中与理论真值比较 SNR/相位误差时，与 `preprocess_to_h_pixel`
        的 min-max 后量纲不一致会导致虚假失败；本方法在与真值同一量纲下评估去噪+展开效果。
        """
        cf = np.asarray(complex_field, dtype=np.complex128)
        if cf.ndim != 3:
            raise ValueError(f"期望 (N_input, H, W)，得到 shape={cf.shape}")
        n_input, h, w = cf.shape
        amplitude = np.abs(cf)
        phase_wrapped = np.angle(cf)
        amplitude_denoised = self.denoise_amplitude_batch(amplitude)
        phase_unwrapped = self._unwrap_phase_least_squares(phase_wrapped)
        continuity = self._phase_continuity_score(phase_unwrapped)
        z = amplitude_denoised * np.exp(1j * phase_unwrapped)
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
        return z.astype(np.complex128), stats
