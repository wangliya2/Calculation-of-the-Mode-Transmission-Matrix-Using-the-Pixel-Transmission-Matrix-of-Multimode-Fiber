"""
数据预处理模块
功能：PTM图像去噪和相位提取
"""

import numpy as np
import cv2
import tifffile
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PTMPreprocessor:
    """PTM数据预处理器"""
    
    def __init__(self, denoise_method: str = 'gaussian', denoise_params: Optional[dict] = None):
        """
        初始化预处理器
        
        Args:
            denoise_method: 去噪方法 ('gaussian', 'bilateral', 'median')
            denoise_params: 去噪参数字典
        """
        self.denoise_method = denoise_method
        self.denoise_params = denoise_params or {}
        
    def load_tiff(self, filepath: str) -> np.ndarray:
        """
        加载TIFF格式的PTM数据
        
        Args:
            filepath: TIFF文件路径
            
        Returns:
            PTM图像数据数组
        """
        try:
            data = tifffile.imread(filepath)
            logger.info(f"成功加载TIFF文件: {filepath}, 形状: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"加载TIFF文件失败: {e}")
            raise
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        PTM图像去噪
        
        Args:
            image: 输入图像数组
            
        Returns:
            去噪后的图像
        """
        if len(image.shape) == 2:
            
            if self.denoise_method == 'gaussian':
                ksize = self.denoise_params.get('ksize', 5)
                sigma = self.denoise_params.get('sigma', 1.0)
                denoised = cv2.GaussianBlur(image, (ksize, ksize), sigma)
            elif self.denoise_method == 'bilateral':
                d = self.denoise_params.get('d', 9)
                sigma_color = self.denoise_params.get('sigma_color', 75)
                sigma_space = self.denoise_params.get('sigma_space', 75)
                denoised = cv2.bilateralFilter(image.astype(np.uint8), d, sigma_color, sigma_space)
                denoised = denoised.astype(image.dtype)
            elif self.denoise_method == 'median':
                ksize = self.denoise_params.get('ksize', 5)
                denoised = cv2.medianBlur(image.astype(np.uint8), ksize)
                denoised = denoised.astype(image.dtype)
            else:
                denoised = image
        else:
            
            denoised = np.zeros_like(image)
            for i in range(image.shape[-1]):
                denoised[..., i] = self.denoise_image(image[..., i])
        
        logger.info(f"图像去噪完成，方法: {self.denoise_method}")
        return denoised
    
    def extract_phase(self, image: np.ndarray, method: str = 'hilbert') -> np.ndarray:
        """
        从PTM图像中提取相位信息
        
        Args:
            image: 输入图像数组
            method: 相位提取方法 ('hilbert', 'fft', 'gradient')
            
        Returns:
            相位数组（弧度）
        """
        if method == 'hilbert':
            
            from scipy import signal
            if len(image.shape) == 2:
                analytic_signal = signal.hilbert(image)
                phase = np.angle(analytic_signal)
            else:
                phase = np.zeros_like(image)
                for i in range(image.shape[-1]):
                    analytic_signal = signal.hilbert(image[..., i])
                    phase[..., i] = np.angle(analytic_signal)
        
        elif method == 'fft':
            
            if len(image.shape) == 2:
                fft_image = np.fft.fft2(image)
                phase = np.angle(fft_image)
            else:
                phase = np.zeros_like(image)
                for i in range(image.shape[-1]):
                    fft_image = np.fft.fft2(image[..., i])
                    phase[..., i] = np.angle(fft_image)
        
        elif method == 'gradient':
            
            if len(image.shape) == 2:
                grad_x = np.gradient(image, axis=1)
                grad_y = np.gradient(image, axis=0)
                phase = np.arctan2(grad_y, grad_x)
            else:
                phase = np.zeros_like(image)
                for i in range(image.shape[-1]):
                    grad_x = np.gradient(image[..., i], axis=1)
                    grad_y = np.gradient(image[..., i], axis=0)
                    phase[..., i] = np.arctan2(grad_y, grad_x)
        else:
            
            phase = image / np.max(np.abs(image)) * np.pi
        
        logger.info(f"相位提取完成，方法: {method}")
        return phase
    
    def preprocess(self, filepath: str, extract_phase_method: str = 'hilbert') -> Tuple[np.ndarray, np.ndarray]:
        """
        完整的预处理流程：加载 -> 去噪 -> 相位提取
        
        Args:
            filepath: TIFF文件路径
            extract_phase_method: 相位提取方法
            
        Returns:
            (去噪后的图像, 相位数组)
        """
        
        raw_data = self.load_tiff(filepath)
        
        
        denoised_data = self.denoise_image(raw_data)
        
        
        phase_data = self.extract_phase(denoised_data, method=extract_phase_method)
        
        logger.info("数据预处理完成")
        return denoised_data, phase_data
