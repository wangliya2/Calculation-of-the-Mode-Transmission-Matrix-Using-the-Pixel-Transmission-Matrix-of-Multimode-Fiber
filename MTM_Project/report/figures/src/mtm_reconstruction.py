"""
模式矩阵重构模块
基于 Plöschner et al. 2015 算法实现MTM计算
"""

import numpy as np
from scipy import linalg, optimize
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MTMReconstructor:
    """
    模式传输矩阵（MTM）重构器
    基于 Plöschner et al. "Seeing through chaos in multimode fibres" (2015)
    """
    
    def __init__(self, num_modes: int = 10, regularization: float = 1e-6):
        """
        初始化MTM重构器
        
        Args:
            num_modes: 模式数量
            regularization: 正则化参数
        """
        self.num_modes = num_modes
        self.regularization = regularization
        self.mtm = None
        
    def generate_lp_modes(self, grid_size: Tuple[int, int], 
                          fiber_params: dict) -> np.ndarray:
        """
        生成LP模式基函数
        
        Args:
            grid_size: 网格大小 (height, width)
            fiber_params: 光纤参数字典，包含：
                - core_radius: 纤芯半径（微米）
                - wavelength: 波长（微米）
                - na: 数值孔径
                
        Returns:
            LP模式基函数数组，形状为 (num_modes, height, width)
        """
        height, width = grid_size
        y, x = np.meshgrid(
            np.linspace(-fiber_params['core_radius'], fiber_params['core_radius'], width),
            np.linspace(-fiber_params['core_radius'], fiber_params['core_radius'], height)
        )
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        
        k0 = 2 * np.pi / fiber_params['wavelength']
        v = k0 * fiber_params['core_radius'] * fiber_params['na']
        
        modes = []
        mode_idx = 0
        
        
        for m in range(self.num_modes // 2 + 1):
            for n in range(1, self.num_modes // (m + 1) + 2):
                if mode_idx >= self.num_modes:
                    break
                
                
                if m == 0:
                    
                    u = (2.405 * n) / fiber_params['core_radius']
                    mode_field = np.exp(-(r / fiber_params['core_radius'])**2) * np.cos(u * r)
                else:
                    
                    u = (2.405 * n + m * np.pi) / fiber_params['core_radius']
                    mode_field = np.exp(-(r / fiber_params['core_radius'])**2) * \
                                np.cos(u * r) * np.cos(m * theta)
                
                
                mode_field = mode_field / np.sqrt(np.sum(mode_field**2))
                modes.append(mode_field)
                mode_idx += 1
                
                if mode_idx >= self.num_modes:
                    break
            if mode_idx >= self.num_modes:
                break
        
        
        while len(modes) < self.num_modes:
            gaussian = np.exp(-(r / (fiber_params['core_radius'] / 2))**2)
            gaussian = gaussian / np.sqrt(np.sum(gaussian**2))
            modes.append(gaussian)
        
        modes = np.array(modes[:self.num_modes])
        logger.info(f"生成了 {len(modes)} 个LP模式基函数")
        return modes
    
    def compute_transmission_matrix(self, input_fields: np.ndarray, 
                                   output_fields: np.ndarray) -> np.ndarray:
        """
        计算传输矩阵（基于Plöschner算法）
        
        Args:
            input_fields: 输入场分布，形状 (num_inputs, height, width)
            output_fields: 输出场分布，形状 (num_outputs, height, width)
            
        Returns:
            传输矩阵，形状 (num_outputs, num_inputs)
        """
        
        input_vectors = input_fields.reshape(input_fields.shape[0], -1)
        output_vectors = output_fields.reshape(output_fields.shape[0], -1)
        
        
        
        X = input_vectors.T
        Y = output_vectors.T
        
        # T = Y * (X^T * X + lambda*I)^(-1) * X^T
        XTX = X.T @ X
        regularization_matrix = self.regularization * np.eye(XTX.shape[0])
        XTX_reg = XTX + regularization_matrix
        
        try:
            XTX_inv = linalg.inv(XTX_reg)
            T = Y @ X.T @ XTX_inv
        except linalg.LinAlgError:
            
            logger.warning("使用伪逆计算传输矩阵")
            T = Y @ linalg.pinv(X)
        
        self.mtm = T
        logger.info(f"传输矩阵计算完成，形状: {T.shape}")
        return T
    
    def reconstruct_output(self, input_field: np.ndarray, 
                          mode_basis: np.ndarray) -> np.ndarray:
        """
        使用MTM重构输出场
        
        Args:
            input_field: 输入场分布，形状 (height, width)
            mode_basis: 模式基函数，形状 (num_modes, height, width)
            
        Returns:
            重构的输出场分布
        """
        if self.mtm is None:
            raise ValueError("传输矩阵未计算，请先调用 compute_transmission_matrix")
        
        
        input_flat = input_field.flatten()
        mode_basis_flat = mode_basis.reshape(self.num_modes, -1)
        
        
        input_coeffs = mode_basis_flat @ input_flat
        
        
        output_coeffs = self.mtm @ input_coeffs
        
        
        output_field = (mode_basis_flat.T @ output_coeffs).reshape(input_field.shape)
        
        return output_field
    
    def compute_mtm_from_phase(self, phase_data: np.ndarray, 
                               mode_basis: np.ndarray) -> np.ndarray:
        """
        从相位数据计算MTM（在模式系数空间中进行正则化最小二乘拟合）
        
        Args:
            phase_data: 相位数据，形状 (num_measurements, height, width) 或 (height, width)
            mode_basis: 模式基函数，形状 (num_modes, height, width)
            
        Returns:
            传输矩阵，形状约为 (num_modes, num_modes)
        """
        
        complex_fields = np.exp(1j * phase_data)

        
        if complex_fields.ndim == 2:
            complex_fields = complex_fields[None, ...]

        num_measurements = complex_fields.shape[0]
        if num_measurements < 2:
            logger.warning("测量数据不足（<2），使用单位矩阵作为初始MTM")
            T = np.eye(self.num_modes, dtype=np.complex128)
            self.mtm = T
            return T

        
        mode_basis_flat = mode_basis.reshape(self.num_modes, -1)
        coeffs = []
        for field in complex_fields:
            field_flat = field.flatten()
            coeffs.append(mode_basis_flat @ field_flat)
        coeffs = np.array(coeffs)  # (num_measurements, num_modes)

        
        split_idx = num_measurements // 2
        num_pairs = min(split_idx, num_measurements - split_idx)
        if num_pairs <= 0:
            logger.warning("无法形成有效输入/输出对，使用单位矩阵作为初始MTM")
            T = np.eye(self.num_modes, dtype=np.complex128)
            self.mtm = T
            return T

        X = coeffs[:num_pairs]           
        Y = coeffs[split_idx:split_idx + num_pairs]  

        
        XT_X = X.T @ X                   # (M, M)
        reg = self.regularization * np.eye(self.num_modes, dtype=XT_X.dtype)
        XT_X_reg = XT_X + reg
        XT_Y = X.T @ Y                   # (M, M)

        try:
            
            T_T = linalg.solve(XT_X_reg, XT_Y)
            T = T_T.T
        except linalg.LinAlgError:
            logger.warning("正则化矩阵奇异，使用伪逆近似MTM")
            T = (linalg.pinv(X) @ Y).T

        self.mtm = T
        logger.info(f"从相位数据估计MTM完成，形状: {T.shape}")
        return T
