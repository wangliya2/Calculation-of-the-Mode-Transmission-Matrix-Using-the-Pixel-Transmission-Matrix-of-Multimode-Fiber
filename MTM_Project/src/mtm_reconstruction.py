"""
模式矩阵重构模块
基于 Plöschner et al. 2015 算法实现 MTM 计算：

    H_modes = M_out^† · H_pixel · M_in

维度约定：
    - H_pixel  : 标准化复振幅 PTM（像素基），形状 (N_out_pix, N_in_pattern)
    - M_out    : 输出 LP 模式基矢（像素基），形状 (N_out_pix, N_modes)
    - M_in     : 输入 LP 模式基矢（激励模式基），形状 (N_in_pattern, N_modes)
    - H_modes  : 模式传输矩阵（MTM），形状 (N_modes, N_modes)
"""

from __future__ import annotations

from typing import Dict, Tuple

import logging
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MTMReconstructor:
    """
    模式传输矩阵（MTM）重构器
    基于 Plöschner et al. "Seeing through chaos in multimode fibres" (2015)
    """

    def __init__(self, num_modes: int = 10) -> None:
        """
        初始化 MTM 重构器

        Args:
            num_modes: LP 模式数量（M_in/M_out 矩阵的列数）
        """
        self.num_modes = num_modes
        self.mtm: np.ndarray | None = None

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def generate_lp_modes(
        self, grid_size: Tuple[int, int], fiber_params: Dict[str, float]
    ) -> np.ndarray:
        """
        生成 LP 模式场分布（简化模型），后续将其展平为像素基矢。

        Returns:
            modes: (num_modes, H, W)
        """
        height, width = grid_size
        y, x = np.meshgrid(
            np.linspace(
                -fiber_params["core_radius"], fiber_params["core_radius"], width
            ),
            np.linspace(
                -fiber_params["core_radius"], fiber_params["core_radius"], height
            ),
        )
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        modes = []
        mode_idx = 0

        
        for m in range(self.num_modes // 2 + 1):
            for n in range(1, self.num_modes // (m + 1) + 2):
                if mode_idx >= self.num_modes:
                    break

                if m == 0:
                    
                    u = (2.405 * n) / fiber_params["core_radius"]
                    mode_field = np.exp(-(r / fiber_params["core_radius"]) ** 2) * np.cos(
                        u * r
                    )
                else:
                    
                    u = (2.405 * n + m * np.pi) / fiber_params["core_radius"]
                    mode_field = (
                        np.exp(-(r / fiber_params["core_radius"]) ** 2)
                        * np.cos(u * r)
                        * np.cos(m * theta)
                    )

                
                norm = np.sqrt(np.sum(np.abs(mode_field) ** 2))
                if norm < 1e-12:
                    continue
                mode_field = mode_field / norm
                modes.append(mode_field)
                mode_idx += 1

                if mode_idx >= self.num_modes:
                    break
            if mode_idx >= self.num_modes:
                break

        
        while len(modes) < self.num_modes:
            gaussian = np.exp(-(r / (fiber_params["core_radius"] / 2)) ** 2)
            gaussian = gaussian / np.sqrt(np.sum(gaussian**2))
            modes.append(gaussian)

        modes_arr = np.asarray(modes[: self.num_modes], dtype=np.float64)
        logger.info("生成 LP 模式场 %d 个，grid_size=(%d, %d)", modes_arr.shape[0], height, width)
        return modes_arr

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def _gram_schmidt_orthonormal(
        vecs: np.ndarray, tol: float = 1e-10
    ) -> Tuple[np.ndarray, float]:
        """
        对列向量执行 Gram-Schmidt 正交归一化。

        Args:
            vecs: (N_pixel, N_modes) 原始列向量

        Returns:
            q: 正交归一后的列向量 (N_pixel, N_modes)
            max_offdiag: 正交性误差（内积矩阵非对角元素最大绝对值）
        """
        n_pix, n_modes = vecs.shape
        q = np.zeros((n_pix, n_modes), dtype=np.complex128)

        col = 0
        for j in range(n_modes):
            v = vecs[:, j].astype(np.complex128)
            for i in range(col):
                proj = np.vdot(q[:, i], v) * q[:, i]
                v = v - proj
            norm = np.linalg.norm(v)
            if norm > tol:
                q[:, col] = v / norm
                col += 1

        if col < n_modes:
            q = q[:, :col]
            logger.warning(
                "Gram-Schmidt 过程中有效模式数不足: 期望=%d, 实际=%d", n_modes, col
            )

        g = q.conj().T @ q  # (k, k)
        offdiag = g - np.eye(g.shape[0], dtype=g.dtype)
        max_offdiag = float(np.max(np.abs(offdiag)))
        return q, max_offdiag

    def build_output_mode_matrix(
        self,
        grid_size: Tuple[int, int],
        fiber_params: Dict[str, float],
    ) -> Tuple[np.ndarray, float]:
        """
        基于 LP 模式场生成输出侧像素基矢矩阵 M_out，并做正交归一化。

        对直光纤情形，一般认为输入/输出 LP 模式族一致，
        但在数值实现中我们仅在输出侧显式构造 M_out，
        输入侧 M_in 通常在「激励模式」空间中用单位阵或给定基展开。

        Returns:
            M_out:    (N_out_pix, N_modes)
            orth_err: 模式正交性误差（内积绝对值最大非对角元素）
        """
        modes = self.generate_lp_modes(grid_size, fiber_params)  # (M, H, W)
        n_modes, h, w = modes.shape
        n_pix = h * w

        
        modes_flat = modes.reshape(n_modes, n_pix).T  # (N_out_pix, N_modes)

        M_out, orth_err = self._gram_schmidt_orthonormal(modes_flat)

        if M_out.shape[1] < self.num_modes:
            logger.warning(
                "正交化后模式数减少：期望=%d, 实际=%d", self.num_modes, M_out.shape[1]
            )

        logger.info(
            "输出侧 M_out 构建完成: N_out_pix=%d, N_modes=%d, 正交性误差=%.3e",
            n_pix,
            M_out.shape[1],
            orth_err,
        )
        return M_out, orth_err

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def _hermitian_conjugate(mat: np.ndarray) -> np.ndarray:
        """
        厄米特共轭：共轭转置。
        """
        return np.conjugate(mat).T

    def compute_mtm(
        self, H_pixel: np.ndarray, M_in: np.ndarray, M_out: np.ndarray
    ) -> np.ndarray:
        """
        根据公式 H_modes = M_out^† · H_pixel · M_in 计算 MTM。

        Args:
            H_pixel: 像素基 PTM，形状 (N_out_pix, N_in_pattern)
            M_in:    输入模式基，形状 (N_in_pattern, N_modes)
            M_out:   输出模式基，形状 (N_out_pix, N_modes)

        Returns:
            H_modes: 模式传输矩阵，形状 (N_modes, N_modes)
        """
        H = np.asarray(H_pixel, dtype=np.complex128)
        Min = np.asarray(M_in, dtype=np.complex128)
        Mout = np.asarray(M_out, dtype=np.complex128)

        if H.shape[0] != Mout.shape[0]:
            raise ValueError(
                f"输出像素维度不匹配: H_pixel={H.shape}, M_out={Mout.shape}"
            )
        if H.shape[1] != Min.shape[0]:
            raise ValueError(
                f"输入模式维度不匹配: H_pixel={H.shape}, M_in={Min.shape}"
            )

        Mout_h = self._hermitian_conjugate(Mout)  # (N_modes, N_out_pix)

        
        H_modes = Mout_h @ H @ Min  # (N_modes, N_modes)
        self.mtm = H_modes

        logger.info(
            "MTM 计算完成: H_pixel=%s, M_in=%s, M_out=%s -> H_modes=%s",
            H.shape,
            Min.shape,
            Mout.shape,
            H_modes.shape,
        )
        return H_modes

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_mtm(H_modes: np.ndarray) -> Dict[str, float]:
        """
        评估 MTM 的关键指标，对直光纤理想情形：
            - 主对角元素模值应接近 1
            - 非对角元素应尽量接近 0

        返回：
            {
              'diag_mean': 主对角元素模值平均,
              'diag_min' : 主对角元素模值最小,
              'offdiag_max': 非对角元素模值最大,
              'cond_number': 条件数,
            }
        """
        H = np.asarray(H_modes, dtype=np.complex128)
        if H.shape[0] != H.shape[1]:
            raise ValueError(f"MTM 必须为方阵，当前形状: {H.shape}")

        diag = np.diag(H)
        diag_abs = np.abs(diag)
        diag_mean = float(np.mean(diag_abs))
        diag_min = float(np.min(diag_abs))

        mask_off = ~np.eye(H.shape[0], dtype=bool)
        offdiag_abs = np.abs(H[mask_off])
        offdiag_max = float(np.max(offdiag_abs)) if offdiag_abs.size > 0 else 0.0

        try:
            cond_number = float(np.linalg.cond(H))
        except Exception:
            cond_number = float("inf")

        stats = {
            "diag_mean": diag_mean,
            "diag_min": diag_min,
            "offdiag_max": offdiag_max,
            "cond_number": cond_number,
        }
        return stats

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def plot_mtm_heatmap(
        H_modes: np.ndarray, save_path: str, title: str = "MTM magnitude"
    ) -> None:
        """
        输出 MTM 矩阵模值的热力图，用于直观验证对角化特征。
        """
        H = np.asarray(H_modes, dtype=np.complex128)
        magnitude = np.abs(H)

        plt.figure(figsize=(5, 4))
        im = plt.imshow(magnitude, cmap="viridis", origin="lower")
        plt.colorbar(im, label="|H_modes|")
        plt.xlabel("Input modes index")
        plt.ylabel("Output modes index")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        logger.info("MTM 热力图已保存: %s", save_path)

