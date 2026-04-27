"""
MTM reconstruction module.

Implements the projection (Ploeschner et al., 2015):
    H_modes = M_out^H * H_pixel * M_in

Shape conventions:
    - H_pixel : complex PTM in pixel basis, shape (N_out_pix, N_in_pattern)
    - M_out   : output LP-mode basis in pixel basis, shape (N_out_pix, N_modes)
    - M_in    : input basis, shape (N_in_pattern, N_modes)
    - H_modes : MTM, shape (N_modes, N_modes)
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import ndimage
from .lp_theory import FiberModel, list_supported_lp_modes, lp_mode_field

logger = logging.getLogger(__name__)


class MTMReconstructor:
    """
    MTM reconstructor (Ploeschner et al., 2015).
    """

    def __init__(self, num_modes: int = 10) -> None:
        """
        Initialize the reconstructor.

        Args:
            num_modes: number of LP modes (columns of M_in/M_out)
        """
        self.num_modes = num_modes
        self.mtm: np.ndarray | None = None
        self.last_basis_correction: Dict[str, float] | None = None

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def generate_lp_modes(
        self,
        grid_size: Tuple[int, int],
        fiber_params: Dict[str, float],
        correction_params: Dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Generate LP mode fields under the weakly guiding approximation.

        Returns:
            modes: (num_modes, H, W)
        """
        height, width = grid_size
        core_radius = float(fiber_params["core_radius"])
        wavelength = float(fiber_params["wavelength"])
        na = float(fiber_params["na"])
        sim_extent_um = float(fiber_params.get("sim_extent_um", 30.0))
        extent_factor = sim_extent_um / max(core_radius, 1e-12)

        fiber = FiberModel(
            core_radius_um=core_radius,
            wavelength_um=wavelength,
            na=na,
        )
        
        preferred_order = [
            (0, 1), (0, 2), (0, 3),
            (1, 1), (1, 2),
            (2, 1), (2, 2),
            (3, 1),
        ]
        supported = list_supported_lp_modes(fiber=fiber, max_l=8, max_m=8)
        mode_order: list[tuple[int, int]] = []
        for lm in preferred_order:
            if lm in supported:
                mode_order.append(lm)
        for lm in supported:
            if lm not in mode_order:
                mode_order.append(lm)
        if not mode_order:
            raise RuntimeError(
                "No supported LP modes found for the current fiber parameters."
            )

        use_modes = min(self.num_modes, len(mode_order))
        if use_modes < self.num_modes:
            logger.warning(
                "Requested modes=%d exceeds supported=%d; falling back to %d",
                self.num_modes, len(mode_order), use_modes,
            )
        modes: list[np.ndarray] = []

        for idx in range(use_modes):
            l, m = mode_order[idx]
            try:
                field, _ = lp_mode_field(
                    l=l,
                    m=m,
                    grid_size=(height, width),
                    fiber=fiber,
                    extent_factor=extent_factor,
                )
                
                norm = np.linalg.norm(field) + 1e-12
                modes.append((field / norm).astype(np.complex128))
            except Exception as exc:
                logger.warning("LP(%d,%d) generation failed; skipping: %s", l, m, exc)
                continue

        if correction_params:
            scale = float(correction_params.get("scale", 1.0))
            shift_x_px = float(correction_params.get("shift_x_px", 0.0))
            shift_y_px = float(correction_params.get("shift_y_px", 0.0))
            rotation_deg = float(correction_params.get("rotation_deg", 0.0))
            corrected = []
            for mode in modes:
                corrected.append(
                    self._apply_basis_transform(
                        mode,
                        scale=scale,
                        shift_x_px=shift_x_px,
                        shift_y_px=shift_y_px,
                        rotation_deg=rotation_deg,
                    )
                )
            modes = corrected

        if not modes:
            raise RuntimeError("Mode generation failed: no usable modes.")
        modes_arr = np.asarray(modes, dtype=np.complex128)
        logger.info(
            "Generated %d LP modes on grid=(%d,%d), core_radius=%.2fum, extent=±%.2fum",
            modes_arr.shape[0], height, width, core_radius, sim_extent_um,
        )
        return modes_arr

    @staticmethod
    def _apply_basis_transform(
        field: np.ndarray,
        scale: float,
        shift_x_px: float,
        shift_y_px: float,
        rotation_deg: float,
    ) -> np.ndarray:
        """Apply scale/rotation/shift to a complex mode field."""
        real = np.asarray(field.real, dtype=np.float64)
        imag = np.asarray(field.imag, dtype=np.float64)

        
        if abs(scale - 1.0) > 1e-8:
            zoomed_r = ndimage.zoom(real, zoom=scale, order=1)
            zoomed_i = ndimage.zoom(imag, zoom=scale, order=1)
            real = MTMReconstructor._center_crop_or_pad(zoomed_r, field.shape)
            imag = MTMReconstructor._center_crop_or_pad(zoomed_i, field.shape)

        
        if abs(rotation_deg) > 1e-8:
            real = ndimage.rotate(real, angle=rotation_deg, reshape=False, order=1, mode="constant", cval=0.0)
            imag = ndimage.rotate(imag, angle=rotation_deg, reshape=False, order=1, mode="constant", cval=0.0)

        
        if abs(shift_x_px) > 1e-8 or abs(shift_y_px) > 1e-8:
            real = ndimage.shift(real, shift=(shift_y_px, shift_x_px), order=1, mode="constant", cval=0.0)
            imag = ndimage.shift(imag, shift=(shift_y_px, shift_x_px), order=1, mode="constant", cval=0.0)

        out = real + 1j * imag
        out = out / (np.linalg.norm(out) + 1e-12)
        return out.astype(np.complex128)

    @staticmethod
    def _center_crop_or_pad(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Center-crop or pad an array back to target shape."""
        th, tw = target_shape
        h, w = arr.shape
        out = np.zeros(target_shape, dtype=arr.dtype)

        src_y0 = max((h - th) // 2, 0)
        src_x0 = max((w - tw) // 2, 0)
        dst_y0 = max((th - h) // 2, 0)
        dst_x0 = max((tw - w) // 2, 0)
        copy_h = min(h, th)
        copy_w = min(w, tw)
        out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = arr[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
        return out

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def _gram_schmidt_orthonormal(
        vecs: np.ndarray, tol: float = 1e-10
    ) -> Tuple[np.ndarray, float]:
        """
        Gram-Schmidt column orthonormalization.

        Args:
            vecs: (N_pixel, N_modes) column vectors

        Returns:
            q: (N_pixel, K) orthonormalized columns (K <= N_modes)
            max_offdiag: max absolute off-diagonal entry of q^H q
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
                "Gram-Schmidt: fewer valid modes than requested: requested=%d, got=%d",
                n_modes, col,
            )

        g = q.conj().T @ q  # (k, k)
        if g.size == 0:
            return q, 0.0
        offdiag = g - np.eye(g.shape[0], dtype=g.dtype)
        max_offdiag = float(np.max(np.abs(offdiag)))
        return q, max_offdiag

    def build_output_mode_matrix(
        self,
        grid_size: Tuple[int, int],
        fiber_params: Dict[str, float],
        correction_params: Dict[str, float] | None = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Build output pixel-basis matrix M_out from LP modes and orthonormalize it.

        Returns:
            M_out: (N_out_pix, N_modes)
            orth_err: orthogonality error after orthonormalization
        """
        modes = self.generate_lp_modes(
            grid_size,
            fiber_params,
            correction_params=correction_params,
        )  # (M, H, W)
        n_modes, h, w = modes.shape
        n_pix = h * w

        
        modes_flat = modes.reshape(n_modes, n_pix).T  # (N_out_pix, N_modes)

        M_out, orth_err = self._gram_schmidt_orthonormal(modes_flat)

        if M_out.shape[1] < self.num_modes:
            logger.warning(
                "After orthonormalization, fewer modes remain: expected=%d, got=%d",
                self.num_modes, M_out.shape[1],
            )

        logger.info(
            "Built M_out: N_out_pix=%d, N_modes=%d, orth_err=%.3e",
            n_pix, M_out.shape[1], orth_err,
        )
        return M_out, orth_err

    def optimize_output_mode_matrix(
        self,
        H_pixel: np.ndarray,
        M_in: np.ndarray,
        grid_size: Tuple[int, int],
        fiber_params: Dict[str, float],
        correction_cfg: Dict[str, float],
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """
        Optimize output-basis scale/shift/rotation to minimize off-diagonal energy ratio.
        """
        scale_min = float(correction_cfg.get("scale_min", 0.9))
        scale_max = float(correction_cfg.get("scale_max", 1.1))
        shift_max = float(correction_cfg.get("shift_max_px", 12.0))
        rot_max = float(correction_cfg.get("rotation_max_deg", 10.0))
        max_iter = int(correction_cfg.get("max_iter", 30))

        H = np.asarray(H_pixel, dtype=np.complex128)
        Min = np.asarray(M_in, dtype=np.complex128)

        def objective(x: np.ndarray) -> float:
            scale, sx, sy, rot = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            if not (scale_min <= scale <= scale_max):
                return 1e9
            if abs(sx) > shift_max or abs(sy) > shift_max or abs(rot) > rot_max:
                return 1e9

            Mout, _ = self.build_output_mode_matrix(
                grid_size=grid_size,
                fiber_params=fiber_params,
                correction_params={
                    "scale": scale,
                    "shift_x_px": sx,
                    "shift_y_px": sy,
                    "rotation_deg": rot,
                },
            )
            k = min(Mout.shape[1], Min.shape[1])
            T = self.compute_mtm(H_pixel=H, M_in=Min[:, :k], M_out=Mout[:, :k])
            mag = np.abs(T)
            diag_mask = np.eye(k, dtype=bool)
            off = mag[~diag_mask]
            diag = mag[diag_mask]
            
            return float(np.linalg.norm(off) / (np.linalg.norm(diag) + 1e-12))

        x0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        bounds = [
            (scale_min, scale_max),
            (-shift_max, shift_max),
            (-shift_max, shift_max),
            (-rot_max, rot_max),
        ]
        result = optimize.minimize(
            objective,
            x0=x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-4},
        )
        x_best = result.x if result.success else x0
        x_best[0] = np.clip(x_best[0], scale_min, scale_max)
        x_best[1] = np.clip(x_best[1], -shift_max, shift_max)
        x_best[2] = np.clip(x_best[2], -shift_max, shift_max)
        x_best[3] = np.clip(x_best[3], -rot_max, rot_max)

        params = {
            "scale": float(x_best[0]),
            "shift_x_px": float(x_best[1]),
            "shift_y_px": float(x_best[2]),
            "rotation_deg": float(x_best[3]),
            "objective": float(objective(x_best)),
            "opt_success": float(1.0 if result.success else 0.0),
        }
        Mout_best, orth_err = self.build_output_mode_matrix(
            grid_size=grid_size,
            fiber_params=fiber_params,
            correction_params=params,
        )
        self.last_basis_correction = params
        logger.info("模式基校正参数: %s", params)
        return Mout_best, orth_err, params

    @staticmethod
    def _reshape_spatial_columns(
        M: np.ndarray, grid_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        若 M 的每一列可 reshape 为 (H,W) 且 H*W 与 grid_size 一致，则返回 (N_modes,H,W)；否则 None。
        """
        h, w = int(grid_size[0]), int(grid_size[1])
        n_pix = h * w
        if M.ndim != 2 or M.shape[0] != n_pix:
            return None
        n_modes = M.shape[1]
        out = np.zeros((n_modes, h, w), dtype=np.complex128)
        for j in range(n_modes):
            col = M[:, j].reshape(h, w)
            out[j] = col
        return out

    def build_input_mode_matrix_from_lp_fields(
        self,
        modes_hw: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        将 (N_modes,H,W) 的复场展平为 M_in: (H*W, N_modes) 并 Gram-Schmidt 正交归一化。
        """
        modes = np.asarray(modes_hw, dtype=np.complex128)
        if modes.ndim != 3:
            raise ValueError("modes_hw 必须是 (N_modes,H,W)")
        n_modes, h, w = modes.shape
        n_pix = h * w
        flat = modes.reshape(n_modes, n_pix).T
        return self._gram_schmidt_orthonormal(flat)

    def optimize_joint_mode_bases(
        self,
        H_pixel: np.ndarray,
        M_in: np.ndarray,
        grid_size: Tuple[int, int],
        fiber_params: Dict[str, float],
        correction_cfg: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
        """
        联合优化：对 LP 模式场使用同一组几何参数同时构造 M_out 与（可空间化的）M_in，
        最小化 MTM 非对角能量占比。

        说明：
        - M_in 必须是“每列对应一个 (H,W) 空间模式”的形式，且行数等于 H*W；
          Hadamard/单位阵输入基不满足该条件，将自动回退为仅优化 M_out。
        """
        scale_min = float(correction_cfg.get("scale_min", 0.9))
        scale_max = float(correction_cfg.get("scale_max", 1.1))
        shift_max = float(correction_cfg.get("shift_max_px", 12.0))
        rot_max = float(correction_cfg.get("rotation_max_deg", 10.0))
        max_iter = int(correction_cfg.get("max_iter", 30))

        H = np.asarray(H_pixel, dtype=np.complex128)
        Min0 = np.asarray(M_in, dtype=np.complex128)

        modes_template = self.generate_lp_modes(
            grid_size, fiber_params, correction_params=None
        )
        spatial_min = self._reshape_spatial_columns(Min0, grid_size)
        joint_ok = spatial_min is not None and spatial_min.shape[0] == modes_template.shape[0]

        if not joint_ok:
            logger.warning(
                "joint basis correction 需要 M_in 为空间模式矩阵 (H*W, N_modes)；"
                "当前 M_in=%s 不满足，回退为仅优化 M_out。",
                Min0.shape,
            )
            Mout, orth_err, params = self.optimize_output_mode_matrix(
                H_pixel=H,
                M_in=Min0,
                grid_size=grid_size,
                fiber_params=fiber_params,
                correction_cfg=correction_cfg,
            )
            params = dict(params)
            params["joint_mode"] = 0.0
            return Mout, Min0, orth_err, params

        def objective(x: np.ndarray) -> float:
            scale, sx, sy, rot = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            if not (scale_min <= scale <= scale_max):
                return 1e9
            if abs(sx) > shift_max or abs(sy) > shift_max or abs(rot) > rot_max:
                return 1e9

            corr = {
                "scale": scale,
                "shift_x_px": sx,
                "shift_y_px": sy,
                "rotation_deg": rot,
            }
            Mout, _ = self.build_output_mode_matrix(
                grid_size=grid_size,
                fiber_params=fiber_params,
                correction_params=corr,
            )
            modes_in_corr = []
            for j in range(spatial_min.shape[0]):
                modes_in_corr.append(
                    self._apply_basis_transform(
                        spatial_min[j],
                        scale=scale,
                        shift_x_px=sx,
                        shift_y_px=sy,
                        rotation_deg=rot,
                    )
                )
            Min_corr, _ = self.build_input_mode_matrix_from_lp_fields(
                np.stack(modes_in_corr, axis=0)
            )
            k = min(Mout.shape[1], Min_corr.shape[1])
            T = self.compute_mtm(H_pixel=H, M_in=Min_corr[:, :k], M_out=Mout[:, :k])
            mag = np.abs(T)
            diag_mask = np.eye(k, dtype=bool)
            off = mag[~diag_mask]
            diag = mag[diag_mask]
            return float(np.linalg.norm(off) / (np.linalg.norm(diag) + 1e-12))

        x0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        bounds = [
            (scale_min, scale_max),
            (-shift_max, shift_max),
            (-shift_max, shift_max),
            (-rot_max, rot_max),
        ]
        result = optimize.minimize(
            objective,
            x0=x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-4},
        )
        x_best = result.x if result.success else x0
        x_best[0] = np.clip(x_best[0], scale_min, scale_max)
        x_best[1] = np.clip(x_best[1], -shift_max, shift_max)
        x_best[2] = np.clip(x_best[2], -shift_max, shift_max)
        x_best[3] = np.clip(x_best[3], -rot_max, rot_max)

        params = {
            "scale": float(x_best[0]),
            "shift_x_px": float(x_best[1]),
            "shift_y_px": float(x_best[2]),
            "rotation_deg": float(x_best[3]),
            "objective": float(objective(x_best)),
            "opt_success": float(1.0 if result.success else 0.0),
            "joint_mode": 1.0,
        }

        corr = {
            "scale": params["scale"],
            "shift_x_px": params["shift_x_px"],
            "shift_y_px": params["shift_y_px"],
            "rotation_deg": params["rotation_deg"],
        }
        Mout_best, orth_err = self.build_output_mode_matrix(
            grid_size=grid_size,
            fiber_params=fiber_params,
            correction_params=corr,
        )
        modes_in_corr = []
        for j in range(spatial_min.shape[0]):
            modes_in_corr.append(
                self._apply_basis_transform(
                    spatial_min[j],
                    scale=params["scale"],
                    shift_x_px=params["shift_x_px"],
                    shift_y_px=params["shift_y_px"],
                    rotation_deg=params["rotation_deg"],
                )
            )
        Min_best, _ = self.build_input_mode_matrix_from_lp_fields(
            np.stack(modes_in_corr, axis=0)
        )
        self.last_basis_correction = params
        logger.info("联合模式基校正参数: %s", params)
        return Mout_best, Min_best, orth_err, params

    def _apply_geom_to_flat_modes(
        self,
        M_flat: np.ndarray,
        grid_size: Tuple[int, int],
        corr: Dict[str, float],
    ) -> Tuple[np.ndarray, float]:
        """
        对形状为 (H*W, N_modes) 的扁平模式矩阵逐列 reshape 为 (H,W)，施加几何变换后再展平并 Gram-Schmidt。
        """
        spatial = self._reshape_spatial_columns(M_flat, grid_size)
        if spatial is None:
            raise ValueError(f"M_flat 形状 {M_flat.shape} 无法按 grid_size={grid_size} reshape")
        modes_corr = []
        for j in range(spatial.shape[0]):
            modes_corr.append(
                self._apply_basis_transform(
                    spatial[j],
                    scale=float(corr["scale"]),
                    shift_x_px=float(corr["shift_x_px"]),
                    shift_y_px=float(corr["shift_y_px"]),
                    rotation_deg=float(corr["rotation_deg"]),
                )
            )
        return self.build_input_mode_matrix_from_lp_fields(np.stack(modes_corr, axis=0))

    def reorthonormalize_flat_spatial_modes(
        self,
        M_flat: np.ndarray,
        grid_size: Tuple[int, int],
        correction_params: Dict[str, float] | None = None,
    ) -> Tuple[np.ndarray, float]:
        """
        对 (H*W, N_modes) 空间模式矩阵施加可选几何参数后，再 Gram–Schmidt 正交归一化。
        correction_params 缺省为恒等变换。
        """
        corr = correction_params or {
            "scale": 1.0,
            "shift_x_px": 0.0,
            "shift_y_px": 0.0,
            "rotation_deg": 0.0,
        }
        return self._apply_geom_to_flat_modes(M_flat, grid_size, corr)

    def optimize_flat_mode_bases(
        self,
        H_pixel: np.ndarray,
        M_in_flat: np.ndarray,
        M_out_flat: np.ndarray,
        grid_size: Tuple[int, int],
        correction_cfg: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
        """
        在给定扁平模式矩阵 (H*W, N_modes) 上优化几何参数，使 MTM 非对角能量占比最小。
        用于外部给定的 modes_in/modes_out（例如论文仓库 conversion_matrices.npz）。
        """
        scale_min = float(correction_cfg.get("scale_min", 0.9))
        scale_max = float(correction_cfg.get("scale_max", 1.1))
        shift_max = float(correction_cfg.get("shift_max_px", 12.0))
        rot_max = float(correction_cfg.get("rotation_max_deg", 10.0))
        max_iter = int(correction_cfg.get("max_iter", 30))

        H = np.asarray(H_pixel, dtype=np.complex128)
        Min0 = np.asarray(M_in_flat, dtype=np.complex128)
        Mout0 = np.asarray(M_out_flat, dtype=np.complex128)

        def objective(x: np.ndarray) -> float:
            scale, sx, sy, rot = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            if not (scale_min <= scale <= scale_max):
                return 1e9
            if abs(sx) > shift_max or abs(sy) > shift_max or abs(rot) > rot_max:
                return 1e9
            corr = {
                "scale": scale,
                "shift_x_px": sx,
                "shift_y_px": sy,
                "rotation_deg": rot,
            }
            Mout, _ = self._apply_geom_to_flat_modes(Mout0, grid_size, corr)
            Min, _ = self._apply_geom_to_flat_modes(Min0, grid_size, corr)
            k = min(Mout.shape[1], Min.shape[1])
            if k < 1:
                return 1e9
            T = self.compute_mtm(H_pixel=H, M_in=Min[:, :k], M_out=Mout[:, :k])
            mag = np.abs(T)
            diag_mask = np.eye(k, dtype=bool)
            off = mag[~diag_mask]
            diag = mag[diag_mask]
            return float(np.linalg.norm(off) / (np.linalg.norm(diag) + 1e-12))

        x0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        bounds = [
            (scale_min, scale_max),
            (-shift_max, shift_max),
            (-shift_max, shift_max),
            (-rot_max, rot_max),
        ]
        result = optimize.minimize(
            objective,
            x0=x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-4},
        )
        x_best = result.x if result.success else x0
        x_best[0] = np.clip(x_best[0], scale_min, scale_max)
        x_best[1] = np.clip(x_best[1], -shift_max, shift_max)
        x_best[2] = np.clip(x_best[2], -shift_max, shift_max)
        x_best[3] = np.clip(x_best[3], -rot_max, rot_max)

        params = {
            "scale": float(x_best[0]),
            "shift_x_px": float(x_best[1]),
            "shift_y_px": float(x_best[2]),
            "rotation_deg": float(x_best[3]),
            "objective": float(objective(x_best)),
            "opt_success": float(1.0 if result.success else 0.0),
            "joint_mode": 12.0,  
        }
        corr = {
            "scale": params["scale"],
            "shift_x_px": params["shift_x_px"],
            "shift_y_px": params["shift_y_px"],
            "rotation_deg": params["rotation_deg"],
        }
        Mout_best, orth_out = self._apply_geom_to_flat_modes(Mout0, grid_size, corr)
        Min_best, orth_in = self._apply_geom_to_flat_modes(Min0, grid_size, corr)
        orth_err = float(max(orth_out, orth_in))
        self.last_basis_correction = params
        logger.info("外部模式矩阵几何校正参数: %s", params)
        return Mout_best, Min_best, orth_err, params

    def optimize_flat_output_only(
        self,
        H_pixel: np.ndarray,
        M_in_flat: np.ndarray,
        M_out_flat: np.ndarray,
        grid_size: Tuple[int, int],
        correction_cfg: Dict[str, float],
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """
        仅对给定 M_out（扁平空间模式列）做几何参数搜索，M_in 保持不变。
        """
        scale_min = float(correction_cfg.get("scale_min", 0.9))
        scale_max = float(correction_cfg.get("scale_max", 1.1))
        shift_max = float(correction_cfg.get("shift_max_px", 12.0))
        rot_max = float(correction_cfg.get("rotation_max_deg", 10.0))
        max_iter = int(correction_cfg.get("max_iter", 30))

        H = np.asarray(H_pixel, dtype=np.complex128)
        Min0 = np.asarray(M_in_flat, dtype=np.complex128)
        Mout0 = np.asarray(M_out_flat, dtype=np.complex128)

        def objective(x: np.ndarray) -> float:
            scale, sx, sy, rot = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            if not (scale_min <= scale <= scale_max):
                return 1e9
            if abs(sx) > shift_max or abs(sy) > shift_max or abs(rot) > rot_max:
                return 1e9
            corr = {
                "scale": scale,
                "shift_x_px": sx,
                "shift_y_px": sy,
                "rotation_deg": rot,
            }
            Mout, _ = self._apply_geom_to_flat_modes(Mout0, grid_size, corr)
            k = min(Mout.shape[1], Min0.shape[1])
            if k < 1:
                return 1e9
            T = self.compute_mtm(H_pixel=H, M_in=Min0[:, :k], M_out=Mout[:, :k])
            mag = np.abs(T)
            diag_mask = np.eye(k, dtype=bool)
            off = mag[~diag_mask]
            diag = mag[diag_mask]
            return float(np.linalg.norm(off) / (np.linalg.norm(diag) + 1e-12))

        x0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        bounds = [
            (scale_min, scale_max),
            (-shift_max, shift_max),
            (-shift_max, shift_max),
            (-rot_max, rot_max),
        ]
        result = optimize.minimize(
            objective,
            x0=x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-4},
        )
        x_best = result.x if result.success else x0
        x_best[0] = np.clip(x_best[0], scale_min, scale_max)
        x_best[1] = np.clip(x_best[1], -shift_max, shift_max)
        x_best[2] = np.clip(x_best[2], -shift_max, shift_max)
        x_best[3] = np.clip(x_best[3], -rot_max, rot_max)

        params = {
            "scale": float(x_best[0]),
            "shift_x_px": float(x_best[1]),
            "shift_y_px": float(x_best[2]),
            "rotation_deg": float(x_best[3]),
            "objective": float(objective(x_best)),
            "opt_success": float(1.0 if result.success else 0.0),
            "joint_mode": 13.0,  
        }
        corr = {
            "scale": params["scale"],
            "shift_x_px": params["shift_x_px"],
            "shift_y_px": params["shift_y_px"],
            "rotation_deg": params["rotation_deg"],
        }
        Mout_best, orth_err = self._apply_geom_to_flat_modes(Mout0, grid_size, corr)
        self.last_basis_correction = params
        logger.info("外部 M_out 几何校正参数: %s", params)
        return Mout_best, orth_err, params

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def _hermitian_conjugate(mat: np.ndarray) -> np.ndarray:
        """厄米共轭（共轭转置）。"""
        return np.conjugate(mat).T

    def compute_mtm(
        self, H_pixel: np.ndarray, M_in: np.ndarray, M_out: np.ndarray
    ) -> np.ndarray:
        """
        使用公式计算MTM：H_modes = M_out^dagger * H_pixel * M_in

        参数:
            H_pixel: 像素基PTM，形状 (N_out_pix, N_in_pattern)
            M_in:    输入模式基，形状 (N_in_pattern, N_modes)
            M_out:   输出模式基，形状 (N_out_pix, N_modes)

        返回:
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
            "MTM计算完成: H_pixel=%s, M_in=%s, M_out=%s -> H_modes=%s",
            H.shape, Min.shape, Mout.shape, H_modes.shape,
        )
        return H_modes

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_mtm(H_modes: np.ndarray) -> Dict[str, float]:
        """
        评估关键MTM指标。对于理想的直纤维：
            - 对角元素的幅值应接近1
            - 非对角元素应接近0

        返回:
            包含diag_mean, diag_min, offdiag_max, cond_number的字典
        """
        H = np.asarray(H_modes, dtype=np.complex128)
        if H.shape[0] != H.shape[1]:
            raise ValueError(f"MTM必须为方阵，当前形状: {H.shape}")

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

    @staticmethod
    def evaluate_mtm_gain_normalized(H_modes: np.ndarray) -> Dict[str, float]:
        """
        将 MTM 按「对角元幅值平均」整体标定后再评估形状指标。

        用于区分「整体功率缩放/吸收」与「模式间串扰」：实验或 dummy 经预处理后
        对角元未必接近 1，但相对非对角仍应小。
        """
        H = np.asarray(H_modes, dtype=np.complex128)
        if H.shape[0] != H.shape[1]:
            raise ValueError(f"MTM必须为方阵，当前形状: {H.shape}")
        scale = float(np.mean(np.abs(np.diag(H))) + 1e-12)
        return MTMReconstructor.evaluate_mtm(H / scale)

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    @staticmethod
    def plot_mtm_heatmap(
        H_modes: np.ndarray, save_path: str, title: str = "MTM magnitude"
    ) -> None:
        """
        生成并保存MTM幅值的热力图，用于视觉验证对角特性。
        """
        H = np.asarray(H_modes, dtype=np.complex128)
        magnitude = np.abs(H)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        
        im1 = axes[0].imshow(magnitude, cmap="viridis", origin="lower")
        plt.colorbar(im1, ax=axes[0], label="|H_modes|")
        axes[0].set_xlabel("输入模式索引")
        axes[0].set_ylabel("输出模式索引")
        axes[0].set_title(f"{title} - 幅值")

        
        phase = np.angle(H)
        im2 = axes[1].imshow(phase, cmap="twilight", origin="lower", vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im2, ax=axes[1], label="相位 (弧度)")
        axes[1].set_xlabel("输入模式索引")
        axes[1].set_ylabel("输出模式索引")
        axes[1].set_title(f"{title} - 相位")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        logger.info("MTM热力图已保存: %s", save_path)
