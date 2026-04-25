from __future__ import annotations

"""
生成物理驱动的虚拟PTM TIFF数据，用于管线测试。

该脚本创建模拟的PTM数据，近似于近直的多模光纤，其中理想的MTM应接近对角矩阵。
模拟包括：
- 基于LP模式（LP mode）的输入和输出场构建
- 小的非对角耦合以模拟真实的模式混合
- 添加噪声以模拟测量误差

用法：
    python make_dummy_ptm.py
"""

import os

import numpy as np
import tifffile

from config import FiberParams, MTMConfig
from src.lp_theory import FiberModel, recommended_num_modes
from src.mtm_calculator import _build_m_in_hadamard
from src.mtm_reconstruction import MTMReconstructor


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def generate_realistic_dummy_ptm(
    grid_size: int = 64,
    num_modes: int = 8,
    coupling_strength: float = 0.08,
    noise_level: float = 0.02,
    seed: int = 123,
) -> np.ndarray:
    """
    生成物理驱动的虚拟PTM，用于近直光纤。

    模拟的PTM构造为：
        H_pixel = M_out @ H_modes_sim @ M_in^dagger

    其中H_modes_sim是一个近对角的MTM，带有小的非对角耦合项，
    模拟真实的直光纤场景。

    参数：
        grid_size: 模拟CCD的空间分辨率（高=宽）
        num_modes: LP模式数量
        coupling_strength: 非对角耦合强度（0表示完美无耦合）
        noise_level: 叠加复数噪声的标准差
        seed: 随机种子，保证可重复性

    返回：
        tiff_data: 形状为 (use_modes, grid_size, grid_size, 2) 的float32数组
    """
    rng = np.random.default_rng(seed)

    fiber_params = {
        "core_radius": FiberParams.core_radius_um,
        "wavelength": FiberParams.wavelength_um,
        "na": FiberParams.na,
        "sim_extent_um": FiberParams.sim_extent_um,
    }

    
    recon = MTMReconstructor(num_modes=int(num_modes))
    M_out, _ = recon.build_output_mode_matrix(
        grid_size=(grid_size, grid_size),
        fiber_params=fiber_params,
    )
    k = M_out.shape[1]
    M_in = _build_m_in_hadamard(n_input=k, n_modes=k)

    
    
    H_modes_sim = np.diag(
        np.exp(1j * rng.uniform(-0.15, 0.15, size=k))
        * (0.92 + 0.08 * rng.random(k))
    ).astype(np.complex128)

    
    coupling = (
        rng.normal(0, coupling_strength, (k, k))
        + 1j * rng.normal(0, coupling_strength, (k, k))
    )
    np.fill_diagonal(coupling, 0)
    H_modes_sim += coupling

    
    H_pixel = M_out @ H_modes_sim @ M_in.conj().T

    
    noise = (
        rng.normal(0, noise_level, H_pixel.shape)
        + 1j * rng.normal(0, noise_level, H_pixel.shape)
    )
    H_pixel += noise

    
    tiff_data = np.zeros((k, grid_size, grid_size, 2), dtype=np.float32)
    for i in range(k):
        field_2d = H_pixel[:, i].reshape(grid_size, grid_size)
        tiff_data[i, :, :, 0] = field_2d.real.astype(np.float32)
        tiff_data[i, :, :, 1] = field_2d.imag.astype(np.float32)

    return tiff_data


def main() -> None:
    out = os.path.join(project_root(), "data", "input_tiff")
    os.makedirs(out, exist_ok=True)

    print("生成物理驱动的虚拟PTM数据...")

    auto_modes = recommended_num_modes(
        fiber=FiberModel(
            core_radius_um=FiberParams.core_radius_um,
            wavelength_um=FiberParams.wavelength_um,
            na=FiberParams.na,
        ),
        preferred_max=MTMConfig.num_modes,
    )
    print(f"自动推荐模式数: {auto_modes}（配置上限={MTMConfig.num_modes}）")

    
    tiff_data = generate_realistic_dummy_ptm(
        grid_size=64,
        num_modes=auto_modes,
        coupling_strength=0.012,
        noise_level=0.005,
        seed=123,
    )

    path = os.path.join(out, "dummy_straight_fiber.tiff")
    tifffile.imwrite(path, tiff_data)
    print(f"完成: {path}, 形状={tiff_data.shape}")

    
    tiff_data_bent = generate_realistic_dummy_ptm(
        grid_size=64,
        num_modes=auto_modes,
        coupling_strength=0.22,
        noise_level=0.045,
        seed=456,
    )

    path_bent = os.path.join(out, "dummy_bent_fiber.tiff")
    tifffile.imwrite(path_bent, tiff_data_bent)
    print(f"完成: {path_bent}, 形状={tiff_data_bent.shape}")

    print("完成。虚拟PTM文件已生成于:", out)


if __name__ == "__main__":
    main()
