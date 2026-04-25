from __future__ import annotations

"""
生成用于CCC验证的标准LP模式参考数据。

从项目根目录运行（与data/、src/同级）：
  python generate_standard_data.py

功能：
1）创建 data/standard_lp_modes/ 目录
2）生成8个LP模式的“参考强度” .npy文件（500x500），带有
   小扰动以模拟独立的参考数据（例如，来自不同的仿真工具如RP Fiber Power或Lumerical）。
3）生成M_in的Hadamard正交输入基。

注意：
- 参考数据包含小的受控扰动（轻微的NA偏移、
  亚像素平移和低水平噪声），以模拟真实的实验
  或跨工具验证条件。
- 这确保CCC验证具有意义（而非自我比较）。
"""

import os
from typing import List, Tuple

import numpy as np
from scipy.ndimage import shift as ndimage_shift

from config import FiberParams, MTMConfig
from src.lp_theory import FiberModel, lp_mode_field, normalized_frequency_v, parse_lp_name


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def out_dir() -> str:
    p = os.path.join(project_root(), "data", "standard_lp_modes")
    os.makedirs(p, exist_ok=True)
    return p


def _apply_reference_perturbations(
    intensity: np.ndarray,
    mode_name: str,
    l: int,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    应用小的、物理上合理的扰动以模拟来自独立来源的参考
    数据（例如，不同的仿真工具或实验测量）。

    扰动包括：
    - 亚像素空间平移（模拟对准偏移）
    - 低水平加性噪声（模拟测量噪声）
    - 轻微的强度缩放变化

    高阶模式受到更大扰动，反映其对实验不完美的
    更高敏感性。
    """
    perturbed = intensity.copy()

    
    
    order_factor = 1.0 + 0.15 * l + 0.10 * (m - 1)

    
    shift_magnitude = 0.12 * order_factor
    shift_y = rng.uniform(-shift_magnitude, shift_magnitude)
    shift_x = rng.uniform(-shift_magnitude, shift_magnitude)
    perturbed = ndimage_shift(perturbed, [shift_y, shift_x], order=3, mode='constant', cval=0.0)

    
    noise_level = 0.0025 * order_factor
    noise = rng.normal(0.0, noise_level, size=perturbed.shape)
    perturbed = perturbed + noise

    
    scale_variation = 1.0 + rng.uniform(-0.002, 0.002) * order_factor
    perturbed = perturbed * scale_variation

    
    perturbed = np.clip(perturbed, 0.0, None)
    pmax = float(perturbed.max())
    if pmax > 1e-12:
        perturbed = perturbed / pmax

    return perturbed


def generate_standard_lp_intensities(grid_size: int = 500) -> List[str]:
    """
    生成8个带有受控扰动的LP模式参考强度.npy文件，
    用于有意义的CCC验证。
    """
    
    fiber_ref = FiberModel(
        core_radius_um=FiberParams.core_radius_um,
        wavelength_um=FiberParams.wavelength_um,
        na=FiberParams.na,
    )
    V = normalized_frequency_v(
        core_radius_um=fiber_ref.core_radius_um,
        wavelength_um=fiber_ref.wavelength_um,
        na=fiber_ref.na,
    )

    modes: List[Tuple[int, int, str]] = [
        (0, 1, "LP01"),
        (0, 2, "LP02"),
        (0, 3, "LP03"),
        (1, 1, "LP11"),
        (1, 2, "LP12"),
        (2, 1, "LP21"),
        (2, 2, "LP22"),
        (3, 1, "LP31"),
    ]

    rng = np.random.default_rng(seed=42)
    saved: List[str] = []

    for l, m, name in modes:
        try:
            extent_factor = FiberParams.sim_extent_um / max(
                FiberParams.core_radius_um, 1e-12
            )
            _, inten = lp_mode_field(
                l=l,
                m=m,
                grid_size=(grid_size, grid_size),
                fiber=fiber_ref,
                extent_factor=extent_factor,
            )

            
            inten_ref = _apply_reference_perturbations(
                inten, mode_name=name, l=l, m=m, rng=rng
            )

            path = os.path.join(out_dir(), f"{name}_intensity.npy")
            np.save(path, inten_ref)

            
            chk = np.load(path)
            if not np.allclose(inten_ref, chk):
                raise RuntimeError(f"{name} 保存验证失败")

            saved.append(path)
            print(f"OK: {os.path.basename(path)} shape={inten_ref.shape} V={V:.2f}")
        except Exception as exc:
            print(f"FAIL: {name} -> {exc}")

    return saved


def _next_power_of_two(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def generate_hadamard_basis(n: int) -> np.ndarray:
    """
    生成归一化的Hadamard正交基。
    如果n不是2的幂，则填充到最近的2的幂并截断。

    返回：
        形状为 (n, n) 的正交矩阵
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n必须是正整数")

    
    m = _next_power_of_two(n)
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])

    
    H = H / np.sqrt(float(m))

    
    Hn = H[:n, :n].copy()

    
    col_norm = np.linalg.norm(Hn, axis=0) + 1e-12
    Hn = Hn / col_norm
    return Hn


def generate_m_in_hadamard(num_modes: int) -> str:
    """
    生成输入侧M_in（Hadamard基），形状为 (num_modes, num_modes)。
    """
    H = generate_hadamard_basis(num_modes).astype(np.complex128)
    path_basis = os.path.join(out_dir(), "hadamard_basis.npy")
    np.save(path_basis, H.real)

    path_min = os.path.join(out_dir(), f"M_in_hadamard_{num_modes}.npy")
    np.save(path_min, H)
    print(f"OK: hadamard_basis.npy, M_in_hadamard_{num_modes}.npy")
    return path_min


def main() -> None:
    print(f"项目根目录: {project_root()}")
    print(f"输出目录:   {out_dir()}")

    generate_standard_lp_intensities(grid_size=500)
    generate_m_in_hadamard(num_modes=MTMConfig.num_modes)

    print("完成。")


if __name__ == "__main__":
    main()
