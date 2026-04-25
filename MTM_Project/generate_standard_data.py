from __future__ import annotations

"""
在项目根目录执行本脚本（与 data/、src/ 同级）：
  python generate_standard_data.py

功能：
1) 自动创建 data/standard_lp_modes/
2) 生成 8 种典型 LP 模式的“标准理论强度” .npy 文件（500×500），供：
   - LP 仿真一致性验证（CCC）
   - 预处理/MTM 模块测试的“已知理论强度分布”
3) 生成 Hadamard 正交输入基（标准正交基）供 MTM 输入侧 M_in 使用：
   - data/standard_lp_modes/hadamard_basis.npy
   - data/standard_lp_modes/M_in_hadamard.npy（默认 num_modes=8）

注意：
- LP 模式强度生成使用阶跃折射率弱导近似的特征方程符号变化点近似（与你给的 ini 脚本一致）。
- 若 V 值下某些模式不存在，会给出明确报错提示，但脚本仍会继续生成其它模式。
"""

import os
from typing import List, Tuple

import numpy as np

from config import FiberParams, MTMConfig
from src.lp_theory import FiberModel, lp_mode_field, normalized_frequency_v, parse_lp_name


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def out_dir() -> str:
    p = os.path.join(project_root(), "data", "standard_lp_modes")
    os.makedirs(p, exist_ok=True)
    return p


def generate_standard_lp_intensities(grid_size: int = 500) -> List[str]:
    """
    生成 8 种典型 LP 模式强度 .npy 文件。
    """
    fiber = FiberModel(
        core_radius_um=FiberParams.core_radius_um,
        wavelength_um=FiberParams.wavelength_um,
        na=FiberParams.na,
    )
    V = normalized_frequency_v(
        core_radius_um=fiber.core_radius_um,
        wavelength_um=fiber.wavelength_um,
        na=fiber.na,
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

    saved: List[str] = []
    for l, m, name in modes:
        try:
            _, inten = lp_mode_field(
                l=l,
                m=m,
                grid_size=(grid_size, grid_size),
                fiber=fiber,
                extent_factor=2.0,
            )
            path = os.path.join(out_dir(), f"{name}_intensity.npy")
            np.save(path, inten)
            # quick verify load
            chk = np.load(path)
            if not np.allclose(inten, chk):
                raise RuntimeError(f"{name} 保存校验失败")
            saved.append(path)
            print(f"OK: {os.path.basename(path)} shape={inten.shape} V={V:.2f}")
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
    生成归一化 Hadamard 正交基（若 n 不是 2 的幂，则升到最近的 2 的幂并截取前 n 行列）。
    返回 shape (n, n) 的正交矩阵（近似，来自更大 Hadamard 的截断）。
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n 必须为正整数")

    # Sylvester construction
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
    生成输入侧 M_in（Hadamard 基），shape=(num_modes, num_modes)。
    """
    H = generate_hadamard_basis(num_modes).astype(np.complex128)
    path_basis = os.path.join(out_dir(), "hadamard_basis.npy")
    np.save(path_basis, H.real)

    path_min = os.path.join(out_dir(), f"M_in_hadamard_{num_modes}.npy")
    np.save(path_min, H)
    print(f"OK: hadamard_basis.npy, M_in_hadamard_{num_modes}.npy")
    return path_min


def main() -> None:
    print(f"Project root: {project_root()}")
    print(f"Output dir:   {out_dir()}")

    generate_standard_lp_intensities(grid_size=500)
    generate_m_in_hadamard(num_modes=MTMConfig.num_modes)

    print("Done.")


if __name__ == "__main__":
    main()

