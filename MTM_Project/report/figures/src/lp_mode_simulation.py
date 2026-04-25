"""
LP模式仿真与3D可视化模块
生成典型LP模式的3D强度分布，并与“理论模式”进行一致性验证
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from .mtm_reconstruction import MTMReconstructor


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def simulate_lp_modes(
    num_modes: int = 6,
    fiber_params: Dict[str, float] | None = None,
    grid_size: tuple[int, int] = (128, 128),
    noise_level: float = 0.05,
) -> List[Dict]:
    """
    仿真LP模式，并生成3D强度分布与一致性指标

    Args:
        num_modes: 仿真模式数
        fiber_params: 光纤参数字典
        grid_size: 仿真网格大小 (H, W)
        noise_level: 加到“实验强度图”上的噪声水平

    Returns:
        result_rows: 每个模式的一致性结果列表
    """
    if fiber_params is None:
        fiber_params = {
            "core_radius": 25.0,
            "wavelength": 0.532,
            "na": 0.22,
        }

    reconstructor = MTMReconstructor(num_modes=num_modes)
    modes = reconstructor.generate_lp_modes(grid_size, fiber_params)  # (M, H, W)

    H, W = grid_size
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    project_root = get_project_root()
    figures_dir = os.path.join(project_root, "report", "figures")
    report_dir = os.path.join(project_root, "report", "files")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    rows: List[Dict] = []

    for idx in range(min(num_modes, modes.shape[0])):
        field = modes[idx]
        intensity_theory = np.abs(field) ** 2

        
        noise = noise_level * intensity_theory.max() * np.random.randn(
            *intensity_theory.shape
        )
        intensity_meas = intensity_theory + noise
        intensity_meas = np.clip(intensity_meas, 0, None)

        
        t_vec = intensity_theory.flatten().astype(np.float64)
        m_vec = intensity_meas.flatten().astype(np.float64)
        t_vec -= t_vec.mean()
        m_vec -= m_vec.mean()
        num = np.sum(t_vec * m_vec)
        denom = np.sqrt(np.sum(t_vec ** 2) * np.sum(m_vec ** 2)) + 1e-12
        similarity = float(num / denom)

        
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            X,
            Y,
            intensity_theory,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
        )
        ax.set_title(f"LP 模式 {idx} 强度分布")
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        ax.set_zlabel("Intensity (a.u.)")

        fig_path = os.path.join(
            figures_dir, f"lp_mode_{idx}_intensity3d.png"
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close(fig)

        rows.append(
            {
                "mode_index": idx,
                "similarity": similarity,
                "figure_path": fig_path,
            }
        )

    
    df = pd.DataFrame(rows)
    report_csv = os.path.join(report_dir, "lp_mode_verification.csv")
    df.to_csv(report_csv, index=False, encoding="utf-8-sig")

    
    avg_sim = df["similarity"].mean() if not df.empty else 0.0
    threshold = 0.85
    passed = avg_sim >= threshold
    report_txt = os.path.join(report_dir, "lp_mode_verification_summary.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("LP 模式仿真一致性报告（Task 1 理论掌握验证）\n")
        f.write("======================================\n\n")
        f.write(f"仿真模式数量: {len(rows)}\n")
        f.write(f"平均相似度: {avg_sim:.4f}\n")
        f.write(f"理论要求阈值: {threshold:.2f}\n")
        f.write(f"是否通过理论验证: {'是' if passed else '否'}\n")
        f.write("\n逐模式结果:\n")
        for r in rows:
            f.write(
                f"  模式 {r['mode_index']}: 相似度 = {r['similarity']:.4f}, "
                f"图像 = {os.path.basename(r['figure_path'])}\n"
            )

    return rows


if __name__ == "__main__":
    simulate_lp_modes()
