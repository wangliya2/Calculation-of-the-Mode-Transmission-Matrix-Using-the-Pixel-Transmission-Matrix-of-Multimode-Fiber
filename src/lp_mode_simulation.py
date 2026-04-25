"""
LP 模式仿真与可视化模块（中期验收导向）

现状说明（重要）：
- 本项目当前的 LP 模式场生成使用的是 `MTMReconstructor.generate_lp_modes` 的简化模型，
  还未实现“阶跃折射率光纤标量模式理论 + 特征方程求根 + Bessel/Hankel 精确模场”的完整推导数值实现。
- 为了满足“可复现、可验收”的工程要求，本模块提供：
    1) 固定光纤参数（默认与验收一致：a=50um, NA=0.22, n2=1.444, λ=1064nm）
    2) 至少 8 个模式的批量可视化输出（2D 强度/相位，3D 强度多视角，径向强度曲线）
    3) 与“参考模式数据”的 CCC(互相关系数) 一致性验证接口

你只需要把外部参考（RP Fiber Power/Lumerical/教材图像导出的数组）放到指定目录，
即可自动计算每个模式的 CCC 并生成 CSV/文本总结。
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.ndimage import center_of_mass
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import shift as nd_shift

try:
    
    from skimage.transform import resize as sk_resize
except Exception:  # pragma: no cover
    sk_resize = None

from .lp_theory import FiberModel, lp_mode_field, parse_lp_name


EXPORT_EPS = False

EXPORT_SVG = False


warnings.filterwarnings(
    "ignore",
    message=r"The PostScript backend does not support transparency.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Glyph .* missing from font\(s\) DejaVu Sans\.",
)


def _save_eps_quiet(path: str) -> None:
    """
    保存EPS时静默PostScript透明度告警（不影响PNG/SVG结果）。
    """
    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore", category=UserWarning)
        plt.savefig(path)


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ccc(a: np.ndarray, b: np.ndarray) -> float:
    """
    对展平且去均值的向量计算互相关系数（Cross-Correlation Coefficient，CCC）。
    """
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    if av.size != bv.size or av.size == 0:
        return float("nan")
    av = av - av.mean()
    bv = bv - bv.mean()
    num = float(np.sum(av * bv))
    den = float(np.sqrt(np.sum(av**2) * np.sum(bv**2)) + 1e-12)
    return float(num / den)


def _resize_to_match(a: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    使用 skimage 将二维数组调整为目标形状。
    如果不可用且形状不匹配，则抛出异常。
    """
    img = np.asarray(a, dtype=np.float64)
    if img.shape == target_shape:
        return img
    if sk_resize is None:
        raise ValueError(
            f"参考形状 {img.shape} != 目标形状 {target_shape}，"
            "且 skimage.transform.resize 不可用，无法插值对齐。"
        )
    return sk_resize(
        img,
        output_shape=target_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float64)


def _ccc_best_alignment(
    sim_intensity: np.ndarray,
    ref_intensity: np.ndarray,
    coarse_step_deg: int = 5,
    refine_half_window_deg: int = 5,
    max_shift_px: int = 20,
) -> tuple[float, float, str, float, float]:
    """
    为解决 LP 模式简并导致的“方向不唯一”（旋转/镜像等价），
    对参考强度图做旋转/翻转搜索，返回最大 CCC 以及对应角度/翻转方式。

    Returns:
        best_ccc, best_angle_deg, best_flip, best_shift_y_px, best_shift_x_px
          - best_flip: "none" | "lr" | "ud"
    """
    sim = np.asarray(sim_intensity, dtype=np.float64)
    ref0 = _resize_to_match(np.asarray(ref_intensity, dtype=np.float64), sim.shape)

    
    sim = sim / (float(sim.max()) + 1e-12)
    ref0 = ref0 / (float(ref0.max()) + 1e-12)

    flips: list[tuple[str, np.ndarray]] = [
        ("none", ref0),
        ("lr", np.fliplr(ref0)),
        ("ud", np.flipud(ref0)),
    ]

    angles_coarse = list(range(0, 360, max(1, int(coarse_step_deg))))
    best = (-1.0, 0.0, "none", 0.0, 0.0)  # ccc, angle, flip, shift_y, shift_x

    def _estimate_center_shift(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
        src_center = np.array(center_of_mass(src))
        dst_center = np.array(center_of_mass(dst))
        if not np.all(np.isfinite(src_center)) or not np.all(np.isfinite(dst_center)):
            return 0.0, 0.0
        delta = dst_center - src_center
        dy = float(np.clip(delta[0], -max_shift_px, max_shift_px))
        dx = float(np.clip(delta[1], -max_shift_px, max_shift_px))
        return dy, dx

    def eval_angles(ref_img: np.ndarray, flip_tag: str, angles: list[int]) -> None:
        nonlocal best
        for ang in angles:
            ref_rot = nd_rotate(
                ref_img,
                angle=float(ang),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            dy0, dx0 = _estimate_center_shift(ref_rot, sim)
            shift_candidates_y = sorted(set([int(round(dy0 + d)) for d in (-2, -1, 0, 1, 2)]))
            shift_candidates_x = sorted(set([int(round(dx0 + d)) for d in (-2, -1, 0, 1, 2)]))
            for dy in shift_candidates_y:
                for dx in shift_candidates_x:
                    ref_aligned = nd_shift(
                        ref_rot,
                        shift=(float(np.clip(dy, -max_shift_px, max_shift_px)), float(np.clip(dx, -max_shift_px, max_shift_px))),
                        order=1,
                        mode="constant",
                        cval=0.0,
                        prefilter=False,
                    )
                    c = _ccc(sim, ref_aligned)
                    if np.isfinite(c) and c > best[0]:
                        best = (
                            float(c),
                            float(ang),
                            flip_tag,
                            float(np.clip(dy, -max_shift_px, max_shift_px)),
                            float(np.clip(dx, -max_shift_px, max_shift_px)),
                        )

    
    for flip_tag, ref_img in flips:
        eval_angles(ref_img, flip_tag, angles_coarse)

    
    center = int(round(best[1]))
    refine_angles = [(center + d) % 360 for d in range(-refine_half_window_deg, refine_half_window_deg + 1)]
    
    ref_best = next(img for tag, img in flips if tag == best[2])
    eval_angles(ref_best, best[2], refine_angles)

    return best[0], best[1], best[2], best[3], best[4]


def _radial_profile(intensity: np.ndarray, center: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过对具有相同半径的像素进行分箱，计算径向平均强度分布 I(r)。
    """
    img = np.asarray(intensity, dtype=np.float64)
    h, w = img.shape
    cy, cx = center
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = r.astype(np.int32)
    r_max = r_int.max()
    sums = np.bincount(r_int.ravel(), weights=img.ravel(), minlength=r_max + 1)
    counts = np.bincount(r_int.ravel(), minlength=r_max + 1)
    prof = sums / (counts + 1e-12)
    return np.arange(r_max + 1), prof


def _load_reference_mode(ref_dir: str, mode_name: str) -> Optional[np.ndarray]:
    """
    如果存在，则从 .npy 文件加载参考模式强度。
    期望文件名格式：{mode_name}_intensity.npy
    """
    path = os.path.join(ref_dir, f"{mode_name}_intensity.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def simulate_lp_modes(
    mode_names: Optional[List[str]] = None,
    fiber_params: Dict[str, float] | None = None,
    grid_size: tuple[int, int] = (500, 500),
    reference_dir: Optional[str] = None,
) -> List[Dict]:
    """
    仿真LP模式，并生成3D强度分布与一致性指标

    Args:
        mode_names: 模式名称列表（至少 8 个），默认生成 8 个典型名称：
            LP01, LP02, LP11, LP12, LP21, LP22, LP31, LP03
        fiber_params: 光纤参数字典（单位 um）
        grid_size: 仿真网格大小 (H, W)，验收要求 ≥ 500×500
        reference_dir: 参考模式数据目录（可选），内部放:
            LP01_intensity.npy ... 等，用于 CCC 验证

    Returns:
        result_rows: 每个模式的一致性结果列表
    """
    if fiber_params is None:
        fiber_params = {
            "core_radius": 25.0,
            "wavelength": 1.55,
            "na": 0.22,
            "sim_extent_um": 35.0,
        }

    if mode_names is None:
        mode_names = ["LP01", "LP02", "LP11", "LP12", "LP21", "LP22", "LP31", "LP03"]

    fiber = FiberModel(
        core_radius_um=float(fiber_params["core_radius"]),
        wavelength_um=float(fiber_params["wavelength"]),
        na=float(fiber_params["na"]),
    )

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

    
    sim_extent_um = float(fiber_params.get("sim_extent_um", fiber_params["core_radius"]))
    extent_factor = sim_extent_um / max(float(fiber_params["core_radius"]), 1e-12)
    xs_um = np.linspace(-sim_extent_um, sim_extent_um, W)
    ys_um = np.linspace(-sim_extent_um, sim_extent_um, H)
    Xum, Yum = np.meshgrid(xs_um, ys_um)

    reference_source = "external"
    if reference_dir is None:
        candidate_ref = os.path.join(project_root, "data", "reference_lp_modes")
        fallback_ref = os.path.join(project_root, "data", "standard_lp_modes")
        if os.path.exists(candidate_ref) and os.listdir(candidate_ref):
            reference_dir = candidate_ref
        else:
            reference_dir = fallback_ref
            reference_source = "internal_fallback"
    else:
        reference_source = "user_specified"
    os.makedirs(reference_dir, exist_ok=True)

    for mode_name in mode_names:
        print(f"[LP Sim] Running {mode_name} ...")
        l, m = parse_lp_name(mode_name)
        field, intensity = lp_mode_field(
            l=l,
            m=m,
            grid_size=grid_size,
            fiber=fiber,
            extent_factor=extent_factor,
        )
        
        phase = np.angle(field.astype(np.complex128))

        ref_intensity = _load_reference_mode(reference_dir, mode_name)
        if ref_intensity is None:
            similarity = float("nan")
            best_ccc = float("nan")
            best_angle = float("nan")
            best_flip = "na"
            best_shift_y = float("nan")
            best_shift_x = float("nan")
        else:
            similarity = _ccc(intensity, _resize_to_match(ref_intensity, intensity.shape))
            best_ccc, best_angle, best_flip, best_shift_y, best_shift_x = _ccc_best_alignment(intensity, ref_intensity)

        
        fig2 = plt.figure(figsize=(6, 5))
        im = plt.imshow(intensity, cmap="viridis", origin="lower",
                        extent=[xs_um.min(), xs_um.max(), ys_um.min(), ys_um.max()])
        plt.colorbar(im, label="归一化强度 (a.u.)")
        plt.title(f"{mode_name} 强度 (2D)")
        plt.xlabel("x (um)")
        plt.ylabel("y (um)")
        plt.tight_layout()
        fig2_base = os.path.join(figures_dir, f"{mode_name}_mode_2D_intensity")
        fig2_path = f"{fig2_base}.png"
        plt.savefig(fig2_path, dpi=300)
        if EXPORT_SVG:
            plt.savefig(f"{fig2_base}.svg")
        if EXPORT_EPS:
            _save_eps_quiet(f"{fig2_base}.eps")
        plt.close(fig2)

        
        figp = plt.figure(figsize=(6, 5))
        imp = plt.imshow(phase, cmap="twilight", origin="lower",
                         vmin=-np.pi, vmax=np.pi,
                         extent=[xs_um.min(), xs_um.max(), ys_um.min(), ys_um.max()])
        plt.colorbar(imp, label="相位 (rad)")
        plt.title(f"{mode_name} 相位 (2D)")
        plt.xlabel("x (um)")
        plt.ylabel("y (um)")
        plt.tight_layout()
        figp_base = os.path.join(figures_dir, f"{mode_name}_mode_2D_phase")
        figp_path = f"{figp_base}.png"
        plt.savefig(figp_path, dpi=300)
        if EXPORT_SVG:
            plt.savefig(f"{figp_base}.svg")
        if EXPORT_EPS:
            _save_eps_quiet(f"{figp_base}.eps")
        plt.close(figp)

        
        r, prof = _radial_profile(intensity, center=(H / 2.0, W / 2.0))
        fig_r = plt.figure(figsize=(6, 4))
        plt.plot(r, prof, lw=2)
        plt.title(f"{mode_name} 径向强度分布")
        plt.xlabel("半径 (像素)")
        plt.ylabel("平均归一化强度")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_r_base = os.path.join(figures_dir, f"{mode_name}_mode_radial_profile")
        fig_r_path = f"{fig_r_base}.png"
        plt.savefig(fig_r_path, dpi=300)
        if EXPORT_SVG:
            plt.savefig(f"{fig_r_base}.svg")
        if EXPORT_EPS:
            _save_eps_quiet(f"{fig_r_base}.eps")
        plt.close(fig_r)

        
        views = [
            ("front", 20, -60),
            ("angle45", 35, -45),
            ("top", 90, -90),
        ]
        fig3 = plt.figure(figsize=(7, 6))
        ax = fig3.add_subplot(111, projection="3d")
        
        step = 6  
        ax.plot_surface(
            Xum[::step, ::step],
            Yum[::step, ::step],
            intensity[::step, ::step],
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_zlabel("归一化强度")
        ax.set_title(f"{mode_name} 强度 (3D)")
        plt.tight_layout()
        fig3_base = os.path.join(figures_dir, f"{mode_name}_mode_3D_intensity")
        for tag, elev, azim in views:
            ax.view_init(elev=elev, azim=azim)
            png_path = f"{fig3_base}_{tag}.png"
            plt.savefig(png_path, dpi=300)
            if EXPORT_SVG:
                plt.savefig(f"{fig3_base}_{tag}.svg")
            if EXPORT_EPS:
                _save_eps_quiet(f"{fig3_base}_{tag}.eps")
        plt.close(fig3)

        rows.append(
            {
                "mode_name": mode_name,
                "ccc_similarity_raw": similarity,
                "ccc_similarity_best": best_ccc,
                "ccc_best_angle_deg": best_angle,
                "ccc_best_flip": best_flip,
                "ccc_best_shift_y_px": best_shift_y,
                "ccc_best_shift_x_px": best_shift_x,
                "intensity_2d_png": os.path.basename(fig2_path),
                "phase_2d_png": os.path.basename(figp_path),
                "radial_profile_png": os.path.basename(fig_r_path),
                "intensity_3d_front_png": os.path.basename(f"{fig3_base}_front.png"),
                "intensity_3d_angle45_png": os.path.basename(f"{fig3_base}_angle45.png"),
                "intensity_3d_top_png": os.path.basename(f"{fig3_base}_top.png"),
            }
        )

    
    df = pd.DataFrame(rows)
    report_csv = os.path.join(report_dir, "lp_mode_verification.csv")
    df.to_csv(report_csv, index=False, encoding="utf-8-sig")

    
    threshold = 0.85
    base_threshold = 0.90  
    avg_sim = float(df["ccc_similarity_best"].dropna().mean()) if not df.empty else 0.0
    lp01_sim = float(
        df.loc[df["mode_name"] == "LP01", "ccc_similarity_best"].dropna().mean()
    ) if not df.empty else float("nan")
    passed = (avg_sim >= threshold) and (
        np.isnan(lp01_sim) or (lp01_sim >= base_threshold)
    )
    report_txt = os.path.join(report_dir, "lp_mode_verification_summary.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("LP 模式仿真一致性报告（Task 1 理论掌握验证）\n")
        f.write("======================================\n\n")
        f.write(f"仿真模式数量: {len(rows)}\n")
        f.write(f"网格大小: {grid_size}\n")
        f.write(f"参考数据目录: {reference_dir}\n")
        f.write(f"参考数据来源: {reference_source}\n")
        f.write(
            "参考数据命名约定: LP01_intensity.npy 等（强度二维数组，尺寸需与仿真一致）\n\n"
        )
        if reference_source == "internal_fallback":
            f.write("注意：当前参考数据来自项目内部生成（standard_lp_modes），属于“内部一致性验证”。\n")
            f.write("若需满足“外部一致性验证”，请将 RP Fiber Power / Lumerical 导出的强度数组放入 data/reference_lp_modes。\n\n")
        f.write("说明：LP(l>0) 模式存在简并，强度图可能出现旋转/镜像等价。\n")
        f.write("本报告使用“平移 + 旋转 + 翻转对齐后取最大 CCC”的结果作为验收指标。\n\n")
        f.write(f"平均 CCC（对齐后，有参考的模式）: {avg_sim:.4f}\n")
        f.write(f"LP01 CCC（对齐后，若有参考）: {lp01_sim:.4f}\n")
        f.write(f"理论要求阈值: 平均≥{threshold:.2f}, LP01≥{base_threshold:.2f}\n")
        f.write(f"是否通过理论验证: {'是' if passed else '否（或缺参考数据）'}\n")
        f.write("\n逐模式结果:\n")
        for r in rows:
            f.write(
                f"  模式 {r['mode_name']}: "
                f"CCC_raw = {r['ccc_similarity_raw']:.4f}, "
                f"CCC_best = {r['ccc_similarity_best']:.4f}, "
                f"best_angle={r['ccc_best_angle_deg']:.0f}度, "
                f"flip={r['ccc_best_flip']}, "
                f"shift=({r['ccc_best_shift_y_px']:.0f}, {r['ccc_best_shift_x_px']:.0f}) px\n"
            )

    return rows


if __name__ == "__main__":
    simulate_lp_modes()
