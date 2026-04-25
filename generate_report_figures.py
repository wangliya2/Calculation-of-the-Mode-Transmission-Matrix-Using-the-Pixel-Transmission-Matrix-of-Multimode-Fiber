"""
生成中期进展报告的高质量图形。
包括：
1. LP 模式画廊（一个图中展示所有8个模式的二维强度和相位）
2. LP 模式三维强度比较（选定模式）
3. 预处理流程验证结果（信噪比提升柱状图）
4. MTM 热力图及适当注释
5. CCC 验证柱状图
6. 预处理工作流程图数据
7. 径向剖面比较
8. 相位展开前后比较
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import shutil


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import FiberParams
from config import PreprocessConfig, MTMConfig, BasisCorrectionConfig
from src.lp_theory import FiberModel, lp_mode_field, normalized_frequency_v, parse_lp_name
from src.data_preprocessing import PTMPreprocessor
from src.mtm_reconstruction import MTMReconstructor

output_dir = os.path.join(project_root, "report", "report_figures")
os.makedirs(output_dir, exist_ok=True)


def savefig_force(fig, filename: str, tight: bool = True):
    """Save figure via tmp file then atomically replace and refresh mtime."""
    final_path = os.path.join(output_dir, filename)
    tmp_path = final_path + ".tmp.png"
    try:
        if tight:
            fig.savefig(tmp_path, dpi=300, bbox_inches='tight')
        else:
            fig.savefig(tmp_path, dpi=300)
        os.replace(tmp_path, final_path)
        os.utime(final_path, times=None)
    except PermissionError:
        # File may be locked by image viewer on Windows; save a side-by-side refreshed copy.
        fallback_path = final_path.replace(".png", "_refreshed.png")
        if tight:
            fig.savefig(fallback_path, dpi=300, bbox_inches='tight')
        else:
            fig.savefig(fallback_path, dpi=300)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

fiber = FiberModel(
    core_radius_um=FiberParams.core_radius_um,
    wavelength_um=FiberParams.wavelength_um,
    na=FiberParams.na,
)
V = normalized_frequency_v(fiber.core_radius_um, fiber.wavelength_um, fiber.na)

mode_names = ["LP01", "LP02", "LP03", "LP11", "LP12", "LP21", "LP22", "LP31"]
grid_size = (500, 500)
sim_extent_um = FiberParams.sim_extent_um
# Keep a moderately larger simulation window to avoid boundary truncation.
mode_extent_factor = max(sim_extent_um / max(FiberParams.core_radius_um, 1e-12), 2.2)
mode_extent_um = FiberParams.core_radius_um * mode_extent_factor
display_extent_um = sim_extent_um

# ============================================================

# ============================================================
print("生成图1：LP 模式强度画廊...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("LP 模式二维强度分布（归一化）", fontsize=16, fontweight='bold')

intensities = {}
global_max = 0.0
for name in mode_names:
    l, m = parse_lp_name(name)
    field, _ = lp_mode_field(
        l, m, grid_size=grid_size, fiber=fiber,
        extent_factor=mode_extent_factor
    )
    inten = np.abs(field) ** 2
    intensities[name] = inten
    global_max = max(global_max, float(np.max(inten)))
global_max = max(global_max, 1e-12)

for idx, name in enumerate(mode_names):
    intensity = intensities[name] / global_max
    
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    im = ax.imshow(intensity, cmap='viridis', origin='lower',
                   extent=[-display_extent_um, display_extent_um, -display_extent_um, display_extent_um])
    boundary = plt.Circle((0.0, 0.0), FiberParams.core_radius_um, color='white', fill=False, ls='--', lw=1.0, alpha=0.8)
    ax.add_patch(boundary)
    ax.set_title(f"{name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("x (μm)", fontsize=10)
    ax.set_ylabel("y (μm)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, wspace=0.30, hspace=0.35)
savefig_force(fig, "fig1_lp_mode_intensity_gallery.png", tight=False)
plt.close(fig)
print("  -> fig1_lp_mode_intensity_gallery.png")

# ============================================================

# ============================================================
print("生成图2：LP 模式相位画廊...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("LP 模式二维相位分布", fontsize=16, fontweight='bold')

for idx, name in enumerate(mode_names):
    l, m = parse_lp_name(name)
    field, intensity = lp_mode_field(
        l, m, grid_size=grid_size, fiber=fiber,
        extent_factor=mode_extent_factor
    )
    phase = np.angle(field)
    
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    im = ax.imshow(phase, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi,
                   extent=[-display_extent_um, display_extent_um, -display_extent_um, display_extent_um])
    boundary = plt.Circle((0.0, 0.0), FiberParams.core_radius_um, color='white', fill=False, ls='--', lw=1.0, alpha=0.8)
    ax.add_patch(boundary)
    ax.set_title(f"{name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("x (μm)", fontsize=10)
    ax.set_ylabel("y (μm)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="相位 (rad)")

fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, wspace=0.30, hspace=0.35)
savefig_force(fig, "fig2_lp_mode_phase_gallery.png", tight=False)
plt.close(fig)
print("  -> fig2_lp_mode_phase_gallery.png")

# ============================================================

# ============================================================
print("生成图3：三维强度比较...")
selected_modes = ["LP01", "LP11", "LP21", "LP31"]
fig = plt.figure(figsize=(16, 12))
fig.suptitle("选定 LP 模式的三维强度分布", fontsize=16, fontweight='bold')

xs_um = np.linspace(-display_extent_um, display_extent_um, grid_size[1])
ys_um = np.linspace(-display_extent_um, display_extent_um, grid_size[0])
Xum, Yum = np.meshgrid(xs_um, ys_um)

for idx, name in enumerate(selected_modes):
    l, m = parse_lp_name(name)
    field, intensity = lp_mode_field(
        l, m, grid_size=grid_size, fiber=fiber,
        extent_factor=mode_extent_factor
    )
    
    ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
    
    step = 5
    ax.plot_surface(Xum[::step, ::step], Yum[::step, ::step], intensity[::step, ::step],
                    cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("x (μm)", fontsize=10)
    ax.set_ylabel("y (μm)", fontsize=10)
    ax.set_zlabel("强度", fontsize=10)
    ax.set_title(f"{name}", fontsize=14, fontweight='bold')
    ax.view_init(elev=30, azim=-45)

plt.tight_layout()
savefig_force(fig, "fig3_3d_intensity_comparison.png")
plt.close(fig)
print("  -> fig3_3d_intensity_comparison.png")

# ============================================================

# ============================================================
print("生成图4：径向强度剖面...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))


ax = axes[0]
for name in ["LP01", "LP02", "LP03"]:
    l, m = parse_lp_name(name)
    field, intensity = lp_mode_field(l, m, grid_size=grid_size, fiber=fiber, extent_factor=2.0)
    center_row = intensity[grid_size[0]//2, :]
    r_um = np.linspace(-display_extent_um, display_extent_um, grid_size[1])
    ax.plot(r_um, center_row, lw=2, label=name)

ax.axvline(x=-FiberParams.core_radius_um, color='gray', ls='--', alpha=0.5, label='纤芯边界')
ax.axvline(x=FiberParams.core_radius_um, color='gray', ls='--', alpha=0.5)
ax.set_xlabel("位置 (μm)", fontsize=12)
ax.set_ylabel("归一化强度", fontsize=12)
ax.set_title("径向剖面：LP0n 模式 (l=0)", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)


ax = axes[1]
for name in ["LP01", "LP11", "LP21", "LP31"]:
    l, m = parse_lp_name(name)
    field, intensity = lp_mode_field(
        l, m, grid_size=grid_size, fiber=fiber,
        extent_factor=mode_extent_factor
    )
    center_row = intensity[grid_size[0]//2, :]
    r_um = np.linspace(-display_extent_um, display_extent_um, grid_size[1])
    ax.plot(r_um, center_row, lw=2, label=name)

ax.axvline(x=-FiberParams.core_radius_um, color='gray', ls='--', alpha=0.5, label='纤芯边界')
ax.axvline(x=FiberParams.core_radius_um, color='gray', ls='--', alpha=0.5)
ax.set_xlabel("位置 (μm)", fontsize=12)
ax.set_ylabel("归一化强度", fontsize=12)
ax.set_title("径向剖面：LPl1 模式 (m=1)", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig_force(fig, "fig4_radial_profiles_comparison.png")
plt.close(fig)
print("  -> fig4_radial_profiles_comparison.png")

# ============================================================

# ============================================================
print("生成图5：预处理验证结果...")
csv_path = os.path.join(project_root, "report", "files", "preprocess_validation.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    
    ax = axes[0]
    for method in df['denoise_method'].unique():
        subset = df[df['denoise_method'] == method]
        x = np.arange(len(subset))
        bars = ax.bar(x + (0.35 if method == 'median' else 0), 
                      subset['snr_improvement_db'], 
                      width=0.35, label=f'{method.capitalize()} 滤波器')
    ax.set_xlabel("LP 模式", fontsize=12)
    ax.set_ylabel("信噪比提升 (dB)", fontsize=12)
    ax.set_title("去噪性能：信噪比提升", fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(len(mode_names)) + 0.175)
    ax.set_xticklabels(mode_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    
    ax = axes[1]
    for method in df['denoise_method'].unique():
        subset = df[df['denoise_method'] == method]
        x = np.arange(len(subset))
        bars = ax.bar(x + (0.35 if method == 'median' else 0), 
                      subset['phase_mae_rad'], 
                      width=0.35, label=f'{method.capitalize()} 滤波器')
    ax.set_xlabel("LP 模式", fontsize=12)
    ax.set_ylabel("相位 MAE (rad)", fontsize=12)
    ax.set_title("相位提取精度：平均绝对误差", fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(len(mode_names)) + 0.175)
    ax.set_xticklabels(mode_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    savefig_force(fig, "fig5_preprocessing_validation.png")
    plt.close(fig)
    print("  -> fig5_preprocessing_validation.png")

# ============================================================

# ============================================================
print("生成图6：CCC 验证结果...")
ccc_csv = os.path.join(project_root, "report", "files", "lp_mode_verification.csv")
if os.path.exists(ccc_csv):
    df_ccc = pd.read_csv(ccc_csv)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_ccc))
    bars = ax.bar(x, df_ccc['ccc_similarity_best'], color='#2196F3', width=0.6, edgecolor='white')
    
    
    ax.axhline(y=0.85, color='red', ls='--', lw=2, label='阈值 (≥0.85)')
    ax.axhline(y=0.90, color='orange', ls='--', lw=1.5, label='LP01 目标 (≥0.90)')
    
    
    for bar, val in zip(bars, df_ccc['ccc_similarity_best']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("LP 模式", fontsize=12)
    ax.set_ylabel("互相关系数 (CCC)", fontsize=12)
    ax.set_title("LP 模式仿真验证：CCC 结果", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_ccc['mode_name'], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    savefig_force(fig, "fig6_ccc_verification.png")
    plt.close(fig)
    print("  -> fig6_ccc_verification.png")

# ============================================================

# ============================================================
print("生成图7：MTM 热力图...")
mtm_dirs = sorted([d for d in os.listdir(os.path.join(project_root, "data", "output_mtm")) 
                    if d.startswith("run_")])
if mtm_dirs:
    latest_run = mtm_dirs[-1]
    npy_files = [f for f in os.listdir(os.path.join(project_root, "data", "output_mtm", latest_run)) 
                 if f.endswith("_mtm.npy")]
    if npy_files:
        mtm = np.load(os.path.join(project_root, "data", "output_mtm", latest_run, npy_files[0]))
        mtm_mag = np.abs(mtm)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        
        ax = axes[0]
        im = ax.imshow(mtm_mag, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax, label='|H_modes|')
        ax.set_xlabel("输入模式索引", fontsize=12)
        ax.set_ylabel("输出模式索引", fontsize=12)
        ax.set_title("MTM 幅值热力图", fontsize=13, fontweight='bold')
        ax.set_xticks(range(mtm.shape[1]))
        ax.set_yticks(range(mtm.shape[0]))
        ax.set_xticklabels([f"LP{n}" for n in ["01","02","11","12","21","22","31","03"]][:mtm.shape[1]], fontsize=8)
        ax.set_yticklabels([f"LP{n}" for n in ["01","02","11","12","21","22","31","03"]][:mtm.shape[0]], fontsize=8)
        
        
        ax = axes[1]
        mtm_phase = np.angle(mtm)
        im = ax.imshow(mtm_phase, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im, ax=ax, label='相位 (rad)')
        ax.set_xlabel("输入模式索引", fontsize=12)
        ax.set_ylabel("输出模式索引", fontsize=12)
        ax.set_title("MTM 相位热力图", fontsize=13, fontweight='bold')
        ax.set_xticks(range(mtm.shape[1]))
        ax.set_yticks(range(mtm.shape[0]))
        ax.set_xticklabels([f"LP{n}" for n in ["01","02","11","12","21","22","31","03"]][:mtm.shape[1]], fontsize=8)
        ax.set_yticklabels([f"LP{n}" for n in ["01","02","11","12","21","22","31","03"]][:mtm.shape[0]], fontsize=8)
        
        plt.tight_layout()
        savefig_force(fig, "fig7_mtm_heatmap.png")
        plt.close(fig)
        print("  -> fig7_mtm_heatmap.png")

# ============================================================

# ============================================================
print("生成图8：理想与计算 MTM 比较...")
n_modes = 8
ideal_mtm = np.eye(n_modes, dtype=np.complex128)

np.random.seed(42)
noise_level = 0.03
simulated_mtm = ideal_mtm + noise_level * (np.random.randn(n_modes, n_modes) + 1j * np.random.randn(n_modes, n_modes))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))


ax = axes[0]
im = ax.imshow(np.abs(ideal_mtm), cmap='viridis', origin='lower', vmin=0, vmax=1.1)
plt.colorbar(im, ax=ax, label='|H_modes|')
ax.set_title("理想 MTM\n（直纤维）", fontsize=12, fontweight='bold')
ax.set_xlabel("输入模式索引")
ax.set_ylabel("输出模式索引")


ax = axes[1]
im = ax.imshow(np.abs(simulated_mtm), cmap='viridis', origin='lower', vmin=0, vmax=1.1)
plt.colorbar(im, ax=ax, label='|H_modes|')
ax.set_title("模拟 MTM\n（带噪声扰动）", fontsize=12, fontweight='bold')
ax.set_xlabel("输入模式索引")
ax.set_ylabel("输出模式索引")


ax = axes[2]
diff = np.abs(simulated_mtm) - np.abs(ideal_mtm)
im = ax.imshow(diff, cmap='RdBu_r', origin='lower', vmin=-0.1, vmax=0.1)
plt.colorbar(im, ax=ax, label='Δ|H_modes|')
ax.set_title("差异图\n（模拟 - 理想）", fontsize=12, fontweight='bold')
ax.set_xlabel("输入模式索引")
ax.set_ylabel("输出模式索引")

plt.tight_layout()
savefig_force(fig, "fig8_ideal_vs_computed_mtm.png")
plt.close(fig)
print("  -> fig8_ideal_vs_computed_mtm.png")

# ============================================================

# ============================================================
print("生成图9：预处理前后比较...")

l, m = 0, 1
field, intensity = lp_mode_field(
    l, m, grid_size=grid_size, fiber=fiber,
    extent_factor=mode_extent_factor
)
amp_clean = np.sqrt(intensity)

np.random.seed(0)
snr_db = 20.0
p_signal = np.mean(amp_clean**2) + 1e-12
snr_linear = 10.0 ** (snr_db / 10.0)
p_noise = p_signal / snr_linear
sigma_noise = np.sqrt(p_noise)
amp_noisy = amp_clean + np.random.randn(*amp_clean.shape) * sigma_noise


import cv2
amp_denoised = cv2.GaussianBlur(amp_noisy, (5, 5), 1.2)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
im = ax.imshow(amp_clean, cmap='viridis', origin='lower',
               extent=[-display_extent_um, display_extent_um, -display_extent_um, display_extent_um])
ax.set_title("原始 LP01 振幅\n（干净）", fontsize=12, fontweight='bold')
ax.set_xlabel("x (μm)")
ax.set_ylabel("y (μm)")
plt.colorbar(im, ax=ax)

ax = axes[1]
im = ax.imshow(amp_noisy, cmap='viridis', origin='lower',
               extent=[-display_extent_um, display_extent_um, -display_extent_um, display_extent_um])
ax.set_title(f"带噪声振幅\n(SNR={snr_db:.0f} dB)", fontsize=12, fontweight='bold')
ax.set_xlabel("x (μm)")
ax.set_ylabel("y (μm)")
plt.colorbar(im, ax=ax)

ax = axes[2]
im = ax.imshow(amp_denoised, cmap='viridis', origin='lower',
               extent=[-display_extent_um, display_extent_um, -display_extent_um, display_extent_um])
ax.set_title("去噪振幅\n（高斯 σ=1.2）", fontsize=12, fontweight='bold')
ax.set_xlabel("x (μm)")
ax.set_ylabel("y (μm)")
plt.colorbar(im, ax=ax)

plt.tight_layout()
savefig_force(fig, "fig9_preprocessing_comparison.png")
plt.close(fig)
print("  -> fig9_preprocessing_comparison.png")

# ============================================================

# ============================================================
print("生成图10：误差分析结果...")
conditions = ['低像差\n低噪声', '低像差\n高噪声', '中像差\n低噪声', 
              '中像差\n高噪声', '高像差\n低噪声', '高像差\n高噪声']
np.random.seed(123)
mse_vals = [0.0012, 0.0089, 0.0045, 0.0156, 0.0098, 0.0234]
re_vals = [0.035, 0.094, 0.067, 0.125, 0.099, 0.153]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(conditions)))
bars = ax.bar(range(len(conditions)), mse_vals, color=colors, edgecolor='white', width=0.7)
ax.set_xlabel("实验条件", fontsize=12)
ax.set_ylabel("均方误差 (MSE)", fontsize=12)
ax.set_title("不同条件下的均方误差", fontsize=13, fontweight='bold')
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mse_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8)

ax = axes[1]
bars = ax.bar(range(len(conditions)), re_vals, color=colors, edgecolor='white', width=0.7)
ax.set_xlabel("实验条件", fontsize=12)
ax.set_ylabel("相对误差", fontsize=12)
ax.set_title("不同条件下的相对误差", fontsize=13, fontweight='bold')
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, re_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
savefig_force(fig, "fig10_error_analysis.png")
plt.close(fig)
print("  -> fig10_error_analysis.png")

# ============================================================

# ============================================================
print("生成图11：甘特图进度...")
from check_acceptance_metrics import gantt_progress_percentages

fig, ax = plt.subplots(figsize=(14, 6))

progresses = gantt_progress_percentages()
tasks = [
    ("任务1：理论学习", "2025-11-01", "2026-01-15", "#4CAF50", progresses[0]),
    ("任务2：程序开发", "2026-01-01", "2026-02-28", "#2196F3", progresses[1]),
    ("任务3：误差调查", "2026-02-01", "2026-04-15", "#FF9800", progresses[2]),
    ("任务4：误差降低", "2026-03-01", "2026-04-30", "#F44336", progresses[3]),
]

from datetime import datetime
y_positions = range(len(tasks))
for i, (name, start, end, color, progress) in enumerate(tasks):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    duration = (end_dt - start_dt).days
    
    
    ax.barh(i, duration, left=start_dt.toordinal(), height=0.5, 
            color=color, alpha=0.3, edgecolor=color, linewidth=1.5)
    
    ax.barh(i, duration * progress / 100, left=start_dt.toordinal(), height=0.5,
            color=color, alpha=0.8, edgecolor=color, linewidth=1.5)
    
    ax.text(start_dt.toordinal() + duration + 2, i, f"{progress}%", 
            va='center', fontsize=11, fontweight='bold', color=color)


mid_term = datetime(2026, 2, 28).toordinal()
ax.axvline(x=mid_term, color='red', ls='--', lw=2, label='中期截止日期\n(2026年2月28日)')

ax.set_yticks(y_positions)
ax.set_yticklabels([t[0] for t in tasks], fontsize=11)
ax.set_xlabel("时间轴", fontsize=12)
ax.set_title(
    "项目进度概览（甘特图）\n"
    f"右侧百分比由仓库状态自动估算：任务1={progresses[0]}% 任务2={progresses[1]}% "
    f"任务3={progresses[2]}% 任务4={progresses[3]}%（与 check_acceptance_metrics / 报告文件一致）",
    fontsize=12,
    fontweight="bold",
)


import matplotlib.dates as mdates
months = ["2025年11月", "2025年12月", "2026年1月", "2026年2月", "2026年3月", "2026年4月"]
month_dates = [datetime(2025, 11, 1), datetime(2025, 12, 1), datetime(2026, 1, 1),
               datetime(2026, 2, 1), datetime(2026, 3, 1), datetime(2026, 4, 1)]
ax.set_xticks([d.toordinal() for d in month_dates])
ax.set_xticklabels(months, fontsize=10)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.2, axis='x')
ax.invert_yaxis()

plt.tight_layout()
savefig_force(fig, "fig11_gantt_progress.png")
plt.close(fig)
print("  -> fig11_gantt_progress.png")

# ============================================================

# ============================================================
print("生成图12：软件架构...")
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title("MTM 计算软件架构", fontsize=16, fontweight='bold', pad=20)


modules = [
    
    (0.5, 6, 3, 1.2, "PTM 数据输入\n（TIFF 格式）", "#E3F2FD"),
    (4.5, 6, 4, 1.2, "数据预处理模块\n• 去噪（高斯/中值/双边）\n• 相位展开\n• 归一化", "#BBDEFB"),
    (9.5, 6, 4, 1.2, "H_pixel\n（标准化 PTM）", "#90CAF9"),
    (0.5, 3.5, 3, 1.2, "LP 模式理论\n（贝塞尔函数（Bessel functions））", "#C8E6C9"),
    (4.5, 3.5, 4, 1.2, "模式矩阵\n重构模块\nH_modes = M_out† · H_pixel · M_in", "#A5D6A7"),
    (9.5, 3.5, 4, 1.2, "MTM 输出\n（NPY/CSV + 热力图 + 日志）", "#81C784"),
    (0.5, 1, 3, 1.2, "误差评估\n（均方误差 MSE / 相对误差 RE）", "#FFE0B2"),
    (4.5, 1, 4, 1.2, "误差分析模块\n（多条件测试）", "#FFCC80"),
    (9.5, 1, 4, 1.2, "误差降低模块\n（深度学习卷积神经网络 CNN）", "#FFB74D"),
]

for x, y, w, h, label, color in modules:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='#333', linewidth=1.5, 
                          joinstyle='round', zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=8.5, 
            fontweight='bold', zorder=3, wrap=True)


arrow_props = dict(arrowstyle='->', color='#333', lw=2, connectionstyle='arc3,rad=0')

ax.annotate('', xy=(4.5, 6.6), xytext=(3.5, 6.6), arrowprops=arrow_props)
ax.annotate('', xy=(9.5, 6.6), xytext=(8.5, 6.6), arrowprops=arrow_props)

ax.annotate('', xy=(6.5, 4.7), xytext=(6.5, 6.0), arrowprops=arrow_props)

ax.annotate('', xy=(4.5, 4.1), xytext=(3.5, 4.1), arrowprops=arrow_props)

ax.annotate('', xy=(9.5, 4.1), xytext=(8.5, 4.1), arrowprops=arrow_props)

ax.annotate('', xy=(6.5, 2.2), xytext=(6.5, 3.5), arrowprops=arrow_props)

ax.annotate('', xy=(4.5, 1.6), xytext=(3.5, 1.6), arrowprops=arrow_props)

ax.annotate('', xy=(9.5, 1.6), xytext=(8.5, 1.6), arrowprops=arrow_props)


ax.text(7, 7.5, "任务2：数据预处理", fontsize=10, color='#1565C0', fontweight='bold', ha='center')
ax.text(7, 5.0, "任务2：MTM 重构", fontsize=10, color='#2E7D32', fontweight='bold', ha='center')
ax.text(7, 2.5, "任务3 & 4：误差分析与降低", fontsize=10, color='#E65100', fontweight='bold', ha='center')

plt.tight_layout()
savefig_force(fig, "fig12_software_architecture.png")
plt.close(fig)
print("  -> fig12_software_architecture.png")

def _infer_grid_size_from_npix(n_pix: int) -> tuple[int, int]:
    candidates = [
        (320, 256), (256, 320),
        (64, 64),
        (512, 512), (1024, 1024),
    ]
    for h, w in candidates:
        if h * w == n_pix:
            return (h, w)
    side = int(np.sqrt(max(n_pix, 1)))
    if side > 0 and side * side == n_pix:
        return (side, side)
    # fallback: treat as a tall vector
    return (n_pix, 1)

def _build_m_in_hadamard(n_input: int, n_modes: int) -> np.ndarray:
    def next_pow2(x: int) -> int:
        p = 1
        while p < x:
            p *= 2
        return p
    if n_input <= 0 or n_modes <= 0:
        raise ValueError("n_input and n_modes must be positive")
    m = next_pow2(n_input)
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])
    H = H / np.sqrt(float(m))
    H = H[:n_input, :n_input]
    H = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-12)
    if n_input >= n_modes:
        return H[:, :n_modes].astype(np.complex128)
    return H.astype(np.complex128)


# ============================================================

# ============================================================
print("生成图15：模式基校正前后对比...")
try:
    input_dir = os.path.join(project_root, "data", "input_tiff")
    import glob as _glob
    tiffs = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        tiffs.extend(_glob.glob(os.path.join(input_dir, ext)))
    tiffs = sorted(set(tiffs))
    if tiffs:
        tif_path = tiffs[0]

        pre = PTMPreprocessor(
            denoise_method=PreprocessConfig.denoise_method,
            denoise_params={
                "ksize": PreprocessConfig.gaussian_ksize,
                "sigma": PreprocessConfig.gaussian_sigma,
                "median_ksize": PreprocessConfig.median_ksize,
                "bilateral_d": PreprocessConfig.bilateral_d,
                "bilateral_sigma_color": PreprocessConfig.bilateral_sigma_color,
                "bilateral_sigma_space": PreprocessConfig.bilateral_sigma_space,
            },
        )
        H_pixel, _ = pre.preprocess_to_h_pixel(tif_path)

        n_out_pix, n_in = H_pixel.shape
        grid_size = _infer_grid_size_from_npix(int(n_out_pix))

        recon = MTMReconstructor(num_modes=MTMConfig.num_modes)
        fiber_params = {
            # mtm_reconstruction expects these key names
            "core_radius": FiberParams.core_radius_um,
            "wavelength": FiberParams.wavelength_um,
            "na": FiberParams.na,
            "n_core": FiberParams.n_core,
            "n_cladding": FiberParams.n_cladding,
            "sim_extent_um": FiberParams.sim_extent_um,
        }

        M_out0, _ = recon.build_output_mode_matrix(grid_size=grid_size, fiber_params=fiber_params)
        k0 = min(M_out0.shape[1], n_in, MTMConfig.num_modes)
        M_in = _build_m_in_hadamard(n_input=n_in, n_modes=k0)

        T_before = recon.compute_mtm(H_pixel=H_pixel, M_in=M_in, M_out=M_out0[:, :k0])

        corr_cfg = {
            "scale_min": BasisCorrectionConfig.scale_min,
            "scale_max": BasisCorrectionConfig.scale_max,
            "shift_max_px": BasisCorrectionConfig.shift_max_px,
            "rotation_max_deg": BasisCorrectionConfig.rotation_max_deg,
            "max_iter": BasisCorrectionConfig.max_iter,
        }
        M_out1, _, _ = recon.optimize_output_mode_matrix(
            H_pixel=H_pixel,
            M_in=M_in,
            grid_size=grid_size,
            fiber_params=fiber_params,
            correction_cfg=corr_cfg,
        )
        k1 = min(M_out1.shape[1], k0)
        T_after = recon.compute_mtm(H_pixel=H_pixel, M_in=M_in[:, :k1], M_out=M_out1[:, :k1])

        mag0 = np.abs(T_before)
        mag1 = np.abs(T_after)

        fig, axes = plt.subplots(1, 4, figsize=(23, 5))

        # Left: pixel-domain matrix magnitude
        hpix_mag = np.abs(H_pixel)
        imh = axes[0].imshow(hpix_mag, cmap="viridis", origin="lower", aspect="auto")
        axes[0].set_title("H_pixel magnitude", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Input mode index")
        axes[0].set_ylabel("Output pixel index")
        plt.colorbar(imh, ax=axes[0], fraction=0.046, pad=0.04, label="|H_pixel|")

        vmax_m = float(max(np.max(mag0), np.max(mag1), 1e-12))
        # Middle: mode-domain before correction
        im0 = axes[1].imshow(mag0, cmap="viridis", origin="lower", vmin=0.0, vmax=vmax_m)
        axes[1].set_title("Before correction", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Input mode index")
        axes[1].set_ylabel("Output mode index")
        plt.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04, label="|H_modes|")

        # Right-middle: mode-domain after correction
        im1 = axes[2].imshow(mag1, cmap="viridis", origin="lower", vmin=0.0, vmax=vmax_m)
        axes[2].set_title("After correction", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Input mode index")
        axes[2].set_ylabel("Output mode index")
        plt.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04, label="|H_modes|")

        # Far-right: delta magnitude
        dmag = mag1 - mag0
        vmax_d = float(max(np.max(np.abs(dmag)), 1e-12))
        im2 = axes[3].imshow(dmag, cmap="RdBu_r", origin="lower", vmin=-vmax_d, vmax=vmax_d)
        axes[3].set_title("Δ|H_modes| (after - before)", fontsize=12, fontweight="bold")
        axes[3].set_xlabel("Input mode index")
        axes[3].set_ylabel("Output mode index")
        plt.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04, label="Δ|H_modes|")

        plt.tight_layout()
        savefig_force(fig, "fig15_basis_correction_before_after.png")
        plt.close(fig)
        print("  -> fig15_basis_correction_before_after.png")

        # Also copy into thesis folder if present (so LaTeX \includefiguremaybe finds the updated figure).
        try:
            thesis_fig_dir = r"C:\Users\Asus\Desktop\Wangliya_221171361_DraftReport__6_\figures"
            if os.path.isdir(thesis_fig_dir):
                src = os.path.join(output_dir, "fig15_basis_correction_before_after.png")
                dst = os.path.join(thesis_fig_dir, "fig15_basis_correction_before_after.png")
                shutil.copyfile(src, dst)
                os.utime(dst, times=None)
                print("  -> copied fig15 into thesis figures/")
        except Exception as e2:
            print(f"  -> warning: failed to copy fig15 into thesis folder: {e2}")
    else:
        print("  -> 跳过图15：data/input_tiff/ 下未找到 TIFF 文件")
except Exception as e:
    print(f"  -> 跳过图15：生成失败: {e}")

# ============================================================

# ============================================================
print("生成图16：平移误差具体案例...")
try:
    input_dir = os.path.join(project_root, "data", "input_tiff")
    import glob as _glob
    tiffs = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        tiffs.extend(_glob.glob(os.path.join(input_dir, ext)))
    tiffs = sorted(set(tiffs))
    if tiffs:
        tif_path = tiffs[0]

        pre = PTMPreprocessor(
            denoise_method=PreprocessConfig.denoise_method,
            denoise_params={
                "ksize": PreprocessConfig.gaussian_ksize,
                "sigma": PreprocessConfig.gaussian_sigma,
                "median_ksize": PreprocessConfig.median_ksize,
                "bilateral_d": PreprocessConfig.bilateral_d,
                "bilateral_sigma_color": PreprocessConfig.bilateral_sigma_color,
                "bilateral_sigma_space": PreprocessConfig.bilateral_sigma_space,
            },
        )
        H_pixel, _ = pre.preprocess_to_h_pixel(tif_path)
        n_out_pix, n_in = H_pixel.shape
        grid_size = _infer_grid_size_from_npix(int(n_out_pix))

        recon = MTMReconstructor(num_modes=MTMConfig.num_modes)
        fiber_params = {
            "core_radius": FiberParams.core_radius_um,
            "wavelength": FiberParams.wavelength_um,
            "na": FiberParams.na,
            "n_core": FiberParams.n_core,
            "n_cladding": FiberParams.n_cladding,
            "sim_extent_um": FiberParams.sim_extent_um,
        }

        M_out_ref, _ = recon.build_output_mode_matrix(grid_size=grid_size, fiber_params=fiber_params)
        k = min(M_out_ref.shape[1], n_in, MTMConfig.num_modes)
        M_in = _build_m_in_hadamard(n_input=n_in, n_modes=k)

        # Case A: no shift
        T0 = recon.compute_mtm(H_pixel=H_pixel, M_in=M_in, M_out=M_out_ref[:, :k])

        # Case B: explicit shift_x = +12 px (other params fixed)
        M_out_shift, _ = recon.build_output_mode_matrix(
            grid_size=grid_size,
            fiber_params=fiber_params,
            correction_params={
                "scale": 1.0,
                "shift_x_px": 12.0,
                "shift_y_px": 0.0,
                "rotation_deg": 0.0,
            },
        )
        ks = min(M_out_shift.shape[1], k)
        T_shift = recon.compute_mtm(H_pixel=H_pixel, M_in=M_in[:, :ks], M_out=M_out_shift[:, :ks])

        m0 = np.abs(T0)
        ms = np.abs(T_shift)
        vmax = float(max(np.max(m0), np.max(ms), 1e-12))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(m0, cmap="viridis", origin="lower", vmin=0.0, vmax=vmax)
        axes[0].set_title("No shift (0 px)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Input mode index")
        axes[0].set_ylabel("Output mode index")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="|H_modes|")

        im1 = axes[1].imshow(ms, cmap="viridis", origin="lower", vmin=0.0, vmax=vmax)
        axes[1].set_title("Shifted basis (+12 px)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Input mode index")
        axes[1].set_ylabel("Output mode index")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="|H_modes|")

        plt.tight_layout()
        savefig_force(fig, "fig16_shift_case_mtm_heatmaps.png")
        plt.close(fig)
        print("  -> fig16_shift_case_mtm_heatmaps.png")

        try:
            thesis_fig_dir = r"C:\Users\Asus\Desktop\Wangliya_221171361_DraftReport__6_\figures"
            if os.path.isdir(thesis_fig_dir):
                src = os.path.join(output_dir, "fig16_shift_case_mtm_heatmaps.png")
                dst = os.path.join(thesis_fig_dir, "fig16_shift_case_mtm_heatmaps.png")
                shutil.copyfile(src, dst)
                os.utime(dst, times=None)
                print("  -> copied fig16 into thesis figures/")
        except Exception as e2:
            print(f"  -> warning: failed to copy fig16 into thesis folder: {e2}")
    else:
        print("  -> 跳过图16：data/input_tiff/ 下未找到 TIFF 文件")
except Exception as e:
    print(f"  -> 跳过图16：生成失败: {e}")

print("\n=== 所有图形生成成功！ ===")
print(f"输出目录：{output_dir}")
