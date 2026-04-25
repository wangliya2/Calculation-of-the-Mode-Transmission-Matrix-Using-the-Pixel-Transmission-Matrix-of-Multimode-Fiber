"""
误差源分析模块（任务3）

功能：
- 计算不同条件下MTM结果的均方误差（MSE）/相对误差（RE）统计
- 区分不同像差和噪声水平的贡献
- 生成“误差源分析报告”

文件命名规范：
    straight_aberrLow_noiseLow_mtm.npy
    straight_aberrHigh_noiseHigh_mtm.npy
  其中 aberr{Low|Med|High}, noise{Low|High} 表示条件。
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .article_reference_mtm import load_mtm_reference_from_file
from .error_metrics import mse, offdiag_energy_ratio, relative_error


def parse_condition_from_name(name: str) -> Tuple[str, str]:
    """
    从文件名解析像差和噪声水平。
    如果未找到匹配模式，则返回 ('Unknown', 'Unknown')。
    """
    lname = name.lower()
    if "article" in lname:
        return "Article", "Repo"
    aberr_match = re.search(r"aberr(Low|Med|High)", name, re.IGNORECASE)
    noise_match = re.search(r"noise(Low|High)", name, re.IGNORECASE)
    
    if aberr_match is None or noise_match is None:
        if "straight" in lname:
            return "Low", "Low"
        if "bent" in lname:
            return "High", "High"
    aberr = aberr_match.group(1).capitalize() if aberr_match else "Unknown"
    noise = noise_match.group(1).capitalize() if noise_match else "Unknown"
    return aberr, noise


def analyze_error_sources(
    mtm_dir: str,
    num_modes: int,
    report_dir: str,
    reference: np.ndarray | None = None,
    reference_path: str | None = None,
    reference_npz_key: str | None = None,
) -> None:
    """
    扫描目录中所有 *_mtm.npy 文件，从文件名解析条件，
    并计算与参考MTM的MSE/RE统计。

    参数：
        mtm_dir: 包含MTM结果文件的目录
        num_modes: 模式数量（用于截断为方阵）
        report_dir: 报告输出目录
        reference: 参考MTM；默认为单位矩阵（理想直纤）
        reference_path: 若提供，从该 .npy / .npz 加载参考 MTM（覆盖 reference）
        reference_npz_key: .npz 内数组名；省略时自动挑选最像 TM 的方阵
    """
    os.makedirs(report_dir, exist_ok=True)

    reference_meta = ""
    if reference_path:
        reference, reference_meta = load_mtm_reference_from_file(
            reference_path, npz_key=reference_npz_key
        )
    if reference is None:
        reference = np.eye(num_modes, dtype=np.complex128)

    rows: List[Dict] = []

    for fname in os.listdir(mtm_dir):
        if not fname.endswith("_mtm.npy"):
            continue
        path = os.path.join(mtm_dir, fname)
        base = os.path.splitext(fname)[0]

        aberr, noise = parse_condition_from_name(base)
        try:
            T = np.load(path)
            T = np.asarray(T)
            if T.shape[0] != T.shape[1]:
                min_dim = min(T.shape[0], T.shape[1], num_modes)
                T_use = T[:min_dim, :min_dim]
                ref_use = reference[:min_dim, :min_dim]
            else:
                min_dim = min(T.shape[0], num_modes)
                T_use = T[:min_dim, :min_dim]
                ref_use = reference[:min_dim, :min_dim]

            mse_val = mse(T_use, ref_use)
            re_val = relative_error(T_use, ref_use)
            offdiag_ratio = offdiag_energy_ratio(T_use)
            energy_total = float(np.linalg.norm(T_use))
            energy_ref = float(np.linalg.norm(ref_use))
            energy_ratio = float(energy_total / (energy_ref + 1e-12))

            rows.append(
                {
                    "file": base,
                    "aberration_level": aberr,
                    "noise_level": noise,
                    "num_modes_used": min_dim,
                    "mse": mse_val,
                    "relative_error": re_val,
                    "offdiag_energy_ratio": offdiag_ratio,
                    "energy_ratio_vs_ref": energy_ratio,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file": base,
                    "aberration_level": aberr,
                    "noise_level": noise,
                    "num_modes_used": 0,
                    "mse": np.nan,
                    "relative_error": np.nan,
                    "error": str(exc),
                }
            )

    txt_path = os.path.join(report_dir, "error_sources_analysis_report.txt")
    if not rows:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("误差源分析报告（任务3）\n")
            f.write("=========================================\n\n")
            f.write(f"分析目录：{mtm_dir}\n")
            f.write("未找到任何 *_mtm.npy 文件，无法计算误差。\n")
            f.write("请确认：1) 已运行 main.py 或 run_article_mtm.py 生成 MTM；\n")
            f.write("      2) --mtm-dir 指向包含 *_mtm.npy 的 run_* 目录。\n")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(report_dir, "error_sources_analysis.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    
    valid = df.dropna(subset=["mse", "relative_error"])
    summary_csv = os.path.join(report_dir, "error_sources_summary.csv")
    empty_summary_cols = [
        "aberration_level",
        "noise_level",
        "count",
        "mse_mean",
        "re_mean",
        "offdiag_energy_ratio_mean",
        "energy_ratio_vs_ref_mean",
    ]
    if valid.empty:
        summary = pd.DataFrame(columns=empty_summary_cols)
    else:
        summary = (
            valid.groupby(["aberration_level", "noise_level"])
            .agg(
                count=("file", "count"),
                mse_mean=("mse", "mean"),
                re_mean=("relative_error", "mean"),
                offdiag_energy_ratio_mean=("offdiag_energy_ratio", "mean"),
                energy_ratio_vs_ref_mean=("energy_ratio_vs_ref", "mean"),
            )
            .reset_index()
        )
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    ref_note = (
        f"外部文件: {reference_path}  ({reference_meta})"
        if reference_path
        else "默认: 单位矩阵 I（近似理想直纤、模式一一对应且无串扰；论文实测 MTM 通常不接近 I，此时应使用 --reference-mtm 提供仿真/真值）"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("误差源分析报告（任务3）\n")
        f.write("=========================================\n\n")
        f.write(f"分析目录：{mtm_dir}\n")
        f.write(f"参考 MTM：{ref_note}\n")
        f.write(f"详细CSV：{os.path.basename(csv_path)}\n")
        f.write(f"汇总CSV：{os.path.basename(summary_csv)}\n\n")
        f.write("指标说明（通俗）：\n")
        f.write("  MSE：重建矩阵与参考矩阵对应元素差的平方平均，越小越接近参考。\n")
        f.write("  RE：整体 Frobenius 意义下的相对误差 ||T-Tref||/||Tref||。\n")
        f.write("  非对角能量占比：能量有多少落在非对角元上；理想无串扰时对角占优则该值小。\n")
        f.write("  相对参考能量比：||T||/||Tref||，可看出整体增益是否与参考一致。\n\n")
        if summary.empty:
            f.write("无有效数值行（可能全部加载失败），请检查 CSV 中的 error 列。\n")
        else:
            f.write("按条件统计的误差（均值）：\n")
            for _, row in summary.iterrows():
                f.write(
                    f"  像差={row['aberration_level']}，噪声={row['noise_level']}，"
                    f"数量={int(row['count'])}，"
                    f"MSE={row['mse_mean']:.3e}，RE={row['re_mean']:.3e}\n"
                    f"    非对角能量占比={row['offdiag_energy_ratio_mean']:.3e}，"
                    f"相对参考能量比={row['energy_ratio_vs_ref_mean']:.3e}\n"
                )
            f.write("\n---- 可写入论文的简要结论（需结合物理含义人工润色）----\n")
            worst = summary.sort_values("mse_mean", ascending=False).iloc[0]
            best = summary.sort_values("mse_mean", ascending=True).iloc[0]
            f.write(
                f"1) 在所扫描条件下，MSE 最高的是：像差={worst['aberration_level']}, "
                f"噪声={worst['noise_level']}（MSE 均值约 {worst['mse_mean']:.3e}）。\n"
            )
            f.write(
                f"2) MSE 最低的是：像差={best['aberration_level']}, "
                f"噪声={best['noise_level']}（MSE 均值约 {best['mse_mean']:.3e}）。\n"
            )
            f.write(
                "3) 若参考为单位阵：该对比更适合 dummy/理想化场景；真实光纤 MTM 应换用 "
                "仿真或文献真值作为 reference。\n"
            )
            f.write(
                "4) 非对角能量占比升高通常表示模式耦合/基失配/噪声增强，可与像差、噪声标签对照解释。\n"
            )
