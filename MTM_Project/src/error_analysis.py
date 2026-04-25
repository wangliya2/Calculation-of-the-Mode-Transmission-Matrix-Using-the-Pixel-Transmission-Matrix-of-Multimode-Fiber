"""
误差来源分析模块（Task 3）

功能：
- 从 MTM 基线误差 CSV 以及多条件 MTM 结果中，计算 MSE/RE 统计量
- 区分不同像差等级、噪声等级的贡献，生成 “Sources of Error Analysis Report”

约定：
- 文件命名规范示例：
    straight_aberrLow_noiseLow_mtm.npy
    straight_aberrHigh_noiseHigh_mtm.npy
  其中 aberr{Low|Med|High}, noise{Low|High} 用于编码条件。
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .error_metrics import mse, relative_error


def parse_condition_from_name(name: str) -> Tuple[str, str]:
    """
    从文件名中解析像差等级和噪声等级，默认返回 ('Unknown', 'Unknown')
    """
    aberr_match = re.search(r"aberr(Low|Med|High)", name, re.IGNORECASE)
    noise_match = re.search(r"noise(Low|High)", name, re.IGNORECASE)
    aberr = aberr_match.group(1).capitalize() if aberr_match else "Unknown"
    noise = noise_match.group(1).capitalize() if noise_match else "Unknown"
    return aberr, noise


def analyze_error_sources(
    mtm_dir: str,
    num_modes: int,
    report_dir: str,
    reference: np.ndarray | None = None,
) -> None:
    """
    遍历指定目录下的 *_mtm.npy 文件，根据命名解析条件，计算 MSE/RE 统计量。

    Args:
        mtm_dir: 存放 MTM 结果的目录（通常为 data/output_mtm）
        num_modes: 模式数，用于截取方阵
        report_dir: 报告输出目录（通常为 report/files）
        reference: 参考 MTM，不提供则默认使用单位对角阵（直光纤理想情况）
    """
    os.makedirs(report_dir, exist_ok=True)

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

            rows.append(
                {
                    "file": base,
                    "aberration_level": aberr,
                    "noise_level": noise,
                    "num_modes_used": min_dim,
                    "mse": mse_val,
                    "relative_error": re_val,
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

    if not rows:
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(report_dir, "error_sources_analysis.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    
    valid = df.dropna(subset=["mse", "relative_error"])
    summary = (
        valid.groupby(["aberration_level", "noise_level"])
        .agg(
            count=("file", "count"),
            mse_mean=("mse", "mean"),
            re_mean=("relative_error", "mean"),
        )
        .reset_index()
    )

    summary_csv = os.path.join(report_dir, "error_sources_summary.csv")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    
    txt_path = os.path.join(report_dir, "error_sources_analysis_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Sources of Error Analysis Report（Task 3）\n")
        f.write("=========================================\n\n")
        f.write(f"分析目录: {mtm_dir}\n")
        f.write(f"结果明细 CSV: {os.path.basename(csv_path)}\n")
        f.write(f"结果汇总 CSV: {os.path.basename(summary_csv)}\n\n")
        f.write("各条件下误差统计（均值）：\n")
        for _, row in summary.iterrows():
            f.write(
                f"  像差={row['aberration_level']}, 噪声={row['noise_level']}, "
                f"样本数={int(row['count'])}, "
                f"MSE≈{row['mse_mean']:.3e}, RE≈{row['re_mean']:.3e}\n"
            )

