"""
误差评估工具模块

支持MTM和强度分布的均方误差（MSE）和相对误差（RE）计算，并具备CSV导出功能。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """两个数组之间的均方误差（MSE）。"""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    diff = a - b
    return float(np.mean(np.abs(diff) ** 2))


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    相对误差（RE），定义为 ||a-b||_F / (||b||_F + eps)。
    适用于MTM矩阵或强度分布。
    """
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    num = np.linalg.norm(a - b)
    denom = np.linalg.norm(b) + 1e-12
    return float(num / denom)


def offdiag_energy_ratio(mat: np.ndarray) -> float:
    """
    非对角能量占比: ||offdiag||_F / (||diag||_F + eps)
    """
    m = np.asarray(mat, dtype=np.complex128)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("offdiag_energy_ratio 仅支持方阵")
    diag_mask = np.eye(m.shape[0], dtype=bool)
    off = m[~diag_mask]
    diag = m[diag_mask]
    return float(np.linalg.norm(off) / (np.linalg.norm(diag) + 1e-12))


def save_error_row_csv(
    csv_path: str,
    row: Dict[str, Any],
) -> None:
    """
    将单条误差结果行追加到CSV文件中。
    如果文件不存在则创建该文件。
    """
    df_row = pd.DataFrame([row])
    try:
        existing = pd.read_csv(csv_path)
        df_all = pd.concat([existing, df_row], ignore_index=True)
    except FileNotFoundError:
        df_all = df_row
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
