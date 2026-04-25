"""
误差评估工具模块
支持 MTM/强度图的 MSE 与相对误差 (RE) 计算，并导出到 CSV 使用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """均方误差 (Mean Squared Error)"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    return float(np.mean(diff ** 2))


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    相对误差 (RE)，定义为 ||a-b||_F / (||b||_F + eps)
    适用于 MTM 或强度分布。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm(a - b)
    denom = np.linalg.norm(b) + 1e-12
    return float(num / denom)


def save_error_row_csv(
    csv_path: str,
    row: Dict[str, Any],
) -> None:
    """
    将一行误差结果追加/写入到 CSV（若不存在则创建）。
    """
    df_row = pd.DataFrame([row])
    try:
        
        existing = pd.read_csv(csv_path)
        df_all = pd.concat([existing, df_row], ignore_index=True)
    except FileNotFoundError:
        df_all = df_row
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

