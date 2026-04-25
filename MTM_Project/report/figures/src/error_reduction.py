"""
误差抑制与验证模块（Task 4）

目标：
- 在直多模光纤场景下，对 MTM 计算误差进行数据驱动的修正
- 使用深度学习模型（TensorFlow/Keras）学习从“受扰动 MTM”到“校正 MTM”的映射
- 对比优化前后的 MSE/RE，期望达到 ≥30% 误差降低

说明：
- 本模块提供模型结构与训练/验证流程框架，具体训练需在采集到足够 PTM/MTM 数据后进行。
"""

from __future__ import annotations

import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:  # noqa: BLE001
    tf = None
    keras = None
    layers = None

from .error_metrics import mse, relative_error


def _check_tf() -> None:
    if tf is None or keras is None:
        raise ImportError("TensorFlow 未安装，无法运行误差抑制模型，请先安装 tensorflow。")


def build_mtm_correction_model(input_dim: int) -> "keras.Model":
    """
    构建一个简单的全连接网络，用于 MTM 矩阵矢量化后的误差校正。
    输入 / 输出：向量化后的 MTM（实部和虚部可拼接，或仅实部）。
    """
    _check_tf()

    inputs = keras.Input(shape=(input_dim,), name="noisy_mtm_vec")
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear", name="corrected_mtm_vec")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mtm_correction_mlp")
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_training_data(
    noisy_mtms: np.ndarray,
    ref_mtms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将一批 MTM 矩阵 (N, M, M) 展平为 (N, M*M) 作为训练输入/标签。
    """
    noisy_flat = noisy_mtms.reshape(noisy_mtms.shape[0], -1).astype("float32")
    ref_flat = ref_mtms.reshape(ref_mtms.shape[0], -1).astype("float32")
    return noisy_flat, ref_flat


def train_error_reduction_model(
    noisy_mtms: np.ndarray,
    ref_mtms: np.ndarray,
    model_save_path: str,
    epochs: int = 50,
    batch_size: int = 32,
) -> "keras.Model":
    """
    训练误差抑制模型，并将模型权重保存到磁盘。
    """
    _check_tf()

    x_train, y_train = prepare_training_data(noisy_mtms, ref_mtms)
    model = build_mtm_correction_model(input_dim=x_train.shape[1])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    return model


def apply_error_reduction(
    model: "keras.Model",
    noisy_mtm: np.ndarray,
) -> np.ndarray:
    """
    使用训练好的模型对单个 MTM 进行校正。
    """
    flat = noisy_mtm.reshape(1, -1).astype("float32")
    corrected_flat = model.predict(flat)
    corrected = corrected_flat.reshape(noisy_mtm.shape)
    return corrected


def run_error_reduction_experiment(
    mtm_dir: str,
    num_modes: int,
    report_dir: str,
    model_path: str,
) -> Dict[str, Any]:
    """
    误差抑制效果验证流程：
      1. 从目录中加载一批 MTM 结果及对应的理想参考（对角阵）
      2. 载入或训练误差抑制模型
      3. 计算优化前后相对于理想 MTM 的 MSE/RE
      4. 输出 CSV + 文本摘要，评估是否达到 ≥30% 误差降低
    """
    _check_tf()
    os.makedirs(report_dir, exist_ok=True)

    
    noisy_list = []
    ref_list = []
    files = []

    ideal = np.eye(num_modes, dtype=np.float64)

    for fname in os.listdir(mtm_dir):
        if not fname.endswith("_mtm.npy"):
            continue
        path = os.path.join(mtm_dir, fname)
        base = os.path.splitext(fname)[0]
        T = np.load(path).astype(np.float64)
        if T.shape[0] != T.shape[1]:
            m = min(T.shape[0], T.shape[1], num_modes)
            T_use = T[:m, :m]
            ideal_use = ideal[:m, :m]
        else:
            m = min(T.shape[0], num_modes)
            T_use = T[:m, :m]
            ideal_use = ideal[:m, :m]

        noisy_list.append(T_use)
        ref_list.append(ideal_use)
        files.append(base)

    if not noisy_list:
        raise RuntimeError("指定目录下未找到任何 *_mtm.npy 文件，无法进行误差抑制实验。")

    noisy_mtms = np.stack(noisy_list, axis=0)
    ref_mtms = np.stack(ref_list, axis=0)

    
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        model = train_error_reduction_model(
            noisy_mtms, ref_mtms, model_save_path=model_path
        )

    
    rows = []
    for idx, base in enumerate(files):
        T_noisy = noisy_mtms[idx]
        T_ref = ref_mtms[idx]
        T_corr = apply_error_reduction(model, T_noisy)

        mse_before = mse(T_noisy, T_ref)
        re_before = relative_error(T_noisy, T_ref)
        mse_after = mse(T_corr, T_ref)
        re_after = relative_error(T_corr, T_ref)

        rows.append(
            {
                "file": base,
                "mse_before": mse_before,
                "re_before": re_before,
                "mse_after": mse_after,
                "re_after": re_after,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(report_dir, "error_reduction_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    
    mse_reduction = (
        1.0
        - (df["mse_after"].mean() + 1e-12) / (df["mse_before"].mean() + 1e-12)
    )
    re_reduction = (
        1.0
        - (df["re_after"].mean() + 1e-12) / (df["re_before"].mean() + 1e-12)
    )

    target_reduction = 0.30
    passed = (mse_reduction >= target_reduction) and (
        re_reduction >= target_reduction
    )

    txt_path = os.path.join(report_dir, "error_reduction_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Error Reduction Verification Report（Task 4）\n")
        f.write("============================================\n\n")
        f.write(f"样本数量: {len(df)}\n")
        f.write(
            f"平均 MSE 降低比例: {mse_reduction*100:.2f}% "
            f"(目标 ≥ {target_reduction*100:.0f}%)\n"
        )
        f.write(
            f"平均 RE 降低比例: {re_reduction*100:.2f}% "
            f"(目标 ≥ {target_reduction*100:.0f}%)\n"
        )
        f.write(f"是否达到优化目标: {'是' if passed else '否'}\n")
        f.write(f"\n详细结果见: {os.path.basename(csv_path)}\n")

    return {
        "mse_reduction": float(mse_reduction),
        "re_reduction": float(re_reduction),
        "target_met": bool(passed),
        "csv": csv_path,
        "summary_txt": txt_path,
    }

