"""
误差减少与验证模块（任务4）

目标：
- 对直线多模光纤（MMF）场景中的多模传输矩阵（MTM）计算误差应用数据驱动的校正
- 使用深度学习模型（TensorFlow/Keras）学习从“扰动的MTM”到“校正后的MTM”的映射
- 比较优化前后的均方误差（MSE）/相对误差（RE），目标是误差减少 >= 30%

注意：
- 本模块提供模型架构及训练/验证框架。
- 实际训练需要足够的扰动传输矩阵（PTM）/多模传输矩阵（MTM）数据收集。
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

from .article_reference_mtm import load_mtm_reference_from_file
from .error_metrics import mse, relative_error


def _check_tf() -> None:
    if tf is None or keras is None:
        raise ImportError(
            "未安装TensorFlow。请安装tensorflow以使用误差减少模块。"
        )


def _encode_complex_batch(mtms: np.ndarray) -> np.ndarray:
    """
    将复数MTM批量编码为实向量：[real_flat, imag_flat]。
    输入: (N, M, M), complex
    输出: (N, 2*M*M), float32
    """
    c = np.asarray(mtms, dtype=np.complex128)
    n = c.shape[0]
    real_flat = c.real.reshape(n, -1)
    imag_flat = c.imag.reshape(n, -1)
    return np.concatenate([real_flat, imag_flat], axis=1).astype("float32")


def _decode_complex_batch(vecs: np.ndarray, matrix_shape: tuple[int, int]) -> np.ndarray:
    """
    将 [real_flat, imag_flat] 还原为复数MTM批量。
    输入: (N, 2*M*M)
    输出: (N, M, M), complex128
    """
    v = np.asarray(vecs, dtype=np.float64)
    n = v.shape[0]
    m2 = matrix_shape[0] * matrix_shape[1]
    real_flat = v[:, :m2]
    imag_flat = v[:, m2:]
    return (real_flat + 1j * imag_flat).reshape(n, matrix_shape[0], matrix_shape[1]).astype(np.complex128)


def _fit_linear_fallback(x: np.ndarray, y: np.ndarray, reg: float = 1e-3) -> Dict[str, np.ndarray]:
    """
    当TensorFlow不可用时的线性回归后备模型。
    """
    xtx = x.T @ x
    d = xtx.shape[0]
    w = np.linalg.solve(xtx + reg * np.eye(d), x.T @ y)
    return {"W": w.astype(np.float64)}


def _predict_linear_fallback(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return x @ model["W"]


def build_mtm_correction_model(input_dim: int) -> "keras.Model":
    """
    构建一个用于MTM误差校正的全连接网络。
    输入/输出：向量化的多模传输矩阵（实部和虚部拼接）。
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
    将一批MTM矩阵（N, M, M）展平为（N, M*M）以用于训练。
    """
    noisy_flat = _encode_complex_batch(noisy_mtms)
    ref_flat = _encode_complex_batch(ref_mtms)
    return noisy_flat, ref_flat


def train_error_reduction_model(
    noisy_mtms: np.ndarray,
    ref_mtms: np.ndarray,
    model_save_path: str,
    epochs: int = 50,
    batch_size: int = 32,
) -> Any:
    """
    训练误差减少模型并将权重保存到磁盘。
    """
    x_train, y_train = prepare_training_data(noisy_mtms, ref_mtms)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    if tf is not None and keras is not None:
        model = build_mtm_correction_model(input_dim=x_train.shape[1])
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        model.save(model_save_path)
        return model

    
    linear_model = _fit_linear_fallback(x_train.astype(np.float64), y_train.astype(np.float64))
    np.savez(model_save_path, W=linear_model["W"])
    return linear_model


def apply_error_reduction(
    model: Any,
    noisy_mtm: np.ndarray,
) -> np.ndarray:
    """应用训练好的模型校正单个MTM。"""
    in_vec = _encode_complex_batch(np.asarray(noisy_mtm, dtype=np.complex128)[None, ...])
    if hasattr(model, "predict"):
        corrected_vec = model.predict(in_vec, verbose=0)
    else:
        corrected_vec = _predict_linear_fallback(model, in_vec.astype(np.float64))
    corrected = _decode_complex_batch(corrected_vec, noisy_mtm.shape)[0]
    return corrected


def run_error_reduction_experiment(
    mtm_dir: str,
    num_modes: int,
    report_dir: str,
    model_path: str,
    reference_path: str | None = None,
    reference_npz_key: str | None = None,
) -> Dict[str, Any]:
    """
    误差减少验证工作流程：
      1. 从目录加载MTM结果和理想参考
      2. 加载或训练误差减少模型
      3. 计算校正前后的均方误差（MSE）和相对误差（RE）
      4. 输出CSV和文本摘要，评估是否达到 >= 30% 的误差减少目标
    """
    os.makedirs(report_dir, exist_ok=True)

    
    noisy_list = []
    ref_list = []
    files = []

    ideal = np.eye(num_modes, dtype=np.complex128)
    ref_from_file: np.ndarray | None = None
    if reference_path:
        ref_from_file, _ref_meta = load_mtm_reference_from_file(
            reference_path, npz_key=reference_npz_key
        )
        ref_from_file = np.asarray(ref_from_file, dtype=np.complex128)

    for fname in os.listdir(mtm_dir):
        if not fname.endswith("_mtm.npy"):
            continue
        path = os.path.join(mtm_dir, fname)
        base = os.path.splitext(fname)[0]
        T = np.load(path).astype(np.complex128)
        if T.shape[0] != T.shape[1]:
            m = min(T.shape[0], T.shape[1], num_modes)
            T_use = T[:m, :m]
        else:
            m = min(T.shape[0], num_modes)
            T_use = T[:m, :m]

        if ref_from_file is not None:
            R = ref_from_file
            if R.shape[0] >= m and R.shape[1] >= m:
                ideal_use = R[:m, :m].astype(np.complex128)
            else:
                ideal_use = ideal[:m, :m]
        else:
            ideal_use = ideal[:m, :m]

        noisy_list.append(T_use)
        ref_list.append(ideal_use)
        files.append(base)

    if not noisy_list:
        raise RuntimeError(
            "指定目录中未找到 *_mtm.npy 文件。无法运行误差减少实验。"
        )

    noisy_mtms = np.stack(noisy_list, axis=0)
    ref_mtms = np.stack(ref_list, axis=0)

    
    n_samples = noisy_mtms.shape[0]
    idx_all = np.arange(n_samples)
    n_test = max(1, int(round(0.2 * n_samples)))
    if n_samples == 1:
        idx_train = idx_all
        idx_test = idx_all
    else:
        idx_test = idx_all[-n_test:]
        idx_train = idx_all[:-n_test]
        if idx_train.size == 0:
            idx_train = idx_all[:1]
    noisy_train = noisy_mtms[idx_train]
    ref_train = ref_mtms[idx_train]

    
    if os.path.exists(model_path):
        if tf is not None and keras is not None and model_path.endswith((".keras", ".h5")):
            model = keras.models.load_model(model_path)
        else:
            loaded = np.load(model_path)
            model = {"W": loaded["W"]}
    else:
        model = train_error_reduction_model(
            noisy_train, ref_train, model_save_path=model_path
        )

    
    rows = []
    for idx, base in enumerate(files):
        T_noisy = noisy_mtms[idx]
        T_ref = ref_mtms[idx]
        T_corr = apply_error_reduction(model, T_noisy)
        split = "test" if idx in set(idx_test.tolist()) else "train"

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
                "split": split,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(report_dir, "error_reduction_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    
    df_eval = df[df["split"] == "test"].copy()
    if df_eval.empty:
        df_eval = df

    mse_reduction = (
        1.0
        - (df_eval["mse_after"].mean() + 1e-12) / (df_eval["mse_before"].mean() + 1e-12)
    )
    re_reduction = (
        1.0
        - (df_eval["re_after"].mean() + 1e-12) / (df_eval["re_before"].mean() + 1e-12)
    )

    target_reduction = 0.30
    passed = (mse_reduction >= target_reduction) and (
        re_reduction >= target_reduction
    )

    txt_path = os.path.join(report_dir, "error_reduction_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("误差减少验证报告（任务4）\n")
        f.write("============================================\n\n")
        f.write(f"样本数量: {len(df)}\n")
        if n_samples == 1:
            f.write(
                "注意：仅 1 个 MTM 样本时，训练/测试划分无法独立验证泛化；"
                "报告中的“误差减少”可能主要来自对单条的过拟合，论文中应说明局限。\n\n"
            )
        model_kind = "TensorFlow MLP（Dense 网络）" if hasattr(model, "predict") else "线性回归后备（npz）"
        f.write(f"模型类型: {model_kind}\n")
        f.write(f"训练数据来源: {os.path.basename(mtm_dir)} 目录下所有 *_mtm.npy（重建出的 noisy MTM）\n")
        if reference_path:
            f.write(f"训练标签来源: 外部参考 MTM 文件 {reference_path}（按各样本维度截断对齐）\n")
        else:
            f.write(
                "训练标签来源: 默认单位阵 I = eye(num_modes)（dummy 场景；真实/论文数据请用 "
                "--reference-mtm 提供仿真或真值 MTM）\n"
            )
        f.write("输入/输出形式: 复数 MTM 展平后拼接实部/虚部为实向量，并学习 noisy->ideal 的映射\n")
        f.write(
            f"训练/测试划分: train={len(df[df['split']=='train'])}, "
            f"test={len(df[df['split']=='test'])}\n"
        )
        f.write(
            f"测试集平均均方误差减少: {mse_reduction*100:.2f}% "
            f"(目标 >= {target_reduction*100:.0f}%)\n"
        )
        f.write(
            f"测试集平均相对误差减少: {re_reduction*100:.2f}% "
            f"(目标 >= {target_reduction*100:.0f}%)\n"
        )
        f.write(f"是否达标: {'是' if passed else '否'}\n")
        f.write(f"\n详细结果见: {os.path.basename(csv_path)}\n")

    return {
        "mse_reduction": float(mse_reduction),
        "re_reduction": float(re_reduction),
        "target_met": bool(passed),
        "csv": csv_path,
        "summary_txt": txt_path,
    }
