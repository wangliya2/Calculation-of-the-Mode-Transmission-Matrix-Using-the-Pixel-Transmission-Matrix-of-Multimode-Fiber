"""
从 wavefrontshaping/article_MMF_disorder 的 Data/ 中加载「模式域」参考 MTM。

该仓库 README 说明：`TM_modes_*.npz` 为 deformation 校正后的 mode-basis 传输矩阵，
可作为你自算 MTM（M_out† H_pixel M_in）的对照参考（论文侧处理链路与本仓库可能略有差异，
答辩中应说明参考来源与可比性）。
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def _as_complex_square(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 2:
        a = a[..., 0] + 1j * a[..., 1]
    elif not np.iscomplexobj(a):
        a = a.astype(np.complex128)
    else:
        a = a.astype(np.complex128)
    if a.ndim != 2:
        raise ValueError(f"期望 2D 或 (…,2) 实虚拆分，得到 shape={a.shape}")
    m, n = a.shape
    s = min(m, n)
    if s < 1:
        raise ValueError("矩阵为空")
    if m != n:
        a = a[:s, :s]
    return a


def _pick_array_from_npz(z: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, str]:
    """在 npz 中自动挑选最像「模式域 TM」的方阵。"""
    best: tuple[float, str, np.ndarray] | None = None  # (-score, key, arr)
    for key in z.files:
        try:
            raw = z[key]
            if not isinstance(raw, np.ndarray):
                continue
            a = _as_complex_square(np.asarray(raw))
            if a.shape[0] < 2:
                continue
        except Exception:
            continue
        score = float(a.shape[0] * a.shape[1])
        lk = key.lower()
        if "tm" in lk or "trans" in lk or "h_" in lk or lk == "h":
            score += 1e6
        cand = (-score, key, a)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        raise ValueError("npz 中未找到可用的 2D 方阵，请用 --reference-npz-key 指定数组名")
    return best[2], best[1]


def load_mtm_reference_from_file(path: str, npz_key: str | None = None) -> Tuple[np.ndarray, str]:
    """
    从 .npy（单个复数方阵）或 .npz（多数组，可指定键名）加载参考 MTM。

    返回:
        (matrix, note)  note 为数据来源说明字符串
    """
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        m = np.asarray(np.load(path), dtype=np.complex128)
        m = _as_complex_square(m)
        return m, f"npy:{os.path.basename(path)}"

    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        try:
            if npz_key:
                if npz_key not in z.files:
                    raise KeyError(f"npz 中无键 {npz_key!r}，可用键: {z.files}")
                m = _as_complex_square(np.asarray(z[npz_key]))
                picked = npz_key
            else:
                m, picked = _pick_array_from_npz(z)
        finally:
            z.close()
        return m, f"npz:{os.path.basename(path)} key={picked}"

    raise ValueError(f"不支持的参考文件类型: {ext}（请使用 .npy 或 .npz）")
