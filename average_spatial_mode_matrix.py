"""
将多份空间模式矩阵 (H*W, N_modes) 做逐元素平均，可选再做 Gram-Schmidt 正交归一化。

用法示例：
  python average_spatial_mode_matrix.py --inputs a.npy b.npy c.npy --output mean_M.npy --orthonormalize
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _gram_schmidt(vecs: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    n_pix, n_modes = vecs.shape
    q = np.zeros((n_pix, n_modes), dtype=np.complex128)
    col = 0
    for j in range(n_modes):
        v = vecs[:, j].astype(np.complex128)
        for i in range(col):
            proj = np.vdot(q[:, i], v) * q[:, i]
            v = v - proj
        norm = np.linalg.norm(v)
        if norm > tol:
            q[:, col] = v / norm
            col += 1
    return q[:, :col]


def main() -> None:
    p = argparse.ArgumentParser(description="平均空间模式矩阵 (H*W,N_modes)")
    p.add_argument("--inputs", nargs="+", required=True, help="输入 .npy 路径列表")
    p.add_argument("--output", required=True, help="输出 .npy 路径")
    p.add_argument(
        "--orthonormalize",
        action="store_true",
        help="平均后做 Gram-Schmidt 正交归一化",
    )
    args = p.parse_args()

    mats: list[np.ndarray] = []
    for path in args.inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        m = np.load(path).astype(np.complex128)
        mats.append(m)

    shape0 = mats[0].shape
    for m in mats[1:]:
        if m.shape != shape0:
            raise ValueError(f"形状不一致: {shape0} vs {m.shape}")

    mean_m = np.mean(np.stack(mats, axis=0), axis=0).astype(np.complex128)
    if args.orthonormalize:
        mean_m = _gram_schmidt(mean_m)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, mean_m)
    print(f"OK: saved {args.output}, shape={mean_m.shape}")


if __name__ == "__main__":
    main()
