"""
将 article_MMF_disorder 仓库 Data/TM_modes_*.npz 转为单个 reference_mtm.npy，
供 run_task3 / run_task4 的 --reference-mtm 使用。

论文 README：TM_modes_*.npz 为 deformation 校正后的 mode-basis 传输矩阵。

用法:
  python export_article_reference_mtm.py --input path/to/Data/TM_modes_5.0.npz --output data/reference_mtm_article.npy
  python export_article_reference_mtm.py --input path/to/TM_modes_5.0.npz --npz-key TM --output data/ref.npy
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from src.article_reference_mtm import load_mtm_reference_from_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="从 article 仓库 TM_modes_*.npz 导出参考 MTM (.npy)")
    p.add_argument("--input", type=str, required=True, help="TM_modes_*.npz 路径")
    p.add_argument(
        "--output",
        type=str,
        default="data/reference_mtm_article.npy",
        help="输出 .npy 路径",
    )
    p.add_argument("--npz-key", type=str, default="", help="可选：npz 内数组名；省略则自动挑选")
    p.add_argument(
        "--num-modes",
        type=int,
        default=0,
        help="可选：截断到前 k×k；0 表示不截断",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    inp = args.input if os.path.isabs(args.input) else os.path.join(root, args.input)
    out = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    key = str(args.npz_key).strip() or None

    mat, meta = load_mtm_reference_from_file(inp, npz_key=key)
    k = int(args.num_modes)
    if k > 0:
        s = min(k, mat.shape[0], mat.shape[1])
        mat = mat[:s, :s]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(out, mat.astype(np.complex128))
    print(f"OK: {out}  shape={mat.shape}  source={meta}")


if __name__ == "__main__":
    main()
