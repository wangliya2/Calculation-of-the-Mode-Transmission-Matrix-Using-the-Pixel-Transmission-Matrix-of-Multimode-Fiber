from __future__ import annotations

import argparse
import os

from config import MTMConfig
from src.error_analysis import analyze_error_sources


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="任务3：误差源分析")
    p.add_argument(
        "--mtm-dir",
        type=str,
        default="data/output_mtm/latest",
        help="包含 *_mtm.npy 的目录（默认自动使用最新 run_*）",
    )
    p.add_argument(
        "--report-dir",
        type=str,
        default="report/files",
        help="误差分析报告输出目录（默认 report/files）",
    )
    p.add_argument(
        "--num-modes",
        type=int,
        default=MTMConfig.num_modes,
        help="模式数量（默认读取配置）",
    )
    p.add_argument(
        "--reference-mtm",
        type=str,
        default="",
        help="可选：参考 MTM 的 .npy 或 article 仓库的 TM_modes_*.npz；不提供时默认与单位阵比较",
    )
    p.add_argument(
        "--reference-npz-key",
        type=str,
        default="",
        help="可选：.npz 内数组名；省略则自动挑选最像 TM 的方阵",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    mtm_dir_arg = str(args.mtm_dir).strip()
    if mtm_dir_arg.endswith("run_YYYYmmdd_HHMMSS") or mtm_dir_arg.endswith("latest"):
        base = os.path.join(project_root, "data", "output_mtm")
        runs = sorted(
            [
                os.path.join(base, d)
                for d in os.listdir(base)
                if d.startswith("run_") and os.path.isdir(os.path.join(base, d))
            ]
        )
        if not runs:
            raise FileNotFoundError(f"未找到任何 run_* 目录: {base}")
        mtm_dir = runs[-1]
        print(f"自动选择最新MTM目录: {mtm_dir}")
    else:
        mtm_dir = (
            mtm_dir_arg
            if os.path.isabs(mtm_dir_arg)
            else os.path.join(project_root, mtm_dir_arg)
        )
    report_dir = (
        args.report_dir
        if os.path.isabs(args.report_dir)
        else os.path.join(project_root, args.report_dir)
    )

    ref_arg = str(args.reference_mtm).strip()
    reference_path: str | None = None
    if ref_arg:
        reference_path = ref_arg if os.path.isabs(ref_arg) else os.path.join(project_root, ref_arg)
        if not os.path.isfile(reference_path):
            raise FileNotFoundError(f"参考 MTM 不存在: {reference_path}")

    npz_key = str(args.reference_npz_key).strip() or None

    analyze_error_sources(
        mtm_dir=mtm_dir,
        num_modes=int(args.num_modes),
        report_dir=report_dir,
        reference_path=reference_path,
        reference_npz_key=npz_key,
    )
    print("任务3分析完成:", report_dir)


if __name__ == "__main__":
    main()
