#!/usr/bin/env python3
"""
在同一 MTM 目录上分别运行任务4：Ridge 线性回归 vs TensorFlow MLP，
并打印测试集 MSE/RE 降幅对比。

用法（在项目根目录）:
  python compare_task4_backends.py
  python compare_task4_backends.py --mtm-dir data/output_mtm/run_20260413_065752
  python compare_task4_backends.py --epochs 80

说明：样本很少时两者都可能极高降幅，差异主要在泛化能力；MLP 需已安装 tensorflow。
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import MTMConfig


def _resolve_latest_mtm_dir(project_root: str) -> str:
    base = os.path.join(project_root, "data", "output_mtm")
    runs = sorted(
        [
            os.path.join(base, d)
            for d in os.listdir(base)
            if d.startswith("run_") and os.path.isdir(os.path.join(base, d))
        ]
    )
    if not runs:
        raise FileNotFoundError(f"未找到 run_* 目录: {base}")
    return runs[-1]


def main() -> None:
    project_root = _PROJECT_ROOT
    p = argparse.ArgumentParser(description="对比任务4：线性 vs MLP 误差减少")
    p.add_argument(
        "--mtm-dir",
        type=str,
        default="",
        help="含 *_mtm.npy 的目录；省略则使用 data/output_mtm 下最新 run_*",
    )
    p.add_argument("--num-modes", type=int, default=MTMConfig.num_modes)
    p.add_argument(
        "--report-dir",
        type=str,
        default="report/files",
        help="报告输出目录",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    mtm_arg = str(args.mtm_dir).strip()
    if not mtm_arg or mtm_arg.endswith("latest"):
        mtm_dir = _resolve_latest_mtm_dir(project_root)
        print(f"使用 MTM 目录: {mtm_dir}")
    else:
        mtm_dir = mtm_arg if os.path.isabs(mtm_arg) else os.path.join(project_root, mtm_arg)

    report_dir = (
        args.report_dir
        if os.path.isabs(args.report_dir)
        else os.path.join(project_root, args.report_dir)
    )
    os.makedirs(report_dir, exist_ok=True)

    
    linear_model = os.path.join(report_dir, "compare_backend_linear.npz")
    mlp_model = os.path.join(report_dir, "compare_backend_mlp.keras")

    from src.error_reduction import run_error_reduction_experiment

    rows: list[dict[str, object]] = []

    print("\n=== Ridge 线性回归 ===")
    out_l = run_error_reduction_experiment(
        mtm_dir=mtm_dir,
        num_modes=int(args.num_modes),
        report_dir=report_dir,
        model_path=linear_model,
        reference_path=None,
        reference_npz_key=None,
        backend="linear",
        results_suffix="_linear",
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        force_retrain=True,
        fit_verbose=0,
    )
    rows.append(
        {
            "backend": "linear",
            "mse_reduction_pct": out_l["mse_reduction"] * 100.0,
            "re_reduction_pct": out_l["re_reduction"] * 100.0,
            "target_met": out_l["target_met"],
            "csv": out_l["csv"],
        }
    )
    print(f"MSE 降幅: {out_l['mse_reduction']*100:.2f}%  RE 降幅: {out_l['re_reduction']*100:.2f}%  达标: {out_l['target_met']}")

    print("\n=== TensorFlow MLP ===")
    try:
        out_m = run_error_reduction_experiment(
            mtm_dir=mtm_dir,
            num_modes=int(args.num_modes),
            report_dir=report_dir,
            model_path=mlp_model,
            reference_path=None,
            reference_npz_key=None,
            backend="mlp",
            results_suffix="_mlp",
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            force_retrain=True,
            fit_verbose=0,
        )
        rows.append(
            {
                "backend": "mlp",
                "mse_reduction_pct": out_m["mse_reduction"] * 100.0,
                "re_reduction_pct": out_m["re_reduction"] * 100.0,
                "target_met": out_m["target_met"],
                "csv": out_m["csv"],
            }
        )
        print(
            f"MSE 降幅: {out_m['mse_reduction']*100:.2f}%  RE 降幅: {out_m['re_reduction']*100:.2f}%  达标: {out_m['target_met']}"
        )
    except ImportError as e:
        print(f"跳过 MLP：{e}")
        print("安装: pip install tensorflow")

    print("\n=== 对比汇总（测试集平均降幅）===")
    print(f"{'后端':<10} {'MSE降幅%':>12} {'RE降幅%':>12} {'>=30%目标':>12}")
    print("-" * 50)
    for r in rows:
        ok = "是" if r["target_met"] else "否"
        print(
            f"{str(r['backend']):<10} {float(r['mse_reduction_pct']):12.2f} {float(r['re_reduction_pct']):12.2f} {ok:>12}"
        )
    print(f"\n明细 CSV: {report_dir}/error_reduction_results_linear.csv , ..._mlp.csv（若已跑 MLP）")


if __name__ == "__main__":
    main()
