from __future__ import annotations

"""
预处理模块验收脚本（SNR + Phase MAE）

执行位置：项目根目录（与 data/、src/ 同级）
运行：
  python validate_preprocessing.py

脚本做什么：
1) 从 data/standard_lp_modes/*_intensity.npy 读取标准 LP 模式强度（建议先运行 generate_standard_data.py）
2) 生成“标准复场”：
   - 幅度 A = sqrt(intensity)
   - 理论相位 Φ_theory：
       为了避免 LP 模式强度本身不含相位信息带来的不可验证性，
       这里采用可复现的“LP 模式方位阶数相关相位”：
           Φ_theory = l * atan2(y, x)
       其中 l 来自 LP 模式名 LP{l}{m}。
   - 复场 U = A * exp(j Φ_theory)
3) 在复场实部/虚部上叠加高斯噪声，使输入 SNR=20dB（按功率定义）
4) 调用 src/data_preprocessing.PTMPreprocessor：
   - 提取与解包裹相位（以及振幅去噪）
5) 输出量化指标：
   - SNR_before / SNR_after / improvement（以“振幅误差”定义噪声）
   - phase_MAE（在有效区域内计算，默认 A > 0.05）

输出：
  report/files/preprocess_validation.csv
  report/files/preprocess_validation_report.txt
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_preprocessing import PTMPreprocessor


@dataclass(frozen=True)
class ValidationConfig:
    target_snr_db: float = 20.0
    amplitude_mask_threshold: float = 0.05
    rng_seed: int = 0


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def standard_lp_dir() -> str:
    return os.path.join(project_root(), "data", "standard_lp_modes")


def report_dir() -> str:
    p = os.path.join(project_root(), "report", "files")
    os.makedirs(p, exist_ok=True)
    return p


def _snr_db(signal: np.ndarray, reference: np.ndarray) -> float:
    """
    SNR(dB) = 10 log10( P_signal / P_noise )
    其中 P_noise 基于 (signal - reference) 的均方功率。
    """
    s = np.asarray(signal, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    noise = s - r
    p_signal = float(np.mean(r**2)) + 1e-12
    p_noise = float(np.mean(noise**2)) + 1e-12
    return float(10.0 * np.log10(p_signal / p_noise))


def _parse_lp_l(mode_name: str) -> int:
    """
    从 'LP11' 提取 l=1。
    """
    name = mode_name.strip().upper()
    if not name.startswith("LP") or len(name) < 4:
        return 0
    try:
        return int(name[2])
    except Exception:
        return 0


def _make_coordinate_phi(h: int, w: int) -> np.ndarray:
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    return np.arctan2(yy - cy, xx - cx)


def _add_complex_gaussian_noise_for_target_snr(
    u: np.ndarray, target_snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """
    对复场 u 的实部/虚部叠加零均值高斯噪声，使功率 SNR 达到 target_snr_db。
    """
    u = np.asarray(u, dtype=np.complex128)
    p_signal = float(np.mean(np.abs(u) ** 2)) + 1e-12
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    p_noise = p_signal / snr_linear

    
    sigma = np.sqrt(p_noise / 2.0)
    noise = rng.normal(0.0, sigma, size=u.shape) + 1j * rng.normal(0.0, sigma, size=u.shape)
    return u + noise


def _phase_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """
    相位 MAE：使用 wrap-to-pi 的差值避免 2π 等价问题。
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return float("nan")
    diff = p - t
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(np.abs(diff[m])))


def _extract_phase_from_h_pixel(h_pixel: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    将 H_pixel (N_pix, N_input) 的第一列还原为相位图用于验证。
    """
    first = np.asarray(h_pixel[:, 0], dtype=np.complex128).reshape(h, w)
    return np.angle(first)


def _extract_amplitude_from_h_pixel(h_pixel: np.ndarray, h: int, w: int) -> np.ndarray:
    first = np.asarray(h_pixel[:, 0], dtype=np.complex128).reshape(h, w)
    return np.abs(first)


def validate_one_mode(
    mode_name: str,
    intensity: np.ndarray,
    cfg: ValidationConfig,
) -> List[Dict]:
    """
    对单个模式分别测试 gaussian/median 两种去噪配置。
    """
    inten = np.asarray(intensity, dtype=np.float64)
    inten = inten / (float(inten.max()) + 1e-12)
    h, w = inten.shape

    amp_true = np.sqrt(inten)
    phi = _make_coordinate_phi(h, w)
    l = _parse_lp_l(mode_name)
    phase_true = (l * phi + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi]

    u_true = amp_true * np.exp(1j * phase_true)

    rng = np.random.default_rng(cfg.rng_seed)
    u_noisy = _add_complex_gaussian_noise_for_target_snr(
        u_true, target_snr_db=cfg.target_snr_db, rng=rng
    )

    
    tiff_like = np.zeros((1, h, w, 2), dtype=np.float64)
    tiff_like[0, :, :, 0] = u_noisy.real
    tiff_like[0, :, :, 1] = u_noisy.imag

    
    amp_noisy = np.abs(u_noisy)
    snr_before = _snr_db(signal=amp_noisy, reference=amp_true)

    methods = [
        ("gaussian", {"ksize": 5, "sigma": 1.2}),
        ("median", {"ksize": 3}),
    ]

    rows: List[Dict] = []

    
    
    import tifffile  # local import to keep requirements minimal

    tmp_dir = os.path.join(project_root(), "report", "files", "_tmp_validation")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_tif = os.path.join(tmp_dir, f"{mode_name}_snr{int(cfg.target_snr_db)}.tiff")
    tifffile.imwrite(tmp_tif, tiff_like.astype(np.float32))

    for method, params in methods:
        pre = PTMPreprocessor(denoise_method=method, denoise_params=params)
        h_pixel, stats = pre.preprocess_to_h_pixel(tmp_tif)

        amp_after = _extract_amplitude_from_h_pixel(h_pixel, h=h, w=w)
        phase_after = _extract_phase_from_h_pixel(h_pixel, h=h, w=w)

        snr_after = _snr_db(signal=amp_after, reference=amp_true)
        improvement = snr_after - snr_before

        mask = amp_true > cfg.amplitude_mask_threshold
        mae = _phase_mae(pred=phase_after, truth=phase_true, mask=mask)

        rows.append(
            {
                "mode_name": mode_name,
                "grid_h": h,
                "grid_w": w,
                "target_snr_db": cfg.target_snr_db,
                "denoise_method": method,
                "denoise_params": str(params),
                "snr_before_db": snr_before,
                "snr_after_db": snr_after,
                "snr_improvement_db": improvement,
                "phase_mae_rad": mae,
                "phase_continuity": float(stats.get("phase_continuity", np.nan)),
                "mask_threshold": cfg.amplitude_mask_threshold,
            }
        )

    return rows


def main() -> None:
    cfg = ValidationConfig()
    lp_dir = standard_lp_dir()
    if not os.path.exists(lp_dir):
        raise FileNotFoundError(
            f"未找到 {lp_dir}，请先运行: python generate_standard_data.py"
        )

    mode_names = ["LP01", "LP02", "LP03", "LP11", "LP12", "LP21", "LP22", "LP31"]
    rows_all: List[Dict] = []

    for name in mode_names:
        path = os.path.join(lp_dir, f"{name}_intensity.npy")
        if not os.path.exists(path):
            print(f"SKIP: 缺少参考强度文件 {path}")
            continue
        intensity = np.load(path)
        rows_all.extend(validate_one_mode(name, intensity=intensity, cfg=cfg))

    if not rows_all:
        raise RuntimeError("没有生成任何验证结果（可能缺少标准 LP 强度数据）。")

    df = pd.DataFrame(rows_all)
    csv_path = os.path.join(report_dir(), "preprocess_validation.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    
    summary = (
        df.groupby("denoise_method")
        .agg(
            count=("mode_name", "count"),
            snr_improvement_db_mean=("snr_improvement_db", "mean"),
            snr_improvement_db_min=("snr_improvement_db", "min"),
            phase_mae_rad_mean=("phase_mae_rad", "mean"),
            phase_mae_rad_max=("phase_mae_rad", "max"),
            phase_continuity_mean=("phase_continuity", "mean"),
        )
        .reset_index()
    )

    txt_path = os.path.join(report_dir(), "preprocess_validation_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Preprocess Validation Report (SNR + Phase MAE)\n")
        f.write("================================================\n\n")
        f.write(f"Target SNR (input): {cfg.target_snr_db:.1f} dB\n")
        f.write(f"Amplitude mask threshold: {cfg.amplitude_mask_threshold:.3f}\n")
        f.write(f"RNG seed: {cfg.rng_seed}\n")
        f.write(f"Standard LP dir: {lp_dir}\n\n")

        f.write("Acceptance targets (mid-term):\n")
        f.write("- Denoise SNR improvement >= 10 dB (on SNR=20dB noisy standard data)\n")
        f.write("- Phase MAE <= 0.02 rad (on known theoretical phase distribution)\n\n")

        f.write("Summary by denoise method:\n")
        for _, r in summary.iterrows():
            f.write(
                f"  {r['denoise_method']}: "
                f"count={int(r['count'])}, "
                f"SNR_improve_mean={r['snr_improvement_db_mean']:.3f} dB, "
                f"SNR_improve_min={r['snr_improvement_db_min']:.3f} dB, "
                f"Phase_MAE_mean={r['phase_mae_rad_mean']:.4f} rad, "
                f"Phase_MAE_max={r['phase_mae_rad_max']:.4f} rad, "
                f"Phase_continuity_mean={r['phase_continuity_mean']:.3f}\n"
            )

        f.write("\nDetailed results: preprocess_validation.csv\n")

    print(f"OK: {csv_path}")
    print(f"OK: {txt_path}")


if __name__ == "__main__":
    main()

