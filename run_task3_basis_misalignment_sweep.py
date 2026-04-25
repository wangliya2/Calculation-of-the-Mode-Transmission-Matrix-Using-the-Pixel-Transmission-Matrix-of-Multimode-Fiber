from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FiberParams, PreprocessConfig
from src.data_preprocessing import PTMPreprocessor
from src.error_metrics import offdiag_energy_ratio
from src.mtm_reconstruction import MTMReconstructor


@dataclass(frozen=True)
class SweepConfig:
    num_modes: int = 8
    input_basis: str = "hadamard"  # "hadamard" | "identity" | "file"
    m_in_path: str = "data/standard_lp_modes/M_in_hadamard_8.npy"

    shift_values_px: tuple[float, ...] = (-12, -9, -6, -3, 0, 3, 6, 9, 12)
    rotation_values_deg: tuple[float, ...] = (-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10)

    denoise_method: str = PreprocessConfig.denoise_method
    denoise_params: dict[str, float] | None = None


def _infer_grid_size(n_out_pix: int) -> tuple[int, int]:
    """Infer CCD grid size from flattened pixel count."""
    s = int(round(float(n_out_pix) ** 0.5))
    if s * s == n_out_pix:
        return (s, s)
    return (n_out_pix, 1)


def _load_m_in(project_root: str, cfg: SweepConfig, n_input: int, n_modes_out: int) -> np.ndarray:
    if cfg.input_basis == "identity":
        return np.eye(n_input, min(n_input, n_modes_out), dtype=np.complex128)
    if cfg.input_basis == "hadamard":
        p = os.path.join(project_root, cfg.m_in_path)
        m = np.load(p).astype(np.complex128)
        return m
    if cfg.input_basis == "file":
        p = os.path.join(project_root, cfg.m_in_path)
        return np.load(p).astype(np.complex128)
    raise ValueError(f"Unsupported input_basis: {cfg.input_basis}")


def compute_h_pixel(tiff_path: str, cfg: SweepConfig) -> np.ndarray:
    if cfg.denoise_params is None:
        # match main.py defaults
        if cfg.denoise_method == "median":
            denoise_params = {"ksize": PreprocessConfig.median_ksize}
        elif cfg.denoise_method == "bilateral":
            denoise_params = {
                "d": PreprocessConfig.bilateral_d,
                "sigma_color": PreprocessConfig.bilateral_sigma_color,
                "sigma_space": PreprocessConfig.bilateral_sigma_space,
            }
        else:
            denoise_params = {
                "ksize": PreprocessConfig.gaussian_ksize,
                "sigma": PreprocessConfig.gaussian_sigma,
            }
    else:
        denoise_params = dict(cfg.denoise_params)

    pre = PTMPreprocessor(denoise_method=cfg.denoise_method, denoise_params=denoise_params)
    data = pre.load_tiff(tiff_path)  # (N_in, H, W, 2)
    complex_field = data[..., 0] + 1j * data[..., 1]
    z, _meta = pre.reconstruct_denoise_unwrap(complex_field)  # (N_in, H, W), complex
    # flatten to (N_out_pix, N_in)
    n_in, h, w = z.shape
    return z.reshape(n_in, h * w).T.astype(np.complex128)


def sweep_one_axis(
    *,
    recon: MTMReconstructor,
    H_pixel: np.ndarray,
    M_in: np.ndarray,
    grid_size: tuple[int, int],
    fiber_params: dict[str, float],
    axis: str,
    values: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for v in values:
        corr = {"scale": 1.0, "shift_x_px": 0.0, "shift_y_px": 0.0, "rotation_deg": 0.0}
        if axis == "shift_x_px":
            corr["shift_x_px"] = float(v)
        elif axis == "rotation_deg":
            corr["rotation_deg"] = float(v)
        else:
            raise ValueError(f"Unsupported sweep axis: {axis}")

        M_out, orth_err = recon.build_output_mode_matrix(
            grid_size=grid_size,
            fiber_params=fiber_params,
            correction_params=corr,
        )

        k = min(M_in.shape[1], M_out.shape[1])
        T = recon.compute_mtm(H_pixel=H_pixel, M_in=M_in[:, :k], M_out=M_out[:, :k])
        stats = recon.evaluate_mtm_gain_normalized(T)
        rows.append(
            {
                "value": float(v),
                "diag_mean_norm": float(stats["diag_mean"]),
                "offdiag_max_norm": float(stats["offdiag_max"]),
                "offdiag_energy_ratio": float(offdiag_energy_ratio(T)),
                "orth_err": float(orth_err),
            }
        )

    df = pd.DataFrame(rows).sort_values("value").reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Task3 supplement: sweep shift/rotation misalignment and quantify MTM impact.")
    p.add_argument(
        "--tiff",
        type=str,
        default="data/input_tiff/dummy_straight_fiber.tiff",
        help="PTM TIFF path (default: dummy_straight_fiber.tiff).",
    )
    p.add_argument("--num-modes", type=int, default=8, help="Number of modes used in reconstruction (default: 8).")
    p.add_argument(
        "--report-dir",
        type=str,
        default="report/files",
        help="Output directory for CSV files.",
    )
    p.add_argument(
        "--fig-dir",
        type=str,
        default="report/report_figures",
        help="Output directory for PNG figures.",
    )
    args = p.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    tiff_path = args.tiff if os.path.isabs(args.tiff) else os.path.join(project_root, args.tiff)
    report_dir = args.report_dir if os.path.isabs(args.report_dir) else os.path.join(project_root, args.report_dir)
    fig_dir = args.fig_dir if os.path.isabs(args.fig_dir) else os.path.join(project_root, args.fig_dir)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    cfg = SweepConfig(num_modes=int(args.num_modes))
    H_pixel = compute_h_pixel(tiff_path, cfg)
    grid_size = _infer_grid_size(int(H_pixel.shape[0]))

    fiber_params = {
        "core_radius": FiberParams.core_radius_um,
        "wavelength": FiberParams.wavelength_um,
        "na": FiberParams.na,
        "sim_extent_um": FiberParams.sim_extent_um,
    }
    recon = MTMReconstructor(num_modes=cfg.num_modes)

    # Determine output mode count, then load M_in
    M_out0, _ = recon.build_output_mode_matrix(grid_size=grid_size, fiber_params=fiber_params)
    M_in = _load_m_in(project_root, cfg, n_input=int(H_pixel.shape[1]), n_modes_out=int(M_out0.shape[1]))

    # Align to same K
    k = min(M_in.shape[1], M_out0.shape[1], cfg.num_modes)
    M_in = M_in[:, :k]

    df_shift = sweep_one_axis(
        recon=recon,
        H_pixel=H_pixel,
        M_in=M_in,
        grid_size=grid_size,
        fiber_params=fiber_params,
        axis="shift_x_px",
        values=cfg.shift_values_px,
    )
    df_rot = sweep_one_axis(
        recon=recon,
        H_pixel=H_pixel,
        M_in=M_in,
        grid_size=grid_size,
        fiber_params=fiber_params,
        axis="rotation_deg",
        values=cfg.rotation_values_deg,
    )

    csv_shift = os.path.join(report_dir, "basis_misalignment_sweep_shift.csv")
    csv_rot = os.path.join(report_dir, "basis_misalignment_sweep_rotation.csv")
    df_shift.to_csv(csv_shift, index=False, encoding="utf-8-sig")
    df_rot.to_csv(csv_rot, index=False, encoding="utf-8-sig")

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot_df(df: pd.DataFrame, x_label: str, title: str, out_name: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

        ax = axes[0]
        ax.plot(df["value"], df["offdiag_energy_ratio"], marker="o", lw=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Off-diagonal energy ratio")
        ax.set_title("Off-diagonal energy ratio")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(df["value"], df["offdiag_max_norm"], marker="o", lw=2, color="#d62728")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Normalized off-diagonal max")
        ax.set_title("Gain-normalized off-diagonal max")
        ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        out_path = os.path.join(fig_dir, out_name)
        fig.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    plot_df(
        df_shift,
        x_label="Output-basis shift_x (px)",
        title="MTM sensitivity to output-basis shift (dummy straight fiber)",
        out_name="fig13_basis_misalignment_shift_sweep.png",
    )
    plot_df(
        df_rot,
        x_label="Output-basis rotation (deg)",
        title="MTM sensitivity to output-basis rotation (dummy straight fiber)",
        out_name="fig14_basis_misalignment_rotation_sweep.png",
    )

    print("OK:")
    print("  CSV:", csv_shift)
    print("  CSV:", csv_rot)
    print("  FIG:", os.path.join(fig_dir, "fig13_basis_misalignment_shift_sweep.png"))
    print("  FIG:", os.path.join(fig_dir, "fig14_basis_misalignment_rotation_sweep.png"))


if __name__ == "__main__":
    main()

