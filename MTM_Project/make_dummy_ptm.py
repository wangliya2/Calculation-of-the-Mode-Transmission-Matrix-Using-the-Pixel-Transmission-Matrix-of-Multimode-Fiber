from __future__ import annotations
import os
import numpy as np
import tifffile

from generate_standard_data import project_root

def main() -> None:
    root = project_root()
    lp_dir = os.path.join(root, "data", "standard_lp_modes")
    out_dir = os.path.join(root, "data", "input_tiff")
    os.makedirs(out_dir, exist_ok=True)

    
    mode_names = ["LP01", "LP02", "LP03", "LP11", "LP12", "LP21", "LP22", "LP31"]

    fields = []
    for name in mode_names:
        path = os.path.join(lp_dir, f"{name}_intensity.npy")
        if not os.path.exists(path):
            print(f"SKIP: {path} 不存在（先运行 generate_standard_data.py）")
            continue
        inten = np.load(path).astype(np.float64)
        inten /= inten.max() + 1e-12
        amp = np.sqrt(inten)

        
        phase = np.zeros_like(amp, dtype=np.float64)

        u = amp * np.exp(1j * phase)
        fields.append(u)

    if not fields:
        raise RuntimeError("没有任何模式被加载，请先确保 standard_lp_modes 下有 *.npy")

    
    stack = np.stack(fields, axis=0)  # (N_input, H, W)
    tiff_data = np.zeros(stack.shape + (2,), dtype=np.float32)
    tiff_data[..., 0] = stack.real.astype(np.float32)
    tiff_data[..., 1] = stack.imag.astype(np.float32)

    out_path = os.path.join(out_dir, "dummy_ptm_lp_modes.tiff")
    tifffile.imwrite(out_path, tiff_data)
    print(f"OK: 写入合成 PTM 到 {out_path}, shape={tiff_data.shape}")

if __name__ == "__main__":
    main()