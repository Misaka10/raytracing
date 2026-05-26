"""Compare noisy vs denoised renders quantitatively.

Usage:
    python compare_denoise.py noisy.png denoised.png [--diffmap diff.png]

Outputs:
    - Per-channel MAE, RMSE, PSNR
    - Noise stddev in smooth regions (quantifies denoising strength)
    - Diagnosis: whether denoising had a meaningful effect
    - Optional: difference heatmap saved to disk
"""

import sys
import math
import argparse
from PIL import Image
import numpy as np


def linear_from_png(path: str) -> np.ndarray:
    """Load PNG, invert gamma (assumes sRGB-like 2.2 encoding), return [0,inf) float array."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Invert gamma: our PNG uses gamma ~2.2 (sqrt actually, but 2.2 is close)
    return np.power(arr, 2.2)


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Return edge mask (True = edge pixel) using simple gradient (no scipy needed)."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[1:-1, 1:-1] = (gray[2:, 1:-1] - gray[:-2, 1:-1]) / 2.0
    gy[1:-1, 1:-1] = (gray[1:-1, 2:] - gray[1:-1, :-2]) / 2.0
    mag = np.sqrt(gx**2 + gy**2)
    return mag > np.percentile(mag, 90)


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """BT.709 luminance."""
    return 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]


def analyze(noisy: np.ndarray, denoised: np.ndarray, edge_mask: np.ndarray) -> dict:
    """Compute per-pixel metrics, separated by smooth vs edge regions."""
    results = {}

    diff = noisy - denoised
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    total_pixels = noisy.shape[0] * noisy.shape[1]
    smooth_mask = ~edge_mask

    for label, mask in [("全图", None), ("平滑区", smooth_mask), ("边缘区", edge_mask)]:
        if mask is None:
            d = diff.reshape(-1, 3)
            ad = abs_diff.reshape(-1, 3)
            sd = sq_diff.reshape(-1, 3)
            n = noisy.reshape(-1, 3)
            dn = denoised.reshape(-1, 3)
        else:
            m3 = np.stack([mask, mask, mask], axis=2)
            d = diff[m3].reshape(-1, 3)
            ad = abs_diff[m3].reshape(-1, 3)
            sd = sq_diff[m3].reshape(-1, 3)
            n = noisy[m3].reshape(-1, 3)
            dn = denoised[m3].reshape(-1, 3)

        n_pix = d.shape[0]

        mae = float(np.mean(ad))
        mse = float(np.mean(sd))
        rmse = math.sqrt(mse) if mse > 0 else 0.0
        psnr = float(20 * math.log10(1.0 / rmse)) if rmse > 0 else float('inf')

        # Per-channel RMSE
        rmse_r = math.sqrt(float(np.mean(sd[:, 0]))) if sd.shape[0] > 0 else 0
        rmse_g = math.sqrt(float(np.mean(sd[:, 1]))) if sd.shape[0] > 0 else 0
        rmse_b = math.sqrt(float(np.mean(sd[:, 2]))) if sd.shape[0] > 0 else 0

        # Smooth region variance — compute on the match for label "平滑区" only
        if label == "平滑区" and n.shape[0] > 0:
            smooth_noisy_var = float(np.var(n))
            smooth_denoised_var = float(np.var(dn))
        else:
            smooth_noisy_var = 0.0
            smooth_denoised_var = 0.0

        results[label] = {
            "mae": mae,
            "rmse": rmse,
            "psnr": psnr,
            "rmse_r": rmse_r,
            "rmse_g": rmse_g,
            "rmse_b": rmse_b,
            "smooth_noisy_var": smooth_noisy_var,
            "smooth_denoised_var": smooth_denoised_var,
        }

    return results


def generate_diffmap(noisy: np.ndarray, denoised: np.ndarray, out_path: str):
    """Save a difference heatmap (amplified 10x for visibility)."""
    diff = np.abs(noisy - denoised)
    # Clamp and magnify for visibility
    max_diff = np.percentile(diff, 99)
    if max_diff > 0:
        diff_vis = np.clip(diff / max(0.001, max_diff * 0.5), 0, 1)
    else:
        diff_vis = np.zeros_like(diff)
    # Apply RGB heatmap
    h, w = diff_vis.shape[:2]
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:,:,0] = diff_vis[:,:,0]  # R diff → red
    img[:,:,1] = diff_vis[:,:,1]  # G diff → green
    img[:,:,2] = diff_vis[:,:,2]  # B diff → blue
    img_uint8 = (img * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(out_path)
    print(f"\n差异热力图已保存: {out_path}")


def diagnose(results: dict) -> list[str]:
    """Generate diagnosis messages based on metrics."""
    msgs = []
    smooth = results["平滑区"]
    full = results["全图"]

    # Criterion 1: Smooth-region variance reduction
    var_ratio = (smooth["smooth_denoised_var"] / smooth["smooth_noisy_var"]
                 if smooth["smooth_noisy_var"] > 0 else 1.0)
    if var_ratio < 0.3:
        msgs.append("[EXCELLENT] Variance reduction in smooth areas: {:.0%}".format(1 - var_ratio))
    elif var_ratio < 0.7:
        msgs.append("[GOOD] Variance reduction in smooth areas: {:.0%}".format(1 - var_ratio))
    elif var_ratio < 0.9:
        msgs.append("[WEAK] Variance reduction in smooth areas: {:.0%}".format(1 - var_ratio))
    else:
        msgs.append("[NONE] No significant denoising detected: {:.0%}".format(1 - var_ratio))

    # Criterion 2: PSNR (high = less difference = over-denoised? or too clean input?)
    if full["psnr"] > 40:
        msgs.append("[LOW-NOISE] Input very clean (PSNR > 40dB), samples may be too high for denoising to matter")
    elif full["psnr"] < 25:
        msgs.append("[ACTIVE] Large difference before/after (PSNR < 25dB), denoiser is actively working")

    # Criterion 3: Edge preservation check
    edge_rmse = results["边缘区"]["rmse"]
    smooth_rmse = results["平滑区"]["rmse"]
    if smooth_rmse > 0:
        ratio = edge_rmse / smooth_rmse
        if ratio < 0.8:
            msgs.append("[EDGE-OK] Edge RMSE lower than smooth RMSE, good edge preservation")
        elif ratio > 1.5:
            msgs.append("[EDGE-WARN] Edge RMSE significantly higher than smooth RMSE, possible edge artifacts")
    # No else (smooth RMSE = 0)

    return msgs


def main():
    parser = argparse.ArgumentParser(description="对比降噪前后渲染图像")
    parser.add_argument("noisy", help="降噪前 PNG")
    parser.add_argument("denoised", help="降噪后 PNG")
    parser.add_argument("--diffmap", default=None, help="差异热力图输出路径")
    args = parser.parse_args()

    print(f"加载 {args.noisy} ...")
    noisy = linear_from_png(args.noisy)
    print(f"加载 {args.denoised} ...")
    denoised = linear_from_png(args.denoised)

    if noisy.shape != denoised.shape:
        print(f"错误: 尺寸不匹配 {noisy.shape} vs {denoised.shape}")
        sys.exit(1)

    h, w = noisy.shape[:2]
    print(f"分辨率: {w}×{h}")

    gray = compute_luminance(noisy)
    edge_mask = sobel_edges(gray)
    edge_pct = np.sum(edge_mask) / edge_mask.size * 100
    print(f"边缘像素: {edge_pct:.1f}%")

    results = analyze(noisy, denoised, edge_mask)

    # Print results
    for label in ["全图", "平滑区", "边缘区"]:
        r = results[label]
        print(f"\n--- {label} ---")
        print(f"  MAE:        {r['mae']:.6f}")
        print(f"  RMSE:       {r['rmse']:.6f}  (R={r['rmse_r']:.6f} G={r['rmse_g']:.6f} B={r['rmse_b']:.6f})")
        print(f"  PSNR:       {r['psnr']:.2f} dB")
        if label == "平滑区" and r["smooth_noisy_var"] > 0:
            var_reduction = 1.0 - r["smooth_denoised_var"] / r["smooth_noisy_var"]
            print(f"  噪声方差:   noisy={r['smooth_noisy_var']:.6f} → denoised={r['smooth_denoised_var']:.6f} ({var_reduction:.1%} 降低)")

    # Diagnosis
    msgs = diagnose(results)
    print("\n=== 诊断 ===")
    for m in msgs:
        print(f"  {m}")

    if args.diffmap:
        generate_diffmap(noisy, denoised, args.diffmap)


if __name__ == "__main__":
    main()
