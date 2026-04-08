#!/usr/bin/env python3
import argparse
import csv
import os
import re
import math
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- SURGICAL ALIGNMENT: COPIED FROM V4 FITTING SCRIPT ---

def parse_size_to_float(label):
    if label.lower() == 'baseline': return 0.0
    match = re.match(r"(\d+(?:\.\d+)?)([MBx])", label, re.IGNORECASE)
    if not match: return 0.0
    val, unit = match.groups()
    val = float(val)
    if unit.upper() == 'M': return val * 1e6
    if unit.upper() == 'B': return val * 1e9
    return val # Handles 'x' as raw float

def parse_params_from_filename(filename):
    match = re.search(r"(\d+(?:\.\d+)?)([mb])", filename.lower())
    if not match: return 0.0
    val, unit = match.groups()
    return float(val) * 1e6 if unit == 'm' else float(val) * 1e9

def metric_matches(target_metric, row_metric_key):
    """
    Flexible metric matching across lm-eval variants.

    Examples:
      --metric perplexity                  -> matches perplexity,none and perplexity,remove_whitespace
      --metric perplexity,none            -> matches both none/remove_whitespace variants
      --metric exact_match,remove_whitespace -> matches both exact_match variants
      --metric brier_score,none           -> exact match only (unless none/remove_whitespace pair exists)
    """
    target = target_metric.strip().lower()
    row_key = row_metric_key.strip().lower()

    if row_key == target:
        return True

    target_base = target.split(",")[0]
    row_base = row_key.split(",")[0]
    if target_base != row_base:
        return False

    # If user passes only base metric (e.g., "perplexity"), accept all variants.
    if "," not in target:
        return True

    # Treat none/remove_whitespace as equivalent variants for same base metric.
    variants = {f"{target_base},none", f"{target_base},remove_whitespace"}
    if target in variants and row_key in variants:
        return True

    return False

# --- ANALYTICAL ENGINE ---

def main():
    parser = argparse.ArgumentParser(description="Empirical RAG Saturation Detector")
    parser.add_argument("--dir", required=True, help="Path to ScalingFiles directory")
    parser.add_argument(
        "--metric",
        default="perplexity,none",
        help="Metric key (or base metric like 'perplexity'/'exact_match') in CSV",
    )
    parser.add_argument(
        "--tasks",
        default="arc_easy,arc_challenge,hellaswag,sciq,openbookqa,nq_open,strategyqa,simpleqa,piqa_generative,commonsense_qa,dclm_val_ppl",
        help="Comma-separated allowlist of tasks to include (matches fit_scaling_law.py defaults).",
    )
    parser.add_argument("--threshold", type=float, default=0.90, help="Target fraction of max gain")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for saturation plots (default: <dir>/saturation_plots_<metric>)",
    )
    args = parser.parse_args()
    allowed_tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    metric_slug = args.metric.replace(",", "_").replace(" ", "")
    out_dir = args.out_dir or os.path.join(args.dir, f"saturation_plots_{metric_slug}")
    os.makedirs(out_dir, exist_ok=True)

    # Structure: task -> N -> d_ratio -> list of (R, y)
    obs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # Per-task aggregated critical points in R/N units
    critical_ratios_by_task = defaultdict(list)
    # Per-task detailed critical points for plotting (N-specific curves over D/N)
    critical_points_by_task = defaultdict(list)

    # 1. Harvest Data (Identical path handling to V4)
    if not os.path.exists(args.dir):
        print(f"Error: Directory {args.dir} not found.")
        return

    for fname in os.listdir(args.dir):
        if not fname.endswith('.csv'): continue
        if fname == "result_completeness_audit.csv": continue
        N = parse_params_from_filename(fname)
        with open(os.path.join(args.dir, fname), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not metric_matches(args.metric, row['metric_key']): continue
                task = row['task']
                if allowed_tasks and task not in allowed_tasks:
                    continue
                d_ratio = parse_size_to_float(row['model_scale'])
                r_val = parse_size_to_float(row['index_label'])
                y_val = float(row['metric_value'])
                # No transformations here (like 1-acc) to keep it raw empirical
                obs[task][N][d_ratio].append((r_val, y_val))

    print("\n" + "="*115)
    print("Task allowlist:", ", ".join(sorted(allowed_tasks)) if allowed_tasks else "ALL")
    print(f"EMPIRICAL CRITICAL POINT ANALYSIS (Metric: {args.metric} | Threshold: {args.threshold*100:.0f}%)")
    print(f"{'BENCHMARK':<18} | {'MODEL':<8} | {'D-RATIO':<8} | {'BASE PPL':<9} | {'MAX GAIN':<9} | {'CRITICAL R':<10} | {'R/N RATIO'}")
    print("-" * 115)

    # 2. Identify the Elbow
    for task in sorted(obs.keys()):
        for N in sorted(obs[task].keys()):
            for d_ratio in sorted(obs[task][N].keys()):
                points = obs[task][N][d_ratio]
                points.sort(key=lambda x: x[0]) # Sort by R size
                
                # Establish Baseline (R=0)
                y0_list = [p[1] for p in points if p[0] == 0]
                if not y0_list: continue
                y0 = sum(y0_list) / len(y0_list)
                
                # Calculate Gains
                gains = [(r, y0 - y) for r, y in points if r > 0]
                if not gains: continue
                
                max_observed_gain = max(g for r, g in gains)
                if max_observed_gain <= 0.0001: continue # Skip if no RAG benefit exists

                # Find Critical R: First R that achieves 'threshold' percent of max observed gain
                critical_r = None
                for r, g in gains:
                    if g >= args.threshold * max_observed_gain:
                        critical_r = r
                        break
                
                if critical_r is not None:
                    r_str = f"{critical_r/1e9:>6.1f}B" if critical_r >= 1e9 else f"{critical_r/1e6:>6.0f}M"
                    r_n_ratio = f"{critical_r/N:.1f}x"
                    print(f"{task:<18} | {N/1e6:>6.0f}M | {d_ratio:>7.1f}x | {y0:<9.4f} | {max_observed_gain:<9.4f} | {r_str:<10} | {r_n_ratio}")
                    if N > 0 and math.isfinite(critical_r):
                        critical_ratios_by_task[task].append(critical_r / N)
                        critical_points_by_task[task].append(
                            {
                                "N": N,
                                "d_ratio": d_ratio,
                                "critical_r": critical_r,
                                "critical_r_over_n": critical_r / N,
                                "max_gain": max_observed_gain,
                            }
                        )

    # 3. Dataset-level summary: one "critical point" per task via aggregated R/N.
    print("\n" + "="*115)
    print("DATASET-LEVEL CRITICAL POINT SUMMARY (aggregated across model sizes and D-ratios)")
    print(f"Threshold criterion: first R where gain >= {args.threshold*100:.0f}% of max observed gain")
    print(f"{'BENCHMARK':<18} | {'N':>3} | {'Median R/N':>12} | {'P75 R/N':>9} | {'P90 R/N':>9} | {'Min R/N':>9} | {'Max R/N':>9}")
    print("-" * 115)

    for task in sorted(critical_ratios_by_task.keys()):
        vals = [v for v in critical_ratios_by_task[task] if math.isfinite(v)]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        arr = vals_sorted
        n = len(arr)
        med = float(arr[n // 2]) if n % 2 == 1 else float((arr[n // 2 - 1] + arr[n // 2]) / 2.0)
        p75 = float(arr[int(round(0.75 * (n - 1)))])
        p90 = float(arr[int(round(0.90 * (n - 1)))])
        mn = float(arr[0])
        mx = float(arr[-1])
        print(f"{task:<18} | {n:>3d} | {med:>11.2f}x | {p75:>8.2f}x | {p90:>8.2f}x | {mn:>8.2f}x | {mx:>8.2f}x")

    # 4. Save per-benchmark plots: critical R vs D/N with one curve per N.
    for task in sorted(critical_points_by_task.keys()):
        pts = critical_points_by_task[task]
        if not pts:
            continue

        # Group by model size N
        by_n = defaultdict(list)
        for p in pts:
            by_n[p["N"]].append(p)

        plt.figure(figsize=(8.8, 5.6))
        for n_val in sorted(by_n.keys()):
            rows = sorted(by_n[n_val], key=lambda x: x["d_ratio"])
            x = np.array([r["d_ratio"] for r in rows], dtype=float)
            y = np.array([r["critical_r"] / 1e9 for r in rows], dtype=float)  # billions
            if len(x) == 0:
                continue
            n_label = f"{n_val/1e9:.1f}B" if n_val >= 1e9 else f"{n_val/1e6:.0f}M"
            plt.plot(x, y, marker="o", linewidth=1.8, markersize=4.2, label=f"N={n_label}")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("D/N (Pretraining ratio, x)")
        plt.ylabel("Critical R (billions of retrieval tokens)")
        plt.title(f"{task}: Critical R vs D/N (threshold={args.threshold:.2f})")
        plt.grid(True, which="both", alpha=0.28, linestyle="--")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{task}_critical_R_vs_D_over_N.png")
        plt.savefig(out_path, dpi=180)
        plt.close()

    print(f"\nSaved per-benchmark saturation plots to: {out_dir}")

if __name__ == "__main__":
    main()
