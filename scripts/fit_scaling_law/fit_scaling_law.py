#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold

# Headless plotting for remote clusters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
COLM 2026: SUBSTITUTION RATIO SUITE (V7)
-----------------------------------------
New Features:
- ARE Decomposition: Reports CV ARE, LOMO ARE, and Overall ARE
- 2D Baseline Saving: Exports parametric anchor to JSON
- Substitution Ratio Calculator: Computes virtual token savings (ΔD and σ)
- Enhanced Reporting: Full transparency of interpolation vs extrapolation performance

python3 scripts/fit_scaling_law/fit_scaling_law.py \
    --dir scripts/eval/csv \
    --metric "perplexity,none" \
    --retrieval_model log  \
    --mode sequential \
    --exclude_dclm_val_ppl > fit_scaling_law_results.log
"""

# Plot style defaults (larger for readability in report figures)
TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14

# --- CORE SCALING FUNCTIONS ---

def scaling_law_2d(x, A, alpha, B, beta, L0):
    N, D = x
    N_hat, D_hat = N/1e9, D/1e9
    return (A * np.power(N_hat, -alpha)) + (B * np.power(D_hat, -beta)) + L0

def scaling_law_3d_power(x, A, alpha, B, beta, C, gamma, L0):
    N, D, R = x
    N_hat, D_hat, R_hat = N/1e9, D/1e9, R/1e9
    return (A * np.power(N_hat, -alpha)) + \
           (B * np.power(D_hat, -beta)) + \
           (C * np.power(R_hat + 1, -gamma)) + L0

def scaling_law_3d_log(x, A, alpha, B, beta, C, eta, L0):
    N, D, R = x
    N_hat, D_hat, R_hat = N/1e9, D/1e9, R/1e9
    parametric = (A * np.power(N_hat, -alpha)) + (B * np.power(D_hat, -beta))
    gain = C * np.log1p(eta * R_hat)
    return parametric - gain + L0

def scaling_law_3d_hill(x, A, alpha, B, beta, delta_max, K, n, L0):
    N, D, R = x
    N_hat, D_hat, R_hat = N/1e9, D/1e9, R/1e9
    parametric = (A * np.power(N_hat, -alpha)) + (B * np.power(D_hat, -beta))
    gain = delta_max * (np.power(R_hat, n) / (np.power(R_hat, n) + np.power(K, n)))
    return parametric - gain + L0

def scaling_law_3d_log_interaction(x, A, alpha, B, beta, C, eta, L0):
    """Interaction Log: Gain is a function of the R/N ratio."""
    N, D, R = x
    N_hat, D_hat = N/1e9, D/1e9
    parametric = (A * np.power(N_hat, -alpha)) + (B * np.power(D_hat, -beta))
    ratio = (R + 1) / N
    gain = C * np.log1p(eta * ratio)
    return parametric - gain + L0

# --- CROSS VALIDATION LOGIC (Enhanced with ARE tracking) ---

def run_with_cv(func, x_data, y_data, p0, bounds):
    """
    Standard 5-fold Random CV (Interpolation check).
    Returns: (popt, mean_test_mse, mean_test_are)
    """
    n_splits = min(5, len(y_data))
    if n_splits < 2:
        try:
            p_final, _ = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds, maxfev=50000)
            y_pred = func(x_data, *p_final)
            are = np.mean(np.abs((y_data - y_pred) / y_data)) * 100
            return p_final, 0.0, are
        except:
            return p0, 0.0, 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_mses, test_ares = [], []

    for train_idx, test_idx in kf.split(y_data):
        if len(train_idx) <= len(p0): continue
        try:
            x_train = tuple(arr[train_idx] for arr in x_data)
            x_test = tuple(arr[test_idx] for arr in x_data)
            y_train, y_test = y_data[train_idx], y_data[test_idx]
            p_cv, _ = curve_fit(func, x_train, y_train, p0=p0, bounds=bounds, maxfev=20000)
            y_pred = func(x_test, *p_cv)
            test_mses.append(np.mean((y_test - y_pred)**2))
            test_ares.append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        except:
            continue

    try:
        p_final, _ = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds, maxfev=50000)
    except:
        p_final = p0

    mean_mse = np.mean(test_mses) if test_mses else 0.0
    mean_are = np.mean(test_ares) if test_ares else 0.0
    return p_final, mean_mse, mean_are

def run_with_lomo_cv(func, x_data, y_data, p0, bounds):
    """
    D-CPT Style: Leave-One-Model-Out CV (Extrapolation check).
    Returns: (mean_test_mse, mean_test_are)
    """
    all_n = np.array(x_data[0])
    unique_n = np.sort(np.unique(all_n))
    if len(unique_n) < 2:
        return 0.0, 0.0

    test_mses, test_ares = [], []

    for hidden_n in unique_n:
        train_mask = (all_n != hidden_n)
        test_mask = (all_n == hidden_n)
        if not any(train_mask) or not any(test_mask):
            continue

        x_train = tuple(arr[train_mask] for arr in x_data)
        x_test = tuple(arr[test_mask] for arr in x_data)
        y_train, y_test = y_data[train_mask], y_data[test_mask]

        try:
            p_lomo, _ = curve_fit(func, x_train, y_train, p0=p0, bounds=bounds, maxfev=20000)
            y_pred = func(x_test, *p_lomo)
            test_mses.append(np.mean((y_test - y_pred)**2))
            test_ares.append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        except:
            continue

    mean_mse = np.mean(test_mses) if test_mses else 0.0
    mean_are = np.mean(test_ares) if test_ares else 0.0
    return mean_mse, mean_are

# --- HELPERS ---

def parse_size_to_float(label):
    if label.lower() == 'baseline': return 0.0
    match = re.match(r"(\d+(?:\.\d+)?)([MBx])", label, re.IGNORECASE)
    if not match: return 0.0
    val, unit = match.groups()
    val = float(val)
    if unit.upper() == 'M': return val * 1e6
    if unit.upper() == 'B': return val * 1e9
    return val

def parse_params_from_filename(filename):
    match = re.search(r"(\d+(?:\.\d+)?)([mb])", filename.lower())
    if not match: return 0.0
    val, unit = match.groups()
    return float(val) * 1e6 if unit == 'm' else float(val) * 1e9


TASK_NAME_MAP = {
    "arc_challenge": "ARC-Challenge",
    "arc_easy": "ARC-Easy",
    "hellaswag": "HellaSwag",
    "sciq": "Science Questions (SCIQ)",
    "openbookqa": "OpenBookQA",
    "nq_open": "Natural Questions",
    "strategyqa": "StrategyQA",
    "simpleqa": "SimpleQA",
    "piqa_generative": "PIQA",
    "commonsense_qa": "CommonsenseQA",
    "dclm_val_ppl": "DCLM Val PPL",
}


def format_task_name(task):
    return TASK_NAME_MAP.get(task, task.replace("_", " ").title())

def save_2d_baseline(task, popt_2d, out_dir, metrics=None):
    """Save the 2D parametric baseline for substitution calculations."""
    baseline_dict = {
        "task": task,
        "A": float(popt_2d[0]),
        "alpha": float(popt_2d[1]),
        "B": float(popt_2d[2]),
        "beta": float(popt_2d[3]),
        "L0": float(popt_2d[4])
    }
    if metrics:
        baseline_dict.update(metrics)
    save_path = os.path.join(out_dir, f"{task}_2d_baseline.json")
    with open(save_path, 'w') as f:
        json.dump(baseline_dict, f, indent=2)
    return save_path

def compute_substitution_ratios(task, data, popt_2d, popt_3d, r_model, out_dir):
    r"""
    MARGINAL SUBSTITUTION RATIO CALCULATOR
    ======================================

    Computes TWO complementary metrics for RAG efficiency:

    σ (SIGMA) - REPLACEMENT COST:
    =============================
    "How much pretraining would I need to match this RAG benefit?"

    For each RAG configuration (N, D, R):
    1. Measure L_rag = perplexity WITH retrieval at (N, D, R)
    2. Project onto 2D curve: D_eff = D needed to achieve L_rag WITHOUT retrieval
    3. Marginal savings: \Delta D = D_eff - D
    4. Replacement cost: \sigma = \Delta D / R

    Interpretation:
        \sigma = 5.0   →  Each retrieval token saves 5 pretraining tokens
        \sigma = 0.2   →  Each retrieval token saves 0.2 pretraining tokens

    NOTE: σ typically INCREASES with D (overtrained regime) because the 2D curve
    flattens, making marginal improvements via pretraining very expensive!

    κ (KAPPA) - MARGINAL BENEFIT:
    ==============================
    "How much loss improvement per BILLION retrieval tokens?"

    For each RAG configuration (N, D, R):
    1. \Delta L = L_baseline - L_rag  ← Absolute loss improvement from RAG
    2. Marginal benefit: \kappa = \Delta L / (R/1e9)  ← Per billion tokens!

    Interpretation:
        \kappa = 0.1   →  Each billion retrieval tokens drops loss by 0.1
        \kappa = 0.01  →  Each billion retrieval tokens drops loss by 0.01 (diminishing returns)

    NOTE: κ typically DECREASES with D (overtrained regime) because the absolute
    gap between RAG and baseline shrinks!

    The 2D inversion formula:
        Given L, solve for D: D = [(L - L_0 - A \cdot N^{-\alpha}) / B]^{-1/\beta}
    """
    A, alpha, B, beta, L0 = popt_2d
    N_all = np.array(data['N'])
    D_all = np.array(data['D'])
    R_all = np.array(data['R'])
    y_all = np.array(data['y'])

    # Build lookup table for R=0 baseline perplexities: {(N, D): L_baseline}
    baseline_lookup = {}
    for i in range(len(N_all)):
        if R_all[i] == 0:
            key = (N_all[i], D_all[i])
            baseline_lookup[key] = y_all[i]

    # Group all RAG configs by (N, D) and find best R for each
    # Key: (N, D) → Value: config with lowest L_rag (best retrieval performance)
    best_rag_per_ND = {}
    for i in range(len(N_all)):
        if R_all[i] == 0:
            continue  # Skip baseline configs

        N, D, R, L_rag = N_all[i], D_all[i], R_all[i], y_all[i]
        key = (N, D)

        # Check if baseline exists for this (N, D)
        if key not in baseline_lookup:
            continue

        # Keep config with lowest L_rag (best performance)
        if key not in best_rag_per_ND or L_rag < best_rag_per_ND[key]['L_rag']:
            best_rag_per_ND[key] = {
                'N': N,
                'D': D,
                'R': R,
                'L_rag': L_rag,
                'L_baseline': baseline_lookup[key]
            }

    # Helper function to invert 2D baseline
    def invert_2d_baseline(N, L):
        """Find D such that L_{2D}(N, D) = L"""
        N_hat = N / 1e9
        model_capacity_term = A * np.power(N_hat, -alpha)
        data_residual = L - L0 - model_capacity_term

        if data_residual <= 0 or B <= 0:
            return np.nan

        D = np.power(data_residual / B, -1.0 / beta) * 1e9
        return D

    # Compute marginal substitution ONLY for best R at each (N, D)
    results = []
    for (N, D), config in best_rag_per_ND.items():
        R = config['R']
        L_rag = config['L_rag']
        L_baseline = config['L_baseline']

        # Project both losses onto 2D curve
        D_eff_baseline = invert_2d_baseline(N, L_baseline)  # For diagnostics
        D_eff_rag = invert_2d_baseline(N, L_rag)

        # Compute marginal savings
        Delta_L = L_baseline - L_rag  # Absolute perplexity improvement from RAG

        if np.isnan(D_eff_rag):
            Delta_D = np.nan
            sigma = np.nan
        else:
            # σ (SIGMA): REPLACEMENT COST
            # "How much pretraining data did retrieval SAVE us?"
            # If D_eff_rag > D: RAG allowed us to achieve better performance than
            # we would have with just D pretraining → positive σ (savings)
            Delta_D = D_eff_rag - D  # Virtual pretraining saved by RAG
            sigma = Delta_D / R if R > 0 else 0.0  # Exchange rate (pretraining tokens per retrieval token)

            # Filter out only extreme numerical outliers (keep negatives to diagnose issues)
            if abs(sigma) > 10000:
                sigma = np.nan

        # κ (KAPPA): MARGINAL BENEFIT
        # "How much loss improvement per BILLION retrieval tokens?"
        # This measures absolute efficiency and typically DECREASES with D (diminishing returns)
        kappa = (Delta_L / (R / 1e9)) if R > 0 else 0.0  # Loss improvement per billion retrieval tokens

        results.append({
            'N': N,
            'D': D,
            'R': R,  # This is the BEST R for this (N, D) pair
            'D_ratio': D/N,
            'R_ratio': R/N,
            'L_baseline': L_baseline,  # Perplexity without RAG (at same D)
            'L_rag': L_rag,            # Perplexity with BEST retrieval
            'Delta_L': Delta_L,        # Marginal perplexity drop (maximum)
            'D_eff_baseline': D_eff_baseline,  # 2D-equivalent of baseline (diagnostic)
            'D_eff_rag': D_eff_rag,            # 2D-equivalent of RAG performance
            'Delta_D': Delta_D,        # Virtual pretraining tokens saved (= D_eff_rag - D)
            'sigma': sigma,            # σ: Replacement cost (increases with D due to saturation)
            'kappa': kappa             # κ: Marginal benefit (decreases with D, diminishing returns)
        })

    if not results:
        return None

    # Save to CSV
    csv_path = os.path.join(out_dir, f"{task}_substitution_ratios.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    return results

def generate_substitution_summary(all_results, metric, r_model, out_dir):
    """
    Generate a summary table aggregating σ by model size and D-ratio bins.

    Note: all_results already contains only BEST R for each (N, D) pair.
    """
    summary_lines = []
    summary_lines.append("")
    summary_lines.append("="*120)
    summary_lines.append(f"MARGINAL SUBSTITUTION RATIO SUMMARY: {metric.upper()} | {r_model.upper()}")
    summary_lines.append("="*120)
    summary_lines.append("")
    summary_lines.append("TWO COMPLEMENTARY METRICS:")
    summary_lines.append("")
    summary_lines.append("σ (SIGMA) - REPLACEMENT COST:")
    summary_lines.append("  For config (N, D, R), we achieve loss L_rag(N,D,R).")
    summary_lines.append("  We find D_eff = the pretraining budget needed to achieve L_rag WITHOUT retrieval.")
    summary_lines.append("  σ = (D_eff - D) / R  ← How much pretraining did retrieval SAVE us?")
    summary_lines.append("  • σ = 5.0   →  Each retrieval token saves 5 pretraining tokens")
    summary_lines.append("  • σ INCREASES with D (pretraining becomes expensive due to saturation)")
    summary_lines.append("")
    summary_lines.append("κ (KAPPA) - MARGINAL BENEFIT:")
    summary_lines.append("  κ = (L_baseline - L_rag) / (R/1B)  ← Loss improvement per BILLION retrieval tokens")
    summary_lines.append("  • κ = 0.1   →  Each billion retrieval tokens drops loss by 0.1")
    summary_lines.append("  • κ DECREASES with D (diminishing absolute returns)")
    summary_lines.append("")
    summary_lines.append("NOTE: D/N ratios (not retrieval sizes!):")
    summary_lines.append("  1x   = D/N ≈ 1    (undertrained: 728M params on 728M tokens)")
    summary_lines.append("  10x  = D/N ≈ 10   (optimal: 728M params on 7.28B tokens)")
    summary_lines.append("  100x = D/N ≈ 100  (overtrained: 728M params on 72.8B tokens)")
    summary_lines.append("")
    summary_lines.append("IMPORTANT: Both metrics computed only for BEST R at each (N,D) pair (maximum RAG benefit)")
    summary_lines.append("")
    summary_lines.append("="*120)
    summary_lines.append("σ (SIGMA) - REPLACEMENT COST:")
    summary_lines.append(f"{'BENCHMARK':<18} | {'Mean σ':<10} | {'σ @ 1x':<10} | {'σ @ 10x':<10} | {'σ @ 100x':<10}")
    summary_lines.append("-"*120)

    for task, results in all_results.items():
        if not results:
            continue

        # Overall mean (only finite values)
        sigmas = [r['sigma'] for r in results if np.isfinite(r['sigma'])]
        mean_sigma = np.mean(sigmas) if sigmas else np.nan

        # Bin by D-ratio (D/N)
        sigma_1x_vals = [r['sigma'] for r in results if 0.5 < r['D_ratio'] < 2.0 and np.isfinite(r['sigma'])]
        sigma_10x_vals = [r['sigma'] for r in results if 8.0 < r['D_ratio'] < 15.0 and np.isfinite(r['sigma'])]
        sigma_100x_vals = [r['sigma'] for r in results if 80.0 < r['D_ratio'] < 120.0 and np.isfinite(r['sigma'])]

        sigma_1x = np.mean(sigma_1x_vals) if sigma_1x_vals else np.nan
        sigma_10x = np.mean(sigma_10x_vals) if sigma_10x_vals else np.nan
        sigma_100x = np.mean(sigma_100x_vals) if sigma_100x_vals else np.nan

        summary_lines.append(f"{task:<18} | {mean_sigma:<10.2f} | {sigma_1x:<10.2f} | {sigma_10x:<10.2f} | {sigma_100x:<10.2f}")

    # Add κ (KAPPA) table
    summary_lines.append("")
    summary_lines.append("="*120)
    summary_lines.append("κ (KAPPA) - MARGINAL BENEFIT (Loss improvement per BILLION retrieval tokens):")
    summary_lines.append(f"{'BENCHMARK':<18} | {'Mean κ':<10} | {'κ @ 1x':<10} | {'κ @ 10x':<10} | {'κ @ 100x':<10}")
    summary_lines.append("-"*120)

    for task, results in all_results.items():
        if not results:
            continue

        # Overall mean (only finite values)
        kappas = [r['kappa'] for r in results if np.isfinite(r['kappa'])]
        mean_kappa = np.mean(kappas) if kappas else np.nan

        # Bin by D-ratio (D/N)
        kappa_1x_vals = [r['kappa'] for r in results if 0.5 < r['D_ratio'] < 2.0 and np.isfinite(r['kappa'])]
        kappa_10x_vals = [r['kappa'] for r in results if 8.0 < r['D_ratio'] < 15.0 and np.isfinite(r['kappa'])]
        kappa_100x_vals = [r['kappa'] for r in results if 80.0 < r['D_ratio'] < 120.0 and np.isfinite(r['kappa'])]

        kappa_1x = np.mean(kappa_1x_vals) if kappa_1x_vals else np.nan
        kappa_10x = np.mean(kappa_10x_vals) if kappa_10x_vals else np.nan
        kappa_100x = np.mean(kappa_100x_vals) if kappa_100x_vals else np.nan

        summary_lines.append(f"{task:<18} | {mean_kappa:<10.4f} | {kappa_1x:<10.4f} | {kappa_10x:<10.4f} | {kappa_100x:<10.4f}")

    # Add model-size breakdown section
    summary_lines.append("")
    summary_lines.append("="*120)
    summary_lines.append("BREAKDOWN BY MODEL SIZE (N):")
    summary_lines.append("  (Averages σ across all D and R configurations for each model size)")
    summary_lines.append("="*120)

    # Auto-detect all unique model sizes across all tasks
    all_model_sizes = set()
    for task, results in all_results.items():
        if results:
            all_model_sizes.update([r['N'] for r in results])

    sorted_model_sizes = sorted(all_model_sizes)

    # Format header dynamically with actual model sizes
    header_parts = [f"{'BENCHMARK':<18}"]
    for N in sorted_model_sizes:
        if N >= 1e9:
            label = f"{N/1e9:.1f}B"
        else:
            label = f"{N/1e6:.0f}M"
        header_parts.append(f"{label:<10}")

    summary_lines.append(" | ".join(header_parts))
    summary_lines.append("-"*120)

    for task, results in all_results.items():
        if not results:
            continue

        # Compute mean sigma for each model size
        model_sigmas = {}
        for N in sorted_model_sizes:
            n_vals = [r['sigma'] for r in results if r['N'] == N and np.isfinite(r['sigma'])]
            model_sigmas[N] = np.mean(n_vals) if n_vals else np.nan

        # Format line with all model sizes
        line_parts = [f"{task:<18}"]
        for N in sorted_model_sizes:
            sig = model_sigmas.get(N, np.nan)
            line_parts.append(f"{sig:<10.2f}")

        summary_lines.append(" | ".join(line_parts))

    # Add κ (KAPPA) model size breakdown
    summary_lines.append("")
    summary_lines.append("="*120)
    summary_lines.append("κ (KAPPA) BREAKDOWN BY MODEL SIZE (N):")
    summary_lines.append("  (Averages κ across all D and R configurations for each model size)")
    summary_lines.append("="*120)

    # Reuse same header
    summary_lines.append(" | ".join(header_parts))
    summary_lines.append("-"*120)

    for task, results in all_results.items():
        if not results:
            continue

        # Compute mean kappa for each model size
        model_kappas = {}
        for N in sorted_model_sizes:
            n_vals = [r['kappa'] for r in results if r['N'] == N and np.isfinite(r['kappa'])]
            model_kappas[N] = np.mean(n_vals) if n_vals else np.nan

        # Format line with all model sizes
        line_parts = [f"{task:<18}"]
        for N in sorted_model_sizes:
            kap = model_kappas.get(N, np.nan)
            line_parts.append(f"{kap:<10.4f}")

        summary_lines.append(" | ".join(line_parts))

    # Add D-by-N breakdown: One table per model size showing σ vs D
    summary_lines.append("")
    summary_lines.append("="*120)
    summary_lines.append("BREAKDOWN BY PRETRAINING DATA (D) FOR EACH MODEL SIZE (N):")
    summary_lines.append("  (Separate table per model size - shows σ vs D trend)")
    summary_lines.append("="*120)

    # Get all unique N values across all tasks
    all_N_values = set()
    for task, results in all_results.items():
        if results:
            all_N_values.update([r['N'] for r in results])
    sorted_N_values = sorted(all_N_values)

    # Create one table per model size
    for N in sorted_N_values:
        if N >= 1e9:
            n_label = f"N={N/1e9:.2f}B"
        else:
            n_label = f"N={N/1e6:.0f}M"

        summary_lines.append("")
        summary_lines.append(f"MODEL SIZE: {n_label}")
        summary_lines.append("-"*120)

        # Collect all (task, D, σ) tuples for this N
        task_data = {}
        for task, results in all_results.items():
            if not results:
                continue

            # Get all D values for this N
            n_results = [r for r in results if r['N'] == N and np.isfinite(r['sigma'])]
            if not n_results:
                continue

            task_data[task] = {}
            for r in n_results:
                task_data[task][r['D']] = r['sigma']

        if not task_data:
            summary_lines.append("  (No data for this model size)")
            continue

        # Get all unique D values for this N (across all tasks)
        all_D_for_N = set()
        for task_d_dict in task_data.values():
            all_D_for_N.update(task_d_dict.keys())
        sorted_D_for_N = sorted(all_D_for_N)

        # Header: D values (show actual values, not rounded!)
        header_parts = [f"{'BENCHMARK':<18}"]
        for D in sorted_D_for_N:
            if D >= 1e9:
                d_label = f"D={D/1e9:.2f}B"[:12].ljust(12)
            else:
                d_label = f"D={D/1e6:.0f}M"[:12].ljust(12)
            header_parts.append(d_label)
        summary_lines.append(" | ".join(header_parts))
        summary_lines.append("-"*120)

        # One row per task
        for task in sorted(task_data.keys()):
            line_parts = [f"{task:<18}"]
            for D in sorted_D_for_N:
                sigma_val = task_data[task].get(D, np.nan)
                line_parts.append(f"{sigma_val:<12.2f}")
            summary_lines.append(" | ".join(line_parts))

    # Add κ (KAPPA) D-by-N breakdown: One table per model size showing κ vs D
    summary_lines.append("")
    summary_lines.append("="*120)
    summary_lines.append("κ (KAPPA) BREAKDOWN BY PRETRAINING DATA (D) FOR EACH MODEL SIZE (N):")
    summary_lines.append("  (Separate table per model size - shows κ vs D trend, should DECREASE with D)")
    summary_lines.append("="*120)

    # Create one table per model size
    for N in sorted_N_values:
        if N >= 1e9:
            n_label = f"N={N/1e9:.2f}B"
        else:
            n_label = f"N={N/1e6:.0f}M"

        summary_lines.append("")
        summary_lines.append(f"MODEL SIZE: {n_label}")
        summary_lines.append("-"*120)

        # Collect all (task, D, κ) tuples for this N
        task_data_kappa = {}
        for task, results in all_results.items():
            if not results:
                continue

            # Get all D values for this N
            n_results = [r for r in results if r['N'] == N and np.isfinite(r['kappa'])]
            if not n_results:
                continue

            task_data_kappa[task] = {}
            for r in n_results:
                task_data_kappa[task][r['D']] = r['kappa']

        if not task_data_kappa:
            summary_lines.append("  (No data for this model size)")
            continue

        # Get all unique D values for this N (across all tasks)
        all_D_for_N_kappa = set()
        for task_d_dict in task_data_kappa.values():
            all_D_for_N_kappa.update(task_d_dict.keys())
        sorted_D_for_N_kappa = sorted(all_D_for_N_kappa)

        # Header: D values (show actual values, not rounded!)
        header_parts_kappa = [f"{'BENCHMARK':<18}"]
        for D in sorted_D_for_N_kappa:
            if D >= 1e9:
                d_label = f"D={D/1e9:.2f}B"[:12].ljust(12)
            else:
                d_label = f"D={D/1e6:.0f}M"[:12].ljust(12)
            header_parts_kappa.append(d_label)
        summary_lines.append(" | ".join(header_parts_kappa))
        summary_lines.append("-"*120)

        # One row per task
        for task in sorted(task_data_kappa.keys()):
            line_parts = [f"{task:<18}"]
            for D in sorted_D_for_N_kappa:
                kappa_val = task_data_kappa[task].get(D, np.nan)
                line_parts.append(f"{kappa_val:<12.4f}")
            summary_lines.append(" | ".join(line_parts))

    summary_text = "\n".join(summary_lines)

    # Save to file
    summary_path = os.path.join(out_dir, f"substitution_summary_{metric.split(',')[0]}_{r_model}.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    # Print to terminal
    print("\n" + summary_text)

    return summary_path

# --- RUNNERS ---

def run_traditional(task, data, metric_key):
    """Fit 2D baseline with ARE decomposition."""
    N, D, y = map(np.array, [data['N'], data['D'], data['y']])
    avg_y = np.mean(y)
    min_y = np.min(y)

    # Data-driven L_0 bounds: allow it to go below minimum observed value
    L0_lower = max(0.5, min_y * 0.5)  # At least 50% below min observed
    L0_upper = avg_y * 1.5  # Up to 150% of average

    p0 = [avg_y*0.1, 0.35, avg_y*0.1, 0.3, min_y * 0.9]
    bounds = ([0, 0, 0, 0, L0_lower], [np.inf, 2.0, np.inf, 2.0, L0_upper])

    popt, rand_mse, cv_are = run_with_cv(scaling_law_2d, (N, D), y, p0, bounds)
    lomo_mse, lomo_are = run_with_lomo_cv(scaling_law_2d, (N, D), y, p0, bounds)

    y_pred = scaling_law_2d((N, D), *popt)
    overall_are = np.mean(np.abs((y - y_pred) / y)) * 100

    return popt, overall_are, cv_are, lomo_are, rand_mse, lomo_mse

def run_sequential(task, data, metric_key, r_model):
    """Fit 3D law with locked 2D baseline and ARE decomposition."""
    N_all, D_all, R_all, y_all = map(np.array, [data['N'], data['D'], data['R'], data['y']])
    mask = R_all == 0

    # 1. Get and save 2D baseline
    popt_2d, base_overall_are, base_cv_are, base_lomo_are, _, _ = run_traditional(
        task, {'N': N_all[mask], 'D': D_all[mask], 'y': y_all[mask]}, metric_key
    )
    f_A, f_alpha, f_B, f_beta, f_L0 = popt_2d

    # 2. Lock and fit retrieval term
    def locked_model(x, A, B, C, exp_term, L0):
        if r_model == 'power':
            return scaling_law_3d_power(x, A, f_alpha, B, f_beta, C, exp_term, f_L0)
        elif r_model == 'log':
            return scaling_law_3d_log(x, A, f_alpha, B, f_beta, C, exp_term, f_L0)
        elif r_model == 'hill':
            return scaling_law_3d_hill(x, A, f_alpha, B, f_beta, C, exp_term, 1.0, f_L0)
        else:  # interactionlog
            return scaling_law_3d_log_interaction(x, A, f_alpha, B, f_beta, C, exp_term, f_L0)

    p0 = [f_A, f_B, 0.1, 1.0, f_L0]
    bounds = ([0, 0, 0, 0, f_L0 - 1e-6], [np.inf, np.inf, np.inf, 10.0, f_L0 + 1e-6])

    popt_seq, rand_mse, cv_are = run_with_cv(locked_model, (N_all, D_all, R_all), y_all, p0, bounds)
    lomo_mse, lomo_are = run_with_lomo_cv(locked_model, (N_all, D_all, R_all), y_all, p0, bounds)

    # 3. Format full parameters
    full_popt = [popt_seq[0], f_alpha, popt_seq[1], f_beta, popt_seq[2], popt_seq[3], popt_seq[4]]

    # 4. Calculate overall ARE
    x_input = (N_all, D_all, R_all)
    if r_model == 'interactionlog':
        y_pred = scaling_law_3d_log_interaction(x_input, *full_popt)
    elif r_model == 'hill':
        y_pred = scaling_law_3d_hill(x_input, full_popt[0], full_popt[1], full_popt[2],
                                     full_popt[3], full_popt[4], full_popt[5], 1.0, full_popt[6])
    else:
        func = scaling_law_3d_power if r_model == 'power' else scaling_law_3d_log
        y_pred = func(x_input, *full_popt)

    overall_are = np.mean(np.abs((y_all - y_pred) / y_all)) * 100

    baseline_metrics = {
        "baseline_overall_are": float(base_overall_are),
        "baseline_cv_are": float(base_cv_are),
        "baseline_lomo_are": float(base_lomo_are),
    }
    return popt_2d, full_popt, overall_are, cv_are, lomo_are, rand_mse, lomo_mse, baseline_metrics

# --- PLOTTING (Same as v6) ---

def plot_intuitive_efficiency_slices(task, data, popt, mode, out_dir, scaling_law_func=scaling_law_3d_log, popt_2d=None):
    """
    User's 'Intuitive' Plot:
    X-axis: Pretraining Ratio (1x, 2x, etc.)
    Curves: Different Retrieval Ratios (0y, 1y, 2y, etc. where y = N)
    Grouping: One plot per Model Size (N)

    If popt_2d is provided, also plots the 2D baseline curve for reference.
    """
    os.makedirs(out_dir, exist_ok=True)
    N = np.array(data['N'])
    D = np.array(data['D'])
    R = np.array(data['R'])
    y = np.array(data['y'])

    unique_n = np.sort(np.unique(N))

    clean_task = format_task_name(task)
    for n_val in unique_n:
        plt.figure(figsize=(8, 6))
        mask_n = (N == n_val)

        # Calculate Index Ratios (R/N) and find unique ones (0, 1, 2, 5...)
        # We round to handle floating point noise
        index_ratios = np.unique(np.round(R[mask_n] / n_val, 1))
        colors = plt.cm.plasma(np.linspace(0, 0.8, len(index_ratios)))

        # Plot 2D baseline first (if provided) as reference
        if popt_2d is not None and mode == 'sequential':
            ratio_smooth = np.linspace(max(1.0, (D[mask_n]/n_val).min()), (D[mask_n]/n_val).max(), 100)
            d_smooth = ratio_smooth * n_val
            y_2d_baseline = scaling_law_2d((n_val, d_smooth), *popt_2d)
            plt.plot(ratio_smooth, y_2d_baseline,
                    color='black', linestyle='--', linewidth=3,
                    label='2D Baseline (Pure Parametric)', alpha=0.8, zorder=10)

        for i, r_ratio in enumerate(index_ratios):
            mask_r = mask_n & (np.isclose(R/n_val, r_ratio, atol=0.1))
            if not any(mask_r): continue

            # Scatter Observations
            label_name = "R=0 (No Retrieval)" if r_ratio == 0 else f"Index: {int(r_ratio)}y"
            plt.scatter(D[mask_r]/n_val, y[mask_r], color=colors[i], alpha=0.7)

            # Plot Predicted Curve
            # Smooth range for pretraining ratios (e.g., 1.0 to 100.0)
            ratio_smooth = np.linspace(max(1.0, (D[mask_n]/n_val).min()), (D[mask_n]/n_val).max(), 100)
            d_smooth = ratio_smooth * n_val
            r_val = r_ratio * n_val

            if mode == 'traditional' and r_ratio == 0:
                y_smooth = scaling_law_func((n_val, d_smooth), *popt)
            elif mode != 'traditional':
                y_smooth = scaling_law_func((n_val, d_smooth, r_val), *popt)
            else:
                continue # Skip RAG curves in traditional mode

            plt.plot(ratio_smooth, y_smooth, color=colors[i], label=label_name, linewidth=2)

        plt.title(
            f"{clean_task}: Model Size N={n_val/1e6:.0f}M\nPretraining vs. Retrieval Tradeoff",
            fontsize=TITLE_FONTSIZE,
        )
        plt.xlabel("Pretraining Ratio (xParams)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("Metric (PPL/Error)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.xscale('log') # Ratios often span orders of magnitude (1x to 100x)
        plt.legend(title="Retrieval Scale", loc='upper right', fontsize='small')
        plt.grid(True, alpha=0.2)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)

        save_path = os.path.join(out_dir, f"{task}_{mode}_potential_N_{int(n_val/1e6)}M.png")
        plt.savefig(save_path, dpi=200)
        plt.close()


def plot_2d_projections(task, data, popt, mode, retrieval_model, out_dir):
    """2D projections: Pretraining slice (R=0) and Retrieval slices (fixed D/N ratios)."""
    os.makedirs(out_dir, exist_ok=True)
    clean_task = format_task_name(task)
    N, D, R, y = map(np.array, [data['N'], data['D'], data['R'], data['y']])
    unique_n = np.sort(np.unique(N))
    ratios = np.unique(np.round(D / N, 1))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_n)))

    # 1. Pretrain Slice
    plt.figure(figsize=(8, 6))
    for i, n_val in enumerate(unique_n):
        mask = (N == n_val) & (R == 0)
        if not any(mask): continue
        plt.scatter(D[mask]/1e9, y[mask], color=colors[i], alpha=0.7, label=f'N={n_val/1e6:.0f}M')
        d_smooth = np.exp(np.linspace(np.log(D[mask].min()), np.log(D[mask].max()), 100))

        if mode == 'traditional':
            y_smooth = scaling_law_2d((n_val, d_smooth), *popt)
        else:
            if retrieval_model == 'power': func = scaling_law_3d_power
            elif retrieval_model == 'log': func = scaling_law_3d_log
            elif retrieval_model == 'hill': func = lambda x, *p: scaling_law_3d_hill(x, p[0], p[1], p[2], p[3], p[4], p[5], 1.0, p[6])
            else: func = scaling_law_3d_log_interaction

            y_smooth = func((n_val, d_smooth, 0), *popt)
        plt.plot(d_smooth/1e9, y_smooth, color=colors[i], linestyle='-')
    plt.title(f"{clean_task}: Pretraining (R=0)", fontsize=TITLE_FONTSIZE)
    plt.xscale('log')
    plt.legend()
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.savefig(os.path.join(out_dir, f"{task}_{mode}_pretrain.png"), dpi=200); plt.close()

    # 2. Retrieval Slices
    if mode != 'traditional':
        for ratio in ratios:
            plt.figure(figsize=(8, 6))
            for i, n_val in enumerate(unique_n):
                mask = (N == n_val) & (np.abs(D/N - ratio) < 0.5)
                if not any(mask): continue
                plt.scatter(R[mask]/1e9, y[mask], color=colors[i], label=f'N={n_val/1e6:.0f}M')
                r_smooth = np.linspace(0, R.max(), 100)

                if retrieval_model == 'power': func = scaling_law_3d_power
                elif retrieval_model == 'log': func = scaling_law_3d_log
                elif retrieval_model == 'hill': func = lambda x, *p: scaling_law_3d_hill(x, p[0], p[1], p[2], p[3], p[4], p[5], 1.0, p[6])
                else: func = scaling_law_3d_log_interaction

                y_smooth = func((n_val, n_val * ratio, r_smooth), *popt)
                plt.plot(r_smooth/1e9, y_smooth, color=colors[i])
            plt.title(f"{clean_task}: Retrieval (Ratio {ratio:.1f}x)", fontsize=TITLE_FONTSIZE)
            plt.legend()
            plt.xticks(fontsize=TICK_FONTSIZE)
            plt.yticks(fontsize=TICK_FONTSIZE)
            plt.savefig(os.path.join(out_dir, f"{task}_{mode}_retrieval_{ratio:.1f}x.png"), dpi=200); plt.close()


def plot_sigma_vs_training_ratio(task, results, out_dir):
    """
    Chinchilla-style plot: σ as a function of D (pretraining tokens).
    Different curves for different model sizes (N).

    Note: Results already contain only BEST R for each (N, D) pair.
    This shows the maximum benefit RAG can provide at each training point.
    """
    if not results:
        return

    os.makedirs(out_dir, exist_ok=True)

    # Organize by model size (results already filtered to best R per (N,D))
    data_by_N = {}
    for r in results:
        if not np.isfinite(r['sigma']):
            continue

        N = r['N']
        D = r['D']
        if N not in data_by_N:
            data_by_N[N] = {'D': [], 'sigma': [], 'R': []}
        data_by_N[N]['D'].append(D)
        data_by_N[N]['sigma'].append(r['sigma'])
        data_by_N[N]['R'].append(r['R'])

    if not data_by_N:
        return

    # Create plot
    plt.figure(figsize=(10, 6))
    clean_task = format_task_name(task)
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_by_N)))

    for i, (N, data) in enumerate(sorted(data_by_N.items())):
        if not data['D']:
            continue

        # Sort by D for clean lines
        sorted_indices = np.argsort(data['D'])
        D_vals = np.array(data['D'])[sorted_indices]
        sigmas = np.array(data['sigma'])[sorted_indices]
        R_vals = np.array(data['R'])[sorted_indices]

        # Format label
        if N >= 1e9:
            label = f"N={N/1e9:.1f}B"
        else:
            label = f"N={N/1e6:.0f}M"

        # Plot scatter + line (one point per (N,D) with best R)
        plt.scatter(D_vals/1e9, sigmas, color=colors[i], alpha=0.7, s=80, edgecolors='white', linewidth=1)
        plt.plot(D_vals/1e9, sigmas, color=colors[i], label=label, linewidth=2.5, alpha=0.9)

    plt.xscale('log')
    plt.xlabel('D (Pretraining Tokens)', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('σ (Substitution Ratio)\n[Pretraining tokens saved per retrieval token]', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(
        f'{clean_task}: RAG Efficiency vs Pretraining\n(Shows best R for each (N,D): maximum RAG benefit)',
        fontsize=TITLE_FONTSIZE,
    )
    plt.legend(title='Model Size', fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{task}_sigma_vs_D.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_kappa_vs_training_ratio(task, results, out_dir):
    """
    Chinchilla-style plot: κ as a function of D (pretraining tokens).
    Different curves for different model sizes (N).

    Note: Results already contain only BEST R for each (N, D) pair.
    This shows the maximum marginal benefit RAG can provide at each training point.
    """
    if not results:
        return

    os.makedirs(out_dir, exist_ok=True)

    # Organize by model size (results already filtered to best R per (N,D))
    data_by_N = {}
    for r in results:
        if not np.isfinite(r['kappa']):
            continue

        N = r['N']
        D = r['D']
        if N not in data_by_N:
            data_by_N[N] = {'D': [], 'kappa': [], 'R': []}
        data_by_N[N]['D'].append(D)
        data_by_N[N]['kappa'].append(r['kappa'])
        data_by_N[N]['R'].append(r['R'])

    if not data_by_N:
        return

    # Create plot
    plt.figure(figsize=(10, 6))
    clean_task = format_task_name(task)
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_by_N)))

    for i, (N, data) in enumerate(sorted(data_by_N.items())):
        if not data['D']:
            continue

        # Sort by D for clean lines
        sorted_indices = np.argsort(data['D'])
        D_vals = np.array(data['D'])[sorted_indices]
        kappas = np.array(data['kappa'])[sorted_indices]
        R_vals = np.array(data['R'])[sorted_indices]

        # Format label
        if N >= 1e9:
            label = f"N={N/1e9:.1f}B"
        else:
            label = f"N={N/1e6:.0f}M"

        # Plot scatter + line (one point per (N,D) with best R)
        plt.scatter(D_vals/1e9, kappas, color=colors[i], alpha=0.7, s=80, edgecolors='white', linewidth=1)
        plt.plot(D_vals/1e9, kappas, color=colors[i], label=label, linewidth=2.5, alpha=0.9)

    plt.xscale('log')
    plt.xlabel('D (Pretraining Tokens)', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('κ (Marginal Benefit)\n[Loss improvement per billion retrieval tokens]', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(
        f'{clean_task}: RAG Marginal Benefit vs Pretraining\n(Shows best R for each (N,D): κ decreases → diminishing returns)',
        fontsize=TITLE_FONTSIZE,
    )
    plt.legend(title='Model Size', fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{task}_kappa_vs_D.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_aggregated_sigma_across_tasks(all_results, out_dir, metric_name):
    """
    Aggregate σ across all benchmarks using GEOMETRIC MEAN (standard for efficiency metrics).

    X-axis: D (pretraining tokens)
    Y-axis: Geometric mean σ across benchmarks
    Curves: One per model size (N)
    Shaded: Geometric std deviation (multiplicative spread)

    Creates two versions: vs D (absolute) and vs D/N (ratio)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect all data: for each (N, D), gather σ values across all benchmarks
    data_by_N_D = {}

    for task, results in all_results.items():
        if not results:
            continue
        for r in results:
            if not np.isfinite(r['sigma']) or r['sigma'] <= 0:  # Geometric mean needs positive values
                continue

            N = r['N']
            D = r['D']
            sigma = r['sigma']

            if N not in data_by_N_D:
                data_by_N_D[N] = {}
            if D not in data_by_N_D[N]:
                data_by_N_D[N][D] = []
            data_by_N_D[N][D].append(sigma)

    if not data_by_N_D:
        return

    # === PLOT 1: σ vs D (absolute) ===
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_by_N_D)))

    for i, (N, D_dict) in enumerate(sorted(data_by_N_D.items())):
        if not D_dict:
            continue

        # Compute geometric mean and geometric std for each D
        D_vals = []
        geom_means = []
        geom_lower = []
        geom_upper = []

        for D in sorted(D_dict.keys()):
            sigmas = D_dict[D]
            if len(sigmas) >= 1:
                log_sigmas = np.log(sigmas)
                mean_log = np.mean(log_sigmas)
                std_log = np.std(log_sigmas)

                D_vals.append(D)
                geom_means.append(np.exp(mean_log))
                geom_lower.append(np.exp(mean_log - std_log))  # Multiplicative lower bound
                geom_upper.append(np.exp(mean_log + std_log))  # Multiplicative upper bound

        if not D_vals:
            continue

        D_vals = np.array(D_vals)
        geom_means = np.array(geom_means)
        geom_lower = np.array(geom_lower)
        geom_upper = np.array(geom_upper)

        # Format label
        if N >= 1e9:
            label = f"N={N/1e9:.1f}B"
        else:
            label = f"N={N/1e6:.0f}M"

        # Plot geometric mean line with markers
        plt.plot(D_vals/1e9, geom_means, color=colors[i], label=label, linewidth=2.5, alpha=0.9, marker='o', markersize=6)

        # Light shaded geometric std region (less intrusive in log scale)
        plt.fill_between(D_vals/1e9, geom_lower, geom_upper, color=colors[i], alpha=0.15)

    plt.xscale('log')
    plt.yscale('log')  # Log scale on Y-axis for orders of magnitude
    plt.xlabel('D (Pretraining Tokens, Billions)', fontsize=12)
    plt.ylabel('σ (Geometric mean, log scale)\n[Pretraining tokens saved per retrieval token]', fontsize=12)
    plt.title(f'Aggregated RAG Efficiency: σ vs D\n(Geometric mean across {len(all_results)} benchmarks, log-log scale)', fontsize=13)
    plt.legend(title='Model Size', fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='σ=1 (break-even)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"aggregated_sigma_vs_D_{metric_name}.png"), dpi=200)
    plt.close()

    # === PLOT 2: σ vs D/N (training ratio) ===
    plt.figure(figsize=(10, 6))

    for i, (N, D_dict) in enumerate(sorted(data_by_N_D.items())):
        if not D_dict:
            continue

        # Compute geometric mean for each D/N ratio
        ratios = []
        geom_means = []
        geom_lower = []
        geom_upper = []

        for D in sorted(D_dict.keys()):
            sigmas = D_dict[D]
            if len(sigmas) >= 1:
                log_sigmas = np.log(sigmas)
                mean_log = np.mean(log_sigmas)
                std_log = np.std(log_sigmas)

                ratios.append(D / N)
                geom_means.append(np.exp(mean_log))
                geom_lower.append(np.exp(mean_log - std_log))
                geom_upper.append(np.exp(mean_log + std_log))

        if not ratios:
            continue

        ratios = np.array(ratios)
        geom_means = np.array(geom_means)
        geom_lower = np.array(geom_lower)
        geom_upper = np.array(geom_upper)

        # Format label
        if N >= 1e9:
            label = f"N={N/1e9:.1f}B"
        else:
            label = f"N={N/1e6:.0f}M"

        # Plot geometric mean line with markers
        plt.plot(ratios, geom_means, color=colors[i], label=label, linewidth=2.5, alpha=0.9, marker='o', markersize=6)

        # Light shaded geometric std region (less intrusive in log scale)
        plt.fill_between(ratios, geom_lower, geom_upper, color=colors[i], alpha=0.15)

    plt.xscale('log')
    plt.yscale('log')  # Log scale on Y-axis for orders of magnitude
    plt.xlabel('D/N (Training Ratio: tokens per parameter)', fontsize=12)
    plt.ylabel('σ (Geometric mean, log scale)\n[Pretraining tokens saved per retrieval token]', fontsize=12)
    plt.title(f'Aggregated RAG Efficiency: σ vs D/N\n(Geometric mean across {len(all_results)} benchmarks, log-log scale)', fontsize=13)
    plt.legend(title='Model Size', fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='σ=1 (break-even)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"aggregated_sigma_vs_D_over_N_{metric_name}.png"), dpi=200)
    plt.close()


def plot_aggregated_sigma_across_tasks_with_powerfit(all_results, out_dir, metric_name):
    """
    Same as plot_aggregated_sigma but with POWER LAW FITS: σ = a × D^b

    Fits in log-log space (2 parameters: a and b) and shows smooth fitted curves.
    Keeps scattered points visible.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect all data (same as original)
    data_by_N_D = {}
    for task, results in all_results.items():
        if not results:
            continue
        for r in results:
            if not np.isfinite(r['sigma']) or r['sigma'] <= 0:
                continue
            N = r['N']
            D = r['D']
            sigma = r['sigma']
            if N not in data_by_N_D:
                data_by_N_D[N] = {}
            if D not in data_by_N_D[N]:
                data_by_N_D[N][D] = []
            data_by_N_D[N][D].append(sigma)

    if not data_by_N_D:
        return

    # === PLOT 1: σ vs D with power law fits ===
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_by_N_D)))

    for i, (N, D_dict) in enumerate(sorted(data_by_N_D.items())):
        if not D_dict:
            continue

        # Compute geometric mean for each D
        D_vals = []
        geom_means = []
        geom_lower = []
        geom_upper = []

        for D in sorted(D_dict.keys()):
            sigmas = D_dict[D]
            if len(sigmas) >= 1:
                log_sigmas = np.log(sigmas)
                mean_log = np.mean(log_sigmas)
                std_log = np.std(log_sigmas)
                D_vals.append(D)
                geom_means.append(np.exp(mean_log))
                geom_lower.append(np.exp(mean_log - std_log))
                geom_upper.append(np.exp(mean_log + std_log))

        if not D_vals or len(D_vals) < 2:
            continue

        D_vals = np.array(D_vals)
        geom_means = np.array(geom_means)
        geom_lower = np.array(geom_lower)
        geom_upper = np.array(geom_upper)

        # Format label
        if N >= 1e9:
            n_label = f"N={N/1e9:.1f}B"
        else:
            n_label = f"N={N/1e6:.0f}M"

        # Fit power law: σ = a × D^b (linear fit in log-log space, 2 parameters)
        log_D = np.log(D_vals)
        log_sigma = np.log(geom_means)
        coeffs = np.polyfit(log_D, log_sigma, 1)
        b = coeffs[0]  # Power law exponent (slope in log-log)
        log_a = coeffs[1]  # Log of coefficient (intercept in log-log)
        a = np.exp(log_a)

        # Compute R² for fit quality
        sigma_fit_points = a * D_vals ** b
        ss_res = np.sum((geom_means - sigma_fit_points) ** 2)
        ss_tot = np.sum((geom_means - np.mean(geom_means)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Generate smooth fitted curve
        D_smooth = np.logspace(np.log10(D_vals.min()), np.log10(D_vals.max()), 100)
        sigma_smooth = a * D_smooth ** b

        # Plot scattered points (actual data)
        plt.scatter(D_vals/1e9, geom_means, color=colors[i], s=100, alpha=0.6,
                   edgecolors='white', linewidth=1.5, zorder=5)

        # Plot fitted power law curve
        label = f"{n_label}: σ∝D^{{{b:.2f}}} (R²={r_squared:.2f})"
        plt.plot(D_smooth/1e9, sigma_smooth, color=colors[i], label=label,
                linewidth=2.5, alpha=0.9, linestyle='-')

        # Very light shaded region
        plt.fill_between(D_vals/1e9, geom_lower, geom_upper, color=colors[i], alpha=0.08)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('D (Pretraining Tokens, Billions)', fontsize=12)
    plt.ylabel('σ (Geometric mean, log scale)\n[Pretraining tokens saved per retrieval token]', fontsize=12)
    plt.title(f'Aggregated RAG Efficiency: σ vs D (Power Law Fits)\n(Fitted: σ = a×D^b, across {len(all_results)} benchmarks)', fontsize=13)
    plt.legend(title='Model Size & Fit', fontsize=9, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"aggregated_sigma_vs_D_{metric_name}_powerfit.png"), dpi=200)
    plt.close()

    # === PLOT 2: σ vs D/N with power law fits ===
    plt.figure(figsize=(10, 6))

    for i, (N, D_dict) in enumerate(sorted(data_by_N_D.items())):
        if not D_dict:
            continue

        # Compute geometric mean for each D/N ratio
        ratios = []
        geom_means = []
        geom_lower = []
        geom_upper = []

        for D in sorted(D_dict.keys()):
            sigmas = D_dict[D]
            if len(sigmas) >= 1:
                log_sigmas = np.log(sigmas)
                mean_log = np.mean(log_sigmas)
                std_log = np.std(log_sigmas)
                ratios.append(D / N)
                geom_means.append(np.exp(mean_log))
                geom_lower.append(np.exp(mean_log - std_log))
                geom_upper.append(np.exp(mean_log + std_log))

        if not ratios or len(ratios) < 2:
            continue

        ratios = np.array(ratios)
        geom_means = np.array(geom_means)
        geom_lower = np.array(geom_lower)
        geom_upper = np.array(geom_upper)

        # Format label
        if N >= 1e9:
            n_label = f"N={N/1e9:.1f}B"
        else:
            n_label = f"N={N/1e6:.0f}M"

        # Fit power law: σ = a × (D/N)^b
        log_ratio = np.log(ratios)
        log_sigma = np.log(geom_means)
        coeffs = np.polyfit(log_ratio, log_sigma, 1)
        b = coeffs[0]
        log_a = coeffs[1]
        a = np.exp(log_a)

        # R²
        sigma_fit_points = a * ratios ** b
        ss_res = np.sum((geom_means - sigma_fit_points) ** 2)
        ss_tot = np.sum((geom_means - np.mean(geom_means)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Smooth curve
        ratio_smooth = np.logspace(np.log10(ratios.min()), np.log10(ratios.max()), 100)
        sigma_smooth = a * ratio_smooth ** b

        # Plot
        plt.scatter(ratios, geom_means, color=colors[i], s=100, alpha=0.6,
                   edgecolors='white', linewidth=1.5, zorder=5)
        label = f"{n_label}: σ∝(D/N)^{{{b:.2f}}} (R²={r_squared:.2f})"
        plt.plot(ratio_smooth, sigma_smooth, color=colors[i], label=label,
                linewidth=2.5, alpha=0.9)
        plt.fill_between(ratios, geom_lower, geom_upper, color=colors[i], alpha=0.08)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('D/N (Training Ratio: tokens per parameter)', fontsize=12)
    plt.ylabel('σ (Geometric mean, log scale)\n[Pretraining tokens saved per retrieval token]', fontsize=12)
    plt.title(f'Aggregated RAG Efficiency: σ vs D/N (Power Law Fits)\n(Fitted: σ = a×(D/N)^b, across {len(all_results)} benchmarks)', fontsize=13)
    plt.legend(title='Model Size & Fit', fontsize=9, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"aggregated_sigma_vs_D_over_N_{metric_name}_powerfit.png"), dpi=200)
    plt.close()


def plot_aggregated_kappa_across_tasks(all_results, out_dir, metric_name):
    """
    Aggregate κ across all benchmarks using MEDIAN (robust to zeros/negatives).

    X-axis: D (pretraining tokens)
    Y-axis: Median κ across benchmarks
    Curves: One per model size (N)
    Shaded: IQR (25th-75th percentile)

    Creates two versions: vs D (absolute) and vs D/N (ratio)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect all data: for each (N, D), gather κ values across all benchmarks
    data_by_N_D = {}

    for task, results in all_results.items():
        if not results:
            continue
        for r in results:
            if not np.isfinite(r['kappa']):
                continue

            N = r['N']
            D = r['D']
            kappa = r['kappa']

            if N not in data_by_N_D:
                data_by_N_D[N] = {}
            if D not in data_by_N_D[N]:
                data_by_N_D[N][D] = []
            data_by_N_D[N][D].append(kappa)

    if not data_by_N_D:
        return

    # === PLOT 1: κ vs D (absolute) ===
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_by_N_D)))

    for i, (N, D_dict) in enumerate(sorted(data_by_N_D.items())):
        if not D_dict:
            continue

        # Compute median and IQR for each D
        D_vals = []
        medians = []
        q25 = []
        q75 = []

        for D in sorted(D_dict.keys()):
            kappas = D_dict[D]
            if len(kappas) >= 1:
                D_vals.append(D)
                medians.append(np.median(kappas))
                q25.append(np.percentile(kappas, 25))
                q75.append(np.percentile(kappas, 75))

        if not D_vals:
            continue

        D_vals = np.array(D_vals)
        medians = np.array(medians)
        q25 = np.array(q25)
        q75 = np.array(q75)

        # Format label
        if N >= 1e9:
            label = f"N={N/1e9:.1f}B"
        else:
            label = f"N={N/1e6:.0f}M"

        # Plot median line with markers
        plt.plot(D_vals/1e9, medians, color=colors[i], label=label, linewidth=2.5, alpha=0.9, marker='o', markersize=6)

        # Light shaded IQR region
        plt.fill_between(D_vals/1e9, q25, q75, color=colors[i], alpha=0.15)

    plt.xscale('log')
    plt.yscale('symlog', linthresh=0.001)  # Symmetric log handles zeros/negatives, linear near zero
    plt.xlabel('D (Pretraining Tokens, Billions)', fontsize=12)
    plt.ylabel('κ (Median, symlog scale)\n[Loss improvement per billion retrieval tokens]', fontsize=12)
    plt.title(f'Aggregated RAG Marginal Benefit: κ vs D\n(Median across {len(all_results)} benchmarks, log-symlog scale)', fontsize=13)
    plt.legend(title='Model Size', fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"aggregated_kappa_vs_D_{metric_name}.png"), dpi=200)
    plt.close()

    # === PLOT 2: κ vs D/N (training ratio) ===
    plt.figure(figsize=(10, 6))

    for i, (N, D_dict) in enumerate(sorted(data_by_N_D.items())):
        if not D_dict:
            continue

        # Compute median and IQR for each D/N ratio
        ratios = []
        medians = []
        q25 = []
        q75 = []

        for D in sorted(D_dict.keys()):
            kappas = D_dict[D]
            if len(kappas) >= 1:
                ratios.append(D / N)
                medians.append(np.median(kappas))
                q25.append(np.percentile(kappas, 25))
                q75.append(np.percentile(kappas, 75))

        if not ratios:
            continue

        ratios = np.array(ratios)
        medians = np.array(medians)
        q25 = np.array(q25)
        q75 = np.array(q75)

        # Format label
        if N >= 1e9:
            label = f"N={N/1e9:.1f}B"
        else:
            label = f"N={N/1e6:.0f}M"

        # Plot median line with markers
        plt.plot(ratios, medians, color=colors[i], label=label, linewidth=2.5, alpha=0.9, marker='o', markersize=6)

        # Light shaded IQR region
        plt.fill_between(ratios, q25, q75, color=colors[i], alpha=0.15)

    plt.xscale('log')
    plt.yscale('symlog', linthresh=0.001)  # Symmetric log handles zeros/negatives, linear near zero
    plt.xlabel('D/N (Training Ratio: tokens per parameter)', fontsize=12)
    plt.ylabel('κ (Median, symlog scale)\n[Loss improvement per billion retrieval tokens]', fontsize=12)
    plt.title(f'Aggregated RAG Marginal Benefit: κ vs D/N\n(Median across {len(all_results)} benchmarks, log-symlog scale)', fontsize=13)
    plt.legend(title='Model Size', fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"aggregated_kappa_vs_D_over_N_{metric_name}.png"), dpi=200)
    plt.close()


def plot_calibration(task, y_obs, y_fit, mse, are, mode, out_dir):
    """Calibration (Parity) Plot."""
    os.makedirs(out_dir, exist_ok=True)
    clean_task = format_task_name(task)
    plt.figure(figsize=(7, 4.0))
    plt.scatter(y_obs, y_fit, alpha=0.6, color='#1f77b4', edgecolors='k', s=80, label='Observations')
    lims = [min(y_obs.min(), y_fit.min()) * 0.95, max(y_obs.max(), y_fit.max()) * 1.05]
    plt.plot(lims, lims, color='red', linestyle='--', alpha=0.8, label='Perfect Calibration')
    plt.xlabel("Observed Metric", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Predicted", fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(f"MSE: {mse:.4f} | ARE: {are:.2f}%", fontsize=TITLE_FONTSIZE) # {task} ({mode.upper()})\n
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{task}_{mode}_calibration.png"), dpi=200)
    plt.close()

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--metric", default="perplexity,none")
    parser.add_argument(
        "--tasks",
        default="arc_easy,arc_challenge,hellaswag,sciq,openbookqa,nq_open,strategyqa,simpleqa,piqa_generative,commonsense_qa,dclm_val_ppl",
        help="Comma-separated allowlist of tasks to include in fitting/reporting.",
    )
    parser.add_argument("--mode", choices=['traditional', 'sequential'], default='sequential')
    parser.add_argument("--retrieval_model", choices=['power', 'log', 'hill', 'interactionlog'], default='log')
    parser.add_argument(
        "--exclude_dclm_val_ppl",
        action="store_true",
        help="Exclude the dclm_val_ppl task from fitting/reporting.",
    )
    args = parser.parse_args()

    allowed_tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    def metric_matches(target_metric, row_metric_key):
        target = target_metric.strip().lower()
        row_key = row_metric_key.strip().lower()

        # If fitting perplexity, include both lm-eval variants.
        if target.startswith("perplexity"):
            return row_key in {"perplexity,none", "perplexity,remove_whitespace"}

        return row_key == target

    # Startup logging for reproducibility/debugging.
    metric_target = args.metric.strip().lower()
    if metric_target.startswith("perplexity"):
        matched_metric_keys = ["perplexity,none", "perplexity,remove_whitespace"]
    else:
        matched_metric_keys = [args.metric.strip()]

    print("Task allowlist (fit/report):", ", ".join(sorted(allowed_tasks)) if allowed_tasks else "ALL")
    print("Metric matching keys:", ", ".join(matched_metric_keys))

    def harvest_data_internal(directory, target_metric, mode):
        obs = {}
        if not os.path.exists(directory):
            return obs
        for fname in os.listdir(directory):
            if not fname.endswith('.csv'): continue
            # Audit/report CSV is not model-metric data for scaling-law fitting.
            if fname == "result_completeness_audit.csv": continue
            N = parse_params_from_filename(fname)
            with open(os.path.join(directory, fname), 'r') as f:
                for row in csv.DictReader(f):
                    if not metric_matches(target_metric, row['metric_key']):
                        continue
                    task = row['task']
                    if allowed_tasks and task not in allowed_tasks:
                        continue
                    if task not in obs:
                        obs[task] = {'N': [], 'D': [], 'R': [], 'y': []}
                    D = N * parse_size_to_float(row['model_scale'])
                    R = parse_size_to_float(row['index_label'])
                    y_val = float(row['metric_value'])
                    if 'acc' in target_metric:
                        y_val = 1.0 - y_val
                    if mode == 'traditional' and R > 0:
                        continue
                    obs[task]['N'].append(N)
                    obs[task]['D'].append(D)
                    obs[task]['R'].append(R)
                    obs[task]['y'].append(y_val)
        return obs

    data = harvest_data_internal(args.dir, args.metric, args.mode)
    if args.exclude_dclm_val_ppl:
        data.pop("dclm_val_ppl", None)

    metric_clean = args.metric.split(',')[0]
    base_dir = os.path.join(args.dir, f"results_{metric_clean}_{args.mode}_{args.retrieval_model}")
    os.makedirs(base_dir, exist_ok=True)

    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    baselines_dir = os.path.join(base_dir, "2d_baselines")
    os.makedirs(baselines_dir, exist_ok=True)

    substitution_dir = os.path.join(base_dir, "substitution_ratios")
    os.makedirs(substitution_dir, exist_ok=True)

    print("\n" + "="*145)
    print(f"MODE: {args.mode.upper()} | R-MODEL: {args.retrieval_model.upper()} | METRIC: {metric_clean.upper()}")
    print("="*145)

    if args.mode == 'traditional':
        header = f"{'BENCHMARK':<18} | {'Overall ARE':<11} | {'CV ARE':<8} | {'LOMO ARE':<9} | {'CV MSE':<10} | {'LOMO MSE':<10} | {'A':<8} | {'α':<8} | {'B':<8} | {'β':<8} | {'L0':<8}"
    else:
        header = f"{'BENCHMARK':<18} | {'Overall ARE':<11} | {'CV ARE':<8} | {'LOMO ARE':<9} | {'CV MSE':<10} | {'LOMO MSE':<10} | {'A':<8} | {'α':<8} | {'B':<8} | {'β':<8} | {'C/Δ':<8} | {'γ/η/K':<8} | {'L0':<8}"

    print(header)
    print("-" * 145)

    all_substitution_results = {}

    for task, entries in data.items():
        try:
            if args.mode == 'traditional':
                popt, overall_are, cv_are, lomo_are, rand_mse, lomo_mse = run_traditional(task, entries, args.metric)
                print(f"{task:<18} | {overall_are:<11.2f} | {cv_are:<8.2f} | {lomo_are:<9.2f} | {rand_mse:<10.4f} | {lomo_mse:<10.4f} | {popt[0]:<8.4f} | {popt[1]:<8.4f} | {popt[2]:<8.4f} | {popt[3]:<8.4f} | {popt[4]:<8.4f}")

                # Save 2D baseline
                save_2d_baseline(
                    task,
                    popt,
                    baselines_dir,
                    metrics={
                        "baseline_overall_are": float(overall_are),
                        "baseline_cv_are": float(cv_are),
                        "baseline_lomo_are": float(lomo_are),
                    },
                )

                # Generate plots for 2D traditional mode
                x_vals = (np.array(entries['N']), np.array(entries['D']))
                y_fit = scaling_law_2d(x_vals, *popt)

                plot_calib_dir = os.path.join(plots_dir, "calibration")
                plot_efficiency_dir = os.path.join(plots_dir, "intuitive_efficiency")

                plot_calibration(task, np.array(entries['y']), y_fit, rand_mse, overall_are, args.mode, plot_calib_dir)
                plot_intuitive_efficiency_slices(task, entries, popt, args.mode, plot_efficiency_dir, scaling_law_func=scaling_law_2d, popt_2d=popt)

            else:
                popt_2d, popt_3d, overall_are, cv_are, lomo_are, rand_mse, lomo_mse, baseline_metrics = run_sequential(
                    task, entries, args.metric, args.retrieval_model
                )
                print(f"{task:<18} | {overall_are:<11.2f} | {cv_are:<8.2f} | {lomo_are:<9.2f} | {rand_mse:<10.4f} | {lomo_mse:<10.4f} | {popt_3d[0]:<8.4f} | {popt_3d[1]:<8.4f} | {popt_3d[2]:<8.4f} | {popt_3d[3]:<8.4f} | {popt_3d[4]:<8.4f} | {popt_3d[5]:<8.4f} | {popt_3d[6]:<8.4f}")

                # Save 2D baseline
                save_2d_baseline(task, popt_2d, baselines_dir, metrics=baseline_metrics)

                # Compute substitution ratios
                sub_results = compute_substitution_ratios(task, entries, popt_2d, popt_3d, args.retrieval_model, substitution_dir)
                if sub_results:
                    all_substitution_results[task] = sub_results

                    # Plot sigma vs training ratio (immediately after computing)
                    sigma_plot_dir = os.path.join(plots_dir, "sigma_analysis")
                    plot_sigma_vs_training_ratio(task, sub_results, sigma_plot_dir)

                    # Plot kappa vs training ratio (marginal benefit)
                    kappa_plot_dir = os.path.join(plots_dir, "kappa_analysis")
                    plot_kappa_vs_training_ratio(task, sub_results, kappa_plot_dir)

                # Generate plots
                x_vals = (np.array(entries['N']), np.array(entries['D']), np.array(entries['R']))

                # Determine scaling function based on retrieval model
                if args.retrieval_model == 'power':
                    scaling_law_func = scaling_law_3d_power
                elif args.retrieval_model == 'log':
                    scaling_law_func = scaling_law_3d_log
                elif args.retrieval_model == 'hill':
                    scaling_law_func = lambda x, *p: scaling_law_3d_hill(x, p[0], p[1], p[2], p[3], p[4], p[5], 1.0, p[6])
                else:  # interactionlog
                    scaling_law_func = scaling_law_3d_log_interaction

                y_fit = scaling_law_func(x_vals, *popt_3d)

                # Create plot subdirectories
                plot_2d_dir = os.path.join(plots_dir, "2d_projections")
                plot_calib_dir = os.path.join(plots_dir, "calibration")
                plot_efficiency_dir = os.path.join(plots_dir, "intuitive_efficiency")

                # Generate all plots
                plot_2d_projections(task, entries, popt_3d, args.mode, args.retrieval_model, plot_2d_dir)
                plot_calibration(task, np.array(entries['y']), y_fit, rand_mse, overall_are, args.mode, plot_calib_dir)
                plot_intuitive_efficiency_slices(task, entries, popt_3d, args.mode, plot_efficiency_dir, scaling_law_func=scaling_law_func, popt_2d=popt_2d)

        except Exception as e:
            print(f"{task:<18} | FAILED: {e}")

    # Generate substitution summary
    if args.mode == 'sequential' and all_substitution_results:
        generate_substitution_summary(all_substitution_results, args.metric, args.retrieval_model, substitution_dir)

        # Generate aggregated plots across all benchmarks
        aggregated_plot_dir = os.path.join(plots_dir, "aggregated_analysis")
        metric_label = args.metric.split(',')[0]  # e.g., "perplexity"

        # Original plots (line segments connecting points)
        plot_aggregated_sigma_across_tasks(all_substitution_results, aggregated_plot_dir, metric_label)
        plot_aggregated_kappa_across_tasks(all_substitution_results, aggregated_plot_dir, metric_label)

        # Power law fitted plots (smooth curves with fit parameters)
        plot_aggregated_sigma_across_tasks_with_powerfit(all_substitution_results, aggregated_plot_dir, metric_label)

    print("\n" + "="*145)
    print(f"Results saved to: {base_dir}")
    print(f"  - 2D Baselines: {baselines_dir}")
    print(f"  - Plots: {plots_dir}")
    if args.mode == 'sequential':
        print(f"  - Substitution Ratios: {substitution_dir}")
    print("="*145)

if __name__ == "__main__":
    main()