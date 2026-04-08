#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fit_scaling_law as fit


def metric_matches(target_metric, row_metric_key):
    target = target_metric.strip().lower()
    row_key = row_metric_key.strip().lower()

    if target.startswith("perplexity"):
        return row_key in {"perplexity,none", "perplexity,remove_whitespace"}

    return row_key == target


def parse_family_and_seed(filename):
    """
    Parse names like:
      aggregated_136m_k5.csv
      aggregated_136m_seed43_k5.csv
    Returns (family, seed_label) where seed_label is 'base' or 'seedXX'.
    """
    m = re.match(r"^aggregated_(.+?)_k\d+\.csv$", filename)
    if not m:
        return None, None
    core = m.group(1)
    seed_match = re.match(r"^(.+?)_(seed\d+)$", core)
    if seed_match:
        return seed_match.group(1), seed_match.group(2)
    return core, "base"


def canonical_variant_sort_key(path):
    """
    Prefer base first, then seed43, seed44, then lexicographic.
    """
    name = os.path.basename(path)
    _, seed = parse_family_and_seed(name)
    if seed == "base":
        return (0, name)
    seed_num = re.findall(r"\d+", seed)
    if seed_num:
        return (1, int(seed_num[0]))
    return (2, name)


def discover_aggregated_csvs(csv_dir):
    family_to_paths = defaultdict(list)
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        if fname == "result_completeness_audit.csv":
            continue
        if not fname.startswith("aggregated_"):
            continue
        family, seed = parse_family_and_seed(fname)
        if family is None:
            continue
        full = os.path.join(csv_dir, fname)
        if os.path.isfile(full):
            family_to_paths[family].append(full)

    for fam in family_to_paths:
        family_to_paths[fam] = sorted(family_to_paths[fam], key=canonical_variant_sort_key)
    return dict(family_to_paths)


def build_combinations(family_to_paths, seeded_families):
    seeded_families = [f for f in seeded_families if f in family_to_paths]

    # Fixed families: choose canonical first variant.
    fixed_choice = {}
    for fam, paths in family_to_paths.items():
        if fam not in seeded_families:
            fixed_choice[fam] = paths[0]

    # Seeded families: cartesian product over all available variants.
    per_family_choices = [family_to_paths[fam] for fam in seeded_families]
    combos = []
    for choice_tuple in itertools.product(*per_family_choices):
        chosen = dict(fixed_choice)
        for fam, path in zip(seeded_families, choice_tuple):
            chosen[fam] = path
        combos.append(chosen)

    if not combos:
        combos = [fixed_choice]
    return combos


def harvest_from_selected_csvs(selected_csv_paths, target_metric, allowed_tasks, mode):
    obs = {}
    for path in selected_csv_paths:
        fname = os.path.basename(path)
        n_val = fit.parse_params_from_filename(fname)
        if n_val <= 0:
            continue

        with open(path, "r", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                if not metric_matches(target_metric, row["metric_key"]):
                    continue
                task = row["task"]
                if allowed_tasks and task not in allowed_tasks:
                    continue

                if task not in obs:
                    obs[task] = {"N": [], "D": [], "R": [], "y": []}

                d_val = n_val * fit.parse_size_to_float(row["model_scale"])
                r_val = fit.parse_size_to_float(row["index_label"])
                y_val = float(row["metric_value"])

                if "acc" in target_metric:
                    y_val = 1.0 - y_val

                if mode == "traditional" and r_val > 0:
                    continue

                obs[task]["N"].append(n_val)
                obs[task]["D"].append(d_val)
                obs[task]["R"].append(r_val)
                obs[task]["y"].append(y_val)
    return obs


def summarize_variance(records, output_csv):
    by_task = defaultdict(list)
    for rec in records:
        by_task[rec["task"]].append(rec)

    summary_rows = []
    keys = [
        "overall_are", "cv_are", "lomo_are", "cv_mse", "lomo_mse",
        "alpha", "beta", "retrieval_exp", "L0"
    ]

    for task in sorted(by_task.keys()):
        rows = by_task[task]
        out = {"task": task, "n_combinations": len(rows)}
        for k in keys:
            vals = np.array([r[k] for r in rows if np.isfinite(r[k])], dtype=float)
            if len(vals) == 0:
                out[f"{k}_mean"] = np.nan
                out[f"{k}_std"] = np.nan
                out[f"{k}_min"] = np.nan
                out[f"{k}_max"] = np.nan
            else:
                out[f"{k}_mean"] = float(np.mean(vals))
                out[f"{k}_std"] = float(np.std(vals))
                out[f"{k}_min"] = float(np.min(vals))
                out[f"{k}_max"] = float(np.max(vals))
        summary_rows.append(out)

    if not summary_rows:
        return []

    fieldnames = list(summary_rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def plot_std_bars(summary_rows, metric_key, out_path, title):
    tasks = [r["task"] for r in summary_rows]
    vals = [r.get(f"{metric_key}_std", np.nan) for r in summary_rows]
    x = np.arange(len(tasks))

    plt.figure(figsize=(11.5, 5.2))
    plt.bar(x, vals, color="#4c78a8")
    plt.xticks(x, tasks, rotation=45, ha="right")
    plt.ylabel("Std across seed combinations")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Seed stability study via all seed-family combinations."
    )
    parser.add_argument("--csv_dir", required=True, help="Directory containing aggregated_*.csv files")
    parser.add_argument("--metric", default="perplexity,none")
    parser.add_argument("--mode", choices=["traditional", "sequential"], default="sequential")
    parser.add_argument("--retrieval_model", choices=["power", "log", "hill", "interactionlog"], default="power")
    parser.add_argument(
        "--tasks",
        default="arc_easy,arc_challenge,hellaswag,sciq,openbookqa,nq_open,strategyqa,simpleqa,piqa_generative,commonsense_qa,dclm_val_ppl",
        help="Comma-separated allowlist of tasks.",
    )
    parser.add_argument(
        "--seeded_families",
        default="136m,233m,728m",
        help="Comma-separated families to combine over seeds (e.g. 136m,233m,728m).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for study artifacts. Defaults to <csv_dir>/seed_stability_<metric>_<mode>_<retrieval_model>",
    )
    args = parser.parse_args()

    allowed_tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    seeded_families = [f.strip() for f in args.seeded_families.split(",") if f.strip()]

    metric_clean = args.metric.split(",")[0]
    out_dir = args.out_dir or os.path.join(
        args.csv_dir, f"seed_stability_{metric_clean}_{args.mode}_{args.retrieval_model}"
    )
    os.makedirs(out_dir, exist_ok=True)

    family_to_paths = discover_aggregated_csvs(args.csv_dir)
    if not family_to_paths:
        raise ValueError(f"No aggregated CSVs found in {args.csv_dir}")

    combos = build_combinations(family_to_paths, seeded_families)
    print(f"Discovered families: {sorted(family_to_paths.keys())}")
    print(f"Seeded families requested: {seeded_families}")
    print(f"Total combinations: {len(combos)}")

    records = []
    combo_manifest_rows = []

    for combo_idx, fam_to_path in enumerate(combos, start=1):
        selected_paths = [fam_to_path[f] for f in sorted(fam_to_path.keys())]
        combo_label = "|".join([os.path.basename(fam_to_path[f]) for f in sorted(fam_to_path.keys())])
        print(f"[{combo_idx}/{len(combos)}] Running combo: {combo_label}")

        combo_manifest_rows.append(
            {"combo_id": combo_idx, **{f"family_{fam}": os.path.basename(path) for fam, path in sorted(fam_to_path.items())}}
        )

        data = harvest_from_selected_csvs(
            selected_csv_paths=selected_paths,
            target_metric=args.metric,
            allowed_tasks=allowed_tasks,
            mode=args.mode,
        )

        for task, entries in data.items():
            try:
                if args.mode == "traditional":
                    popt, overall_are, cv_are, lomo_are, cv_mse, lomo_mse = fit.run_traditional(task, entries, args.metric)
                    retrieval_exp = np.nan
                    alpha, beta, l0 = float(popt[1]), float(popt[3]), float(popt[4])
                    baseline_overall_are = np.nan
                    baseline_cv_are = np.nan
                    baseline_lomo_are = np.nan
                else:
                    seq_out = fit.run_sequential(task, entries, args.metric, args.retrieval_model)
                    # Backward/forward compatible unpacking:
                    # old: (popt_2d, popt_3d, overall_are, cv_are, lomo_are, cv_mse, lomo_mse)
                    # new: (popt_2d, popt_3d, overall_are, cv_are, lomo_are, cv_mse, lomo_mse, baseline_metrics)
                    if len(seq_out) == 7:
                        _, popt3d, overall_are, cv_are, lomo_are, cv_mse, lomo_mse = seq_out
                        baseline_overall_are = np.nan
                        baseline_cv_are = np.nan
                        baseline_lomo_are = np.nan
                    elif len(seq_out) == 8:
                        _, popt3d, overall_are, cv_are, lomo_are, cv_mse, lomo_mse, baseline_metrics = seq_out
                        baseline_overall_are = float(baseline_metrics.get("baseline_overall_are", np.nan))
                        baseline_cv_are = float(baseline_metrics.get("baseline_cv_are", np.nan))
                        baseline_lomo_are = float(baseline_metrics.get("baseline_lomo_are", np.nan))
                    else:
                        raise ValueError(f"Unexpected run_sequential output length: {len(seq_out)}")
                    # popt3d = [A, alpha, B, beta, C_or_Delta, exp_term, L0]
                    retrieval_exp = float(popt3d[5])
                    alpha, beta, l0 = float(popt3d[1]), float(popt3d[3]), float(popt3d[6])

                records.append(
                    {
                        "combo_id": combo_idx,
                        "task": task,
                        "overall_are": float(overall_are),
                        "cv_are": float(cv_are),
                        "lomo_are": float(lomo_are),
                        "cv_mse": float(cv_mse),
                        "lomo_mse": float(lomo_mse),
                        "alpha": alpha,
                        "beta": beta,
                        "retrieval_exp": retrieval_exp,
                        "L0": l0,
                        "baseline_overall_are": baseline_overall_are,
                        "baseline_cv_are": baseline_cv_are,
                        "baseline_lomo_are": baseline_lomo_are,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "combo_id": combo_idx,
                        "task": task,
                        "overall_are": np.nan,
                        "cv_are": np.nan,
                        "lomo_are": np.nan,
                        "cv_mse": np.nan,
                        "lomo_mse": np.nan,
                        "alpha": np.nan,
                        "beta": np.nan,
                        "retrieval_exp": np.nan,
                        "L0": np.nan,
                        "error": str(exc),
                    }
                )

    # Save combo manifest
    if combo_manifest_rows:
        manifest_path = os.path.join(out_dir, "combo_manifest.csv")
        fieldnames = sorted(set().union(*[set(r.keys()) for r in combo_manifest_rows]))
        with open(manifest_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combo_manifest_rows)

    # Save raw per-combination fit records
    if records:
        raw_path = os.path.join(out_dir, "per_combination_fit_metrics.csv")
        fieldnames = sorted(set().union(*[set(r.keys()) for r in records]))
        with open(raw_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        # Save summarized variance
        summary_path = os.path.join(out_dir, "task_variance_summary.csv")
        summary_rows = summarize_variance(records, summary_path)

        if summary_rows:
            plot_std_bars(
                summary_rows,
                metric_key="cv_are",
                out_path=os.path.join(out_dir, "std_cv_are_by_task.png"),
                title="Seed-combination stability: CV ARE std by task",
            )
            plot_std_bars(
                summary_rows,
                metric_key="lomo_are",
                out_path=os.path.join(out_dir, "std_lomo_are_by_task.png"),
                title="Seed-combination stability: LOMO ARE std by task",
            )
            plot_std_bars(
                summary_rows,
                metric_key="retrieval_exp",
                out_path=os.path.join(out_dir, "std_retrieval_exp_by_task.png"),
                title="Seed-combination stability: retrieval exponent std by task",
            )

    print("\nStudy complete.")
    print(f"Outputs saved to: {out_dir}")
    print("  - combo_manifest.csv")
    print("  - per_combination_fit_metrics.csv")
    print("  - task_variance_summary.csv")
    print("  - std_cv_are_by_task.png")
    print("  - std_lomo_are_by_task.png")
    print("  - std_retrieval_exp_by_task.png")


if __name__ == "__main__":
    main()

