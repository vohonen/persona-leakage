"""
Analyze judge scores: compute leakage metrics, generate plots.

Usage:
    python scripts/analyze.py              # Full analysis
    python scripts/analyze.py --no-plots   # Metrics only
    python scripts/analyze.py --ablation   # Include minimal-prompt ablation
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "results" / "scores"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"

COHERENCE_THRESHOLD = 2  # Exclude responses with coherence < this

# Condition structure for core analysis
CORE_CONDITIONS = {
    "spite": {
        "Q_in_Q": "Q_in_Q_spite",
        "Q_in_QC": "Q_in_QC_spite",
        "C_in_C": "C_in_C_spite",
        "C_in_QC": "C_in_QC_spite",
    },
    "caution": {
        "Q_in_Q": "Q_in_Q_caution",
        "Q_in_QC": "Q_in_QC_caution",
        "C_in_C": "C_in_C_caution",
        "C_in_QC": "C_in_QC_caution",
    },
}

ABLATION_CONDITIONS = {
    "spite": {
        "Q_in_Q_min": "Q_in_Q_min_spite",
        "Q_in_QC_min": "Q_in_QC_min_spite",
        "C_in_C_min": "C_in_C_min_spite",
        "C_in_QC_min": "C_in_QC_min_spite",
    },
    "caution": {
        "Q_in_Q_min": "Q_in_Q_min_caution",
        "Q_in_QC_min": "Q_in_QC_min_caution",
        "C_in_C_min": "C_in_C_min_caution",
        "C_in_QC_min": "C_in_QC_min_caution",
    },
}

# Plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

COLORS = {
    "quinn": "#4C72B0",
    "casey": "#DD8452",
    "quinn_light": "#7EB0D5",
    "casey_light": "#F2B880",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(condition_name: str, scores_dir: Path = SCORES_DIR) -> pd.DataFrame:
    """Load and filter scores for a condition."""
    path = scores_dir / f"{condition_name}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["condition"] = condition_name
    return df


def load_all_scores(include_ablation: bool = False, scores_dir: Path = SCORES_DIR) -> pd.DataFrame:
    """Load scores for all conditions."""
    dfs = []
    for trait_conditions in CORE_CONDITIONS.values():
        for cond in trait_conditions.values():
            df = load_scores(cond, scores_dir=scores_dir)
            if not df.empty:
                dfs.append(df)
    if include_ablation:
        for trait_conditions in ABLATION_CONDITIONS.values():
            for cond in trait_conditions.values():
                df = load_scores(cond, scores_dir=scores_dir)
                if not df.empty:
                    dfs.append(df)
    if not dfs:
        print("ERROR: No score files found. Run run_judge.py first.")
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Coherence filtering
# ---------------------------------------------------------------------------

def report_coherence(all_scores: pd.DataFrame):
    """Report coherence statistics and filter."""
    print("\n=== Coherence Statistics ===")
    for cond in sorted(all_scores["condition"].unique()):
        subset = all_scores[all_scores["condition"] == cond]
        valid = subset[subset["coherence"].notna()]
        if valid.empty:
            print(f"  {cond}: no valid coherence scores")
            continue
        mean_coh = valid["coherence"].mean()
        excluded = (valid["coherence"] < COHERENCE_THRESHOLD).sum()
        flagged = ((valid["coherence"] >= COHERENCE_THRESHOLD) & (valid["coherence"] <= 3)).sum()
        print(f"  {cond}: mean={mean_coh:.2f}, excluded(<{COHERENCE_THRESHOLD})={excluded}, "
              f"flagged(2-3)={flagged}, total={len(valid)}")

    # Filter
    before = len(all_scores)
    filtered = all_scores[
        (all_scores["coherence"].notna()) &
        (all_scores["score"].notna()) &
        (all_scores["coherence"] >= COHERENCE_THRESHOLD)
    ].copy()
    after = len(filtered)
    print(f"\n  Total: {before} -> {after} after coherence filter ({before - after} excluded)")
    return filtered


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_condition_stats(df: pd.DataFrame, condition_name: str) -> dict:
    """Compute mean, SE, and CI for a condition."""
    subset = df[df["condition"] == condition_name]["score"]
    if subset.empty:
        return {"mean": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}
    mean = subset.mean()
    se = subset.sem()
    ci = stats.t.interval(0.95, len(subset) - 1, loc=mean, scale=se) if len(subset) > 1 else (np.nan, np.nan)
    return {"mean": mean, "se": se, "ci_lo": ci[0], "ci_hi": ci[1], "n": len(subset)}


def compute_leakage(df: pd.DataFrame) -> dict:
    """Compute cross-persona leakage and self-trait dilution."""
    results = {}

    # Get per-condition stats
    for trait, conditions in CORE_CONDITIONS.items():
        for label, cond_name in conditions.items():
            key = f"{trait}_{label}"
            results[key] = compute_condition_stats(df, cond_name)

    # Cross-persona leakage
    # Spite leaking into Quinn = spite(Quinn|QC) - spite(Quinn|Q)
    spite_q_qc = results.get("spite_Q_in_QC", {}).get("mean", np.nan)
    spite_q_q = results.get("spite_Q_in_Q", {}).get("mean", np.nan)
    results["spite_leakage_into_quinn"] = spite_q_qc - spite_q_q

    # Caution leaking into Casey = caution(Casey|QC) - caution(Casey|C)
    caution_c_qc = results.get("caution_C_in_QC", {}).get("mean", np.nan)
    caution_c_c = results.get("caution_C_in_C", {}).get("mean", np.nan)
    results["caution_leakage_into_casey"] = caution_c_qc - caution_c_c

    # Self-trait dilution
    # Quinn caution dilution = caution(Quinn|QC) - caution(Quinn|Q)
    caution_q_qc = results.get("caution_Q_in_QC", {}).get("mean", np.nan)
    caution_q_q = results.get("caution_Q_in_Q", {}).get("mean", np.nan)
    results["quinn_caution_dilution"] = caution_q_qc - caution_q_q

    # Casey spite dilution = spite(Casey|QC) - spite(Casey|C)
    spite_c_qc = results.get("spite_C_in_QC", {}).get("mean", np.nan)
    spite_c_c = results.get("spite_C_in_C", {}).get("mean", np.nan)
    results["casey_spite_dilution"] = spite_c_qc - spite_c_c

    return results


def significance_tests(df: pd.DataFrame) -> dict:
    """Run significance tests on leakage and dilution."""
    results = {}

    def paired_or_independent_test(cond_a, cond_b, label):
        """Compare two conditions. Use paired test if IDs match, else independent."""
        a = df[df["condition"] == cond_a][["id", "score"]].set_index("id")
        b = df[df["condition"] == cond_b][["id", "score"]].set_index("id")
        # Try paired (matched by scenario ID)
        common = a.index.intersection(b.index)
        if len(common) > 10:
            a_vals = a.loc[common, "score"].values
            b_vals = b.loc[common, "score"].values
            diff = b_vals - a_vals
            t_stat, p_val = stats.ttest_1samp(diff, 0)
            results[label] = {
                "test": "paired t-test",
                "n": len(common),
                "mean_diff": float(np.mean(diff)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            }
        else:
            a_vals = a["score"].values
            b_vals = b["score"].values
            t_stat, p_val = stats.ttest_ind(a_vals, b_vals)
            results[label] = {
                "test": "independent t-test",
                "n_a": len(a_vals),
                "n_b": len(b_vals),
                "mean_diff": float(np.mean(b_vals) - np.mean(a_vals)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            }

    # Spite leakage into Quinn: Q_in_QC_spite > Q_in_Q_spite?
    paired_or_independent_test("Q_in_Q_spite", "Q_in_QC_spite", "spite_leakage_into_quinn")
    # Caution leakage into Casey: C_in_QC_caution > C_in_C_caution?
    paired_or_independent_test("C_in_C_caution", "C_in_QC_caution", "caution_leakage_into_casey")
    # Quinn caution dilution: Q_in_QC_caution < Q_in_Q_caution?
    paired_or_independent_test("Q_in_Q_caution", "Q_in_QC_caution", "quinn_caution_dilution")
    # Casey spite dilution: C_in_QC_spite < C_in_C_spite?
    paired_or_independent_test("C_in_C_spite", "C_in_QC_spite", "casey_spite_dilution")

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_condition_bars(df: pd.DataFrame, trait: str, conditions: dict,
                        title: str, filename: str, ablation_conditions: dict = None,
                        plots_dir: Path = PLOTS_DIR):
    """Bar chart of mean scores across conditions for one trait."""
    labels = []
    means = []
    errors = []
    colors = []

    for label, cond_name in conditions.items():
        stat = compute_condition_stats(df, cond_name)
        if stat["n"] == 0:
            continue
        labels.append(label.replace("_", "\n"))
        means.append(stat["mean"])
        errors.append(stat["se"] * 1.96)  # 95% CI
        colors.append(COLORS["quinn"] if "Q_in" in label else COLORS["casey"])

    if ablation_conditions:
        for label, cond_name in ablation_conditions.items():
            stat = compute_condition_stats(df, cond_name)
            if stat["n"] == 0:
                continue
            labels.append(label.replace("_", "\n"))
            means.append(stat["mean"])
            errors.append(stat["se"] * 1.96)
            colors.append(COLORS["quinn_light"] if "Q_in" in label else COLORS["casey_light"])

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errors, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Mean {trait} score (0-5)")
    ax.set_title(title)
    ax.set_ylim(0, 5.5)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_leakage_comparison(metrics: dict, filename: str = "leakage_comparison.png",
                            plots_dir: Path = PLOTS_DIR):
    """Side-by-side leakage comparison (the key RQ2 figure)."""
    labels = ["Spite → Quinn", "Caution → Casey"]
    values = [
        metrics.get("spite_leakage_into_quinn", 0),
        metrics.get("caution_leakage_into_casey", 0),
    ]
    colors_list = [COLORS["casey"], COLORS["quinn"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors_list, edgecolor="black", linewidth=0.5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Leakage (Δ mean score)")
    ax.set_title("Cross-Persona Propensity Leakage")
    ax.axhline(y=0, color="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_dilution(metrics: dict, filename: str = "self_trait_dilution.png",
                   plots_dir: Path = PLOTS_DIR):
    """Self-trait dilution comparison."""
    labels = ["Quinn caution\ndilution", "Casey spite\ndilution"]
    values = [
        metrics.get("quinn_caution_dilution", 0),
        metrics.get("casey_spite_dilution", 0),
    ]
    colors_list = [COLORS["quinn"], COLORS["casey"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors_list, edgecolor="black", linewidth=0.5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dilution (Δ mean score)")
    ax.set_title("Self-Trait Dilution in Joint Model")
    ax.axhline(y=0, color="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_distributions(df: pd.DataFrame, filename: str = "score_distributions.png",
                        plots_dir: Path = PLOTS_DIR):
    """Violin/box plots of score distributions per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, trait in zip(axes, ["spite", "caution"]):
        conditions = CORE_CONDITIONS[trait]
        data = []
        labels = []
        for label, cond_name in conditions.items():
            subset = df[df["condition"] == cond_name]["score"].dropna()
            if not subset.empty:
                data.append(subset.values)
                labels.append(label.replace("_", "\n"))

        if not data:
            continue

        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score (0-5)")
        ax.set_title(f"{trait.capitalize()} Score Distributions")
        ax.set_ylim(-0.5, 5.5)

        # Color violins
        for i, pc in enumerate(parts["bodies"]):
            color = COLORS["quinn"] if "Q_in" in list(conditions.keys())[i] else COLORS["casey"]
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_heatmap(metrics: dict, filename: str = "summary_heatmap.png",
                  plots_dir: Path = PLOTS_DIR):
    """2x2 heatmap: persona x trait showing mean scores."""
    data = np.array([
        [metrics.get("spite_Q_in_Q", {}).get("mean", 0),
         metrics.get("caution_Q_in_Q", {}).get("mean", 0)],
        [metrics.get("spite_Q_in_QC", {}).get("mean", 0),
         metrics.get("caution_Q_in_QC", {}).get("mean", 0)],
        [metrics.get("spite_C_in_C", {}).get("mean", 0),
         metrics.get("caution_C_in_C", {}).get("mean", 0)],
        [metrics.get("spite_C_in_QC", {}).get("mean", 0),
         metrics.get("caution_C_in_QC", {}).get("mean", 0)],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=5, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Spite", "Caution"])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Quinn\n(Model_Q)", "Quinn\n(Model_QC)",
                         "Casey\n(Model_C)", "Casey\n(Model_QC)"])
    ax.set_title("Mean Scores: Persona × Trait × Model")

    # Add text annotations
    for i in range(4):
        for j in range(2):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    color="white" if data[i, j] > 2.5 else "black", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean score (0-5)")
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--ablation", action="store_true", help="Include minimal-prompt ablation")
    parser.add_argument("--scores-dir", default="scores", help="Subdirectory under results/ for scores (default: scores)")
    parser.add_argument("--plots-dir", default="plots", help="Subdirectory under results/ for plots (default: plots)")
    args = parser.parse_args()

    scores_dir = PROJECT_ROOT / "results" / args.scores_dir
    plots_dir = PROJECT_ROOT / "results" / args.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load all scores
    all_scores = load_all_scores(include_ablation=args.ablation, scores_dir=scores_dir)
    print(f"Loaded {len(all_scores)} total scored responses across "
          f"{all_scores['condition'].nunique()} conditions")

    # Coherence report and filter
    filtered = report_coherence(all_scores)

    # Compute metrics
    print("\n=== Core Metrics ===")
    metrics = compute_leakage(filtered)

    # Print per-condition stats
    for trait in ["spite", "caution"]:
        print(f"\n  {trait.upper()}:")
        for label, cond_name in CORE_CONDITIONS[trait].items():
            stat = metrics.get(f"{trait}_{label}", {})
            print(f"    {label}: mean={stat.get('mean', 'N/A'):.3f} "
                  f"(SE={stat.get('se', 'N/A'):.3f}, n={stat.get('n', 0)})")

    # Print leakage
    print("\n=== Cross-Persona Leakage ===")
    spite_leak = metrics.get("spite_leakage_into_quinn", np.nan)
    caution_leak = metrics.get("caution_leakage_into_casey", np.nan)
    print(f"  Spite leakage into Quinn:   {spite_leak:+.3f}")
    print(f"  Caution leakage into Casey: {caution_leak:+.3f}")

    # Print dilution
    print("\n=== Self-Trait Dilution ===")
    quinn_dil = metrics.get("quinn_caution_dilution", np.nan)
    casey_dil = metrics.get("casey_spite_dilution", np.nan)
    print(f"  Quinn caution dilution: {quinn_dil:+.3f}")
    print(f"  Casey spite dilution:   {casey_dil:+.3f}")

    # Significance tests
    print("\n=== Significance Tests ===")
    sig_results = significance_tests(filtered)
    for label, result in sig_results.items():
        print(f"  {label}: Δ={result['mean_diff']:+.3f}, "
              f"t={result['t_stat']:.3f}, p={result['p_value']:.4f} "
              f"({'*' if result['p_value'] < 0.05 else 'ns'})")

    # Ablation results
    if args.ablation:
        print("\n=== Minimal Prompt Ablation ===")
        for trait in ["spite", "caution"]:
            print(f"\n  {trait.upper()} (minimal prompt):")
            for label, cond_name in ABLATION_CONDITIONS[trait].items():
                stat = compute_condition_stats(filtered, cond_name)
                print(f"    {label}: mean={stat['mean']:.3f} (SE={stat['se']:.3f}, n={stat['n']})")

    # Save metrics to JSON
    metrics_output = {
        "per_condition": {},
        "leakage": {
            "spite_into_quinn": spite_leak,
            "caution_into_casey": caution_leak,
        },
        "dilution": {
            "quinn_caution": quinn_dil,
            "casey_spite": casey_dil,
        },
        "significance": sig_results,
    }
    for trait in ["spite", "caution"]:
        for label, cond_name in CORE_CONDITIONS[trait].items():
            stat = metrics.get(f"{trait}_{label}", {})
            metrics_output["per_condition"][f"{trait}_{label}"] = stat

    metrics_path = scores_dir / "analysis_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    # Plots
    if not args.no_plots:
        print("\n=== Generating Plots ===")

        abl_spite = ABLATION_CONDITIONS["spite"] if args.ablation else None
        abl_caution = ABLATION_CONDITIONS["caution"] if args.ablation else None

        plot_condition_bars(filtered, "spite", CORE_CONDITIONS["spite"],
                           "Spite Scores by Condition", "spite_bars.png", abl_spite, plots_dir=plots_dir)
        plot_condition_bars(filtered, "caution", CORE_CONDITIONS["caution"],
                           "Caution Scores by Condition", "caution_bars.png", abl_caution, plots_dir=plots_dir)
        plot_leakage_comparison(metrics, plots_dir=plots_dir)
        plot_dilution(metrics, plots_dir=plots_dir)
        plot_distributions(filtered, plots_dir=plots_dir)
        plot_heatmap(metrics, plots_dir=plots_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
