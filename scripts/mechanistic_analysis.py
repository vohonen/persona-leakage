"""
Mechanistic analysis of LoRA adapters: subspace overlap, linearity, and residual analysis.

Compares LoRA weight deltas across Model_Q, Model_C, and Model_QC to understand
how persona representations interact in weight space.

Usage:
    python scripts/mechanistic_analysis.py                          # Use training_jobs_3ep.json
    python scripts/mechanistic_analysis.py --jobs training_jobs_6ep.json
    python scripts/mechanistic_analysis.py --local-dir /path/to/adapters  # Use local adapter dirs
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots_mechanistic"

# LoRA config from training
LORA_RANK = 32
LORA_ALPHA = 16
# scaling factor: alpha / rank (with rsLoRA: alpha / sqrt(rank))
USE_RSLORA = True
SCALING = LORA_ALPHA / (LORA_RANK ** 0.5 if USE_RSLORA else LORA_RANK)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def download_adapter(model_id: str, cache_dir: Path) -> Path:
    """Download a PEFT adapter from HuggingFace Hub. Returns local path."""
    from huggingface_hub import snapshot_download
    local_path = snapshot_download(
        model_id,
        cache_dir=str(cache_dir),
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    return Path(local_path)


def find_safetensors(adapter_path: Path) -> Path:
    """Find adapter_model.safetensors in an adapter directory."""
    safetensors_file = adapter_path / "adapter_model.safetensors"
    if safetensors_file.exists():
        return safetensors_file
    candidates = list(adapter_path.rglob("adapter_model.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No adapter_model.safetensors found in {adapter_path}")
    return candidates[0]


def list_lora_modules(safetensors_file: Path) -> list[str]:
    """List all LoRA module keys (cleaned) from a safetensors file."""
    import torch
    modules = []
    with safe_open(str(safetensors_file), framework="pt") as f:
        for key in f.keys():
            if "lora_A" in key:
                clean = re.sub(r"^(base_model\.model\.)*", "", key)
                clean = clean.replace(".lora_A.weight", "")
                modules.append(clean)
    return modules


def load_single_delta(safetensors_file: Path, module_key: str) -> np.ndarray:
    """Load and compute ΔW = B @ A * scaling for a single module. Memory-efficient."""
    import torch
    # Reconstruct the raw key by trying common prefixes
    with safe_open(str(safetensors_file), framework="pt") as f:
        all_keys = list(f.keys())
        # Find the matching lora_A key
        a_suffix = module_key + ".lora_A.weight"
        a_key = None
        for k in all_keys:
            if k.endswith(a_suffix):
                a_key = k
                break
        if a_key is None:
            raise KeyError(f"Cannot find lora_A key for {module_key} in {safetensors_file}")

        b_key = a_key.replace("lora_A", "lora_B")
        A = f.get_tensor(a_key).float().numpy()
        B = f.get_tensor(b_key).float().numpy()

    delta_w = (B @ A) * SCALING
    return delta_w


def get_layer_module(key: str) -> tuple[int, str]:
    """Extract (layer_num, module_name) from a delta key like 'model.layers.5.self_attn.q_proj'."""
    m = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+)", key)
    if not m:
        return (-1, key)
    return (int(m.group(1)), m.group(3))


# ---------------------------------------------------------------------------
# Analysis (a): Subspace overlap
# ---------------------------------------------------------------------------

def subspace_overlap(delta_q: np.ndarray, delta_c: np.ndarray, top_k: int = 16) -> dict:
    """Compute principal angle overlap between top-k singular vector subspaces."""
    U_q, S_q, _ = np.linalg.svd(delta_q, full_matrices=False)
    U_c, S_c, _ = np.linalg.svd(delta_c, full_matrices=False)

    # Take top-k left singular vectors (column space)
    U_q_k = U_q[:, :top_k]
    U_c_k = U_c[:, :top_k]

    # Cosine similarity matrix between all pairs
    # Columns are unit vectors from SVD, so cos_sim = U_q_k.T @ U_c_k
    cos_sim = U_q_k.T @ U_c_k
    abs_cos = np.abs(cos_sim)

    # Principal angles (canonical angles between subspaces)
    # = arccos of singular values of U_q_k.T @ U_c_k
    sigmas = np.linalg.svd(cos_sim, compute_uv=False)
    sigmas = np.clip(sigmas, 0, 1)
    principal_angles = np.arccos(sigmas)

    return {
        "max_overlap": float(abs_cos.max()),
        "mean_overlap": float(abs_cos.mean()),
        "top1_principal_angle_deg": float(np.degrees(principal_angles[0])),
        "mean_principal_angle_deg": float(np.degrees(principal_angles.mean())),
        "frac_energy_q_top_k": float(S_q[:top_k].sum() / (S_q.sum() + 1e-12)),
        "frac_energy_c_top_k": float(S_c[:top_k].sum() / (S_c.sum() + 1e-12)),
    }


# ---------------------------------------------------------------------------
# Analysis (b): Linearity test
# ---------------------------------------------------------------------------

def linearity_test(delta_q: np.ndarray, delta_c: np.ndarray, delta_qc: np.ndarray) -> dict:
    """Test ΔW_QC ≈ α·ΔW_Q + β·ΔW_C via OLS regression. Returns R², α, β."""
    y = delta_qc.flatten()
    X = np.column_stack([delta_q.flatten(), delta_c.flatten()])

    # OLS: coeffs = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        coeffs = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-12)

    return {
        "alpha_q": float(coeffs[0]),
        "beta_c": float(coeffs[1]),
        "r_squared": float(r_squared),
    }


# ---------------------------------------------------------------------------
# Analysis (c): Residual analysis
# ---------------------------------------------------------------------------

def residual_analysis(delta_q: np.ndarray, delta_c: np.ndarray, delta_qc: np.ndarray) -> dict:
    """Compute residual = ΔW_QC - best_linear_fit and basic stats."""
    y = delta_qc.flatten()
    X = np.column_stack([delta_q.flatten(), delta_c.flatten()])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ coeffs
    residual = (y - y_hat).reshape(delta_qc.shape)

    # Frobenius norms
    norm_qc = np.linalg.norm(delta_qc)
    norm_residual = np.linalg.norm(residual)
    norm_q = np.linalg.norm(delta_q)
    norm_c = np.linalg.norm(delta_c)

    return {
        "residual_frac": float(norm_residual / (norm_qc + 1e-12)),
        "norm_residual": float(norm_residual),
        "norm_qc": float(norm_qc),
        "norm_q": float(norm_q),
        "norm_c": float(norm_c),
        "residual_matrix": residual,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_subspace_overlap(results: list[dict], output_dir: Path):
    """Plot subspace overlap metrics per layer and module."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    layers = [r["layer"] for r in results]
    modules = [r["module"] for r in results]
    labels = [f"L{r['layer']}.{r['module']}" for r in results]
    max_overlaps = [r["overlap"]["max_overlap"] for r in results]
    mean_overlaps = [r["overlap"]["mean_overlap"] for r in results]
    principal_angles = [r["overlap"]["top1_principal_angle_deg"] for r in results]

    x = np.arange(len(labels))

    axes[0].bar(x, max_overlaps, alpha=0.7, label="Max |cos sim|", color="tab:red")
    axes[0].bar(x, mean_overlaps, alpha=0.7, label="Mean |cos sim|", color="tab:blue")
    axes[0].set_ylabel("Subspace overlap")
    axes[0].set_title("Quinn vs Casey LoRA subspace overlap (top-16 singular vectors)")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=1/np.sqrt(LORA_RANK), color="gray", linestyle="--", alpha=0.5, label="random baseline")

    axes[1].bar(x, principal_angles, alpha=0.7, color="tab:green")
    axes[1].set_ylabel("Top-1 principal angle (°)")
    axes[1].set_xlabel("Layer.Module")
    axes[1].set_ylim(0, 90)
    axes[1].axhline(y=90, color="gray", linestyle="--", alpha=0.3)

    # Only label every Nth tick to avoid clutter
    n_ticks = len(labels)
    step = max(1, n_ticks // 30)
    axes[1].set_xticks(x[::step])
    axes[1].set_xticklabels([labels[i] for i in range(0, n_ticks, step)], rotation=90, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / "subspace_overlap.png", dpi=150)
    plt.close()
    print(f"  Saved subspace_overlap.png")


def plot_linearity(results: list[dict], output_dir: Path):
    """Plot R² and coefficients per layer."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    labels = [f"L{r['layer']}.{r['module']}" for r in results]
    r_squareds = [r["linearity"]["r_squared"] for r in results]
    alphas = [r["linearity"]["alpha_q"] for r in results]
    betas = [r["linearity"]["beta_c"] for r in results]

    x = np.arange(len(labels))

    axes[0].bar(x, r_squareds, alpha=0.7, color="tab:purple")
    axes[0].set_ylabel("R²")
    axes[0].set_title("Linearity: ΔW_QC ≈ α·ΔW_Q + β·ΔW_C")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.3)

    axes[1].bar(x - 0.15, alphas, width=0.3, alpha=0.7, label="α (Quinn)", color="tab:blue")
    axes[1].bar(x + 0.15, betas, width=0.3, alpha=0.7, label="β (Casey)", color="tab:orange")
    axes[1].set_ylabel("Coefficient")
    axes[1].set_xlabel("Layer.Module")
    axes[1].legend()
    axes[1].axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    n_ticks = len(labels)
    step = max(1, n_ticks // 30)
    axes[1].set_xticks(x[::step])
    axes[1].set_xticklabels([labels[i] for i in range(0, n_ticks, step)], rotation=90, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / "linearity.png", dpi=150)
    plt.close()
    print(f"  Saved linearity.png")


def plot_residual_norms(results: list[dict], output_dir: Path):
    """Plot residual fraction per layer."""
    fig, ax = plt.subplots(figsize=(14, 5))

    labels = [f"L{r['layer']}.{r['module']}" for r in results]
    fracs = [r["residual"]["residual_frac"] for r in results]

    x = np.arange(len(labels))
    ax.bar(x, fracs, alpha=0.7, color="tab:red")
    ax.set_ylabel("‖Residual‖ / ‖ΔW_QC‖")
    ax.set_title("Nonlinear residual: what joint training adds beyond α·Quinn + β·Casey")
    ax.set_xlabel("Layer.Module")

    n_ticks = len(labels)
    step = max(1, n_ticks // 30)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([labels[i] for i in range(0, n_ticks, step)], rotation=90, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / "residual_norms.png", dpi=150)
    plt.close()
    print(f"  Saved residual_norms.png")


def plot_summary_by_module_type(results: list[dict], output_dir: Path):
    """Aggregate metrics by module type (q_proj, k_proj, etc.)."""
    from collections import defaultdict
    by_module = defaultdict(lambda: {"overlaps": [], "r2s": [], "res_fracs": []})

    for r in results:
        mod = r["module"]
        by_module[mod]["overlaps"].append(r["overlap"]["mean_overlap"])
        by_module[mod]["r2s"].append(r["linearity"]["r_squared"])
        by_module[mod]["res_fracs"].append(r["residual"]["residual_frac"])

    modules = sorted(by_module.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(modules))
    for idx, (metric, label, color) in enumerate([
        ("overlaps", "Mean subspace overlap", "tab:blue"),
        ("r2s", "R² (linearity)", "tab:purple"),
        ("res_fracs", "Residual fraction", "tab:red"),
    ]):
        means = [np.mean(by_module[m][metric]) for m in modules]
        stds = [np.std(by_module[m][metric]) for m in modules]
        axes[idx].bar(x, means, yerr=stds, alpha=0.7, color=color, capsize=3)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(modules, rotation=45, fontsize=8)
        axes[idx].set_title(label)
        if "R²" in label:
            axes[idx].set_ylim(0, 1.05)

    plt.suptitle("Mechanistic metrics aggregated by module type (mean ± std across layers)")
    plt.tight_layout()
    plt.savefig(output_dir / "summary_by_module.png", dpi=150)
    plt.close()
    print(f"  Saved summary_by_module.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mechanistic analysis of LoRA adapters")
    parser.add_argument("--jobs", default="training_jobs_3ep.json",
                        help="Training jobs JSON filename in results/")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="Local directory containing adapter subdirs (Model_Q, Model_C, Model_QC)")
    parser.add_argument("--top-k", type=int, default=16,
                        help="Number of top singular vectors for subspace overlap (default: 16)")
    parser.add_argument("--output-dir", default="plots_mechanistic",
                        help="Output subdirectory under results/ for plots")
    args = parser.parse_args()

    output_dir = RESULTS_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve adapter paths
    if args.local_dir:
        local = Path(args.local_dir)
        adapter_paths = {
            "Model_Q": local / "Model_Q",
            "Model_C": local / "Model_C",
            "Model_QC": local / "Model_QC",
        }
    else:
        jobs_file = RESULTS_DIR / args.jobs
        if not jobs_file.exists():
            print(f"ERROR: {jobs_file} not found.")
            sys.exit(1)
        with open(jobs_file) as f:
            jobs = json.load(f)

        cache_dir = PROJECT_ROOT / ".adapter_cache"
        cache_dir.mkdir(exist_ok=True)
        adapter_paths = {}
        for job in jobs:
            name = job["name"]
            model_id = job.get("model_id")
            if not model_id:
                print(f"ERROR: {name} has no model_id. Run `train.py status` first.")
                sys.exit(1)
            print(f"Downloading {name} adapter from {model_id}...")
            adapter_paths[name] = download_adapter(model_id, cache_dir)

    # Find safetensors files and list modules
    print("\n--- Locating adapters ---")
    st_files = {}
    for name, path in adapter_paths.items():
        st_files[name] = find_safetensors(path)
        print(f"  {name}: {st_files[name]}")

    modules_q = set(list_lora_modules(st_files["Model_Q"]))
    modules_c = set(list_lora_modules(st_files["Model_C"]))
    modules_qc = set(list_lora_modules(st_files["Model_QC"]))
    common_keys = modules_q & modules_c & modules_qc
    sorted_keys = sorted(common_keys, key=lambda k: get_layer_module(k))
    print(f"  {len(sorted_keys)} common modules across all 3 models")

    # Run analyses per module (streaming — only 3 matrices in memory at a time)
    print("\n--- Running analyses (streaming, one module at a time) ---")
    results = []
    for i, key in enumerate(sorted_keys):
        layer_num, module_name = get_layer_module(key)
        dq = load_single_delta(st_files["Model_Q"], key)
        dc = load_single_delta(st_files["Model_C"], key)
        dqc = load_single_delta(st_files["Model_QC"], key)

        overlap = subspace_overlap(dq, dc, top_k=args.top_k)
        lin = linearity_test(dq, dc, dqc)
        res = residual_analysis(dq, dc, dqc)

        results.append({
            "key": key,
            "layer": layer_num,
            "module": module_name,
            "overlap": overlap,
            "linearity": lin,
            "residual": {k: v for k, v in res.items() if k != "residual_matrix"},
        })

        if (i + 1) % 36 == 0:
            print(f"  {i + 1}/{len(sorted_keys)} modules done")

    # Print summary table
    print(f"\n{'Module':<45} {'Overlap(max)':<14} {'Overlap(mean)':<14} {'R²':<8} {'α(Q)':<8} {'β(C)':<8} {'Res%':<8}")
    print("-" * 110)
    for r in results:
        print(f"{r['key']:<45} {r['overlap']['max_overlap']:<14.3f} {r['overlap']['mean_overlap']:<14.3f} "
              f"{r['linearity']['r_squared']:<8.3f} {r['linearity']['alpha_q']:<8.3f} {r['linearity']['beta_c']:<8.3f} "
              f"{r['residual']['residual_frac']:<8.3f}")

    # Aggregate stats
    mean_r2 = np.mean([r["linearity"]["r_squared"] for r in results])
    mean_overlap = np.mean([r["overlap"]["mean_overlap"] for r in results])
    mean_res_frac = np.mean([r["residual"]["residual_frac"] for r in results])
    print(f"\n--- Aggregates ---")
    print(f"  Mean R²:              {mean_r2:.4f}")
    print(f"  Mean subspace overlap: {mean_overlap:.4f}")
    print(f"  Mean residual frac:    {mean_res_frac:.4f}")

    # Save results JSON
    results_file = output_dir / "mechanistic_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # Plots
    print("\n--- Generating plots ---")
    plot_subspace_overlap(results, output_dir)
    plot_linearity(results, output_dir)
    plot_residual_norms(results, output_dir)
    plot_summary_by_module_type(results, output_dir)

    # Interpretation summary
    print("\n--- Interpretation ---")
    if mean_r2 > 0.9:
        print("  HIGH LINEARITY: Joint model ≈ linear combination of single-persona models.")
        print("  -> Personas are superposed, not reorganized. Leakage (if any) is structural.")
    elif mean_r2 > 0.5:
        print("  MODERATE LINEARITY: Partial superposition with some nonlinear interaction.")
    else:
        print("  LOW LINEARITY: Joint training found a qualitatively different solution.")
        print("  -> Representational reorganization occurred.")

    if mean_overlap > 0.3:
        print("  HIGH OVERLAP: Quinn and Casey modify overlapping weight subspaces.")
        print("  -> Perfect compartmentalization is structurally impossible.")
    elif mean_overlap > 0.1:
        print("  MODERATE OVERLAP: Some shared subspace, some separation.")
    else:
        print("  LOW OVERLAP: Personas modify mostly orthogonal subspaces.")
        print("  -> Model has capacity to compartmentalize personas.")


if __name__ == "__main__":
    main()
