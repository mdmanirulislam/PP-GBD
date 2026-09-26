"""Generate result figures and CSV tables from aggregate_results.json."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
r = json.loads((RES / "aggregate_results.json").read_text())


def roc_points(y, s):
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    return np.r_[0.0, fp / max((y == 0).sum(), 1)], np.r_[0.0, tp / max((y == 1).sum(), 1)]


def pr_points(y, s):
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max((y == 1).sum(), 1)
    return np.r_[0.0, recall], np.r_[1.0, precision]


def roc_auc(y, s):
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y == 1].mean())


# Fig. 3: ROC and precision-recall curves for seed 11.
fig, axes = plt.subplots(2, 1, figsize=(3.383, 5.033), dpi=300)
for name in ["LR", "MLP", "GCN-poly", "GCN"]:
    d = r["curve_data_seed_11"][name]
    fpr, tpr = roc_points(d["y"], d["s"])
    rec, prec = pr_points(d["y"], d["s"])
    axes[0].plot(fpr, tpr, lw=1.25, label=f"{name} ({roc_auc(d['y'], d['s']):.3f})")
    axes[1].plot(rec, prec, lw=1.25, label=f"{name} ({average_precision(d['y'], d['s']):.3f})")
axes[0].plot([0, 1], [0, 1], ":", lw=0.7)
axes[0].text(0.01, 0.03, "ROC-AUC", transform=axes[0].transAxes, fontsize=6.2)
axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01))
axes[1].axhline(r["config"]["n_bot"] / (r["config"]["n_bg"] + r["config"]["n_bot"]),
                ls="--", lw=0.7)
axes[1].text(0.01, 0.03, "Average precision", transform=axes[1].transAxes, fontsize=6.2)
axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
for ax in axes:
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=True, fontsize=5.4, loc="lower right")
    ax.tick_params(labelsize=6.5)
    ax.xaxis.label.set_size(7.0)
    ax.yaxis.label.set_size(7.0)
fig.subplots_adjust(left=0.19, right=0.98, top=0.985, bottom=0.08, hspace=0.34)
fig.savefig(FIG / "fig3_roc_pr.png", facecolor="white")
plt.close(fig)

# Fig. 4: privacy-utility plot.
eps = [1.0, 2.0, 4.0, 8.0]
f1m = [r["dp"]["by_epsilon"][str(e)]["f1"]["mean"] for e in eps]
f1s = [r["dp"]["by_epsilon"][str(e)]["f1"]["ci95"] for e in eps]
aucm = [r["dp"]["by_epsilon"][str(e)]["auc"]["mean"] for e in eps]
aucs = [r["dp"]["by_epsilon"][str(e)]["auc"]["ci95"] for e in eps]
fig, ax = plt.subplots(figsize=(3.41, 2.51), dpi=300)
ax.axhline(r["models"]["GCN"]["f1"]["mean"], ls="--", lw=0.75, label="Non-private F1")
ax.errorbar(eps, f1m, yerr=f1s, marker="o", lw=1.25, capsize=2.5, label="F1")
ax.errorbar(eps, aucm, yerr=aucs, marker="s", lw=1.25, capsize=2.5, label="ROC-AUC")
ax.set_xscale("log")
ax.set_xticks(eps, ["1", "2", "4", "8"])
ax.minorticks_off()
ax.set(xlabel=r"Privacy budget $\epsilon$ ($\delta=10^{-3}$)", ylabel="Test metric", ylim=(0, 1.04))
ax.grid(alpha=0.25, lw=0.4)
ax.legend(frameon=True, fontsize=6.2, loc="lower right")
ax.tick_params(labelsize=7)
ax.xaxis.label.set_size(7.5)
ax.yaxis.label.set_size(7.5)
fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.20)
fig.savefig(FIG / "fig4_privacy_utility.png", facecolor="white")
plt.close(fig)


def write_csv(filename, header, rows):
    with (RES / filename).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def pm(rec):
    return f"{rec['mean']:.4f}+/-{rec['ci95']:.4f}"


write_csv(
    "table_models.csv",
    ["model", "precision_95ci", "recall_95ci", "f1_95ci", "roc_auc_95ci", "average_precision_95ci", "train_s", "infer_ms_per_graph"],
    [[name] + [pm(r["models"][name][k]) for k in ["precision", "recall", "f1", "auc", "average_precision"]]
     + [f"{r['models'][name]['train_time_s']['mean']:.2f}",
        f"{r['models'][name]['infer_time_per_graph_ms']['mean']:.3f}"]
     for name in ["LR", "MLP", "GCN", "GCN-poly"]]
)
write_csv(
    "table_per_topology.csv",
    ["model", "topology", "precision_95ci", "recall_95ci", "f1_95ci", "roc_auc_95ci"],
    [[name, topo] + [pm(r["models_per_topology"][name][topo][k])
                     for k in ["precision", "recall", "f1", "auc"]]
     for name in ["LR", "MLP", "GCN"] for topo in r["config"]["topologies"]]
)
write_csv(
    "table_dp.csv",
    ["epsilon", "sigma", "precision_95ci", "recall_95ci", "f1_95ci", "roc_auc_95ci", "host_loss_diagnostic_auc_95ci"],
    [[e, f"{float(r['dp']['sigma_map'][str(e)]):.4f}"]
     + [pm(r["dp"]["by_epsilon"][str(e)][k]) for k in ["precision", "recall", "f1", "auc"]]
     + [pm(r["dp"]["by_epsilon"][str(e)]["node_loss_mia_auc"])]
     for e in eps]
)

write_csv(
    "table_clipping.csv",
    ["clip_norm", "f1_95ci", "roc_auc_95ci", "clipped_fraction_95ci"],
    [[c,
      pm(r["ablations"]["clipping_sweep"][str(c)]["f1"]),
      pm(r["ablations"]["clipping_sweep"][str(c)]["auc"]),
      pm(r["ablations"]["clipping_sweep"][str(c)]["clipped_fraction"])]
     for c in [0.25, 0.5, 1.0, 2.0]]
)

write_csv(
    "table_private_baseline.csv",
    ["model", "epsilon", "f1_95ci", "roc_auc_95ci"],
    [["DP-MLP", 4.0,
      pm(r["ablations"]["dp_mlp_eps4"]["f1"]),
      pm(r["ablations"]["dp_mlp_eps4"]["auc"])],
     ["DP-GCN", 4.0,
      pm(r["dp"]["by_epsilon"]["4.0"]["f1"]),
      pm(r["dp"]["by_epsilon"]["4.0"]["auc"])]]
)

write_csv(
    "table_overlay_edges.csv",
    ["topology", "contact_records", "unique_undirected", "edges_added"],
    [[topo,
      f"{r['overlay_edges'][topo]['overlay_contact_records']['mean']:.1f}",
      f"{r['overlay_edges'][topo]['overlay_unique_edges']['mean']:.1f}",
      f"{r['overlay_edges'][topo]['overlay_added_edges']['mean']:.1f}"]
     for topo in r["config"]["topologies"]]
)

if "cost_benchmark" in r:
    write_csv(
        "table_cost.csv",
        ["mode", "train_s", "infer_ms_per_graph", "peak_traced_memory_mb",
         "model_kib_float32", "communication_mib_float32"],
        [[name,
          f"{row['train_time_s']:.2f}",
          f"{row['infer_time_per_graph_ms']:.3f}",
          f"{row['peak_traced_memory_mb']:.2f}",
          f"{row['model_bytes_float32'] / 1024:.2f}",
          f"{row.get('communication_bytes_float32', 0) / (1024 ** 2):.2f}"]
         for name, row in r["cost_benchmark"]["rows"].items()]
    )
print(f"Wrote figures and CSV tables under {ROOT}")
