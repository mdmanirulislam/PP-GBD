"""Generate static PP-GBD figures: the architecture schematic and C&C topologies.

Equation artwork is intentionally not generated. Equations are maintained as
editable mathematical objects in the manuscript rather than raster images.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def rounded_box(ax, x, y, w, h, text, dashed=False, fontsize=10.5):
    """Draw a white rounded box using normalized figure coordinates."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.3, color="black",
            shrinkA=0, shrinkB=0,
        )
    )


# ---------------------------------------------------------------- Fig. 1: architecture
fig, ax = plt.subplots(figsize=(6.7, 8.2), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

rounded_box(ax, 0.10, 0.855, 0.80, 0.095,
            "Encrypted traffic records\n(payload unavailable)", fontsize=12.0)
rounded_box(ax, 0.10, 0.710, 0.80, 0.105,
            "TLS-visible metadata and\ncommunication graph G=(V,E,X)", fontsize=11.5)
rounded_box(ax, 0.10, 0.565, 0.80, 0.095,
            "Three-layer GCN\nper-host bot scores", fontsize=11.5)

arrow(ax, 0.50, 0.855, 0.50, 0.815)
arrow(ax, 0.50, 0.710, 0.50, 0.660)

# Branch line and arrows, positioned outside boxes to prevent overlap.
ax.plot([0.17, 0.83], [0.535, 0.535], color="black", linewidth=1.3)
ax.plot([0.50, 0.50], [0.565, 0.535], color="black", linewidth=1.3)

rounded_box(ax, 0.02, 0.325, 0.30, 0.135,
            "Fixed-slot\ngraph-level DP\n(training)", dashed=True, fontsize=10.5)
rounded_box(ax, 0.35, 0.325, 0.30, 0.135,
            "FedAvg across\noperators\n(deployment)", dashed=True, fontsize=10.5)
rounded_box(ax, 0.68, 0.325, 0.30, 0.135,
            "Polynomial GCN\n(encrypted X;\nknown topology)", dashed=True, fontsize=10.5)

for x in (0.17, 0.50, 0.83):
    ax.plot([x, x], [0.535, 0.475], color="black", linewidth=1.3)
    arrow(ax, x, 0.475, x, 0.460)

rounded_box(ax, 0.10, 0.075, 0.80, 0.105,
            "Ranked alerts for analyst triage", fontsize=11.5)
for x in (0.17, 0.50, 0.83):
    arrow(ax, x, 0.325, x, 0.180)

plt.tight_layout(pad=0.15)
plt.savefig(FIG / "fig1_architecture.png", bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------- Fig. 2: C&C topologies (2 x 2 IEEE-column layout)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import botnet_overlay_edges  # noqa: E402

rng = np.random.default_rng(7)
fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.6), dpi=300)
titles = {
    "C2": "(a) Centralized C2",
    "DEBRUIJN": "(b) de Bruijn",
    "KADEMLIA": "(c) Kademlia-style",
    "CHORD": "(d) Chord-style",
}
for axis, topology in zip(axes.flat, ["C2", "DEBRUIJN", "KADEMLIA", "CHORD"]):
    graph = nx.Graph()
    graph.add_nodes_from(range(128))
    graph.add_edges_from(botnet_overlay_edges(np.arange(128), topology, rng))
    if topology in ("DEBRUIJN", "CHORD"):
        positions = nx.circular_layout(graph)
    else:
        positions = nx.spring_layout(graph, seed=3, k=0.18)
    degree = dict(graph.degree())
    sizes = [8 + 2.2 * degree[node] for node in graph.nodes()]
    nx.draw_networkx_edges(
        graph, positions, ax=axis, width=0.4, alpha=0.45, edge_color="#4a5568"
    )
    nx.draw_networkx_nodes(
        graph, positions, ax=axis, node_size=sizes,
        node_color="#c53030", linewidths=0.2, edgecolors="white"
    )
    axis.set_title(titles[topology], fontsize=9)
    axis.axis("off")
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.12, hspace=0.20)
plt.savefig(FIG / "fig2_topologies.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("static figures generated (architecture and 2 x 2 topologies; no equation images)")
