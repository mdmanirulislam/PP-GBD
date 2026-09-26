"""Generate the synthetic C&C topology visualization used as Figure 2.

Figure 1 is a manually prepared architecture schematic and is distributed as
the static asset ``figures/fig1_architecture.png``.
"""
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

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
        graph,
        positions,
        ax=axis,
        node_size=sizes,
        node_color="#c53030",
        linewidths=0.2,
        edgecolors="white",
    )
    axis.set_title(titles[topology], fontsize=9)
    axis.axis("off")

plt.subplots_adjust(
    left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.12, hspace=0.20
)
plt.savefig(FIG / "fig2_topologies.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("Figure 2 topology visualization generated")
