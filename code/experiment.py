"""
PP-GBD: Privacy-Preserving Graph Neural Botnet Detection in Encrypted Traffic
====================================================================================
Reproducible experimental pipeline (pure NumPy/SciPy implementation).

Components:
  1. Synthetic encrypted-traffic communication-graph benchmark (4 C&C topologies:
     centralized C2, de Bruijn P2P, Kademlia-style P2P, Chord-style P2P) overlaid
     on power-law background graphs, with TLS-metadata node features.
  2. 3-layer GCN (symmetric-normalized adjacency) trained with weighted BCE.
  3. Fixed-slot graph-level DP-SGD (per-graph clipping + Gaussian noise) under
     add/remove adjacency with a public denominator.
  4. Polynomial-arithmetic GCN variant for future encrypted-feature inference
     when the propagation topology is provider-known.
  5. Federated (FedAvg) training across 4 non-IID clients (one topology each).
  6. Auxiliary host-loss membership diagnostic (not aligned with graph-level adjacency).

Run: python experiment.py models|dp|fed|ablations <seed>
     python experiment.py cost [seed]
     python experiment.py agg
"""

import json, math, os, platform, time, tracemalloc
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy import stats

RNG_SEEDS = [11, 23, 47, 59, 83]
N_BG = 3000            # background (benign) nodes per graph
N_BOT = 128            # bots per graph
FEAT_DIM = 10
TOPOLOGIES = ["C2", "DEBRUIJN", "KADEMLIA", "CHORD"]
GRAPHS_PER_TOPO = 12   # 8 train / 2 val / 2 test
HID = 64
EPOCHS = 150
DP_EPOCHS = 100
LR = 0.01
DELTA = 1e-3
PUBLIC_SLOTS = 32      # public graph-contribution slots for add/remove DP adjacency
PUBLIC_POS_WEIGHT = N_BG / N_BOT  # fixed from the public benchmark design
CLIP_NORMS = [0.25, 0.5, 1.0, 2.0]
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

# ----------------------------------------------------------------------------
# 1. Synthetic benchmark generation
# ----------------------------------------------------------------------------

def botnet_overlay_edges(bot_ids, topo, rng):
    """Edges among bot nodes according to a C&C topology."""
    n = len(bot_ids)
    edges = set()
    if topo == "C2":                              # centralized: 3 C2 servers
        c2 = rng.choice(n, size=3, replace=False)
        for i in range(n):
            if i in c2:
                continue
            for s in rng.choice(c2, size=rng.integers(1, 3), replace=False):
                edges.add((i, int(s)))
        for a in c2:                              # C2 servers interlink
            for b in c2:
                if a < b:
                    edges.add((int(a), int(b)))
        for i in range(n):                        # sparse lateral bot links
            if rng.random() < 0.15:
                j = int(rng.integers(0, n))
                if j != i:
                    edges.add((i, j))
    elif topo == "DEBRUIJN":                      # binary de Bruijn graph
        for i in range(n):
            edges.add((i, (2 * i) % n))
            edges.add((i, (2 * i + 1) % n))
    elif topo == "KADEMLIA":                      # XOR-metric bucket contacts
        bits = int(math.log2(n))
        for i in range(n):
            for k in range(bits):
                lo, hi = 2 ** k, 2 ** (k + 1)
                cands = [j for j in range(n) if lo <= (i ^ j) < hi]
                if cands:
                    edges.add((i, int(rng.choice(cands))))
    elif topo == "CHORD":                         # ring + finger tables
        bits = int(math.log2(n))
        for i in range(n):
            edges.add((i, (i + 1) % n))
            for k in range(1, bits, 2):           # every other finger
                edges.add((i, (i + 2 ** k) % n))
    return [(a, b) for (a, b) in edges if a != b]


def sample_features(is_bot, m, rng):
    """TLS-observable flow-metadata features (log/standardized later).
    Columns: log-flows, mean pkt size, std pkt size, log mean IAT, log CV-IAT,
    log duration, log up/down byte ratio, log distinct peers/ports,
    nocturnal fraction, TLS-handshake meta score."""
    if is_bot:  # periodic beaconing over TLS: regular timing, uniform sizes
        f = np.column_stack([
            rng.lognormal(4.25, 0.95, m),
            rng.normal(605, 200, m),
            np.abs(rng.normal(195, 100, m)),
            rng.lognormal(2.1, 0.85, m),
            rng.lognormal(-0.85, 0.75, m),      # low inter-arrival jitter
            rng.lognormal(2.7, 1.25, m),
            rng.lognormal(-0.15, 0.75, m),      # near-symmetric up/down
            rng.lognormal(2.65, 0.85, m),
            rng.beta(4, 6, m),
            rng.normal(0.35, 1.0, m),
        ])
    else:
        f = np.column_stack([
            rng.lognormal(4.0, 1.0, m),
            rng.normal(720, 260, m),
            np.abs(rng.normal(310, 130, m)),
            rng.lognormal(1.6, 1.2, m),
            rng.lognormal(0.0, 0.65, m),
            rng.lognormal(3.0, 1.5, m),
            rng.lognormal(-0.8, 0.9, m),        # download-heavy
            rng.lognormal(2.5, 0.9, m),
            rng.beta(2, 8, m),
            rng.normal(0.0, 1.0, m),
        ])
    for c in [0, 3, 4, 5, 6, 7]:
        f[:, c] = np.log1p(np.maximum(f[:, c], 1e-6))
    return f


def make_graph(topo, rng):
    g = nx.powerlaw_cluster_graph(N_BG + N_BOT, m=2, p=0.10, seed=int(rng.integers(1e9)))
    n = g.number_of_nodes()
    bot_ids = rng.choice(n, size=N_BOT, replace=False)
    overlay = botnet_overlay_edges(bot_ids, topo, rng)
    overlay_unique = {tuple(sorted((int(bot_ids[a]), int(bot_ids[b])))) for a, b in overlay}
    overlay_added = sum(not g.has_edge(a, b) for a, b in overlay_unique)
    for a, b in overlay_unique:
        g.add_edge(a, b)
    y = np.zeros(n, dtype=np.float64); y[bot_ids] = 1.0
    X = np.empty((n, FEAT_DIM))
    X[y == 0] = sample_features(False, int((y == 0).sum()), rng)
    X[y == 1] = sample_features(True, int((y == 1).sum()), rng)
    # A positive feature-wise affine perturbation is applied before per-graph
    # z-standardization. Standardization cancels this transform almost exactly,
    # so it is not interpreted as a persistent cross-domain shift.
    X = X * rng.lognormal(0.0, 0.08, FEAT_DIM) + rng.normal(0.0, 0.15, FEAT_DIM)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    A = nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)
    A = A + sp.eye(n, format="csr")
    d = np.asarray(A.sum(1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(d))
    A_hat = (Dm @ A @ Dm).tocsr()
    return {"A": A_hat, "X": X, "y": y, "AX": A_hat @ X, "topo": topo,
            "overlay_contact_records": len(overlay),
            "overlay_unique_edges": len(overlay_unique),
            "overlay_added_edges": int(overlay_added)}


def build_dataset(seed):
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for topo in TOPOLOGIES:
        graphs = [make_graph(topo, rng) for _ in range(GRAPHS_PER_TOPO)]
        splits["train"] += graphs[:8]
        splits["val"] += graphs[8:10]
        splits["test"] += graphs[10:12]
    return splits

# ----------------------------------------------------------------------------
# 2. GCN in NumPy (manual backprop) + baselines
# ----------------------------------------------------------------------------

def init_params(rng, dims):
    P = {}
    for i in range(len(dims) - 1):
        lim = math.sqrt(6.0 / (dims[i] + dims[i + 1]))
        P[f"W{i}"] = rng.uniform(-lim, lim, (dims[i], dims[i + 1]))
        P[f"b{i}"] = np.zeros(dims[i + 1])
    return P


def act_fn(z, kind):
    if kind == "relu":
        return np.maximum(z, 0.0), (z > 0).astype(np.float64)
    # CKKS-friendly degree-2 polynomial approximation of ReLU on ~[-3, 3]
    return 0.25 + 0.5 * z + 0.125 * z * z, 0.5 + 0.25 * z


def n_layers(P):
    return sum(1 for k in P if k.startswith("W"))


def gcn_forward(P, g, kind, use_graph=True):
    """Layered propagation: M_l = A H_l (or H_l), Z = M W + b, H = act(Z)."""
    L = n_layers(P)
    A = g["A"]
    Ms, Ds = [], []
    H = g["X"]
    for l in range(L):
        M = (g["AX"] if l == 0 else A @ H) if use_graph else H
        Ms.append(M)
        Z = M @ P[f"W{l}"] + P[f"b{l}"]
        if l < L - 1:
            H, D = act_fn(Z, kind); Ds.append(D)
        else:
            z_out = Z.ravel()
    return {"Ms": Ms, "Ds": Ds, "z": z_out}


def loss_and_grads(P, g, kind, pos_w, use_graph=True):
    c = gcn_forward(P, g, kind, use_graph)
    L = n_layers(P)
    y = g["y"]; p = 1.0 / (1.0 + np.exp(-c["z"]))
    w = np.where(y == 1, pos_w, 1.0); sw = w.sum()
    eps = 1e-12
    loss = -(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).sum() / sw
    dZ = (w * (p - y) / sw)[:, None]
    G = {}
    for l in range(L - 1, -1, -1):
        G[f"W{l}"] = c["Ms"][l].T @ dZ; G[f"b{l}"] = dZ.sum(0)
        if l > 0:
            dH = (g["A"] @ dZ if use_graph else dZ) @ P[f"W{l}"].T
            dZ = dH * c["Ds"][l - 1]
    return loss, G, p


def flat(G, keys):   return np.concatenate([G[k].ravel() for k in keys])
def unflat(v, P, keys):
    out, i = {}, 0
    for k in keys:
        s = P[k].size
        out[k] = v[i:i + s].reshape(P[k].shape); i += s
    return out


def train(graphs, rng, kind="relu", use_graph=True, epochs=EPOCHS, lr=LR,
          clip=None, sigma=0.0, dims=None, pos_w=None, public_slots=None,
          diagnostics=None):
    dims = dims or [FEAT_DIM, HID, HID, 1]
    if pos_w is None:
        # The class weight is fixed by the public benchmark design rather than
        # estimated from the protected training slots.
        pos_w = PUBLIC_POS_WEIGHT
    P = init_params(rng, dims)
    keys = sorted(P.keys())
    m = {k: np.zeros_like(P[k]) for k in keys}
    v = {k: np.zeros_like(P[k]) for k in keys}
    b1, b2, adam_eps = 0.9, 0.999, 1e-8
    B = len(graphs) if public_slots is None else int(public_slots)
    if B < len(graphs):
        raise ValueError("public_slots cannot be smaller than the number of supplied graphs")
    if diagnostics is not None:
        diagnostics.update({"gradient_records": 0, "clipped_records": 0,
                            "clip_norm": clip, "noise_multiplier": sigma})
    # When public_slots > len(graphs), absent slots contribute the zero gradient.
    # The denominator is public and fixed, which is required by the add/remove
    # graph-adjacency proof used for the DP experiments.
    for t in range(1, epochs + 1):
        acc = np.zeros(sum(P[k].size for k in keys))
        for g in graphs:
            _, G, _ = loss_and_grads(P, g, kind, pos_w, use_graph)
            gv = flat(G, keys)
            if clip is not None:
                norm = np.linalg.norm(gv)
                if diagnostics is not None:
                    diagnostics["gradient_records"] += 1
                    diagnostics["clipped_records"] += int(norm > clip)
                gv = gv * min(1.0, clip / (norm + 1e-12))
            acc += gv
        if sigma > 0:
            acc += rng.normal(0.0, sigma * clip, size=acc.shape)
        acc /= B
        Gm = unflat(acc, P, keys)
        for k in keys:
            m[k] = b1 * m[k] + (1 - b1) * Gm[k]
            v[k] = b2 * v[k] + (1 - b2) * Gm[k] ** 2
            mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
            P[k] -= lr * mh / (np.sqrt(vh) + adam_eps)
    if diagnostics is not None and diagnostics["gradient_records"]:
        diagnostics["clipped_fraction"] = (
            diagnostics["clipped_records"] / diagnostics["gradient_records"]
        )
    return P, pos_w


def predict(P, g, kind, use_graph=True):
    z = gcn_forward(P, g, kind, use_graph)["z"]
    return 1.0 / (1.0 + np.exp(-z))

# ----------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------

def roc_auc(y, s):
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(y, s):
    """Average precision (area under the stepwise precision-recall curve)."""
    y = np.asarray(y, dtype=int); s = np.asarray(s, dtype=float)
    order = np.argsort(-s); y = y[order]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y == 1].mean())


def prf(y, s, thr):
    yp = (s >= thr).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def best_threshold(y, s):
    thrs = np.quantile(s, np.linspace(0.5, 0.999, 120))
    return max(thrs, key=lambda t: prf(y, s, t)[2])


def evaluate(P, kind, val, test, use_graph=True, per_topo=False):
    yv = np.concatenate([g["y"] for g in val])
    sv = np.concatenate([predict(P, g, kind, use_graph) for g in val])
    thr = best_threshold(yv, sv)
    yt = np.concatenate([g["y"] for g in test])
    st = np.concatenate([predict(P, g, kind, use_graph) for g in test])
    p, r, f1 = prf(yt, st, thr)
    res = {"precision": p, "recall": r, "f1": f1, "auc": roc_auc(yt, st),
           "average_precision": average_precision(yt, st)}
    if per_topo:
        res["per_topo"] = {}
        for topo in TOPOLOGIES:
            gs = [g for g in test if g["topo"] == topo]
            y2 = np.concatenate([g["y"] for g in gs])
            s2 = np.concatenate([predict(P, g, kind, use_graph) for g in gs])
            p2, r2, f2 = prf(y2, s2, thr)
            res["per_topo"][topo] = {"precision": p2, "recall": r2, "f1": f2,
                                     "auc": roc_auc(y2, s2)}
    return res, (yt, st)

# ----------------------------------------------------------------------------
# 4. RDP accounting for the fixed-slot full-batch Gaussian mechanism
# ----------------------------------------------------------------------------

def eps_from_sigma(sigma, T, delta=DELTA):
    """RDP composition for add/remove graph adjacency over fixed public slots.

    Per-graph gradients are clipped to C; replacing an occupied slot by the
    null record changes the clipped sum by at most C. The mechanism adds
    N(0, sigma^2 C^2 I) before division by the public slot count.
    Replace-one adjacency would require sensitivity 2C and is not claimed.
    """
    alphas = np.concatenate([np.linspace(1.01, 64, 4000)])
    eps = T * alphas / (2 * sigma ** 2) + np.log(1 / delta) / (alphas - 1)
    return float(eps.min())


def sigma_for_eps(target, T, delta=DELTA):
    lo, hi = 0.2, 500.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if eps_from_sigma(mid, T, delta) > target:
            lo = mid
        else:
            hi = mid
    return hi

# ----------------------------------------------------------------------------
# 5. Auxiliary node-loss membership diagnostic (not graph-unit aligned)
# ----------------------------------------------------------------------------

def node_loss_mia_auc(P, kind, train_g, test_g, pos_w, rng):
    def node_losses(gs):
        ls = []
        for g in gs:
            p = predict(P, g, kind)
            y = g["y"]; e = 1e-12
            ls.append(-(y * np.log(p + e) + (1 - y) * np.log(1 - p + e)))
        return np.concatenate(ls)
    lm, ln = node_losses(train_g), node_losses(test_g)
    k = min(len(lm), len(ln))
    lm = rng.choice(lm, k, replace=False); ln = rng.choice(ln, k, replace=False)
    labels = np.concatenate([np.ones(k), np.zeros(k)])
    scores = -np.concatenate([lm, ln])      # lower loss -> more likely member
    return roc_auc(labels, scores)

# ----------------------------------------------------------------------------
# 6. Federated (FedAvg) with non-IID topology clients
# ----------------------------------------------------------------------------

def fedavg(splits, rng, rounds=60, local_steps=4, lr_local=0.15, kind="relu"):
    dims = [FEAT_DIM, HID, HID, 1]
    P = init_params(rng, dims); keys = sorted(P.keys())
    pos_w = PUBLIC_POS_WEIGHT
    clients = {t: [g for g in splits["train"] if g["topo"] == t] for t in TOPOLOGIES}
    for r in range(1, rounds + 1):
        new = {k: np.zeros_like(P[k]) for k in keys}
        for t in TOPOLOGIES:
            L = {k: P[k].copy() for k in keys}
            for _ in range(local_steps):
                acc = {k: np.zeros_like(P[k]) for k in keys}
                for g in clients[t]:
                    _, G, _ = loss_and_grads(L, g, kind, pos_w)
                    for k in keys:
                        acc[k] += G[k] / len(clients[t])
                for k in keys:
                    L[k] -= lr_local * acc[k]
            for k in keys:
                new[k] += L[k] / len(TOPOLOGIES)   # equal-size clients
        P = new
    return P


def inference_ms(P, kind, graphs, use_graph=True, repeats=5):
    # Warm the sparse kernels before timing and average several passes because
    # a single eight-graph pass is too noisy for an overhead comparison.
    for g in graphs:
        predict(P, g, kind, use_graph)
    t0 = time.perf_counter()
    for _ in range(repeats):
        for g in graphs:
            predict(P, g, kind, use_graph)
    return (time.perf_counter() - t0) / (len(graphs) * repeats) * 1e3


def parameter_summary(P):
    count = int(sum(v.size for v in P.values()))
    return {"parameters": count,
            "model_bytes_float64": int(sum(v.nbytes for v in P.values())),
            "model_bytes_float32": int(count * 4)}


def dataset_overlay_summary(splits):
    out = {}
    all_graphs = splits["train"] + splits["val"] + splits["test"]
    for topo in TOPOLOGIES:
        gs = [g for g in all_graphs if g["topo"] == topo]
        out[topo] = {
            key: {"mean": float(np.mean([g[key] for g in gs])),
                  "min": int(np.min([g[key] for g in gs])),
                  "max": int(np.max([g[key] for g in gs]))}
            for key in ["overlay_contact_records", "overlay_unique_edges", "overlay_added_edges"]
        }
    return out

# ----------------------------------------------------------------------------
# 7. Staged experimental protocol
#    python experiment.py models <seed> | dp <seed> | fed <seed> | agg
# ----------------------------------------------------------------------------

EPS_TARGETS = [1.0, 2.0, 4.0, 8.0]


def part_path(part, seed):
    names = {
        "models": f"seed_{seed}_models.json",
        "dp": f"seed_{seed}_dp.json",
        "fed": f"seed_{seed}_federated.json",
        "ablations": f"seed_{seed}_ablations.json",
    }
    return os.path.join(OUT, names[part])


def run_models(seed):
    splits = build_dataset(seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    specs = {"LR":  dict(kind="relu", use_graph=False, dims=[FEAT_DIM, 1]),
             "MLP": dict(kind="relu", use_graph=False, dims=[FEAT_DIM, HID, HID, 1]),
             "GCN": dict(kind="relu", use_graph=True,  dims=[FEAT_DIM, HID, HID, 1]),
             "GCN-poly": dict(kind="poly", use_graph=True, dims=[FEAT_DIM, HID, HID, 1])}
    out = {"models": {}, "roc": {},
           "overlay_summary": dataset_overlay_summary(splits)}
    for name, spec in specs.items():
        t0 = time.perf_counter()
        P, pw = train(tr, np.random.default_rng(seed + 1000), kind=spec["kind"],
                      use_graph=spec["use_graph"], dims=spec["dims"])
        t_train = time.perf_counter() - t0
        per_topo = name in ("LR", "MLP", "GCN")
        res, (yt, st) = evaluate(P, spec["kind"], va, te,
                                 use_graph=spec["use_graph"], per_topo=per_topo)
        res["train_time_s"] = t_train
        res["infer_time_per_graph_ms"] = inference_ms(
            P, spec["kind"], te, spec["use_graph"]
        )
        res.update(parameter_summary(P))
        if name == "GCN":
            res["node_loss_mia_auc"] = node_loss_mia_auc(P, "relu", tr, te, pw, np.random.default_rng(seed + 5))
        out["models"][name] = res
        out["roc"][name] = {"y": yt.astype(int).tolist(),
                            "s": np.round(st, 5).tolist()}
        print(f"  {name:9s} F1={res['f1']:.3f} AUC={res['auc']:.3f}", flush=True)
    with open(part_path("models", seed), "w") as f:
        json.dump(out, f)


def run_dp(seed):
    splits = build_dataset(seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    sigmas = {e: sigma_for_eps(e, DP_EPOCHS) for e in EPS_TARGETS}
    out = {"sigmas": {str(e): sigmas[e] for e in EPS_TARGETS}, "by_eps": {}}
    for e in EPS_TARGETS:
        diag = {}
        t0 = time.perf_counter()
        P, pw = train(tr, np.random.default_rng(seed + 2000), kind="relu",
                      epochs=DP_EPOCHS, clip=1.0, sigma=sigmas[e],
                      public_slots=PUBLIC_SLOTS, diagnostics=diag)
        diag["train_time_s"] = time.perf_counter() - t0
        res, _ = evaluate(P, "relu", va, te)
        res["node_loss_mia_auc"] = node_loss_mia_auc(P, "relu", tr, te, pw, np.random.default_rng(seed + 6))
        res["infer_time_per_graph_ms"] = inference_ms(P, "relu", te)
        res.update(parameter_summary(P))
        res["training_diagnostics"] = diag
        out["by_eps"][str(e)] = res
        print(f"  DP eps={e:<4} sigma={sigmas[e]:.2f} F1={res['f1']:.3f} "
              f"node-loss diagnostic={res['node_loss_mia_auc']:.3f}", flush=True)
    with open(part_path("dp", seed), "w") as f:
        json.dump(out, f)


def run_fed(seed):
    splits = build_dataset(seed)
    t0 = time.perf_counter()
    Pf = fedavg(splits, np.random.default_rng(seed + 3000))
    train_time_s = time.perf_counter() - t0
    resf, _ = evaluate(Pf, "relu", splits["val"], splits["test"])
    resf["train_time_s"] = train_time_s
    resf["infer_time_per_graph_ms"] = inference_ms(Pf, "relu", splits["test"])
    resf.update(parameter_summary(Pf))
    resf["communication_bytes_float32"] = int(
        60 * len(TOPOLOGIES) * resf["parameters"] * 4 * 2
    )
    print(f"  FedAvg F1={resf['f1']:.3f} AUC={resf['auc']:.3f}", flush=True)
    with open(part_path("fed", seed), "w") as f:
        json.dump({"final": resf}, f)


def run_ablations(seed):
    """Clipping sweep and a matched feature-only private baseline at eps=4."""
    splits = build_dataset(seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    sigma = sigma_for_eps(4.0, DP_EPOCHS)
    out = {"epsilon": 4.0, "sigma": sigma, "clipping_sweep": {}}
    for clip in CLIP_NORMS:
        diag = {}
        t0 = time.perf_counter()
        P, _ = train(tr, np.random.default_rng(seed + 2000), kind="relu",
                     epochs=DP_EPOCHS, clip=clip, sigma=sigma,
                     public_slots=PUBLIC_SLOTS, diagnostics=diag)
        diag["train_time_s"] = time.perf_counter() - t0
        res, _ = evaluate(P, "relu", va, te)
        res["infer_time_per_graph_ms"] = inference_ms(P, "relu", te)
        res["training_diagnostics"] = diag
        out["clipping_sweep"][str(clip)] = res
        print(f"  clip={clip:<4} F1={res['f1']:.3f} clipped={diag['clipped_fraction']:.3f}", flush=True)

    diag = {}
    t0 = time.perf_counter()
    Pm, _ = train(tr, np.random.default_rng(seed + 2100), kind="relu",
                  use_graph=False, epochs=DP_EPOCHS, clip=1.0, sigma=sigma,
                  public_slots=PUBLIC_SLOTS, diagnostics=diag)
    diag["train_time_s"] = time.perf_counter() - t0
    resm, _ = evaluate(Pm, "relu", va, te, use_graph=False)
    resm["infer_time_per_graph_ms"] = inference_ms(Pm, "relu", te, False)
    resm["training_diagnostics"] = diag
    resm.update(parameter_summary(Pm))
    out["dp_mlp"] = resm
    print(f"  DP-MLP eps=4 F1={resm['f1']:.3f}", flush=True)
    with open(part_path("ablations", seed), "w") as f:
        json.dump(out, f)


def run_cost(seed=11):
    """Single-machine cost benchmark for all implemented deployment profiles."""
    splits = build_dataset(seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    sigma = sigma_for_eps(4.0, DP_EPOCHS)
    rows = {}

    def measure(name, fn, kind="relu", use_graph=True):
        tracemalloc.start()
        t0 = time.perf_counter()
        P = fn()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        row = {"train_time_s": elapsed,
               "infer_time_per_graph_ms": inference_ms(P, kind, te, use_graph),
               "peak_traced_memory_mb": peak / (1024 ** 2)}
        row.update(parameter_summary(P))
        rows[name] = row
        print(f"  {name:12s} train={elapsed:.1f}s peak={row['peak_traced_memory_mb']:.1f}MB", flush=True)
        return P

    measure("GCN", lambda: train(tr, np.random.default_rng(seed + 1000))[0])
    measure("GCN-poly", lambda: train(tr, np.random.default_rng(seed + 1000), kind="poly")[0], kind="poly")
    measure("DP-GCN-eps4", lambda: train(
        tr, np.random.default_rng(seed + 2000), epochs=DP_EPOCHS,
        clip=1.0, sigma=sigma, public_slots=PUBLIC_SLOTS)[0])
    measure("DP-MLP-eps4", lambda: train(
        tr, np.random.default_rng(seed + 2100), use_graph=False,
        epochs=DP_EPOCHS, clip=1.0, sigma=sigma,
        public_slots=PUBLIC_SLOTS)[0], use_graph=False)
    pf = measure("FedAvg", lambda: fedavg(splits, np.random.default_rng(seed + 3000)))
    rows["FedAvg"]["communication_bytes_float32"] = int(
        60 * len(TOPOLOGIES) * rows["FedAvg"]["parameters"] * 4 * 2
    )
    # Inference cost depends on architecture, not learned weights. Re-benchmark
    # the three identical ReLU GCN paths with one common initialized model to
    # avoid thermal/load drift after their different training procedures.
    gcn_probe = init_params(np.random.default_rng(seed + 9000), [FEAT_DIM, HID, HID, 1])
    gcn_ms = inference_ms(gcn_probe, "relu", te, True, repeats=20)
    for name in ["GCN", "DP-GCN-eps4", "FedAvg"]:
        rows[name]["infer_time_per_graph_ms"] = gcn_ms
    rows["GCN-poly"]["infer_time_per_graph_ms"] = inference_ms(
        gcn_probe, "poly", te, True, repeats=20
    )
    rows["DP-MLP-eps4"]["infer_time_per_graph_ms"] = inference_ms(
        gcn_probe, "relu", te, False, repeats=20
    )
    output = os.path.join(OUT, "cost_benchmark.json")
    with open(output, "w") as f:
        json.dump({"seed": seed, "platform": platform.platform(),
                   "python": platform.python_version(), "rows": rows}, f, indent=2)


def aggregate():
    """Aggregate registered per-seed results with sample SD and 95% t CI."""
    def summ(vals):
        v = np.asarray(vals, dtype=float)
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        ci = float(stats.t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v))) if len(v) > 1 else 0.0
        return {"mean": float(v.mean()), "std": sd, "ci95": ci, "n": int(len(v))}

    results = {
        "config": {
            "n_bg": N_BG, "n_bot": N_BOT, "feat_dim": FEAT_DIM,
            "graphs_per_topology": GRAPHS_PER_TOPO, "hidden": HID,
            "epochs": EPOCHS, "dp_epochs": DP_EPOCHS, "learning_rate": LR,
            "delta": DELTA, "seeds": RNG_SEEDS, "clip_norm": 1.0,
            "topologies": TOPOLOGIES,
            "dp_adjacency": "add/remove over fixed public contribution slots",
            "public_slots": PUBLIC_SLOTS,
            "public_positive_class_weight": PUBLIC_POS_WEIGHT,
            "dispersion": "sample standard deviation and two-sided 95% Student-t confidence interval"
        },
        "models": {}, "models_per_topology": {}, "dp": {}, "federated": {},
        "node_loss_diagnostic": {}, "ablations": {}, "overlay_edges": {}
    }

    mods = [json.load(open(part_path("models", s))) for s in RNG_SEEDS]
    dps = [json.load(open(part_path("dp", s))) for s in RNG_SEEDS]
    feds = [json.load(open(part_path("fed", s))) for s in RNG_SEEDS]
    abls = [json.load(open(part_path("ablations", s))) for s in RNG_SEEDS]

    metric_keys = ["precision", "recall", "f1", "auc", "average_precision",
                   "train_time_s", "infer_time_per_graph_ms"]
    for name in ["LR", "MLP", "GCN", "GCN-poly"]:
        results["models"][name] = {
            k: summ([m["models"][name][k] for m in mods]) for k in metric_keys
        }

    for name in ["LR", "MLP", "GCN"]:
        results["models_per_topology"][name] = {
            topo: {
                k: summ([m["models"][name]["per_topo"][topo][k] for m in mods])
                for k in ["precision", "recall", "f1", "auc"]
            }
            for topo in TOPOLOGIES
        }

    results["node_loss_diagnostic"] = {
        "unit": "host-level loss; not aligned with graph-level DP adjacency",
        "nonprivate": summ([m["models"]["GCN"]["node_loss_mia_auc"] for m in mods])
    }

    results["dp"]["sigma_map"] = dps[0]["sigmas"]
    results["dp"]["epsilon_check"] = {
        eps: eps_from_sigma(float(sigma), DP_EPOCHS)
        for eps, sigma in dps[0]["sigmas"].items()
    }
    results["dp"]["by_epsilon"] = {}
    for eps in EPS_TARGETS:
        key = str(eps)
        rec = {
            k: summ([d["by_eps"][key][k] for d in dps])
            for k in ["precision", "recall", "f1", "auc"]
        }
        rec["node_loss_mia_auc"] = summ(
            [d["by_eps"][key]["node_loss_mia_auc"] for d in dps]
        )
        results["dp"]["by_epsilon"][key] = rec

    results["federated"]["final"] = {
        k: summ([f["final"][k] for f in feds])
        for k in ["precision", "recall", "f1", "auc"]
    }
    fed_params = sum(v.size for v in init_params(
        np.random.default_rng(0), [FEAT_DIM, HID, HID, 1]
    ).values())
    results["federated"]["parameters"] = int(fed_params)
    results["federated"]["communication_bytes_float32"] = int(
        60 * len(TOPOLOGIES) * fed_params * 4 * 2
    )
    results["federated"]["protocol_note"] = (
        "FedAvg and centralized training use different optimizers and budgets; "
        "the performance gap is descriptive."
    )
    results["ablations"]["epsilon"] = 4.0
    results["ablations"]["clipping_sweep"] = {
        str(c): {
            k: summ([a["clipping_sweep"][str(c)][k] for a in abls])
            for k in ["precision", "recall", "f1", "auc"]
        } | {
            "clipped_fraction": summ([
                a["clipping_sweep"][str(c)]["training_diagnostics"]["clipped_fraction"]
                for a in abls
            ])
        }
        for c in CLIP_NORMS
    }
    results["ablations"]["dp_mlp_eps4"] = {
        k: summ([a["dp_mlp"][k] for a in abls])
        for k in ["precision", "recall", "f1", "auc"]
    }

    overlay_summaries = [
        m.get("overlay_summary") or dataset_overlay_summary(build_dataset(seed))
        for seed, m in zip(RNG_SEEDS, mods)
    ]
    for topo in TOPOLOGIES:
        results["overlay_edges"][topo] = {}
        for key in ["overlay_contact_records", "overlay_unique_edges", "overlay_added_edges"]:
            vals = [summary[topo][key]["mean"] for summary in overlay_summaries]
            results["overlay_edges"][topo][key] = summ(vals)

    cost_path = os.path.join(OUT, "cost_benchmark.json")
    if os.path.exists(cost_path):
        results["cost_benchmark"] = json.load(open(cost_path))
    results["curve_data_seed_11"] = mods[0]["roc"]
    results["per_seed"] = {
        "models": {name: [m["models"][name] for m in mods]
                   for name in ["LR", "MLP", "GCN", "GCN-poly"]},
        "dp": [d["by_eps"] for d in dps],
        "federated": [f["final"] for f in feds],
        "ablations": abls
    }

    output = os.path.join(OUT, "aggregate_results.json")
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results/aggregate_results.json")


if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    part = sys.argv[1]
    if part == "agg":
        aggregate()
    elif part == "cost":
        run_cost(int(sys.argv[2]) if len(sys.argv) > 2 else 11)
    else:
        seed = int(sys.argv[2])
        print(f"=== {part} seed {seed} ===", flush=True)
        {"models": run_models, "dp": run_dp, "fed": run_fed,
         "ablations": run_ablations}[part](seed)
