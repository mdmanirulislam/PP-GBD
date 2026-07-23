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

Run:  python experiment.py models|dp|fed <seed>, then python experiment.py agg
"""

import json, math, os, time
import numpy as np
import networkx as nx
import scipy.sparse as sp

RNG_SEEDS = [11, 23, 47]
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
    for (a, b) in botnet_overlay_edges(bot_ids, topo, rng):
        g.add_edge(int(bot_ids[a]), int(bot_ids[b]))
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
    return {"A": A_hat, "X": X, "y": y, "AX": A_hat @ X, "topo": topo}


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
          clip=None, sigma=0.0, dims=None, pos_w=None, public_slots=None):
    dims = dims or [FEAT_DIM, HID, HID, 1]
    if pos_w is None:
        pos = sum(g["y"].sum() for g in graphs); tot = sum(len(g["y"]) for g in graphs)
        pos_w = (tot - pos) / pos
    P = init_params(rng, dims)
    keys = sorted(P.keys())
    m = {k: np.zeros_like(P[k]) for k in keys}
    v = {k: np.zeros_like(P[k]) for k in keys}
    b1, b2, adam_eps = 0.9, 0.999, 1e-8
    B = len(graphs) if public_slots is None else int(public_slots)
    if B < len(graphs):
        raise ValueError("public_slots cannot be smaller than the number of supplied graphs")
    # When public_slots > len(graphs), absent slots contribute the zero gradient.
    # The denominator is public and fixed, which is required by the add/remove
    # graph-adjacency proof used for the DP experiments.
    for t in range(1, epochs + 1):
        acc = np.zeros(sum(P[k].size for k in keys))
        for g in graphs:
            _, G, _ = loss_and_grads(P, g, kind, pos_w, use_graph)
            gv = flat(G, keys)
            if clip is not None:
                gv = gv * min(1.0, clip / (np.linalg.norm(gv) + 1e-12))
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
    pos = sum(g["y"].sum() for g in splits["train"]); tot = sum(len(g["y"]) for g in splits["train"])
    pos_w = (tot - pos) / pos
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
    }
    return os.path.join(OUT, names[part])


def run_models(seed):
    splits = build_dataset(seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    specs = {"LR":  dict(kind="relu", use_graph=False, dims=[FEAT_DIM, 1]),
             "MLP": dict(kind="relu", use_graph=False, dims=[FEAT_DIM, HID, HID, 1]),
             "GCN": dict(kind="relu", use_graph=True,  dims=[FEAT_DIM, HID, HID, 1]),
             "GCN-poly": dict(kind="poly", use_graph=True, dims=[FEAT_DIM, HID, HID, 1])}
    out = {"models": {}, "roc": {}}
    for name, spec in specs.items():
        t0 = time.time()
        P, pw = train(tr, np.random.default_rng(seed + 1000), kind=spec["kind"],
                      use_graph=spec["use_graph"], dims=spec["dims"])
        t_train = time.time() - t0
        per_topo = name in ("LR", "MLP", "GCN")
        res, (yt, st) = evaluate(P, spec["kind"], va, te,
                                 use_graph=spec["use_graph"], per_topo=per_topo)
        res["train_time_s"] = t_train
        t0 = time.time()
        for g in te:
            predict(P, g, spec["kind"], spec["use_graph"])
        res["infer_time_per_graph_ms"] = (time.time() - t0) / len(te) * 1e3
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
        P, pw = train(tr, np.random.default_rng(seed + 2000), kind="relu",
                      epochs=DP_EPOCHS, clip=1.0, sigma=sigmas[e],
                      public_slots=PUBLIC_SLOTS)
        res, _ = evaluate(P, "relu", va, te)
        res["node_loss_mia_auc"] = node_loss_mia_auc(P, "relu", tr, te, pw, np.random.default_rng(seed + 6))
        out["by_eps"][str(e)] = res
        print(f"  DP eps={e:<4} sigma={sigmas[e]:.2f} F1={res['f1']:.3f} "
              f"node-loss diagnostic={res['node_loss_mia_auc']:.3f}", flush=True)
    with open(part_path("dp", seed), "w") as f:
        json.dump(out, f)


def run_fed(seed):
    splits = build_dataset(seed)
    Pf = fedavg(splits, np.random.default_rng(seed + 3000))
    resf, _ = evaluate(Pf, "relu", splits["val"], splits["test"])
    print(f"  FedAvg F1={resf['f1']:.3f} AUC={resf['auc']:.3f}", flush=True)
    with open(part_path("fed", seed), "w") as f:
        json.dump({"final": resf}, f)


def aggregate():
    """Aggregate registered per-seed results using sample standard deviation."""
    def summ(vals):
        v = np.asarray(vals, dtype=float)
        return {"mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    results = {
        "config": {
            "n_bg": N_BG, "n_bot": N_BOT, "feat_dim": FEAT_DIM,
            "graphs_per_topology": GRAPHS_PER_TOPO, "hidden": HID,
            "epochs": EPOCHS, "dp_epochs": DP_EPOCHS, "learning_rate": LR,
            "delta": DELTA, "seeds": RNG_SEEDS, "clip_norm": 1.0,
            "topologies": TOPOLOGIES,
            "dp_adjacency": "add/remove over fixed public contribution slots",
            "public_slots": PUBLIC_SLOTS,
            "dispersion": "sample standard deviation (ddof=1)"
        },
        "models": {}, "models_per_topology": {}, "dp": {}, "federated": {},
        "node_loss_diagnostic": {}
    }

    mods = [json.load(open(part_path("models", s))) for s in RNG_SEEDS]
    dps = [json.load(open(part_path("dp", s))) for s in RNG_SEEDS]
    feds = [json.load(open(part_path("fed", s))) for s in RNG_SEEDS]

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
    results["federated"]["protocol_note"] = (
        "FedAvg and centralized training use different optimizers and budgets; "
        "the performance gap is descriptive."
    )
    results["curve_data_seed_11"] = mods[0]["roc"]
    results["per_seed"] = {
        "models": {name: [m["models"][name] for m in mods]
                   for name in ["LR", "MLP", "GCN", "GCN-poly"]},
        "dp": [d["by_eps"] for d in dps],
        "federated": [f["final"] for f in feds]
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
    else:
        seed = int(sys.argv[2])
        print(f"=== {part} seed {seed} ===", flush=True)
        {"models": run_models, "dp": run_dp, "fed": run_fed}[part](seed)
