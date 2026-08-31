# PP-GBD Reproducibility Package

**Paper:** *PP-GBD: Privacy-Preserving Graph Neural Botnet Detection in Encrypted Traffic*  
**Version:** 1.0.0  

This repository contains the synthetic benchmark generator, NumPy/SciPy implementation, per-seed results, privacy accountant, figure-generation scripts, and result tables used in the paper.

## Scientific scope

- **Graph-level differential privacy:** add/remove adjacency is defined over 32 public graph-contribution slots. An empty slot contributes a zero gradient, and the optimizer always divides by the fixed public slot count. The clipped gradient sum has L2 sensitivity `C`, and the averaged gradient has sensitivity `C/B`. Replace-one adjacency would require sensitivity `2C` for the sum and is not claimed.
- **Protected dataset:** the formal privacy guarantee covers the 32 training contribution slots. Validation and test graphs are held out and are outside the protected dataset. Selecting a threshold from sensitive validation data would require an additional private selection mechanism.
- **Polynomial inference architecture:** the polynomial GCN uses additions and multiplications suitable for future CKKS-based encrypted-feature inference when the graph propagation operator is known to the service provider. This release does not include a CKKS implementation or a confidential-topology protocol.
- **Federated learning:** FedAvg is evaluated across four topology-partitioned clients. Secure aggregation is compatible with the design but is not implemented. Centralized and federated training use different optimization protocols, so their performance difference is descriptive.
- **Privacy diagnostic:** the loss-threshold diagnostic operates on host losses. It is not aligned with the protected whole-graph record and does not validate graph-level privacy.
- **Data scope:** all benchmark graphs are synthetic. The release contains no operational network traffic, personal network records, or human-subject data.

## Repository structure

```text
PP-GBD/
├── README.md
├── CITATION.cff
├── .zenodo.json
├── LICENSE
├── requirements.txt
├── run_all.sh
├── SHA256SUMS
├── code/
│   ├── experiment.py
│   ├── figures_static.py
│   └── figures_results.py
├── results/
│   ├── seed_*_models.json
│   ├── seed_*_dp.json
│   ├── seed_*_federated.json
│   ├── aggregate_results.json
│   └── table_*.csv
└── figures/
    ├── fig1_architecture.png
    ├── fig2_topologies.png
    ├── fig3_roc_pr.png
    └── fig4_privacy_utility.png
```

## Installation

Python 3.13.5 was used for release validation.

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce aggregate results and figures

The supplied per-seed files allow the aggregate tables and figures to be regenerated without retraining:

```bash
python code/experiment.py agg
python code/figures_static.py
python code/figures_results.py
```

To rerun the complete CPU-based experiment suite:

```bash
bash run_all.sh
```

The complete training suite is more computationally expensive than regenerating the aggregate outputs.

## Methodological notes

Each graph contains 3,128 hosts: 3,000 benign background nodes and 128 bots. The benchmark includes centralized C2, de Bruijn, Kademlia-style, and Chord-style overlays. Features are standardized within each graph. A positive feature-wise affine perturbation is applied before standardization; because z-standardization is affine invariant, this perturbation does not represent a persistent cross-domain shift.

Summary values use the mean and sample standard deviation (`ddof=1`) across seeds 11, 23, and 47. The three-seed dispersion is descriptive and should not be interpreted as a precise confidence interval.

## Integrity check

```bash
sha256sum -c SHA256SUMS
```

On Windows PowerShell, individual files can be checked with `Get-FileHash -Algorithm SHA256`.

## License and citation

The complete repository is distributed under the MIT License. Citation metadata are provided in `CITATION.cff` and `.zenodo.json`.
