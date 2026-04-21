# QualCF

Steering the Flow: Quality-Aware Personalized Prior Construction for Generative Collaborative Filtering.

## Requirements

- Python >= 3.9
- PyTorch >= 1.12
- [RecBole](https://github.com/RUCAIBox/RecBole) >= 1.2.0

```bash
pip install recbole torch numpy scipy
```

## Datasets

We use three public benchmark datasets: **Amazon-Beauty**, **MovieLens-1M**, and **MovieLens-20M**.

Datasets can be downloaded automatically by RecBole, or manually placed under `dataset/`. See `dataset/README.md` for details.

## Quick Start

### Run QualCF

```bash
# Amazon-Beauty
python run.py --config configs/qualcf_beauty.yaml

```

### Run Baselines

```bash
# Example: run MultiVAE on Amazon-Beauty
python baseline/run_baseline.py --config baseline/multivae/multivae_beauty.yaml
```

Supported baselines: BPR, EASE, LightGCN, SGL, NCL, DGCL, MultiVAE, DiffRec, L-DiffRec, CDiff4Rec, GiffCF.


## Project Structure

```
QualCF/
├── model/
│   └── qualcf.py          # QualCF model
├── baseline/
│   ├── run_baseline.py    # Baseline runner
│   ├── cdiff4rec/         # CDiff4Rec
│   ├── dgcl/              # DGCL
│   ├── giffcf/            # GiffCF
│   └── ...                # BPR, EASE, LightGCN, SGL, NCL, MultiVAE, DiffRec, L-DiffRec
├── configs/
│   ├── qualcf_beauty.yaml # QualCF on Amazon-Beauty 
├── run.py                 # Main entry point
├── utils.py               # Data utilities
└── download_datasets.py   # Dataset download helper
```

## Hyperparameters

Core hyperparameters are **fixed across all datasets**:

| Parameter | Value |
|---|---|
| Flow steps $T$ | 9 |
| Sampling steps $S$ | 2 |
| Time embedding size | 10 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 4096 |
| Early stopping patience | 10 |

Dataset-specific hyperparameters are tuned on the validation set. See config files for details.
