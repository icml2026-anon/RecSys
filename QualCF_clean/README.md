# QualCF

Quality-Aware Collaborative Filtering via Rectified Flow Matching.

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

# MovieLens-1M
python run.py --config configs/qualcf.yaml

# MovieLens-20M
python run.py --config configs/qualcf_ml20m.yaml
```

### Run Baselines

```bash
# Example: run MultiVAE on Amazon-Beauty
python baseline/run_baseline.py --config baseline/multivae/multivae_beauty.yaml
```

Supported baselines: BPR, EASE, LightGCN, SGL, NCL, DGCL, MultiVAE, DiffRec, L-DiffRec, CDiff4Rec, GiffCF.

### Ablation Studies

```bash
# Example: QualCF w/o quality network on Amazon-Beauty
python run.py --config configs/ablation/qualcf_beauty_ablation_wo_quality_net.yaml
```

See `configs/ablation/` for all ablation configurations.

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
│   ├── qualcf.yaml        # QualCF on MovieLens-1M
│   ├── qualcf_ml20m.yaml  # QualCF on MovieLens-20M
│   └── ablation/          # Ablation study configs
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
