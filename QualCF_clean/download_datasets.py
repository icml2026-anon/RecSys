"""
Download datasets for QualCF project using RecBole
"""
import torch
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_logger, init_seed

def download_dataset(dataset_name, config_file):
    """Download a single dataset"""
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}...")
    print(f"{'='*60}\n")

    # Use a simple model name that RecBole recognizes
    config = Config(model='BPR', config_file_list=[config_file])
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    # This will trigger the download if dataset doesn't exist
    dataset = create_dataset(config)

    print(f"\n✓ {dataset_name} downloaded successfully!")
    print(f"  Users: {dataset.user_num}")
    print(f"  Items: {dataset.item_num}")
    print(f"  Interactions: {len(dataset.inter_feat)}")

    return dataset

if __name__ == '__main__':
    datasets = [
        ('MovieLens-1M', 'qualcf.yaml'),
        ('MovieLens-20M', 'qualcf_ml20m.yaml'),
        ('Amazon-Beauty', 'qualcf_beauty.yaml'),
    ]

    for name, config_file in datasets:
        try:
            download_dataset(name, config_file)
        except Exception as e:
            print(f"\n✗ Error downloading {name}: {e}")
            continue

    print(f"\n{'='*60}")
    print("All datasets download complete!")
    print(f"{'='*60}")
