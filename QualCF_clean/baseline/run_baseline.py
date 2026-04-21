"""
Baseline experiment runner for QualCF comparison
Runs all baseline models with the same data preprocessing and evaluation.

Supports both custom baseline models (GiffCF, DGCL, CDiff4Rec) and
built-in RecBole models (LightGCN, NCL, SGL, MultiVAE, LDiffRec, BPR, EASE).

Usage:
    python run_baseline.py --config baseline/lightgcn/lightgcn_beauty.yaml
    python run_baseline.py --config baseline/giffcf/giffcf_beauty.yaml
"""

import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from baseline import GiffCF, DGCL, CDiff4Rec

# Custom models that require their own class (not in RecBole registry)
CUSTOM_MODELS = {
    'GiffCF'   : GiffCF,
    'DGCL'     : DGCL,
    'CDiff4Rec': CDiff4Rec,
}

# Built-in RecBole models — resolved via get_model()
RECBOLE_MODELS = {
    'LightGCN', 'NCL', 'SGL', 'MultiVAE', 'LDiffRec', 'BPR', 'EASE',
    'DiffRec', 'NGCF', 'RecVAE', 'MacridVAE', 'NeuMF', 'ItemKNN',
}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Baseline Models for QualCF')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    model_name = yaml_config.get('model', None)

    if model_name is None:
        raise ValueError('YAML config must specify a `model` field.')

    # ── resolve model class ────────────────────────────────────────────────
    if model_name in CUSTOM_MODELS:
        model_class = CUSTOM_MODELS[model_name]
        config = Config(model=model_class, config_file_list=[args.config])
    elif model_name in RECBOLE_MODELS:
        # Let RecBole resolve the model internally
        config = Config(model=model_name, config_file_list=[args.config])
        model_class = get_model(model_name)
    else:
        # Try RecBole registry as fallback
        try:
            model_class = get_model(model_name)
            config = Config(model=model_name, config_file_list=[args.config])
            print(f'[info] Loaded {model_name} from RecBole registry.')
        except Exception:
            raise ValueError(
                f'Unknown model: {model_name}.\n'
                f'Custom models : {list(CUSTOM_MODELS.keys())}\n'
                f'RecBole models: {sorted(RECBOLE_MODELS)}'
            )

    # ── initialise ────────────────────────────────────────────────────────
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # ── dataset ───────────────────────────────────────────────────────────
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # ── model ─────────────────────────────────────────────────────────────
    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config['device'], logger, transform)
    logger.info(set_color('FLOPs', 'blue') + f': {flops}')

    # ── train ─────────────────────────────────────────────────────────────
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config['show_progress']
    )

    # ── evaluate ──────────────────────────────────────────────────────────
    test_result = trainer.evaluate(
        test_data, show_progress=config['show_progress']
    )

    logger.info(
        'The running environment of this training is as follows:\n'
        + get_environment(config).draw()
    )
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
