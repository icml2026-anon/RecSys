from recbole.data.utils import get_dataloader # modify the register table in this function!!!
import torch
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})
import sys
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
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
from model import *

import yaml

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run QualCF')
    parser.add_argument('--config', type=str, default='qualcf.yaml',
                        help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        yaml_config = yaml.safe_load(file)
    model_config = yaml_config.get('model', None)

    model_cls = locals().get(model_config)
    if model_cls is None:
        model_cls = get_model(model_config)
    config = Config(model=model_cls, config_file_list=[args.config])
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model_cls = locals().get(config['model'])
    if model_cls is None:
        model_cls = get_model(config['model'])
    model = model_cls(config, train_data.dataset).to(config['device'])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # ---- periodically rebuild retrieval index ----
    import types

    _original_fit = trainer.fit

    def fit_with_retrieval_update(self, train_data, valid_data=None, verbose=True,
                                  saved=True, show_progress=False, callback_fn=None):
        """Wrap Trainer.fit to rebuild retrieval index every N epochs."""
        from recbole.trainer import Trainer

        update_freq = self.model.retrieval_update_freq

        # Initial build with training data only (no data leakage)
        logger.info(set_color("[QualCF] Building initial retrieval index from training data", "green"))
        self.model.build_retrieval_index(train_data.dataset.inter_feat)

        class _Callback:
            def __init__(self, model, freq, train_inter_feat):
                self.model = model
                self.freq = freq
                self.epoch = 0
                self.train_inter_feat = train_inter_feat

            def __call__(self, epoch_idx, valid_score):
                self.epoch = epoch_idx
                if epoch_idx > 0 and epoch_idx % self.freq == 0:
                    logger.info(
                        set_color(f"[QualCF] Rebuilding retrieval index at epoch {epoch_idx}", "green")
                    )
                    self.model.build_retrieval_index(self.train_inter_feat)
                return False  # do not early stop

        cb = _Callback(self.model, update_freq, train_data.dataset.inter_feat)
        return _original_fit(
            train_data, valid_data,
            verbose=verbose, saved=saved,
            show_progress=show_progress,
            callback_fn=cb,
        )

    trainer.fit = types.MethodType(fit_with_retrieval_update, trainer)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
