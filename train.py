"""
    This script was made by Nick at 19/07/20.
    To implement code for training your model.
"""
import torch

import json
import argparse
from attrdict import AttrDict
import pytorch_lightning as pl

from src.model.net import DCNNClassifier


def main():
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('-c', '--config_file', type=str, required=True)
    cli_args = cli_parser.parse_args()

    hparams = AttrDict(json.load(open(cli_args.config_file)))

    pl.seed_everything(hparams.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = DCNNClassifier(hparams)

    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == '__main__':
    main()
