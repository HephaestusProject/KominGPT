"""
    This script was made by Nick at 19/07/20.
    To implement code of your network using operation from ops.py.
"""

import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from ..data import (TextClassificationDataset,
                    TextClassificationCollate)


class DCNNClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(DCNNClassifier, self).__init__()
        self.hparams = hparams

        self._build_model()
        self._build_loss()

    def _build_model(self):
        self.embeddings = nn.Embedding(self.hparams.num_embeddings, self.hparams.word_embedding_dim)

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(self.hparams.word_embedding_dim, self.hparams.hidden_dim,
                          kernel_size=self.hparams.kernel_size, stride=1,
                          padding=int((self.hparams.kernel_size - 1) / 2)),
                nn.ReLU(),
                nn.BatchNorm1d(self.hparams.hidden_dim)
            ))

        for _ in range(1, self.hparams.n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(self.hparams.hidden_dim, self.hparams.hidden_dim,
                              kernel_size=self.hparams.kernel_size, stride=1,
                              padding=int((self.hparams.kernel_size - 1) / 2)),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hparams.hidden_dim)
                ))

        self.linear = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
        self.logit = nn.Linear(self.hparams.hidden_dim, self.hparams.classes)

    def _build_loss(self):
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        word_embeddings = self.embeddings(inputs)
        conv_hiddens = torch.transpose(word_embeddings, 1, 2)
        conv_hiddens = self.convolutions[0](conv_hiddens)

        for conv in self.convolutions[1:]:
            conv_hiddens = conv(conv_hiddens) + conv_hiddens

        conv_hiddens = torch.transpose(conv_hiddens, 1, 2)

        hiddens = F.relu(self.linear(torch.mean(conv_hiddens, dim=1)))
        logits = self.logit(hiddens)

        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        train_dataset = json.load(open(self.hparams.train_path))

        return DataLoader(TextClassificationDataset(train_dataset, self.hparams),
                          batch_size=self.hparams.batch_size,
                          collate_fn=TextClassificationCollate(self.hparams),
                          drop_last=True)

    def val_dataloader(self):
        if not self.hparams.val_path == '':
            val_dataset = json.load(open(self.hparams.val_path))

            return DataLoader(TextClassificationDataset(val_dataset, self.hparams),
                              batch_size=self.hparams.batch_size,
                              collate_fn=TextClassificationCollate(self.hparams),
                              drop_last=False)
        else:
            raise NotImplementedError()

    def test_dataloader(self):
        if not self.hparams.test_path == '':
            test_dataset = json.load(open(self.hparams.test_path))

            return DataLoader(TextClassificationDataset(test_dataset, self.hparams),
                              batch_size=self.hparams.batch_size,
                              collate_fn=TextClassificationCollate(self.hparams),
                              drop_last=False)
        else:
            raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x, m, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, m, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=-1).detach().cpu()

        return {'val_loss': self.loss(y_hat, y), 'val_acc': accuracy(preds, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, m, y = batch
        y_hat = self(x)

        return {'test_loss': self.loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}

        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}
