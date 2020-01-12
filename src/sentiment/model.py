# Standard import
import argparse
import json
import datetime as dt

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributions as dist

# Data science imports
import numpy as np
import pandas as pd

# Pytorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

# Pretrained BERT
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Datasets
from dataset import TwitterCSVDataset, TwitterJSONDataset

class SentimentBERT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.hparams = argparse.Namespace(**config)

        print(f"[{dt.datetime.now()}] Loading tweets dataset")
        dataset = TwitterCSVDataset(config['training_dataset'])
        self.prediction_dataset = TwitterJSONDataset(config['prediction_dataset'])

        training_dataset_size = int(len(dataset) * 0.8)
        [training_dataset, validation_dataset] = torch.utils.data.random_split(dataset, [training_dataset_size, len(dataset) - training_dataset_size])

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        print(f"[INFO] Training dataset size: {len(self.training_dataset)}")
        print(f"[INFO] Validation dataset size: {len(self.validation_dataset)}")
        print(f"[INFO] Prediction dataset size: {len(self.prediction_dataset)}")

        print(f"[{dt.datetime.now()}] Building model")
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        loss, _ = self.bert(batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], labels=batch['sentiment'])
        
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, logits = self.bert(batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], labels=batch['sentiment'])
        
        y_hat = torch.argmax(logits, dim=1).reshape(-1,1)
        acc = (batch['sentiment'] == y_hat).sum()/batch['sentiment'].shape[0]

        logs = {
            'loss': loss,
            'acc': acc
        }

        return logs

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        pass

    def test_end(self, outputs):
        pass

    def configure_optimizers(self):
        # Get optimizer parameters from BERT model
        # Adapted from https://github.com/sobamchan/pytorch-lightning-transformers/blob/master/mrpc.py
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.01
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5,)
        return [optimizer]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=TwitterCSVDataset.collate)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=TwitterCSVDataset.collate)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.prediction_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=TwitterJSONDataset.collate)

def run(model, config):
    tt_logger = TestTubeLogger(
        save_dir="logs",
        name=f"{model.name}",
        debug=False,
        create_git_tag=False
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=config['patience'],
        verbose=False,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=f'logs/{tt_logger.name}/version_{tt_logger.experiment.version}/checkpoints',
        save_best_only=True,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = Trainer(max_nb_epochs=100, early_stop_callback=early_stop_callback, checkpoint_callback=checkpoint_callback, logger=tt_logger, gpus=[0])
    trainer.fit(model)
    trainer.test(model)

def test(model, config):
    tt_logger = TestTubeLogger(
        save_dir="logs/tests",
        name=f"sentiment_bert",
        debug=False,
        create_git_tag=False
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=f'{tt_logger.save_dir}/{tt_logger.name}/version_{tt_logger.experiment.version}/checkpoints',
        save_best_only=True,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = Trainer(max_nb_epochs=1, early_stop_callback=None, train_percent_check=0.1, checkpoint_callback=checkpoint_callback, logger=tt_logger, gpus=[0])
    trainer.fit(model)
    # trainer.test(model)

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    model = SentimentBERT(config)

    if args.test:
        test(model, config)
    else:
        run(model, config)

if __name__ == "__main__":
    # Read arguments
    parser = argparse.ArgumentParser(description="Training SentimentBERT")
    parser.add_argument('-c', '--config', dest='config', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    main(args)