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
    name = 'sentiment_bert'
    
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
        acc = (batch['sentiment'] == y_hat).sum().float()/batch['sentiment'].shape[0]

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

# ================================================================
# Main
# ================================================================
def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f)

def get_model_dir(logger):
    return f"{logger.save_dir}/{logger.name}/version_{logger.experiment.version}/"

def get_logger(model_name, version: int = None, test: bool = False):
    tt_logger = TestTubeLogger(
        save_dir="logs/tests" if test else "logs",
        name=model_name,
        debug=False,
        create_git_tag=False,
        version=version
    )

    return tt_logger

def get_checkpoint_callback(model_dir):
    checkpoint_callback = ModelCheckpoint(
        filepath=f'{model_dir}/checkpoints',
        save_best_only=True,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    return checkpoint_callback

def get_early_stopping_callback(patience):
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0.00,
        patience=patience,
        verbose=False,
    )

    return early_stop_callback

def main(args):
    # Create trainer
    trainer_args = {}
    trainer_args['logger'] = get_logger(SentimentBERT.name, version=args.version, test=args.test)
    trainer_args['checkpoint_callback'] = get_checkpoint_callback(model_dir=get_model_dir(trainer_args['logger']))
    trainer_args['early_stop_callback'] = get_early_stopping_callback(args.patience) if not args.test or args.patience is not None else None
    trainer_args['gpus'] = [0]

    if args.test:
        trainer = Trainer(max_nb_epochs=1, train_percent_check=0.01, val_percent_check=0.01, **trainer_args)
    else:
        trainer = Trainer(max_nb_epochs=100, **trainer_args)

    # Load config
    if args.version is not None:
        with open(f"{get_model_dir(trainer_args['logger'])}/config.json", 'r') as f:
            config = json.load(f)
    else:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Save config for record keeping
        with open(f"{get_model_dir(trainer_args['logger'])}/config.json", 'w') as f:
            json.dump(config, f, indent=4)

    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Create model
    # model = SentimentBERT(config)

    # Train and predict
    # trainer.fit(model)
    # trainer.test(model)

if __name__ == "__main__":
    # Read arguments
    parser = argparse.ArgumentParser(description="Training SentimentBERT")
    parser.add_argument('-c', '--config', dest='config', type=str, required=True)
    parser.add_argument('-v', '--version', dest='version', type=int, default=None)
    parser.add_argument('-p', '--patience', dest='patience', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    main(args)