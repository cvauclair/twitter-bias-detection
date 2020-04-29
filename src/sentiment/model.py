# Standard import
import argparse
import yaml
import datetime as dt
import glob
import sys
import os

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
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Datasets
from sentiment.dataset import *

class SentimentBERT(pl.LightningModule):
    name = 'sentiment_bert'
    
    def __init__(self, hparams, test_dataset=None):
        super().__init__()

        self.hparams = hparams

        print(f"[{dt.datetime.now()}] Building model")
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.data_loaded = False

    def load_data(self):
        # Load data if not already loaded
        if not self.data_loaded:
            print(f"[{dt.datetime.now()}] Loading tweets dataset")
            dataset = TwitterCSVDataset(self.hparams.training_dataset)

            validation_dataset_size = int(len(dataset) * 0.15)
            testing_dataset_size = int(len(dataset) * 0.05)
            sizes = [len(dataset) - validation_dataset_size - testing_dataset_size, validation_dataset_size, testing_dataset_size]
            self.training_dataset, self.validation_dataset, self.testing_dataset = torch.utils.data.random_split(dataset, sizes)

            print(f"[INFO] Training dataset size: {len(self.training_dataset)}")
            print(f"[INFO] Validation dataset size: {len(self.validation_dataset)}")
            print(f"[INFO] Testing dataset size: {len(self.testing_dataset)}")
            
            self.data_loaded = True

    def forward(self, batch):
        _, logits = self.bert(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['sentiment'])
        y_hat = torch.argmax(logits, dim=1).reshape(-1,1)
        
        return y_hat

    def training_step(self, batch, batch_idx):
        # loss, _ = self.bert(batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], labels=batch['sentiment'])
        loss, _ = self.bert(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['sentiment'])
        
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # loss, logits = self.bert(batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], labels=batch['sentiment'])
        loss, logits = self.bert(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['sentiment'])
        
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
        print(f"[INFO] avg_acc = {avg_acc.item()}")

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        loss, logits = self.bert(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['sentiment'])
        
        y_hat = torch.argmax(logits, dim=1).reshape(-1,1)
        acc = (batch['sentiment'] == y_hat).sum().float()/batch['sentiment'].shape[0]
        spec = ((batch['sentiment'] == y_hat) & (y_hat == 1)).sum()/(y_hat == 1).sum().float()
        sens = ((batch['sentiment'] == y_hat) & (y_hat == 0)).sum()/(y_hat == 0).sum().float()

        logs = {
            'loss': loss,
            'acc': acc,
            'specificity': spec,
            'sensitivity': sens
        }

        return logs

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_specificity = torch.stack([x['specificity'] for x in outputs]).mean()
        avg_sensitivity = torch.stack([x['sensitivity'] for x in outputs]).mean()
        print(f"[INFO] avg_acc = {avg_acc.item()}")
        print(f"[INFO] avg_specificity = {avg_specificity.item()}")
        print(f"[INFO] avg_sensitivity = {avg_sensitivity.item()}")
        test_results = {'accuracy': avg_acc, 'specificity': avg_specificity, 'sensitivity': avg_sensitivity}
        
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        return {'loss': avg_loss, 'log': tensorboard_logs, **test_results}

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
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-6,)
        return [optimizer]

    def train_dataloader(self):
        self.load_data()
        return DataLoader(self.training_dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=TwitterCSVDataset.collate)

    def val_dataloader(self):
        self.load_data()
        return DataLoader(self.validation_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=TwitterCSVDataset.collate)

    def test_dataloader(self):
        self.load_data()
        return DataLoader(self.testing_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=TwitterJSONDataset.collate)

# ================================================================
# Model Wrapper
# ================================================================
class BERTWrapper(object):
    def __init__(self, model_path=None):
        if model_path is not None:
            self.model = SentimentBERT.load_from_checkpoint(model_path).cuda()
        else:
            self.model = None

    def save_config(self, config, path):
        with open(path, 'w') as f:
            json.dump(config, f)

    def get_model_dir(self, logger):
        return f"{logger.save_dir}/{logger.name}/version_{logger.experiment.version}/"

    def get_logger(self, model_name, version: int = None, test: bool = False):
        tt_logger = TestTubeLogger(
            save_dir="logs/tests" if test else "logs",
            name=model_name,
            debug=False,
            create_git_tag=False,
            version=version
        )

        return tt_logger

    def get_checkpoint_callback(self, model_dir):
        checkpoint_callback = ModelCheckpoint(
            filepath=f'{model_dir}/checkpoints',
            save_top_k=1,
            verbose=False,
            monitor='val_loss',
            mode='min',
            prefix=''
        )

        return checkpoint_callback

    def get_early_stopping_callback(self, patience):
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            mode='min',
            min_delta=0.00,
            patience=patience,
            verbose=False,
        )

        return early_stop_callback

    @staticmethod
    def new_model(self, config_path, log_dir):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        
        # Save config for record keeping
        with open(f"{log_dir}/config.yaml", 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper)

        # Create model
        model =  SentimentBERT(argparse.Namespace(**config)).cuda()

        return BERTWrapper(model)

    def test(self, args):
        pass

    def train(self, args):
        # Create trainer
        trainer_args = {}
        trainer_args['logger'] = self.get_logger(SentimentBERT.name, version=args.version, test=args.test)
        trainer_args['checkpoint_callback'] = self.get_checkpoint_callback(model_dir=self.get_model_dir(trainer_args['logger']))
        trainer_args['early_stop_callback'] = self.get_early_stopping_callback(args.patience) if not args.test or args.patience is not None else None
        trainer_args['gpus'] = [0]

        if args.test:
            trainer = Trainer(max_nb_epochs=1, train_percent_check=0.001, val_percent_check=0.001, test_percent_check=0.001, **trainer_args)
        else:
            trainer = Trainer(max_nb_epochs=100, **trainer_args)

        # Load or create model
        if args.version is not None:
            # Find checkpoint file
            ckpt_files = glob.glob(f"{self.get_model_dir(trainer_args['logger'])}/*.ckpt")
            ckpt_files.sort(reverse=True, key=os.path.getmtime)
            if len(ckpt_files) == 0:
                print(f"[WARNING] No checkpoint found for model version {args.version}, creating new model")

                # If no checkpoints are found, attempt to create a new model using the specified config
                try:
                    model = self.create_model(args.config, self.get_model_dir(trainer_args['logger']))
                except AttributeError as e:
                    print(f"[ERROR] No config specified, cannot create new model")
                    sys.exit()
            else: 
                ckpt_path = glob.glob(f"{self.get_model_dir(trainer_args['logger'])}/*.ckpt")[0]
                
                # Warn user if multiple checkpoints are found
                if len(ckpt_files) > 1:
                    print(f"[WARNING] Multiple checkpoints found for model version {args.version}, using latest checkpoint {ckpt_path}")

                # Load model from checkpoint
                print(f"[{dt.datetime.now()}] Loading model state from {ckpt_path}")
                model = SentimentBERT.load_from_checkpoint(ckpt_path)
        else:
            # Create new model from config
            model = self.create_model(args.config, self.get_model_dir(trainer_args['logger']))

        # Train model
        if not args.eval:
            trainer.fit(model)
        trainer.test(model)

    def predict(self, tweets):
        if self.model is None:
            raise Exception("No model specified")

        # Predict
        dataset = TwitterBaseDataset(tweets, cuda=True)
        dataloader = DataLoader(dataset, batch_size=self.model.hparams.batch_size, shuffle=False, collate_fn=TwitterBaseDataset.collate)

        sentiment = []
        for batch in dataloader:
            sentiment.append(self.model(batch))
        
        all_sents = torch.cat(sentiment, dim=0)
        
        return [{'tweet': t, 'sentiment': all_sents[i].item()} for i, t in enumerate(tweets)]

# ================================================================
# Main
# ================================================================

def main(args):
    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    bert_wrapper = BERTWrapper()
    s = bert_wrapper.predict(version=args.version, tweets=["\nObama\u2019s own State Dept. was so concerned about conflicts of interest from Hunter Biden\u2019s role at Burisma that they raised it themselves while prepping the Ambassador for her confirmation. Yet our Democratic colleagues & Adam Schiff cry foul when we dare ask that same question.pic.twitter.com/jZ0UVItU3e\n"])
    print(s)

if __name__ == "__main__":
    # Read arguments
    parser = argparse.ArgumentParser(description="Training SentimentBERT")
    parser.add_argument('-c', '--config', dest='config', type=str, required=True)
    parser.add_argument('-v', '--version', dest='version', type=int, default=None)
    parser.add_argument('-p', '--patience', dest='patience', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    main(args)