import argparse

import torch
import numpy as np

from model import BERTWrapper
from dataset import TwitterBaseDataset

def main(args):
    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    bert_wrapper = BERTWrapper()
    bert_wrapper.train(args)

if __name__ == "__main__":
    # Read arguments
    parser = argparse.ArgumentParser(description="Training SentimentBERT")
    parser.add_argument('-c', '--config', dest='config', type=str, required=True)
    parser.add_argument('-v', '--version', dest='version', type=int, default=None)
    parser.add_argument('-p', '--patience', dest='patience', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    main(args)