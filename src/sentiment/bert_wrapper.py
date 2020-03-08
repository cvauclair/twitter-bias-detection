import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TwitterListDataset
from model import SentimentBERT

class BERTWrapper(nn.Module):
    MAX_BATCH_SIZE = 8

    def __init__(self, model_filename:str):
        self.model = SentimentBERT.load_state_dict(torch.load(model_filename))

    def forward(self, tweets: list):
        dataset = TwitterListDataset(tweets)
        dataloader = DataLoader(dataset, batch_size=BERTWrapper.MAX_BATCH_SIZE, shuffle=False, collate_fn=TwitterListDataset.collate)

        sentiment = []
        for batch in dataloader:
            sentiment.append(self.model(batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask']))

        try:
            return 0.0
        except
            return 0.0

