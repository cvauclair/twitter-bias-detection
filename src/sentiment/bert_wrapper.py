import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TwitterListDataset
from model import SentimentBERT

class BERTWrapper(nn.Module):
    MAX_BATCH_SIZE = 4

    def __init__(self, checkpoint_path: str):
        # Load model config

        self.model = SentimentBERT.load_from_checkpoint(checkpoint_path)

    def forward(self, tweets: list):
        dataset = TwitterListDataset(tweets)
        dataloader = DataLoader(dataset, batch_size=BERTWrapper.MAX_BATCH_SIZE, shuffle=False, collate_fn=TwitterListDataset.collate)

        sentiment = []
        for batch in dataloader:
            sentiment.append(self.model(batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask']))
        
        all_sents = torch.stack(sentiment, dim=0)
        for i, t in enumerate(tweets):
            t['sentiment'] = all_sents[i]

        return tweets 

