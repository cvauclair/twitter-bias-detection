# Standard imports
import json

# Data manipulation imports
import pandas as pd
import numpy as np

# PyTorch imports
import torch
from torch.utils.data import Dataset

from transformers import DistilBertTokenizer

MAX_LENGTH = 100

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Preprocessing adapted from "How to fine-tune BERT with pytorch-lightning"
# Reference: https://towardsdatascience.com/how-to-fine-tune-bert-with-pytorch-lightning-ba3ad2f928d2
# Code: https://gist.github.com/sobamchan/93ed747097898a75193096e0f91766f6#file-pl-bert-data-preprocessing-py
def process_tweet(tweet):
    inputs = tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=100)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    attention_mask = [1] * len(input_ids)

    # Padd input 
    padding_length = MAX_LENGTH - len(input_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_id] * padding_length)

    # Sanity check
    assert len(input_ids) == MAX_LENGTH, "Error with input length {} vs {}".format(len(input_ids), MAX_LENGTH)
    assert len(attention_mask) == MAX_LENGTH, "Error with input length {} vs {}".format(len(attention_mask), MAX_LENGTH)
    assert len(token_type_ids) == MAX_LENGTH, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LENGTH)

    processed_tweet = {
        'input_ids': torch.cuda.LongTensor([input_ids]),
        'attention_mask': torch.cuda.LongTensor([attention_mask]),
        'token_type_ids': torch.cuda.LongTensor([token_type_ids])
    }

    return processed_tweet

class TwitterCSVDataset(Dataset):
    columns = ['sentiment', 'tweetid', 'datetime', 'query_flag', 'username', 'tweet']

    def __init__(self, filename):
        df = pd.read_csv(filename, encoding="ISO-8859-1", names=TwitterCSVDataset.columns)
        
        # GET FIRST 100 TWEETS, FOR TESTING ONLY, REMOVE LATER
        # df = df.iloc[:100]

        # Store tweets
        self.tweets = df['tweet'].values
        
        # Store sentiment
        # negative = (df['sentiment'].values == 0).astype(int).reshape(-1,1)
        # positive = (df['sentiment'].values == 4).astype(int).reshape(-1,1)
        # self.sentiment = torch.cuda.LongTensor(np.hstack([negative, positive]))

        self.sentiment = df['sentiment'].values

    def __len__(self):
        return self.sentiment.shape[0]

    def __getitem__(self, index):
        processed_tweet = process_tweet(self.tweets[index])

        item = {
            'input_ids': processed_tweet['input_ids'],
            'attention_mask': processed_tweet['attention_mask'],
            'token_type_ids': processed_tweet['token_type_ids'],
            'sentiment': 0 if self.sentiment[index] == 0 else 1
        }
        
        return item

    @staticmethod
    def collate(items):
        batch = {
            'input_ids': torch.cat([item['input_ids'] for item in items], dim=0),
            'attention_mask': torch.cat([item['attention_mask'] for item in items], dim=0),
            'token_type_ids': torch.cat([item['token_type_ids'] for item in items], dim=0),
            'sentiment': torch.LongTensor([item['sentiment'] for item in items]).reshape(-1,1)
        }

        return batch

class TwitterJSONDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.tweets = json.load(f)

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        processed_tweet = process_tweet(self.tweets[index]['text'])

        item = {
            'url': self.tweets[index]['url'],       # Include URL to be able to add back sentiment to database
            'input_ids': processed_tweet['input_ids'],
            'attention_mask': processed_tweet['attention_mask'],
            'token_type_ids': processed_tweet['token_type_ids']
        }
        
        return item

    @staticmethod
    def collate(items):
        batch = {
            'input_ids': torch.cat([item['input_ids'] for item in items], dim=0),
            'attention_mask': torch.cat([item['attention_mask'] for item in items], dim=0),
            'token_type_ids': torch.cat([item['token_type_ids'] for item in items], dim=0)
        }

        return batch

class TwitterListDataset(Dataset):
    def __init__(self, tweets):
        self.tweets = tweets

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        processed_tweet = process_tweet(self.tweets[index]['text'])

        item = {
            'url': self.tweets[index]['url'],       # Include URL to be able to add back sentiment to database
            'input_ids': processed_tweet['input_ids'],
            'attention_mask': processed_tweet['attention_mask'],
            'token_type_ids': processed_tweet['token_type_ids']
        }
        
        return item

    @staticmethod
    def collate(items):
        batch = {
            'input_ids': torch.cat([item['input_ids'] for item in items], dim=0),
            'attention_mask': torch.cat([item['attention_mask'] for item in items], dim=0),
            'token_type_ids': torch.cat([item['token_type_ids'] for item in items], dim=0)
        }

        return batch