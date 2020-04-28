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

class TwitterBaseDataset(Dataset):
    """
        Base class that represents a dataset of tweets. Designed to be used with a DistilBERT model.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

    def __init__(self, tweets: list, sentiments: list = None, cuda: bool = True):
        self.cuda = cuda

        self.tweets = [TwitterBaseDataset.preprocess_tweet(t) for t in tweets]
        if sentiments is not None:
            self.sentiments = sentiments
            assert len(self.tweets) == len(self.sentiments)
        else:
            self.sentiments = [0 for _ in self.tweets]

    @staticmethod
    def preprocess_tweet(tweet):
        words = tweet.split()
        
        def is_username(word):
            return '@' in word

        url_indicators = ['http', 'www']
        def is_url(word):
            return any([k in word for k in url_indicators])

        filtered_words = list(filter(lambda w: not (is_url(w) or is_username(w)), words))
        return " ".join(filtered_words)

    # Processing adapted from "How to fine-tune BERT with pytorch-lightning"
    # Reference: https://towardsdatascience.com/how-to-fine-tune-bert-with-pytorch-lightning-ba3ad2f928d2
    # Code: https://gist.github.com/sobamchan/93ed747097898a75193096e0f91766f6#file-pl-bert-data-preprocessing-py
    def process_tweet(self, tweet):
        inputs = TwitterBaseDataset.tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=100)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        attention_mask = [1] * len(input_ids)

        # Padd input 
        padding_length = MAX_LENGTH - len(input_ids)
        pad_id = TwitterBaseDataset.tokenizer.pad_token_id
        input_ids = input_ids + ([pad_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_id] * padding_length)

        # Sanity check
        assert len(input_ids) == MAX_LENGTH, "Error with input length {} vs {}".format(len(input_ids), MAX_LENGTH)
        assert len(attention_mask) == MAX_LENGTH, "Error with input length {} vs {}".format(len(attention_mask), MAX_LENGTH)
        # assert len(token_type_ids) == MAX_LENGTH, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LENGTH)

        if self.cuda:
            processed_tweet = {
                'input_ids': torch.cuda.LongTensor([input_ids]),
                'attention_mask': torch.cuda.LongTensor([attention_mask]),
                # 'token_type_ids': torch.cuda.LongTensor([token_type_ids])
            }
        else:
            processed_tweet = {
                'input_ids': torch.LongTensor([input_ids]),
                'attention_mask': torch.LongTensor([attention_mask]),
                # 'token_type_ids': torch.cuda.LongTensor([token_type_ids])
            }

        return processed_tweet

    def get_processed_tweet(self, index, to_numpy: bool = False):
        processed_tweet = self.process_tweet(self.tweets[index])
        if to_numpy:
            return processed_tweet['input_ids'].numpy()
        return processed_tweet

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        processed_tweet = self.process_tweet(self.tweets[index])

        item = {
            'input_ids': processed_tweet['input_ids'],
            'attention_mask': processed_tweet['attention_mask'],
            # 'token_type_ids': processed_tweet['token_type_ids'],
            'sentiment': 0 if self.sentiments[index] == 0 else 1
        }
        
        return item

    @staticmethod
    def collate(items):
        batch = {
            'input_ids': torch.cat([item['input_ids'] for item in items], dim=0),
            'attention_mask': torch.cat([item['attention_mask'] for item in items], dim=0),
            # 'token_type_ids': torch.cat([item['token_type_ids'] for item in items], dim=0),
            'sentiment': torch.cuda.LongTensor([item['sentiment'] for item in items]).reshape(-1,1)
        }

        return batch

class TwitterCSVDataset(TwitterBaseDataset):
    columns = ['sentiment', 'tweetid', 'datetime', 'query_flag', 'username', 'tweet']

    def __init__(self, filename, cuda: bool = True):
        df = pd.read_csv(filename, encoding="ISO-8859-1", names=TwitterCSVDataset.columns)
        
        # Store tweets
        self.tweets = [TwitterBaseDataset.preprocess_tweet(t) for t in df['tweet'].values]
        
        # Store sentiment
        # negative = (df['sentiment'].values == 0).astype(int).reshape(-1,1)
        # positive = (df['sentiment'].values == 4).astype(int).reshape(-1,1)
        # self.sentiment = torch.cuda.LongTensor(np.hstack([negative, positive]))

        self.sentiments = df['sentiment'].values
        self.cuda = cuda

    def __len__(self):
        return self.sentiments.shape[0]

class TwitterJSONDataset(TwitterBaseDataset):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            tweets_meta = json.load(f)
        
        self.tweets = [TwitterBaseDataset.preprocess_tweet(t['text']) for t in tweets_meta]
        self.sentiments = [t['sentiment'] if 'sentiment' in t else 0 for t in tweets_meta]

    @staticmethod
    def collate(items):
        batch = {
            'input_ids': torch.cat([item['input_ids'] for item in items], dim=0),
            'attention_mask': torch.cat([item['attention_mask'] for item in items], dim=0),
            'token_type_ids': torch.cat([item['token_type_ids'] for item in items], dim=0)
        }

        return batch
