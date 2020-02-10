import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import gensim


class LDAHelper:
    def __init__(self):
        self.version = 0
        self.stopwords = [
            'RT',
            'rt'
        ] + stopwords.words('english') + list(punctuation)

    def clean_up(self, word):
        """
        Clean up punctuation.
        :param word: full length tweet
        :return: Cleaned up tweet
        """
        return word.translate(str.maketrans('', '', punctuation))

    def lemmatize_stemming(self, word_without_punctuation):
        """
        Lemmatize and stemming a word.
        :param word_without_punctuation:
        :return: resulting word
        """
        word_without_punctuation = self.clean_up(word_without_punctuation)
        stemmer = SnowballStemmer('english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(word_without_punctuation, pos='v'))

    def valid(self, word):
        """
        Verify the validity of a word.
        :param word: word in a tweet
        :return: True if the word passes a set of rules. To be formalized later
        """
        return word not in self.stopwords and word[:1] != '@' and word[:4] != 'http'

    def preprocess(self, tweet):
        """
        Full preprocess pipeline.
        Applying lemmatization and stemming to any word that is not part of
        our stopwords
        :param tweet: Full tweet
        :return: Array of processed words
        """
        return [self.lemmatize_stemming(word)
                for word in gensim.utils.simple_preprocess(tweet, min_len=2)
                if word not in self.stopwords]

# ONLY HERE FOR TESTING
if __name__ == '__main__':
    l = LDAHelper()
