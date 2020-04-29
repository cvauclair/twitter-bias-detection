import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import gensim
import string
from collections import OrderedDict
from gensim import corpora, models, similarities
from gensim.corpora.dictionary import Dictionary
import seaborn as sns
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from topic_analysis.text_preprocessing import LDAHelper

class TopicAnalysisController(object):
    def __init__(self, model_path=None, dictionary_path=None, username=None):
        if model_path is not None:
            self.model = models.LdaModel.load(model_path.format(username=username))
        else:
            self.model = None

        if dictionary_path is not None:
            self.dictionary = Dictionary.load(dictionary_path.format(username=username))
        else:
            self.dictionary = None
            
    def compute_topic_id_for_tweets(self, tweets):

        lda_helper = LDAHelper()
        corpus_exp = [lda_helper.preprocess_without_LS(t) for t in tweets]
        bow_corpus = [self.dictionary.doc2bow(text) for text in corpus_exp]
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        '''
        Here we get a list of list of tuples for each tweet in the form
        L1 = [(0, 0.03669953),
             (1, 0.45451155),
             (2, 0.092942655),
             (3, 0.035580307),
             (4, 0.38026598)]
        
        The first number of the tuple is the TOPIC ID, the second is the PROBABILITY THAT
        SAID TWEET BELONGS TO TOPIC ID. We get a list like L1 for each tweet.
        '''
        prob_distributions = [self.model[corpus_tfidf[i]] for i in range(len(tweets))]
        topics = self.get_all_most_probable_topic_ids(prob_distributions)
        return topics

    def get_all_most_probable_topic_ids(self, prob_distributions):
        result = []
        for p in prob_distributions:
            result.append(self.get_most_max_probability_topic(p))
        return result

    def get_top_words_n_per_topic(self, topic_id, n):
        top_words = self.model.show_topic(topic_id, n)
        top_n = [w[0] for w in top_words]
        return top_n

    @staticmethod
    def get_most_max_probability_topic(prob_distribution):
        max_prob = 0
        max_topic = prob_distribution[0][0]

        for topic in prob_distribution:
            if topic[1] > max_prob:
                max_prob = topic[1]
                max_topic = topic[0]

        return max_topic




