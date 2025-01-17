import datetime as dt
import argparse
from pprint import pprint
import glob
import os

import yaml
from scraper import Scraper, Target
import json
import pymysql
from rds_controller import RDSController
from sentiment.model import BERTWrapper
from topic_analysis.topic_analysis_controller import TopicAnalysisController
from bias.bias import Bias

class Pipeline:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as yaml_stream:
            self.config = yaml.safe_load(yaml_stream)

        self.filename = self.build_filename()

        # Init RDS Controller
        print(f"[{dt.datetime.now()}] Connecting to RDS database")
        self.rds_controller = RDSController()

        # Init scraper
        print(f"[{dt.datetime.now()}] Initializing scraper")
        self.scraper = Scraper(**self.config['scraper_config'], output_filename=self.filename)

        # Init BERT wraper
        print(f"[{dt.datetime.now()}] Initializing BERT sentiment model")
        self.bert_wrapper = BERTWrapper(**self.config['sentiment_config'])

        # Init LDA
        print(f"[{dt.datetime.now()}] Initializing BERT sentiment model")
        self.lda_controller = TopicAnalysisController(**self.config['lda_config'])

        # Init Bias Inference
        # TODO

    # This needs to change because eventually we will have more than
    # one user per file
    def build_filename(self):
        """
        Builds a filename in required format
        :param twitter_username: user who's tweets have been scraped
        :return: formatted filename
        """
        current_date = dt.datetime.now().strftime("%m_%d_%Y")
        return f"scraped_tweets_{current_date}"

    @staticmethod
    def read_json_file(file):
        """
        Reads the existing output file that has just been created
        and should be present in EC2.
        :param file: filename
        :return: list of tweets
        """
        output = []
        with open(file, 'r') as f:
            data = json.load(f)
            for t in data:
                output.append(t)
        return output

    def write_output(self, tweets):
        """
        Write output
        :param tweets:
        :return:
        """
        if self.filename:
            with open(self.filename, 'w') as f:
                json.dump(tweets, f, indent=4)

    def run_pipeline(self, accounts, recompute=False, ignore_date=False):
        # ----------------------------------------------
        # STAGE 1
        # Scraping of tweets
        # ----------------------------------------------

        print(f"[{dt.datetime.now()}] Scraping tweets")
        scraped_tweets = []
        for u in accounts:
            # Check if user exists. If not, create user.
            username = u['username'] if type(u) == dict else u

            target = Target(username)
            user_id = target.get_user_id()

            user = self.rds_controller.get_user(user_id)
            # print(f"[DEBUG] user = {user}")
            if len(user) == 0:
                try:
                    user_profile = target.get_profile()
                    pprint(user_profile)
                    self.rds_controller.create_user(**user_profile)

                except pymysql.err.IntegrityError as err:
                    print(f"[Error] Unable to create user {user_profile['username']} in RDS")
                    print(format(err))
                    continue

            # Get the date of the most recent tweet in the DB
            if ignore_date:
                scraped_tweets += self.scraper.scrape_target(target)                
            else:
                latest_date = self.rds_controller.get_latest_tweet_date(user_id)
                scraped_tweets += self.scraper.scrape_target(target, latest_date)

        # Update DB
        print(f"[{dt.datetime.now()}] Uploading scraped tweets to RDS database")
        for tweet in scraped_tweets:
            try:
                self.rds_controller.create_tweet(**tweet)
            except Exception as e:
                print(f"ERROR: {e}")
                pass

        # ----------------------------------------------
        # STAGE 2
        # BERT
        # ----------------------------------------------
        print(f"[{dt.datetime.now()}] Computing tweets sentiment")

        if recompute:
            all_tweets = []
            for acc in accounts:
                username = acc['username'] if type(acc) == dict else acc
                all_tweets += self.rds_controller.get_tweets_from_user(username)

        else:
            all_tweets = scraped_tweets

        print(f"[DEBUG] num tweets to analyze sentiment: {len(all_tweets)}")

        try:
            tweet_sent = self.bert_wrapper.predict([t['content'] for t in all_tweets])
        except Exception as e:
            print(f"[ERROR] Could not predict tweet sentiment, no sentiment computed")
            print(e)
            tweet_sent = []

        for i, tweet in enumerate(all_tweets):
            tweet['sentiment'] = tweet_sent[i]['sentiment']

        # pprint(all_tweets)

        # Update tweet db
        print(f"[{dt.datetime.now()}] Updating tweets sentiment in database")
        for tweet in all_tweets:
            self.rds_controller.set_tweet_sentiment(
                tweet_id=tweet['tweet_id'],
                sentiment=tweet['sentiment']
            )


        # ----------------------------------------------
        # STAGE 3
        # LDA
        # ----------------------------------------------

        print(f"[{dt.datetime.now()}] Computing tweets topics")

        # Get available LDA models
        models = [f.split('/')[-1] for f in glob.glob('resource/models/lda/*') if os.path.isdir(f)]

        for u in accounts:
            username = u['username'] if type(u) == dict else u
            if username in models:
                user_tweets = list(filter(lambda tweet: tweet['author_username'] == username, all_tweets))
                user_tweets_content = [t['content'] for t in user_tweets]
                user_tweets_id = [t['tweet_id'] for t in user_tweets]

                if user_tweets:
                    # This happens per user
                    try:
                        tweets_topics = self.lda_controller.compute_topic_id_for_tweets(tweets=user_tweets_content, username=username)
                    except:
                        print(f"WARNING: Could not compute topics for {username}")
                        continue

                for i, tweet in enumerate(user_tweets):
                    tweet['topic_id'] = tweets_topics[i]

                # pprint(user_tweets)

                # for id, topic in user_tweets_id, tweets_topics:
                # for i in range(len(user_tweets)):
                #     self.rds_controller.set_tweet_topic(tweet_id=user_tweets[i]['tweet_id'], topic_id=tweets_topics[i])

        # ----------------------------------------------
        # STAGE 4
        # Bias Inference
        # ----------------------------------------------
        biases = {}
        for u in accounts:
            username = u['username'] if type(u) == dict else u
            print(f"[{dt.datetime.now()}] Computing biases for user {username}")

            user_tweets = list(filter(lambda tweet: (tweet['author_username'] == username) and ('topic_id' in tweet), all_tweets))
            user_biases = {}
            topic_tweets = {}

            # Prepare bias inference
            for tweet in user_tweets:
                if tweet['topic_id'] not in user_biases:
                    user_biases[tweet['topic_id']] = Bias()
                if tweet['topic_id'] not in topic_tweets:
                    topic_tweets[tweet['topic_id']] = []

                topic_tweets[tweet['topic_id']].append(tweet)

            for topic in user_biases:
                user_biases[topic].infer2(sentiments=[t['sentiment'] for t in topic_tweets[topic]], **self.config['inference_config'])

            biases[username] = {topic: {
                'topic': self.lda_controller.get_top_words_n_per_topic(topic, 5, username),
                'bias': user_biases[topic].export()
            } for topic in user_biases}

        pprint(all_tweets)
        pprint(biases)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Bias Inference Pipeline")
    parser.add_argument('-c', '--config', dest='config_path', type=str, default='config.yaml')
    parser.add_argument('-a', '--account', dest='account', type=str, default=None)
    parser.add_argument('--accounts-path', dest='accounts_path', type=str, default='accounts.json')
    parser.add_argument('-r', '--recompute', action='store_true')
    parser.add_argument('--ignore-date', dest='ignore_date', action='store_true')
    args = parser.parse_args()
    
    p = Pipeline(args.config_path)

    if args.account is None:
        accounts = Pipeline.read_json_file(args.accounts_path)
        p.run_pipeline(accounts, args.recompute, args.ignore_date)
    else:
        p.run_pipeline([args.account], args.recompute, args.ignore_date)
