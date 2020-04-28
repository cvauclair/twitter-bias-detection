import yaml
from datetime import datetime
from scraper import Scraper
import json
import pymysql
from rds_controller import RDSController
from .topic_analysis.topic_analysis_controller import TopicAnalysisController


class Pipeline:
    def __init__(self):
        self.scraper_configs = self.get_scraper_configs()
        self.lda_configs = self.get_lda_configs()
        self.filename = self.build_filename()

        # Scraper
        self.scraper = Scraper(num_pages=self.scraper_configs['num_pages'],
                               include_retweets=self.scraper_configs['include_retweets'],
                               checkpoint=self.scraper_configs['checkpoint'],
                               oldest_date=self.scraper_configs['oldest_date'],
                               output_filename=self.filename)

        # RDS Controller
        self.rds_controller = RDSController()

    @staticmethod
    def get_scraper_configs():
        """
        Get scraper configs from YAML file
        :return: configs as a dict
        """
        with open('config.yaml', 'r') as yaml_stream:
            configs = yaml.safe_load(yaml_stream)
        return configs['scraper_configs']

    @staticmethod
    def get_lda_configs():
        """
        Get scraper configs from YAML file
        :return: configs as a dict
        """
        with open('config.yaml', 'r') as yaml_stream:
            configs = yaml.safe_load(yaml_stream)
        return configs['lda_configs']

    # This needs to change because eventually we will have more than
    # one user per file
    def build_filename(self):
        """
        Builds a filename in required format
        :param twitter_username: user who's tweets have been scraped
        :return: formatted filename
        """
        current_date = datetime.now().strftime("%m_%d_%Y")
        return "{}_{}".format('scraped_tweets', current_date)

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

    def run_pipeline(self):
        # ----------------------------------------------
        # STAGE 1
        # Scraping of tweets
        # ----------------------------------------------

        accounts = self.read_json_file(self.scraper_configs['accounts_file'])

        all_tweets = {}
        for u in accounts:
            # Check if user exists. If not, create user.
            user_id = self.scraper.extract_user_id(u['username'], None)
            user = self.rds_controller.get_user(user_id)

            if len(user) == 0:
                try:
                    user_profile = self.scraper.extract_user_profile(self.scraper, u['username'])
                    self.rds_controller.create_user(
                        user_profile['id'],
                        user_profile['username'],
                        user_profile['followers_count'],
                        user_profile['following_count'],
                        user_profile['tweets_count'],
                        user_profile['bio'],
                        user_profile['location'],
                        user_profile['fullname']
                    )
                except pymysql.err.IntegrityError as err:
                    print(f"[Error] Unable to create user {user_profile['username']} in RDS")
                    print(format(err))
                    continue

            user_tweets = self.scraper.scrape_tweets(u['username'])

            if user_tweets:
                all_tweets[u['username']] = user_tweets

        self.write_output(tweets=all_tweets)

        # ----------------------------------------------
        # STAGE 2
        # BERT
        # ----------------------------------------------


        # ----------------------------------------------
        # STAGE 3
        # LDA
        # ----------------------------------------------
        lda_controller = TopicAnalysisController()

        #models = self.read_json_file(self.lda_configs['accounts_file'])
        models = ['realDonaldTrump']
        for u in accounts:
            if u['username'] in models:
                user_tweets = all_tweets[u['username']]
                user_tweets_content = [user_tweets[i][1] for i in range(len(user_tweets))]
                user_tweets_id = [user_tweets[i][0] for i in range(len(user_tweets))]
                if user_tweets:
                    tweets_topics = lda_controller.compute_topic_id_for_tweets(tweets= user_tweets_content, username= u['username'])

                for id, topic in user_tweets_id, tweets_topics:
                    RDSController.set_tweet_topic(tweet_id= id, topic_id= topic)



        # ----------------------------------------------
        # STAGE 4
        # Saving tweets to RDS
        # ----------------------------------------------


        # TODO: This is all changing

# ONLY HERE FOR TESTING
if __name__ == '__main__':
    p = Pipeline()
    p.run_pipeline()
