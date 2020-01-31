import yaml
from datetime import datetime
from scraper import Scraper
import json
from rds_controller import RDSController


class Pipeline:
    def __init__(self):
        self.configs = self.get_scraper_configs()
        self.filename = self.build_filename(self.configs['twitter_username'])

        # Scraper
        self.scraper = Scraper(username=self.configs['twitter_username'],
                          num_pages=self.configs['num_pages'],
                          include_retweets=self.configs['include_retweets'],
                          checkpoint=self.configs['checkpoint'],
                          oldest_date=self.configs['oldest_date'],
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

    # This needs to change because eventually we will have more than
    # one user per file
    def build_filename(self, twitter_username):
        """
        Builds a filename in required format
        :param twitter_username: user who's tweets have been scraped
        :return: formatted filename
        """
        current_date = datetime.now().strftime("%m_%d_%Y")
        return "{}_{}".format(twitter_username, current_date)

    @staticmethod
    def read_output(file):
        """
        Reads the existing output file that has just been created
        and should be present in EC2.
        :param file: filename
        :return: list of tweets
        """
        output = []
        with open(file, 'r') as f:
            tweets = json.load(f)
            for t in tweets:
                output.append(t)
        print(output)
        return output


    def run_pipeline(self):
        # ----------------------------------------------
        # STAGE 1
        # Scraping of tweets
        # ----------------------------------------------

        self.scraper.scrape_and_output()
        scraped_tweets = self.read_output(self.filename)

        # ----------------------------------------------
        # STAGE 2
        # Saving tweets to RDS
        # ----------------------------------------------


        #This needs to change once we have different users per tweet
        for t in scraped_tweets:
            id = t['url'].split('/')[-1]
            tweet_author = t['url'].split('/')[-3]
            # Create user if not existent here
            self.rds_controller.create_tweet(id=id,
                                        author_id=tweet_author,
                                        tweeted_on=t['originalDate'],
                                        posted_by_id=self.configs['twitter_username'],
                                        is_retweet=t['isRetweet'],
                                        num_likes=t['num_likes'],
                                        num_retweets=t['num_retweets'],
                                        num_replies=t['num_replies'],
                                        content=t['text'])


# ONLY HERE FOR TESTING
if __name__ == '__main__':
    p = Pipeline()
    p.run_pipeline()
