import yaml
from datetime import datetime
from scraper import Scraper


class Pipeline:
    @staticmethod
    def get_scraper_configs():
        with open('config.yaml', 'r') as yaml_stream:
            configs = yaml.safe_load(yaml_stream)
        return configs['scraper_configs']

    def scrape(self, scraper):
        scraper.scrape_and_output()

    def build_filename(self, twitter_username):
        current_date = datetime.now().strftime("%m_%d_%Y")
        return "{}_{}".format(twitter_username, current_date)

    def run_pipeline(self):
        # STAGE 1
        configs = self.get_scraper_configs()
        scraper = Scraper(username=configs['twitter_username'],
                          num_pages=configs['num_pages'],
                          include_retweets=configs['include_retweets'],
                          checkpoint=configs['checkpoint'],
                          oldest_date=configs['oldest_date'],
                          output_filename=self.build_filename(configs['twitter_username']))
        self.scrape(scraper=scraper)

        # STAGE 2 ..


# ONLY HERE FOR TESTING
if __name__ == '__main__':
    p = Pipeline()
    p.run_pipeline()
