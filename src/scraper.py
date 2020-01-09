from bs4 import BeautifulSoup as soup

import argparse
import sys, traceback
import json
import parsedatetime as pdt
import time
import scraper_utils

TWITTER_ROOT_URL = 'https://twitter.com'


class Scraper:
    def __init__(self,
                 username: str,
                 num_pages: int,
                 include_retweets: bool,
                 checkpoint: str,
                 oldest_date: str,
                 output_filename: str):

        self.username = username
        self.num_pages = num_pages
        self.include_retweets = include_retweets
        self.checkpoint = checkpoint
        self.oldest_date = oldest_date
        self.output_filename = output_filename

    @staticmethod
    def get_url(tweet):
        try:
            return tweet.find('div')['data-permalink-path']
        except:
            return 'ERROR'

    @staticmethod
    def get_text(tweet):
        try:
            return tweet.find('div', {'class': 'js-tweet-text-container'}).text
        except:
            return 'ERROR'

    @staticmethod
    def get_date(tweet):
        try:
            return tweet.find('small', {'class': 'time'}).a['title']
        except:
            return 'ERROR'

    @staticmethod
    def get_num_likes(tweet):
        try:
            tweet_stats = [count['data-tweet-stat-count'] for count in tweet.find_all('span', {'class': 'ProfileTweet-actionCount'}) if count.has_attr('data-tweet-stat-count')]
            return tweet_stats[2]
        except:
            return 'ERROR'

    @staticmethod
    def get_num_replies(tweet):
        try:
            tweet_stats = [count['data-tweet-stat-count'] for count in tweet.find_all('span', {'class': 'ProfileTweet-actionCount'}) if count.has_attr('data-tweet-stat-count')]
            return tweet_stats[0]
        except:
            return 'ERROR'

    @staticmethod
    def get_num_retweets(tweet):
        try:
            tweet_stats = [count['data-tweet-stat-count'] for count in tweet.find_all('span', {'class': 'ProfileTweet-actionCount'}) if count.has_attr('data-tweet-stat-count')]
            return tweet_stats[1]
        except:
            return 'ERROR'

    @staticmethod
    def is_tweet_old(tweet: str, oldest_date: str):
        # Compares if tweet date is older than oldest_date
        split_date = tweet['originalDate'].split(" - ")
        calendar = pdt.Calendar()
        return calendar.parseDT(split_date[1]) < calendar.parseDT(oldest_date)

    def extract_tweets(self, page_soup, include_retweets: bool, oldest_date: str):
        # Set inclusion function depending on flags
        if include_retweets:
            include = lambda t: True
        else:
            include = lambda t: not t.div.has_attr('data-retweeter')

        # Extract tweets
        tweets = [{
            'url': TWITTER_ROOT_URL + self.get_url(t),
            'originalDate': self.get_date(t),
            'text': self.get_text(t),
            'num_replies': self.get_num_replies(t),
            'num_retweets': self.get_num_retweets(t),
            'num_likes': self.get_num_likes(t),
            'isRetweet': t.div.has_attr('data-retweeter')
        } for t in page_soup.find_all('li', {'data-item-type': 'tweet'}) if include(t)]

        # Remove tweets that are older than oldest_date
        # Note: Twitter does NOT give date when tweet was retweeted. So retweets are straight away added.
        fast_exit = False
        filtered_tweets = []
        if oldest_date is not None:
            for tweet in tweets:
                if tweet['isRetweet']:
                    filtered_tweets.append(tweet)
                elif self.is_tweet_old(tweet, oldest_date):
                    fast_exit = True
                    break
                else:
                    filtered_tweets.append(tweet)
            return filtered_tweets, fast_exit

        return tweets, fast_exit

    def scrape_tweets(self):
        # Get user twitter url
        twitter_user_url = f'{TWITTER_ROOT_URL}/{self.username}'

        tweets = []
        try:
            fast_exit = False
            while len(tweets) < self.num_pages:
                if len(tweets) == 0:
                    # If first page scraped, simply get root page of user
                    html = scraper_utils.get_page(twitter_user_url)
                    page_soup = soup(html, 'html.parser')

                    # Check for error on twitter page
                    if page_soup.find("div", {"class": "errorpage-topbar"}):
                        print(f"[Error] Invalid username {self.username}")
                        sys.exit(1)

                    # Add tweets
                    result = self.extract_tweets(page_soup, self.include_retweets, self.oldest_date)
                    tweets += result[0]
                    fast_exit = result[1]

                    # Ptr to next page of tweets
                    ptr = page_soup.find("div", {"class": "stream-container"})["data-min-position"]
                else:
                    # Make request to get next tweets
                    if self.checkpoint:
                        raw_data = json.loads(scraper_utils.get_page(self.checkpoint))
                        checkpoint = None
                    else:
                        raw_data = json.loads(scraper_utils.get_page(f'https://twitter.com/i/profiles/show/{self.username}/timeline/tweets?include_available_features=1&include_entities=1&max_position={ptr}&reset_error_state=false'))

                    if not raw_data["has_more_items"] and not raw_data["min_position"]:
                        print("[INFO] No more tweets returned")
                        break

                    page_soup = soup(raw_data['items_html'], 'html.parser')

                    # Add tweets
                    result = self.extract_tweets(page_soup, self.include_retweets, self.oldest_date)
                    tweets += result[0]
                    fast_exit = result[1]

                    ptr = raw_data['min_position']

                # Perform fast exit if we have passed oldest_date during scraping
                if fast_exit:
                    break

                time.sleep(1)
        except:
            print(f"[ERROR] {sys.exc_info()[0]}")
            traceback.print_exc(file=sys.stdout)

        print(f"[INFO] {len(tweets)} scraped!")

        return tweets

    def write_output(self, tweets):
        if self.output_filename:
            with open(self.output_filename, 'a+') as f:
                json.dump(tweets, f, indent=4)

    def scrape_and_output(self):
        tweets = self.scrape_tweets()
        self.write_output(tweets=tweets)


# Only here for testing
def main():
    # Check args
    parser = argparse.ArgumentParser(description='Scrape tweets')
    parser.add_argument('twitter_username', type=str)
    parser.add_argument('-n', dest='num_pages', type=int)
    parser.add_argument('-o', dest='output_filename', type=str)
    parser.add_argument('--include_retweets', default=True, type=bool)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--oldest_date', default=None, type=str)
    args = parser.parse_args()

    scraper = Scraper(username=args.twitter_username,
                      num_pages=args.num_pages,
                      include_retweets=args.include_retweets,
                      checkpoint=args.checkpoint,
                      oldest_date=args.oldest_date,
                      output_filename=args.output_filename)

    tweets = scraper.scrape_tweets()

    if args.output_filename:
        with open(args.output_filename, 'a+') as f:
            json.dump(tweets, f, indent=4)

