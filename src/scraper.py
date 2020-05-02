from bs4 import BeautifulSoup as soup

import datetime as dt
import argparse
import sys, traceback
import json
import parsedatetime as pdt
import time
import scraper_utils

TWITTER_ROOT_URL = 'https://twitter.com'

class Target(object):
    def __init__(self, username):
        self.username = username
        
        # Init user
        twitter_user_url = f"{TWITTER_ROOT_URL}/{self.username}"
        html = scraper_utils.get_page(twitter_user_url)
        page_soup = soup(html, 'html.parser')

        profile_nav = page_soup.find('div', {'role': 'navigation'})
        user_id = profile_nav['data-user-id']

        try:
            # Get counts of tweets, following, and followers
            num_tweets, num_following, num_followers = [
                profile_nav['data-count']
                for profile_nav in page_soup.find_all('span', {'class': 'ProfileNav-value'}, limit=3)
            ]
        except:
            num_tweets, num_following, num_followers = 0, 0, 0

        try:
            # Get bio
            bio = page_soup.find('p', {'class': 'ProfileHeaderCard-bio u-dir'}).text
        except:
            bio = ""

        try:
            # Get location
            location = page_soup.find('span', {'class': 'ProfileHeaderCard-locationText u-dir'}).text
        except:
            location = ""
        
        try:
            # Get full name
            fullname = page_soup.find('a', {'class': 'ProfileHeaderCard-nameLink u-textInheritColor js-nav'}).text
        except:
            fullname = ""

        self.profile = {
            'user_id': user_id,
            'username': username,
            'num_tweets': num_tweets,
            'num_following': num_following,
            'num_followers': num_followers,
            'bio': bio,
            'location': location,
            'fullname': fullname
        }

        self.next_page_soup = page_soup
        self.page_counter = 0
        self.ptr = None

    def get_profile(self):
        return self.profile

    def get_user_id(self):
        return self.profile['user_id']

    def get_next_page_soup(self):
        if self.next_page_soup is None:
            return None

        # Set the current page soup to be returned
        current_page_soup = self.next_page_soup

        # Ptr to next page of tweets
        if self.page_counter == 0:
            ptr = current_page_soup.find("div", {"class": "stream-container"})["data-min-position"]
        else:
            ptr = self.ptr

        # Prepare the next page soup
        raw_data = json.loads(scraper_utils.get_page(
            f'https://twitter.com/i/profiles/show/{self.username}/timeline/tweets?include_available_features=1&include_entities=1&max_position={ptr}&reset_error_state=false'
        ))
        self.ptr = raw_data['min_position']

        if not raw_data["has_more_items"] and not raw_data["min_position"]:
            print("[INFO] No more tweets returned")
            self.next_page_soup = None
        else:
            self.next_page_soup = soup(raw_data['items_html'], 'html.parser')

        self.page_counter += 1

        return current_page_soup

class Scraper(object):
    def __init__(self,
                 num_pages: int,
                 include_retweets: bool,
                 checkpoint: str,
                 oldest_date: str,
                 output_filename: str):

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
        split_date = tweet['tweeted_on'].split(" - ")
        calendar = pdt.Calendar()
        return calendar.parseDT(split_date[1]) < calendar.parseDT(oldest_date)

    @staticmethod
    def extract_author_username(url: str):
        return url.split('/')[-3]

    @staticmethod
    def extract_tweet_id(url: str):
        return url.split('/')[-1]

    def extract_tweets(self, page_soup, include_retweets: bool, target: Target):
        # Set inclusion function depending on flags
        if include_retweets:
            include = lambda t: True
        else:
            include = lambda t: not t.div.has_attr('data-retweeter')

        # Initialize user id
        user_id = target.get_user_id()

        # Extract tweets
        tweets = []
        for t in page_soup.find_all('li', {'data-item-type': 'tweet'}):
            if not include(t):
                continue
            
            try:
                tweets.append({
                    'tweet_id': self.extract_tweet_id(self.get_url(t)),
                    'author_username': self.extract_author_username(self.get_url(t)),
                    'tweeted_on': self.get_date(t),
                    'posted_by_id': user_id,
                    'is_retweet': t.div.has_attr('data-retweeter'),
                    'num_likes': self.get_num_likes(t),
                    'num_retweets': self.get_num_retweets(t),
                    'num_replies': self.get_num_replies(t),
                    'content': self.get_text(t),
                    # 'posted_by_username': username,
                    # 'url': TWITTER_ROOT_URL + self.get_url(t),
                })
            except:
                continue

        return tweets

    def scrape_target(self, target, stop_date=None):
        tweets = []
        page_counter = 0
        while page_counter < self.num_pages:
            page_soup = target.get_next_page_soup()
            if page_soup is None:
                break

            new_tweets = self.extract_tweets(page_soup, self.include_retweets, target)
            tweets += new_tweets

            # Stop scraping if reached stop date
            last_scraped_date = dt.datetime.strptime(tweets[-1]['tweeted_on'], "%I:%M %p - %d %b %Y")
            # print(f"[DEBUG] stop_date: {stop_date}, last_scraped_date: {last_scraped_date}")
            if (stop_date is not None) and (last_scraped_date < stop_date):
                break

            page_counter += 1

        return tweets

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

    scraper = Scraper(num_pages=args.num_pages,
                      include_retweets=args.include_retweets,
                      checkpoint=args.checkpoint,
                      oldest_date=args.oldest_date,
                      output_filename=args.output_filename)

    tweets = scraper.scrape_tweets(args.username)

    if args.output_filename:
        with open(args.output_filename, 'a+') as f:
            json.dump(tweets, f, indent=4)


