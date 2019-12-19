from bs4 import BeautifulSoup as soup

import argparse
import sys, traceback
import json
import time

import utils

TWITTER_ROOT_URL = 'https://twitter.com'

def get_url(tweet):
    try:
        return tweet.find('div')['data-permalink-path']
    except:
        return 'ERROR'

def get_text(tweet):
    try:
        return tweet.find('div', {'class': 'js-tweet-text-container'}).text
    except:
        return 'ERROR'

def extract_tweets(page_soup, include_retweets: bool):
    # Set inclusion function depending on flags
    if include_retweets:
        include = lambda t: True
    else:
        include = lambda t: not t.div.has_attr('data-retweeter')

    # Extract tweets
    tweets = [{
        'url': TWITTER_ROOT_URL + get_url(t), 
        'text': get_text(t)
    } for t in page_soup.find_all('li', {'data-item-type': 'tweet'}) if include(t)]

    return tweets

def scrape_tweets(username: str, num_tweets: int, include_retweets: bool, checkpoint: str):
    # Get user twitter url
    twitter_user_url = f'{TWITTER_ROOT_URL}/{username}'
    
    tweets = []
    try:
        while len(tweets) < num_tweets:
            if len(tweets) == 0:
                # If first page scraped, simply get root page of user
                html = utils.get_page(twitter_user_url)
                page_soup = soup(html, 'html.parser')

                # Check for error on twitter page
                if page_soup.find("div", {"class": "errorpage-topbar"}):
                    print(f"[Error] Invalid username {username}")
                    sys.exit(1)

                # Add tweets
                tweets += extract_tweets(page_soup, include_retweets)
                
                # Ptr to next page of tweets
                ptr = page_soup.find("div", {"class": "stream-container"})["data-min-position"]
            else:
                # Make request to get next tweets
                if checkpoint:
                    raw_data = json.loads(utils.get_page(checkpoint))
                    checkpoint = None
                else:
                    raw_data = json.loads(utils.get_page(f'https://twitter.com/i/profiles/show/{username}/timeline/tweets?include_available_features=1&include_entities=1&max_position={ptr}&reset_error_state=false'))

                if not raw_data["has_more_items"] and not raw_data["min_position"]:
                    print("[INFO] No more tweets returned")
                    break
                
                page_soup = soup(raw_data['items_html'], 'html.parser')

                # Add tweets
                tweets += extract_tweets(page_soup, include_retweets)

                ptr = raw_data['min_position']

            time.sleep(1)
    except:
        print(f"[ERROR] {sys.exc_info()[0]}")
        traceback.print_exc(file=sys.stdout)

    print(f"[INFO] {len(tweets)} scraped!")

    return tweets

def main():
    # Check args
    parser = argparse.ArgumentParser(description='Scrape tweets')
    parser.add_argument('twitter_username', type=str)
    parser.add_argument('-n', dest='num_tweets', type=int)
    parser.add_argument('-o', dest='output_filename', type=str)
    parser.add_argument('--include_retweets', default=True, type=bool)
    parser.add_argument('--checkpoint', default=None, type=str)
    args = parser.parse_args()

    tweets = scrape_tweets(username=args.twitter_username, num_tweets=args.num_tweets, include_retweets=args.include_retweets, checkpoint=args.checkpoint)

    if args.output_filename:
        with open(args.output_filename, 'w') as f:
            json.dump(tweets, f, indent=4)

if __name__ == "__main__":
    main()