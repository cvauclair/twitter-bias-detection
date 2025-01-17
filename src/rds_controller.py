import sys
import pymysql
import json
from config import config

import datetime as dt

class RDSController:
    def __init__(self):
        try:
            self.conn = pymysql.connect(
                config['rds']['host'], 
                user=config['rds']['user'],
                passwd=config['rds']['password'], 
                db=config['rds']['db_name'],
                connect_timeout=5
            )
        except:
            print("FAILURE: Failed to connect to server.")
            print('Got error {!r}, errno is {}'.format(e, e.args[0]))
            sys.exit()

        print("SUCCESS: Connection to RDS mysql instance succeeded")

    def get_user(self, user_id):
        with self.conn.cursor() as cur:
            query = (f'SELECT * FROM users WHERE id = {user_id}')
            cur.execute(query)
            user = cur.fetchall()
            print(f"SUCCESS: Successfully fetched user with id: {user_id}")
            return user

    def get_all_users(self):
        with self.conn.cursor() as cur:
            query = ('SELECT * FROM users')
            cur.execute(query)
            users = cur.fetchall()
            print("SUCCESS: Successfully fetched all users")
            return users

    def get_tweets_from_user(self, author_username):
        with self.conn.cursor() as cur:
            query = f"SELECT id, content FROM tweets WHERE authorUsername = '{author_username}'"
            cur.execute(query)
            content = cur.fetchall()
            print("SUCCESS: Successfully fetched tweets from:", author_username)
        
        tweets = [{'tweet_id': t[0], 'content': t[1], 'author_username': author_username} for t in content]
        return tweets

    def create_topic(self, name):
        with self.conn.cursor() as cur:
            query = f'INSERT INTO topics (topic) VALUES ("{name}")'
            cur.execute(query)
            self.conn.commit()
            print(f"SUCCESS: Successfully added new topic: {name}")

    def delete_topic(self, name):
        with self.conn.cursor() as cur:
            query = 'DELETE FROM topics WHERE topic="{}"'.format(name)
            cur.execute(query)
            self.conn.commit()
            print("SUCCESS: Successfully deleted topic:", name)

    def delete_user(self, id):
        with self.conn.cursor() as cur:
            query = 'DELETE FROM users WHERE id="{}"'.format(id)
            cur.execute(query)
            self.conn.commit()
            print("SUCCESS: Successfully deleted user with ID:", id)

    def create_user(self, user_id, username, num_followers, num_following, num_tweets, bio, location,
                    fullname, photo_url=None):
        with self.conn.cursor() as cur:
            query = """
                INSERT INTO users (id, handle, followersCount, followingCount, tweetsCount, bio, location, photoURL, fullname)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            cur.execute(query, (user_id, username, num_followers, num_following, num_tweets, bio, location, photo_url, fullname))
            self.conn.commit()
            print(f"SUCCESS: Creation of user with ID {user_id} succeeded")

    def create_tweet(self, tweet_id, author_username, tweeted_on, posted_by_id, is_retweet, num_likes, num_retweets,
                     num_replies, content=None, sentiment=None):
        with self.conn.cursor() as cur:
            query = """
                INSERT INTO tweets (id, content, authorUsername, tweetedOn, postedById, isRetweet, likesCount, retweetsCount, repliesCount, sentiment)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            cur.execute(query, (tweet_id, content, author_username, tweeted_on, posted_by_id, is_retweet, num_likes, num_retweets, num_replies, sentiment))
            self.conn.commit()
            # print(f"SUCCESS: Creation of tweet with ID {tweet_id} succeeded")

    def get_all_tweets(self):
        tweets = []
        with self.conn.cursor() as cur:
            query = "SELECT id, content FROM tweets"
            cur.execute(query)
            self.conn.commit()
        
            for t in cur.fetchall():
                tweets.append({'tweet_id': t[0], 'content': t[1]})

        return tweets

    def get_latest_tweet_date(self, user_id):
        with self.conn.cursor() as cur:
            query = f"""
                SELECT tweetedOn
                FROM tweets
                WHERE postedById = {user_id}
            """

            cur.execute(query)
            tweet_dates = cur.fetchall()
            
            self.conn.commit()

        if len(tweet_dates) == 0:
            return None

        tweet_dates = [dt.datetime.strptime(d[0], "%I:%M %p - %d %b %Y") for d in tweet_dates]

        return max(tweet_dates)

    def set_tweet_sentiment(self, tweet_id, sentiment):
        with self.conn.cursor() as cur:
            query = (
                'UPDATE tweets '
                f'SET sentiment = "{sentiment}" '
                f'WHERE id = {tweet_id}'
            )

            cur.execute(query)
            self.conn.commit()
            # print(f"SUCCESS: Set sentiment of tweet with ID {tweet_id} to {sentiment} succeeded")

    def set_tweet_topic(self, tweet_id, topic_id):
        with self.conn.cursor() as cur:
            query = (
                'INSERT INTO isAbout (tweetId, topicId) '
                f'VALUES ("{tweet_id}", "{topic_id}")'
            )

            cur.execute(query)
            self.conn.commit()
            print(f"SUCCESS: Set topic of tweet with ID {tweet_id} to topic with ID {topic_id} succeeded")

    def set_user_topic_bias_score(self, user_id, topic_id, bias_score, num_mentions, last_updated):
        with self.conn.cursor() as cur:
            user_topic_bias_count_query = (
                'SELECT mentionCount '
                'FROM isBiased '
                f'WHERE userId={user_id} AND topicId={topic_id}'
            )
            cur.execute(user_topic_bias_count_query)
            res = cur.fetchone()
            if (res):
                num_mentions += res[0]
                query = (
                    'UPDATE isBiased '
                    f'SET biasScore={bias_score}, mentionCount={num_mentions}, lastUpdated={last_updated} '
                    f'WHERE userId={user_id} AND topicId={topic_id}'
                )
            else:
                query = (
                    'INSERT INTO isBiased (userId, topicId, biasScore, mentionCount, lastUpdated) '
                    f'VALUES ("{user_id}", "{topic_id}", "{bias_score}", "{num_mentions}", "{last_updated}")'
                )

            cur.execute(query)
            self.conn.commit()
            print(f"SUCCESS: Set bias of topic with {topic_id} to user with ID {user_id} with score {bias_score} succeeded")
