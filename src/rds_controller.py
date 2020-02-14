import sys
import pymysql
import json
from config import config


class RDSController:
    def __init__(self):
        try:
            self.conn = pymysql.connect(config['rds']['host'], user=config['rds']['user'],
                                        passwd=config['rds']['password'], db=config['rds']['db_name'],
                                        connect_timeout=5)
        except:
            print("FAILURE: Failed to connect to server.")
            sys.exit()

        print("SUCCESS: Connection to RDS mysql instance succeeded")

    def get_user(self, user_id):
        with self.conn.cursor() as cur:
            query = ('SELECT * FROM users WHERE id = {user_id}').format(user_id=user_id)
            cur.execute(query)
            user = cur.fetchall()
            print("SUCCESS: Successfully fetched user with id:", user_id)
            return user

    def create_topic(self, name):
        with self.conn.cursor() as cur:
            query = 'INSERT INTO topics (topic) VALUES ("{}")'.format(name)
            cur.execute(query)
            self.conn.commit()
            print("SUCCESS: Successfully added new topic:", name)

    def create_user(self, id, handle, num_followers, num_following, num_tweets, bio, location,
                    fullname, photo_url=None):
        with self.conn.cursor() as cur:
            query = (
                'INSERT INTO users (id, handle, followersCount, followingCount, tweetsCount, bio, location, photoURL, fullname) '
                'VALUES ("{id}", "{handle}", "{num_followers}", "{num_following})", "{num_tweets}", "{bio}", "{location}", "{photo_url}", "{fullname}")'
                .format(id=id, handle=handle, num_followers=num_followers, num_following=num_following,
                        num_tweets=num_tweets, bio=bio, location=location, photo_url=photo_url, fullname=fullname))
            cur.execute(query)
            self.conn.commit()
            print("SUCCESS: Creation of user with ID {} succeeded".format(id))

    def create_tweet(self, id, author_username, tweeted_on, posted_by_id, is_retweet, num_likes, num_retweets,
                     num_replies, content=None, sentiment=None):
        with self.conn.cursor() as cur:
            query = (
                'INSERT INTO tweets (id, content, authorUsername, tweetedOn, postedById, isRetweet, likesCount, retweetsCount, repliesCount, sentiment) '
                'VALUES ("{id}", "{content}", "{author_username}", "{tweeted_on}", "{posted_by_id}", "{is_retweet}", "{num_likes}", "{num_retweets}", "{num_replies}", "{sentiment}")'
                .format(id=id, content=content, author_username=author_username, tweeted_on=tweeted_on,
                        posted_by_id=posted_by_id, is_retweet=is_retweet, num_likes=num_likes, num_replies=num_replies,
                        num_retweets=num_retweets, sentiment=sentiment))

            cur.execute(query)
            self.conn.commit()
            print("SUCCESS: Creation of tweet with ID {} succeeded".format(id))

    def set_tweet_sentiment(self, tweet_id, sentiment):
        with self.conn.cursor() as cur:
            query = ('UPDATE tweets '
                     'SET sentiment = "{sentiment}" '
                     'WHERE id = {tweet_id}'
                     .format(sentiment=sentiment, tweet_id=tweet_id))
            cur.execute(query)
            self.conn.commit()
            print(
                "SUCCESS: Set sentiment of tweet with ID {tweet_id} to {sentiment} succeeded".format(tweet_id=tweet_id,
                                                                                                     sentiment=sentiment))

    def set_tweet_topic(self, tweet_id, topic_id):
        with self.conn.cursor() as cur:
            query = ('INSERT INTO isAbout (tweetId, topicId) '
                     'VALUES ("{tweet_id}", "{topic_id}")'
                     .format(tweet_id=tweet_id, topic_id=topic_id))
            cur.execute(query)
            self.conn.commit()
            print("SUCCESS: Set topic of tweet with ID {tweet_id} to topic with ID {topic_id} succeeded".format(
                tweet_id=tweet_id, topic_id=topic_id))

    def set_user_topic_bias_score(self, user_id, topic_id, bias_score, num_mentions, last_updated):
        with self.conn.cursor() as cur:
            user_topic_bias_count_query = ('SELECT mentionCount '
                                           'FROM isBiased '
                                           'WHERE userId={user_id} AND topicId={topic_id}'
                                           .format(user_id=user_id, topic_id=topic_id))
            cur.execute(user_topic_bias_count_query)
            res = cur.fetchone();
            if (res):
                num_mentions += res[0]
                query = ('UPDATE isBiased '
                         'SET biasScore={bias_score}, mentionCount={num_mentions}, lastUpdated={last_updated} '
                         'WHERE userId={user_id} AND topicId={topic_id}'
                         .format(bias_score=bias_score, num_mentions=num_mentions, last_updated=last_updated,
                                 user_id=user_id, topic_id=topic_id))
            else:
                query = ('INSERT INTO isBiased (userId, topicId, biasScore, mentionCount, lastUpdated) '
                         'VALUES ("{user_id}", "{topic_id}", "{bias_score}", "{num_mentions}", "{last_updated}")'
                         .format(user_id=user_id, topic_id=topic_id, bias_score=bias_score, num_mentions=num_mentions,
                                 last_updated=last_updated))

            cur.execute(query)
            self.conn.commit()
            print(
                "SUCCESS: Set bias of topic with {topic_id} to user with ID {user_id} with score {bias_score} succeeded".format(
                    topic_id=topic_id, user_id=user_id, bias_score=bias_score))