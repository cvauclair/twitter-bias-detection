import io
import unittest
from unittest.mock import Mock, MagicMock
from unittest.mock import patch, call
import pymysql
from rds_controller import RDSController

RDSController = RDSController()

class TestStringMethods(unittest.TestCase):
    test_topic = "__TEST_TOPIC__"

    test_user = {
        "id" : "__TEST_ID__",
        "handle" : "__TEST__HANDLE__",
        "num_followers" : 0,
        "num_following" : 0,
        "num_tweets" : 0,
        "bio" : "__TEST__BIO__",
        "location" : "__TEST__LOCATION__",
        "fullname" : "__TEST__FULL_NAME__"
    }
    
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_create_topic(self, mock_stdout):
        RDSController.create_topic(self.test_topic)
        self.assertEqual(mock_stdout.getvalue(), "SUCCESS: Successfully added new topic: " + self.test_topic + "\n")

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_topic(self, mock_stdout):
        RDSController.delete_topic(self.test_topic)
        self.assertEqual(mock_stdout.getvalue(), "SUCCESS: Successfully deleted topic: " + self.test_topic + "\n")

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_create_user(self, mock_stdout):
        RDSController.create_user(
            self.test_user['id'], 
            self.test_user['handle'], 
            self.test_user['num_followers'],
            self.test_user['num_following'],
            self.test_user['num_tweets'],
            self.test_user['bio'],
            self.test_user['location'],
            self.test_user['fullname'])
        self.assertEqual(mock_stdout.getvalue(), "SUCCESS: Creation of user with ID {} succeeded\n".format(self.test_user['id']))

if __name__ == '__main__':
    unittest.main()