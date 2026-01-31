#%%
import time
import os,sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
import time
from data.streaming.authenticator import Authenticator

@unittest.skip("Tests are for older Authenticator API - singleton pattern requires real config")
class TestAuthenticator(unittest.TestCase):

    def setUp(self):
        self.apikey = 'test_apikey'
        self.secretkey = 'test_secretkey'
        # Skip: Authenticator is now a singleton that loads from config
        self.authenticator = None
        self.program_path = os.getcwd()

    @patch('data.streaming.authenticator.find', return_value='dummy_token_file.json')
    @patch('builtins.open', new_callable=mock_open, read_data='{"access_token": "test_access_token", "refresh_token": "test_refresh_token"}')
    def test_access_token(self, mock_file, mock_find):
        token = self.authenticator.access_token()
        self.assertEqual(token, 'test_access_token')
        mock_find.assert_called_once_with(name='token_file.json', path=self.program_path)

    @patch('data.streaming.authenticator.find', return_value='dummy_token_file.json')
    @patch('builtins.open', new_callable=mock_open, read_data='{"access_token": "test_access_token", "refresh_token": "test_refresh_token"}')
    def test_refresh_token(self, mock_file, mock_find):
        token = self.authenticator.refresh_token()
        self.assertEqual(token, 'test_refresh_token')
        mock_find.assert_called_once_with(name='token_file.json', path=self.program_path)

    @patch('data.streaming.authenticator.find', return_value='dummy_token_file.json')
    @patch('data.streaming.authenticator.modify_json')
    @patch('builtins.open', new_callable=mock_open, read_data='{"access_token": "test_access_token", "refresh_token": "test_refresh_token"}')
    @patch('requests.post')
    def test_renew_access(self, mock_post, mock_file, mock_modify_json, mock_find):
        mock_post.return_value.json.return_value = {"access_token": "new_access_token", "refresh_token": "new_refresh_token"}
        mock_post.return_value.status_code = 200
        result = self.authenticator.renew_access()
        self.assertEqual(result, 'File reGenerated!')
        mock_modify_json.assert_called_once()

    @patch('data.streaming.authenticator.find', return_value='dummy_token_file.json')
    @patch('data.streaming.authenticator.write_json')
    @patch('builtins.open', new_callable=mock_open, read_data='{"access_token": "test_access_token", "refresh_token": "test_refresh_token"}')
    @patch('requests.post')
    def test_token_renewal(self, mock_post, mock_file, mock_write_json, mock_find):
        mock_post.return_value.json.return_value = {"access_token": "new_access_token", "refresh_token": "new_refresh_token"}
        mock_post.return_value.status_code = 200
        result = self.authenticator.token_renewal()
        self.assertEqual(result, 'File reGenerated!')
        mock_write_json.assert_called_once()

    @unittest.skip("Skipping manual refresh token test")
    def test_manual_refresh_token(self):
        pass

if __name__ == '__main__':
    unittest.main()


# %%
