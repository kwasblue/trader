import os
import json
import time
import base64
import asyncio
import aiohttp
import json
from dotenv import load_dotenv
from data.output.writer import FileWriter
from loggers.logger import Logger
from utils.configloader import ConfigLoader
from urllib.parse import urlencode

class Authenticator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance
    

    def _initialize(self, apikey=None, secretkey=None):
       
        self.config =  ConfigLoader().load_config()
        env_path = self.config["folders"]["env"]
        load_dotenv(dotenv_path=env_path)
        self.apikey = os.getenv('SCHWAB_API_KEY')
        self.secret = os.getenv('SCHWAB_SECRET')
        self.redirect_url = os.getenv('SCHWAB_REDIRECT_URL')
        self.auth_url = self.config["auth"]["authentication_url"]
        self.logs_dir = self.config["folders"]["logs"]
        self.token_endpoint = self.config["auth"]["token_endpoint"]
        self.program_path = self.config['folders']['app_path']
        self.logger = Logger('app.log', 'Authenticator', log_dir=self.logs_dir).get_logger()
        self.writer = FileWriter(log_file='app.log',logger_name='FileWriter',log_dir=self.logs_dir)
        self.token_path = self.config["folders"]["tokens"]
        self.rate_lim = 0.5

    async def _throttle_requests(self):
        await asyncio.sleep(self.rate_lim)
         
    async def _get(self, endpoint: str, headers: dict, params: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url=endpoint, headers=headers, data=params) as response:
                    tokens = await response.json()  # Parse the response as JSON
                    if response.status == 200:
                        self.logger.info("Tokens successfully retrieved.")
                        return tokens
                    else:
                        self.logger.error(f"API returned status code {response.status}")
                        return {"error": f"API returned status code {response.status}"}
            except aiohttp.ClientError as e:
                self.logger.error(f"HTTP request failed: {str(e)}")
                return {"error": f"HTTP request failed: {str(e)}"}
            finally:
                await self._throttle_requests()  # Enforce rate limit
    

    def _set_headers_params(self, endpoint: str, apikey: str = None, secret: str = None, grant_type: str = None, code: str = None, redrirect_url = '', refresh_token = None):
        params = {}
        if refresh_token:
            params['refresh_token'] = refresh_token
        if grant_type:
            params['grant_type'] = grant_type
        if code:
            params['code'] = code
        if redrirect_url:
            params['redirect_uri'] = redrirect_url
        headers = {
            'Authorization': f'Basic {base64.b64encode(bytes(f"{apikey}:{secret}", "utf-8")).decode("utf-8")}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        return endpoint, headers, params

    def _read_token_file(self) -> dict:
        file = self.writer.find(name='token_file', path=self.program_path)
        try:
            with open(file) as f:
                token_file = json.load(f)
            return token_file
        except FileNotFoundError:
            self.logger.warning("Token file not found. Starting fresh.")
            return {}
        except json.JSONDecodeError:
            self.logger.error("Token file is corrupted.")
            return {}

    def _is_access_token_expired(self, token_data: dict) -> bool:
        access_time = token_data.get("access_time")
        expires_in = token_data.get("expires_in")

        if not access_time or not expires_in:
            self.logger.warning("Access token data is incomplete. Assuming expired.")
            return True

        return time.time() >= (access_time + expires_in)
    
    def _contains_error(self, token_data: dict) -> bool:
        if 'error' in token_data:
            self.logger.error(f'{token_data['error']}')
            return True
        else:
            return False

        
    def _is_refresh_token_expired(self, token_data: dict, refresh_interval_days: int=6) -> bool:
        refresh_time = token_data.get("refresh_time")

        if not refresh_time:
            self.logger.warning("Refresh token data is incomplete. Assuming expired.")
            return True

        refresh_interval_seconds = refresh_interval_days * 24 * 60 * 60
        return time.time() >= (refresh_time + refresh_interval_seconds)


    async def manual_refresh_token(self):
        params = {
            'client_id': self.apikey,
            'redirect_uri': self.redirect_url
        }
        auth_url = f'{self.config['auth']['authentication_url']}{urlencode(params)}'
        print(f"Use this URL to log in and authenticate: {auth_url}")
        returned_link = input("Complete the login process and paste the redirect URL here:")
        
        try:    
            # Extract code from the returned URL
            code = returned_link[returned_link.index('code=') + 5:returned_link.index('%40')] + "@"
        except Exception as e:
            self.logger.error(f"Error extracting authorization code: {str(e)}")
            return {"error": "Failed to extract code from URL."}
        
        endpoint, headers, params = self._set_headers_params(
            endpoint=self.token_endpoint, 
            grant_type='authorization_code', 
            code=code, 
            redrirect_url=self.redirect_url, 
            secret=self.secret, 
            apikey=self.apikey
        )
        
        token_dict = await self._get(endpoint=endpoint, headers=headers, params=params)
        token_dict['refresh_time'] = time.time()
        token_dict['access_time'] = time.time()

        if 'error' in token_dict:
            self.logger.error(f"Error during manual refresh: {token_dict['error']}")
            return token_dict
        
        try:
            self.writer.write_json(target_path=f'{self.program_path}/tokens/', target_file='token_file', data=token_dict)
            self.logger.info("Manual refresh token successfully saved.")
        except Exception as e:
            self.logger.error(f"Error saving token: {str(e)}")
            return {"error": f"Error saving token: {str(e)}"}
        else:
            return True

    def access_token(self) -> str:
            token_file = self._read_token_file()
            self.logger.info("Access token retrieved.")
            return token_file['access_token']
    
    def refresh_token(self) -> str:
            token_file = self._read_token_file()
            self.logger.info("Refresh token retrieved.")
            return token_file['refresh_token']
    
    async def renew_access(self) -> bool:
            endpoint, headers, params = self._set_headers_params(endpoint=self.token_endpoint, apikey=self.apikey, secret=self.secret, grant_type='refresh_token', refresh_token=self.refresh_token(), redrirect_url=self.redirect_url)
            token_dict = await self._get(endpoint, headers, params)
            token_dict['access_time'] = time.time()
            try:
                self.writer.modify_json(f'{self.program_path}/tokens/', 'token_file', token_dict)
                self.logger.info("Access token successfully renewed.")
            except Exception as e:
                self.logger.error(f"Error renewing access token: {str(e)}")
                return {"error": str(e)}
            else:
                return True

    async def token_renewal(self) -> None:
        """
        Periodically checks and renews tokens (access and refresh) as necessary.
        """
        while True:
            try:
                if not os.path.exists(self.token_path):
                    self.logger.error("Token file not found. Cannot renew tokens.")
                    await asyncio.sleep(60)
                    continue

                # Read and validate token data
                token_data = self._read_token_file()
                if not token_data:
                    self.logger.error("Token file is empty or invalid. Cannot proceed.")
                    await asyncio.sleep(60)
                    continue

                # Check and renew access token
                if self._is_access_token_expired(token_data):
                    self.logger.info("Access token expired. Attempting renewal...")
                    if not await self.renew_access():
                        self.logger.error("Access token renewal failed. Initiating manual refresh...")
                        await self.manual_refresh_token()

                # Check and renew refresh token
                if self._is_refresh_token_expired(token_data):
                    self.logger.info("Refresh token expired. Performing manual refresh...")
                    await self.manual_refresh_token()

                if self._contains_error(token_data):
                    self.logger.error('There is an error in the file attempting manual refresh....')
                    await self.manual_refresh_token()
    
                await asyncio.sleep(60)  # Wait 1 minute before the next check

            except Exception as e:
                self.logger.error(f"Unexpected error in token renewal: {str(e)}")
                await asyncio.sleep(60)
