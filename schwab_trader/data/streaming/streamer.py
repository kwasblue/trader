import json
import requests
import websockets
from data.streaming.authenticator import Authenticator
from utils.logger import Logger
from utils.configloader import ConfigLoader
from data.streaming.schwab_client import SchwabClient   
from core.eventhandler import EventHandler
import asyncio

class SchwabStreamingClient():
    def __init__(self, apikey, secretkey):
        self.authenticator = Authenticator()
        self.config = ConfigLoader().load_config()
        self.apikey = apikey
        self.secretkey = secretkey
        self.streamer_info = None
        self.connection = None
        self.streaming_logger = Logger('app.log', 'SchwabStreamingClient', log_dir=f'{self.config['folders']['logs']}').get_logger()
        self.price_dict = {}

    async def websocket_client(self, symbols):

        url = r"https://api.schwabapi.com/trader/v1/userPreference"
        headers = {'Authorization': f"Bearer {self.authenticator.access_token()}"}
        try:
            response = requests.get(headers=headers, url=url)
            response.raise_for_status()
            user_preference = response.json()
            self.streamer_info = user_preference['streamerInfo'][0]
            self.streaming_logger.info("Retrieved user preferences successfully")
        except Exception as e:
            self.streaming_logger.error(f"Failed to retrieve user preferences: {e}")
            return

        login_request = {
            'service': 'ADMIN',
            'requestid': 0,
            'command': 'LOGIN',
            'SchwabClientCustomerId': self.streamer_info['schwabClientCustomerId'],
            'SchwabClientCorrelId': self.streamer_info['schwabClientCorrelId'],
            'parameters': {
                'Authorization': self.authenticator.access_token(),
                'SchwabClientChannel': self.streamer_info['schwabClientChannel'],
                'SchwabClientFunctionId': self.streamer_info['schwabClientFunctionId'],
            }
        }

        symbol_request = {
            'service': 'LEVELONE_EQUITIES',
            'requestid': 1,
            'command': 'SUBS',
            'SchwabClientCustomerId': self.streamer_info['schwabClientCustomerId'],
            'SchwabClientCorrelId': self.streamer_info['schwabClientCorrelId'],
            'parameters': {
                'keys': ','.join(symbols),
                'fields': ','.join(str(field) for field in range(0, 42))
            }
        }

        async with websockets.connect(self.streamer_info['streamerSocketUrl']) as ws:
            # Send login request
            await ws.send(json.dumps(login_request))

            while True:
                try:
                    message = await ws.recv()
                    self.streaming_logger.info("Login Message Received:")
                    self.streaming_logger.info(message)
                    break
                except websockets.ConnectionClosed:
                    self.streaming_logger.error("Connection closed")
                    return

            # Send symbol request
            await ws.send(json.dumps(symbol_request))
            quote_dict = {}
            while True:
                try:
                    message = await ws.recv()
                    self.streaming_logger.info("\nReceived message:")
                    self.streaming_logger.info(message)

                    # Check if the message is a JSON string and parse it
                    try:
                        parsed_message = json.loads(message)
                    except json.JSONDecodeError:
                        self.streaming_logger.error("Failed to decode message as JSON")
                        continue  # Skip this message if it isn't JSON

                    content = parsed_message.get('data', [{}])[0].get('content', [])

                    for item in content:
                        symbol = item.get('key')
                        last_price = item.get('1')
                        bid_price = item.get('2')
                        ask_price = item.get('3')
                        close_price = item.get('29')
                    
                        quote_dict[symbol] = {
                            'last_price': last_price,
                            'bid_price': bid_price,
                            'ask_price': ask_price,
                            'close_price': close_price
                        }
                    print(quote_dict)
                        
                except websockets.ConnectionClosed:
                    self.streaming_logger.error("Connection closed")
                    break

    async def run(self, symbols):
        await self.websocket_client(symbols)

