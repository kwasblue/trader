# run_stream.py
# system level stuff to make sure we get the right root and can import the stuff we want
import sys
from pathlib import Path
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#
import asyncio
from data.streaming.streamer import SchwabStreamingClient   # adjust if filename is different
from dotenv import load_dotenv

# Replace with your real API keys or load them from env/secret manager
load_dotenv(r'C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\venv\.env')
API_KEY = "SCHWAB_API_KEY"
SECRET_KEY = "SCHWAB_SECRET"

async def main():
    # Instantiate the client
    client = SchwabStreamingClient(API_KEY, SECRET_KEY)

    # Symbols you want to test with
    symbols = ["AAPL", "MSFT", "TSLA"]

    # Run the websocket stream
    await client.run(symbols)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Streaming stopped by user.")
