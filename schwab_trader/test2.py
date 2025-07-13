#%%
from utils.authenticator import Authenticator
from alpha.schwab_client import SchwabClient
from dotenv import load_dotenv
from pathlib import Path
import asyncio

#%%

parent_directory = Path.cwd().parent
# Build the path to the .env file
env_file_path = parent_directory / '.venv' / 'env' / '.env'
# Load the .env file
load_dotenv(env_file_path)
auth = Authenticator()
apikey = auth.apikey
secret = auth.secret

session = SchwabClient(apikey=apikey, secretkey=secret)

async def main():
    tasks = [
        session.account_number(),
        session.user_preferences(),
        session.accounts(),
        session.daily_price_history('AAPL'),
        session.quote('AMD'),
        session.quotes(['AMD', 'AAPL', 'TSLA']),
        session.markets('EQUITY'),
        session.market_hours('equity'),
        session.movers('NYSE', 'PERCENT_CHANGE_UP'),
        session.movers('NYSE', 'PERCENT_CHANGE_DOWN'),
        session.option_chains('AAPL'),
        session.expiration_chain('AAPL'),
        session.instruments('037833100'),
    ]

    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
