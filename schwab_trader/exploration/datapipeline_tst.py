#%%
import sys
import os

# Dynamically set root path (one level up from 'exploration')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#fill the data pipline
from data.datapipeline import StockDataPipeline
from sp500 import sp500_tickers

pipline = StockDataPipeline(sp500_tickers)
pipline.run()
# %%
