#%%
#fill the data pipline
from utils.datapipeline import StockDataPipeline
from sp500 import sp500_tickers

pipline = StockDataPipeline(sp500_tickers)
pipline.run()
# %%
