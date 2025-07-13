#%%
import pandas as pd
import numpy as np
import finnhub
import time
from transformers import pipeline
import time
import random
import requests

class SentimentAnalyzer:
    SOURCE_WEIGHTS = {
        'Bloomberg': 1.2,
        'Reuters': 1.2,
        'CNBC': 1.1,
        'Yahoo Finance': 1.0,
        'Seeking Alpha': 0.9,
        'Unknown': 0.8
    }

    def __init__(self, finnhub_api_key):
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def fetch_stock_news(self, ticker, from_date, to_date, retries=5, backoff_factor=2):
        """Fetch stock news from Finnhub API with retry mechanism."""
        for attempt in range(retries):
            try:
                news = self.finnhub_client.company_news(symbol=ticker, _from=from_date, to=to_date)
                return news if news else []
            except (requests.exceptions.RequestException, ConnectionError) as e:
                print(f"Error fetching news: {e}. Retrying ({attempt + 1}/{retries})...")
                time.sleep(backoff_factor ** attempt + random.uniform(0, 1))  # Exponential backoff with jitter
        print(f"Failed to fetch news after {retries} attempts.")
        return []

    def analyze_sentiment(self, news_articles):
        """Analyze news sentiment using FinBERT and apply source weighting."""
        results = []

        for article in news_articles:
            text = article.get("headline", "")
            date = article.get("datetime", "")
            source = article.get("source", "Unknown")

            # Skip empty headlines
            if not text:
                continue

            try:
                # Convert the date to datetime format (with error handling)
                date = pd.to_datetime(date, unit='s', errors='raise')
            except (ValueError, pd.errors.OutOfBoundsDatetime) as e:
                # Skip articles with invalid or out-of-bounds dates
                print(f"Skipping article due to invalid date: {date}. Error: {e}")
                continue

            # Run sentiment analysis
            sentiment_result = self.sentiment_pipeline(text)[0]
            score = sentiment_result["score"] if sentiment_result["label"] == "positive" else -sentiment_result["score"]

            # Apply source weighting
            weighted_score = score * self.SOURCE_WEIGHTS.get(source, 1.0)

            # Append results
            results.append({
                "Date": date,
                "Ticker": article.get("symbol", ""),
                "Sentiment": weighted_score,
                "Headline": text,
                "Source": source
            })

    # Convert to DataFrame and aggregate sentiment
        df = pd.DataFrame(results)
        return self.aggregate_sentiment(df)


    def aggregate_sentiment(self, df: pd.DataFrame):
        """Normalize sentiment by the number of articles per day."""
        if df.empty:
            return df

        df["Weighted_Sentiment"] = df["Sentiment"] * df.groupby("Date")["Sentiment"].transform("count")
        df = df.groupby("Date").agg({"Weighted_Sentiment": "sum", "Headline": lambda x: list(x)}).reset_index()
        df["Sentiment"] = df["Weighted_Sentiment"] / df["Weighted_Sentiment"].abs().max()
        return df[["Date", "Sentiment", "Headline"]]

    def compute_sentiment_trend(self, df, window=5):
        """Calculate rolling sentiment trend to detect momentum shifts."""
        df["Sentiment_Trend"] = df["Sentiment"].rolling(window=window, min_periods=1).mean()
        return df

    def fetch_sector_sentiment(self, sector_ticker, from_date, to_date):
        """Fetch and analyze sentiment for a sector ETF (e.g., QQQ for tech)."""
        news = self.fetch_stock_news(sector_ticker, from_date, to_date)
        sector_sentiment_df = self.analyze_sentiment(news)
        return sector_sentiment_df["Sentiment"].mean() if not sector_sentiment_df.empty else 0

    def correlate_sentiment_with_returns(self, df, stock_prices):
        """Correlate sentiment with stock price movements."""
        df = df.merge(stock_prices, on="Date", how="inner")
        df["Next_Day_Return"] = df["Close"].pct_change().shift(-1)
        correlation = df[["Sentiment", "Next_Day_Return"]].corr().iloc[0, 1]
        return correlation

    def generate_trade_signal(self, sentiment_score):
        """Convert sentiment score into a trade signal."""
        if sentiment_score > 0.75:
            return "BUY"
        elif 0.25 <= sentiment_score <= 0.75:
            return "HOLD"
        elif -0.25 < sentiment_score < 0.25:
            return "NEUTRAL"
        elif -0.75 <= sentiment_score <= -0.25:
            return "CAUTION"
        else:
            return "SHORT SELL"

    def get_combined_sentiment(self, ticker, sector_ticker, from_date, to_date, stock_prices):
        """Fetch sentiment data in 10-day chunks to avoid API limits."""
        start_date = pd.to_datetime(from_date)
        end_date = pd.to_datetime(to_date)
        all_sentiments = []

        while start_date <= end_date:
            chunk_end = min(start_date + pd.Timedelta(days=10), end_date)
            print(f"Fetching news from {start_date.date()} to {chunk_end.date()}...")
            chunk_sentiment = self.analyze_sentiment(self.fetch_stock_news(ticker, start_date.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            if not chunk_sentiment.empty:
                all_sentiments.append(chunk_sentiment)
            start_date = chunk_end + pd.Timedelta(days=1)
            time.sleep(1)

        if not all_sentiments:
            return None

        news_sentiment_df = pd.concat(all_sentiments)
        sector_sentiment = self.fetch_sector_sentiment(sector_ticker, from_date, to_date)
        news_sentiment_df = self.compute_sentiment_trend(news_sentiment_df)
        news_sentiment_df["Date"] = pd.to_datetime(news_sentiment_df["Date"]).dt.date
        stock_prices["Date"] = stock_prices["Date"].dt.date

        grouped_sentiment = news_sentiment_df.groupby("Date").agg({
            "Sentiment": "mean",
            "Sentiment_Trend": "mean",
            "Headline": lambda x: list(x)
        }).reset_index()

        date_range = pd.date_range(start=stock_prices["Date"].min(), end=stock_prices["Date"].max())
        date_range = pd.DataFrame({"Date": date_range.date})
        grouped_sentiment = date_range.merge(grouped_sentiment, on="Date", how="left").fillna({"Sentiment": 0, "Sentiment_Trend": 0})
        grouped_sentiment["Headline"] = grouped_sentiment["Headline"].apply(lambda x: x if isinstance(x, list) else [])
        grouped_sentiment.to_csv('grouped_sentiment.csv')

        final_sentiment = (grouped_sentiment["Sentiment"].mean() * 0.8) + (sector_sentiment * 0.2)
        trade_signal = self.generate_trade_signal(final_sentiment)
        sentiment_correlation = self.correlate_sentiment_with_returns(grouped_sentiment, stock_prices)

        return {
            "Final_Sentiment": final_sentiment,
            "Trade_Signal": trade_signal,
            "Sentiment_Trend": grouped_sentiment["Sentiment_Trend"].tolist(),
            "Correlation_With_Returns": sentiment_correlation
        }


#%% Example usage:
# Your Finnhub API Key
FINNHUB_API_KEY = 'cvlj8e9r01qj3umdo9agcvlj8e9r01qj3umdo9b0'

# Initialize Sentiment Analyzer
analyzer = SentimentAnalyzer(FINNHUB_API_KEY)

# Define stock and sector
TICKER = "AAPL"             # Stock to analyze
SECTOR_TICKER = "QQQ"       # Sector ETF (e.g., QQQ for tech stocks)

# Define date range


# Fetch stock prices (Replace with real data source)
# Expecting a DataFrame with columns ['Date', 'Close']
stock_prices = pd.read_json(r'C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\proc_data\proc_AAPL_file.json')
FROM_DATE = stock_prices["Date"].min().strftime("%Y-%m-%d")
TO_DATE = stock_prices["Date"].max().strftime("%Y-%m-%d")
# Get sentiment analysis results
result = analyzer.get_combined_sentiment(TICKER, SECTOR_TICKER, FROM_DATE, TO_DATE, stock_prices)

# Print results
if result:
    print(f"Final Sentiment Score: {result['Final_Sentiment']:.2f}")
    print(f"Trade Signal: {result['Trade_Signal']}")
    print(f"Sentiment Trend (Rolling Avg): {result['Sentiment_Trend'][-5:]}")  # Last 5 values
    print(f"Correlation with Returns: {result['Correlation_With_Returns']:.2f}")
else:
    print("No sentiment data available.")

# %%
