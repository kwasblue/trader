from abc import ABC, abstractmethod
from typing import List, Dict, Union


class SentimentModelBase(ABC):
    """
    Abstract base class for all sentiment models.
    Sentiment models transform textual or numerical inputs into a sentiment score or label.
    """

    @abstractmethod
    def analyze_text(self, text: str) -> Union[float, str]:
        """
        Analyze the sentiment of a single text string.

        Args:
            text (str): The text to analyze.

        Returns:
            Union[float, str]: A sentiment score or label.
        """
        pass

    @abstractmethod
    def analyze_batch(self, texts: List[str]) -> List[Union[float, str]]:
        """
        Analyze sentiment for a batch of text strings.

        Args:
            texts (List[str]): List of text strings.

        Returns:
            List[Union[float, str]]: List of sentiment scores or labels.
        """
        pass

    @abstractmethod
    def analyze_asset(self, symbol: str) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment for a given asset symbol by aggregating relevant text (e.g., news, tweets).

        Args:
            symbol (str): Asset symbol (e.g., "AAPL").

        Returns:
            Dict[str, Union[float, str]]: Aggregated sentiment result for the asset.
        """
        pass
