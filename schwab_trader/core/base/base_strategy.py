# core/base/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Flexible base: accepts **kwargs so strategies that don't implement __init__
    won't crash when the router injects params. Subclasses can either:
      - read from self.params, OR
      - declare explicit args and still call super().__init__(**kwargs).
    """
    def __init__(self, **kwargs):
        # keep everything; subclasses can pull what they want
        self.params = dict(kwargs)
        self.name = kwargs.get("name", self.__class__.__name__)

    @abstractmethod
    def generate_signal(self, data):
        """
        Return a signal: +1 (long), -1 (short), 0 (hold).
        Accepts either a pandas DataFrame or a dict-like bar.
        """
        ...

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Each strategy must implement generate_signal.")
