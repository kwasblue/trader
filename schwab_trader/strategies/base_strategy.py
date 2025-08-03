import pandas as pd

class BaseStrategy:
    def __init__(self, params=None):
        self.params = params or {}

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Each strategy must implement generate_signal.")
