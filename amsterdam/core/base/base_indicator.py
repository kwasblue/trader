import pandas as pd

class BaseIndicator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute(self) -> pd.DataFrame:
        raise NotImplementedError("Each indicator must implement compute().")
