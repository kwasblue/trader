import os

# === 1. Set up full paths ===
PROJECT_DIR = r"C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader"
STRATEGY_DIR = os.path.join(PROJECT_DIR, "strategies", "signal")
os.makedirs(STRATEGY_DIR, exist_ok=True)

# === 2. Base strategy ===
base_strategy_code = '''\
import pandas as pd

class BaseStrategy:
    def __init__(self, params=None):
        self.params = params or {}

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Each strategy must implement generate_signal.")
'''

base_path = os.path.join(PROJECT_DIR, "strategies", "base_strategy.py")
with open(base_path, "w") as f:
    f.write(base_strategy_code)

# === 3. Individual strategies ===
strategies = {
    "sma_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class SMAStrategy(BaseStrategy):
    def generate_signal(self, data):
        short_window = self.params.get("short_window", 20)
        long_window = self.params.get("long_window", 50)
        data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
        data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()
        data["Signal"] = np.where(data["SMA_Short"] > data["SMA_Long"], 1, -1)
        return data
''',

    "ema_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class EMAStrategy(BaseStrategy):
    def generate_signal(self, data):
        short_window = self.params.get("short_window", 20)
        long_window = self.params.get("long_window", 50)
        data["EMA_Short"] = data["Close"].ewm(span=short_window, adjust=False).mean()
        data["EMA_Long"] = data["Close"].ewm(span=long_window, adjust=False).mean()
        data["Signal"] = np.where(data["EMA_Short"] > data["EMA_Long"], 1, -1)
        return data
''',

    "mean_reversion_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 10)
        data["Rolling_Mean"] = data["Close"].rolling(window=window).mean()
        data["Signal"] = np.where(data["Close"] > data["Rolling_Mean"], -1, 1)
        return data
''',

    "breakout_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 20)
        data["Rolling_High"] = data["High"].rolling(window=window).max()
        data["Rolling_Low"] = data["Low"].rolling(window=window).min()
        data["Signal"] = np.where(data["Close"] > data["Rolling_High"].shift(1), 1,
                                  np.where(data["Close"] < data["Rolling_Low"].shift(1), -1, 0))
        return data
''',

    "momentum_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 10)
        data["Momentum"] = data["Close"].diff(window)
        data["Signal"] = np.where(data["Momentum"] > 0, 1, -1)
        return data
''',

    "rsi_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 14)
        oversold = self.params.get("oversold", 30)
        overbought = self.params.get("overbought", 70)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['Signal'] = np.where(data['RSI'] < oversold, 1, np.where(data['RSI'] > overbought, -1, 0))
        return data
''',

    "macd_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    def generate_signal(self, data):
        fast = self.params.get("fast_window", 12)
        slow = self.params.get("slow_window", 26)
        signal = self.params.get("signal_window", 9)
        data["EMA_Fast"] = data["Close"].ewm(span=fast, adjust=False).mean()
        data["EMA_Slow"] = data["Close"].ewm(span=slow, adjust=False).mean()
        data["MACD"] = data["EMA_Fast"] - data["EMA_Slow"]
        data["MACD_Signal"] = data["MACD"].ewm(span=signal, adjust=False).mean()
        data["Signal"] = np.where(data["MACD"] > data["MACD_Signal"], 1, -1)
        return data
''',

    "bollinger_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class BollingerStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 20)
        num_std = self.params.get("num_std", 2)
        data["Rolling_Mean"] = data["Close"].rolling(window=window).mean()
        data["Rolling_Std"] = data["Close"].rolling(window=window).std()
        data["Upper_Band"] = data["Rolling_Mean"] + (data["Rolling_Std"] * num_std)
        data["Lower_Band"] = data["Rolling_Mean"] - (data["Rolling_Std"] * num_std)
        data["Signal"] = np.where(data["Close"] < data["Lower_Band"], 1,
                                  np.where(data["Close"] > data["Upper_Band"], -1, 0))
        return data
''',

    "vwap_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class VWAPStrategy(BaseStrategy):
    def generate_signal(self, data):
        data["VWAP"] = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()
        data["Signal"] = np.where(data["Close"] < data["VWAP"], 1, -1)
        return data
''',

    "donchian_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class DonchianStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 20)
        data["Donchian_High"] = data["High"].rolling(window=window).max()
        data["Donchian_Low"] = data["Low"].rolling(window=window).min()
        data["Signal"] = np.where(data["Close"] > data["Donchian_High"].shift(1), 1,
                                  np.where(data["Close"] < data["Donchian_Low"].shift(1), -1, 0))
        return data
''',

    "psar_strategy": '''
import numpy as np
from ta.trend import PSARIndicator
from strategies.base_strategy import BaseStrategy

class PSARStrategy(BaseStrategy):
    def generate_signal(self, data):
        psar = PSARIndicator(data["High"], data["Low"], data["Close"])
        data["PSAR"] = psar.psar()
        data["Signal"] = np.where(data["Close"] > data["PSAR"], 1, -1)
        return data
''',

    "stochastic_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class StochasticStrategy(BaseStrategy):
    def generate_signal(self, data):
        k_window = self.params.get("k_window", 14)
        d_window = self.params.get("d_window", 3)
        data["Lowest_Low"] = data["Low"].rolling(window=k_window).min()
        data["Highest_High"] = data["High"].rolling(window=k_window).max()
        data["%K"] = 100 * (data["Close"] - data["Lowest_Low"]) / (data["Highest_High"] - data["Lowest_Low"])
        data["%D"] = data["%K"].rolling(window=d_window).mean()
        data["Signal"] = np.where((data["%K"] > data["%D"]) & (data["%K"] < 20), 1,
                                  np.where((data["%K"] < data["%D"]) & (data["%K"] > 80), -1, 0))
        return data
''',

    "ichimoku_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class IchimokuStrategy(BaseStrategy):
    def generate_signal(self, data):
        data["Tenkan_Sen"] = (data["High"].rolling(window=9).max() + data["Low"].rolling(window=9).min()) / 2
        data["Kijun_Sen"] = (data["High"].rolling(window=26).max() + data["Low"].rolling(window=26).min()) / 2
        data["Senkou_Span_A"] = ((data["Tenkan_Sen"] + data["Kijun_Sen"]) / 2).shift(26)
        data["Senkou_Span_B"] = ((data["High"].rolling(window=52).max() + data["Low"].rolling(window=52).min()) / 2).shift(26)
        data["Signal"] = np.where(data["Close"] > data["Senkou_Span_A"], 1,
                                  np.where(data["Close"] < data["Senkou_Span_B"], -1, 0))
        return data
''',

    "adx_strategy": '''
import numpy as np
from ta.trend import ADXIndicator
from strategies.base_strategy import BaseStrategy

class ADXStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 14)
        threshold = self.params.get("threshold", 25)
        adx = ADXIndicator(data["High"], data["Low"], data["Close"], window=window)
        data["ADX"] = adx.adx()
        data["+DI"] = adx.adx_pos()
        data["-DI"] = adx.adx_neg()
        data["Signal"] = np.where((data["ADX"] > threshold) & (data["+DI"] > data["-DI"]), 1,
                                  np.where((data["ADX"] > threshold) & (data["-DI"] > data["+DI"]), -1, 0))
        return data
''',
    "combined_strategy": '''
import numpy as np
import inspect
from strategies.base_strategy import BaseStrategy

class CombinedStrategy(BaseStrategy):
    def generate_signal(self, data):
        strategy_methods = self.params.get("strategy_methods", [])
        weights = self.params.get("weights", None)
        combine_method = self.params.get("combine_method", "vote")
        strategy_obj = self.params.get("strategy_instance")

        if not strategy_obj:
            raise ValueError("strategy_instance must be passed in params for CombinedStrategy.")

        signals = []

        for method_name in strategy_methods:
            method = getattr(strategy_obj, method_name, None)
            if not method:
                raise ValueError(f"Strategy '{method_name}' not found in provided instance.")

            sig = inspect.signature(method)
            valid_kwargs = {k: v for k, v in self.params.items() if k in sig.parameters}
            strat_result = method(data.copy(), **valid_kwargs)

            if "Signal" not in strat_result.columns:
                raise ValueError(f"Strategy '{method_name}' did not return a 'Signal' column.")

            signals.append(strat_result["Signal"])

        signals_array = np.array(signals)

        if combine_method == "vote":
            combined_signal = np.sign(np.sum(signals_array, axis=0))
        elif combine_method == "weighted":
            if weights is None or len(weights) != len(strategy_methods):
                raise ValueError("Weights must be provided and match number of strategy methods.")
            combined_signal = np.sign(np.sum(signals_array * np.array(weights)[:, None], axis=0))
        else:
            raise ValueError("Unknown combine method. Use 'vote' or 'weighted'.")

        data["Signal"] = combined_signal
        return data
''',

    "logistic_regression_strategy": '''
import numpy as np
from strategies.base_strategy import BaseStrategy

class LogisticRegressionStrategy(BaseStrategy):
    def generate_signal(self, data):
        model = self.params.get("model")
        if model is None:
            raise ValueError("Model must be provided in params for LogisticRegressionStrategy.")

        try:
            preprocessor = model.named_steps["preprocessor"]
            raw_features = preprocessor.feature_names_in_
            X = data[raw_features].copy()
            X_transformed = preprocessor.transform(X)
            preds_proba = model.named_steps["model"].predict_proba(X_transformed)[:, 1]
            data["Signal"] = np.where(preds_proba > 0.52, 1, np.where(preds_proba < 0.48, -1, 0))
            return data
        except Exception as e:
            print(f"Prediction failed: {e}")
            data["Signal"] = 0
            return data
'''

}

# === 4. Write each strategy file ===
for filename, content in strategies.items():
    with open(os.path.join(STRATEGY_DIR, f"{filename}.py"), "w") as f:
        f.write(content.strip())

# === 5. Generate __init__.py with imports + factory ===
init_lines = [
    "from .sma_strategy import SMAStrategy",
    "from .rsi_strategy import RSIStrategy",
    "",
    "STRATEGY_MAP = {",
    "    'sma': SMAStrategy,",
    "    'rsi': RSIStrategy,",
    "}",
    "",
    "def load_strategy(name, params=None):",
    "    cls = STRATEGY_MAP.get(name)",
    "    if not cls:",
    "        raise ValueError(f\"Strategy '{name}' not found\")",
    "    return cls(params)",
]

with open(os.path.join(STRATEGY_DIR, "__init__.py"), "w") as f:
    f.write("\n".join(init_lines))