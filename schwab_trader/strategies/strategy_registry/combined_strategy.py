import numpy as np
import inspect
from core.base.base_strategy import BaseStrategy

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