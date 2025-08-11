#%%
import ast
import os
from pathlib import Path
import pathspec

def load_gitignore(project_root):
    gitignore_path = Path(project_root) / ".gitignore"
    if not gitignore_path.exists():
        return None
    with open(gitignore_path, "r") as f:
        return pathspec.PathSpec.from_lines("gitwildmatch", f)

def extract_defs_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        try:
            tree = ast.parse(file.read(), filename=filepath)
        except SyntaxError as e:
            print(f"SyntaxError in {filepath}: {e}")
            return []

    attach_parents(tree)

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            results.append({"type": "class", "name": node.name, "methods": methods})
        elif isinstance(node, ast.FunctionDef) and not isinstance(getattr(node, "parent", None), ast.ClassDef):
            results.append({"type": "function", "name": node.name})
    return results

def attach_parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

def scan_directory(base_path):
    base_path = Path(base_path)
    gitignore = load_gitignore(base_path)
    code_structure = {}

    for root, _, files in os.walk(base_path):
        rel_root = os.path.relpath(root, base_path)
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_path)

            if file.endswith(".py") and (not gitignore or not gitignore.match_file(rel_path)):
                try:
                    defs = extract_defs_from_file(full_path)
                    if defs:
                        code_structure[rel_path] = defs
                except Exception as e:
                    print(f"Failed to parse {rel_path}: {e}")
    return code_structure

def print_structure(structure):
    for file, defs in structure.items():
        print(f"\n📄 {file}")
        for item in defs:
            if item["type"] == "class":
                print(f"  🧱 class {item['name']}")
                for method in item["methods"]:
                    print(f"     └── def {method}()")
            elif item["type"] == "function":
                print(f"  ⚙️ def {item['name']}()")

if __name__ == "__main__":
    project_dir = "schwab_trader"  # Or set via CLI or os.getcwd()
    structure = scan_directory(project_dir)
    print_structure(structure)

#%%
import os

# Re-defining the structure due to kernel reset
structure = {
    "schwab_trader": {
        "core": {
            "files": ["backtester.py", "executor.py", "mock_executor.py", "eventhandler.py", "position_sizer.py", "runner.py"],
            "simulator": ["gbm_simulator.py", "strategy_router.py", "live_plotter.py", "historical_loader.py", "drawdown_monitor.py"],
            "logic": ["trade_logic_base.py", "default_trade_logic.py", "execution_engine.py", "trade_logic_manager.py", "strategy_routing_manager.py"],
            "broker": ["broker_interface.py", "alpaca_broker.py"],
            "base": ["strategy_base.py", "indicator_base.py", "execution_engine_base.py", "position_sizer_base.py", "risk_model_base.py", 
                     "executor_base.py", "event_handler_base.py", "sentiment_model_base.py", "trade_logger_base.py"]
        },
        "strategies": {
            "files": ["base_strategy.py"],
            "strategy_registry": ["adx_strategy.py", "bollinger_strategy.py", "breakout_strategy.py", "combined_strategy.py", "donchian_strategy.py", 
                                  "ema_strategy.py", "ichimoku_strategy.py", "logistic_regression_strategy.py", "macd_strategy.py", 
                                  "mean_reversion_strategy.py", "momentum_strategy.py", "psar_strategy.py", "rsi_strategy.py", 
                                  "sma_strategy.py", "stochastic_strategy.py", "vwap_strategy.py", "__init__.py"]
        },
        "indicators": ["atr.py", "base_indicator.py", "bollinger.py", "ema.py", "macd.py", "momentum.py", "obv.py", "percent_change.py", 
                       "price_change.py", "psar.py", "roc.py", "rsi.py", "sma.py", "technical_indicators.py", "vwap.py"],
        "data": {
            "files": ["datapipeline.py", "processor.py", "aggregate.py", "datastorage.py", "datautils.py"],
            "output": ["writer.py"],
            "streaming": ["schwab_client.py", "streamer.py", "authenticator.py"]
        },
        "utils": ["logger.py", "framemanager.py", "configloader.py", "risk_metrics.py"],
        "cache": ["cache.py"],
        "monitoring": ["monitor.py"],
        "tests": ["test_aggregator.py", "test_authenticator.py", "test_trade_logic.py"],
        "exploration": ["trader.py", "sentiment_analyzer.py", "simulator.py", "test1.py", "backtest_1.py", "test7.py", "test8.py", 
                        "muilt_model_eval.py", "logistic_regression_eval.py", "logistic_regress_tst.py"],
        "notebooks": ["data_analysis.ipynb", "signal_testing.ipynb", "live_vs_backtest_comparison.ipynb", 
                      "sentiment_correlation.ipynb", "strategy_experiments.ipynb", "feature_engineering.ipynb"],
        "archive": ["strategy.py", "indicators.py", "schwab_client_old.py", "old_authenticator.py"],
        "files": ["structure.py", "requirements.txt"]
    }
}

def create_structure(base_path, tree):
    for key, value in tree.items():
        path = os.path.join(base_path, key)
        os.makedirs(path, exist_ok=True)
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                subpath = os.path.join(path, subkey)
                os.makedirs(subpath, exist_ok=True)
                if isinstance(subvalue, list):
                    for f in subvalue:
                        open(os.path.join(subpath, f), 'a').close()
        elif isinstance(value, list):
            for f in value:
                open(os.path.join(path, f), 'a').close()

create_structure(".", structure)


import os
from pathlib import Path

# Redefine after code execution environment reset
base_dir = Path("schwab_trader")

structure = {
    "core": [
        "backtester.py",
        "executor.py",
        "mock_executor.py",
        "eventhandler.py",
        "position_sizer.py",
        "runner.py",
        "simulator/gbm_simulator.py",
        "simulator/strategy_router.py",
        "simulator/live_plotter.py",
        "simulator/historical_loader.py",
        "simulator/drawdown_monitor.py",
        "logic/trade_logic_base.py",
        "logic/default_trade_logic.py",
        "logic/execution_engine.py",
        "logic/trade_logic_manager.py",
        "logic/strategy_routing_manager.py",
        "broker/broker_interface.py",
        "broker/alpaca_broker.py",
        "base/strategy_base.py",
        "base/indicator_base.py",
        "base/execution_engine_base.py",
        "base/position_sizer_base.py",
        "base/risk_model_base.py",
        "base/executor_base.py",
        "base/event_handler_base.py",
        "base/sentiment_model_base.py",
        "base/trade_logger_base.py"
    ]
}

for parent, files in structure.items():
    for file in files:
        file_path = base_dir / parent / file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch(exist_ok=True)

"`core/` structure fully created."
