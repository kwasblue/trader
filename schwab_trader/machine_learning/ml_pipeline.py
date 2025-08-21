# %% Imports
import os
import sys



# Dynamically set root path (one level up from 'exploration')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, precision_recall_curve,
    auc, confusion_matrix
)

# LightGBM (optional)
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# Your data accessors
from data.datautils import load_stock_Data
from exploration.sp500 import sp500_tickers

SEED = 42
# %% Load and Preprocess

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(by='Date').copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # drop useless all-NaN cols
    df.dropna(axis=1, how='all', inplace=True)

    # Binary next-day up label
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    X = df.drop(columns=['Target'])
    y = df['Target']
    return X, y

length = 1  # how many S&P symbols to include
store = load_stock_Data(sp500_tickers[:length])

frames = []
for ticker in sp500_tickers[:length]:
    try:
        tmp = store.get_dataframe(ticker)
        if tmp is not None and not tmp.empty:
            tmp = tmp.copy()
            tmp['Ticker'] = ticker
            frames.append(tmp)
    except KeyError:
        print(f"Skipping {ticker}: No DataFrame found.")

combined_df = pd.concat(frames, ignore_index=True)
combined_df['Date'] = pd.to_datetime(combined_df['Date'])
combined_df = combined_df.sort_values(by='Date')

X, y = preprocess_data(combined_df)

# chronological split (avoid leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# %% Preprocessing (ColumnTransformer)

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()
if 'Ticker' not in cat_features and 'Ticker' in X.columns:
    cat_features.append('Ticker')

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),  # scale only numeric features
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# %% Run directory helpers (artifacts + metrics persistence)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

RUN_ID = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
RUN_TAG = f"daily_sp500_{length}"
RUN_DIR = os.path.join('experiments', f"{RUN_TAG}_{RUN_ID}")

# subfolders
DIR_MODELS = os.path.join(RUN_DIR, 'models')
DIR_SEARCH = os.path.join(RUN_DIR, 'search')
DIR_METRICS = os.path.join(RUN_DIR, 'metrics')
DIR_REPORTS = os.path.join(RUN_DIR, 'reports')
for d in [RUN_DIR, DIR_MODELS, DIR_SEARCH, DIR_METRICS, DIR_REPORTS]:
    _ensure_dir(d)

# Save dataset + run metadata
meta = {
    'run_id': RUN_ID,
    'tag': RUN_TAG,
    'seed': SEED,
    'tickers': sp500_tickers[:length],
    'date_min': str(X.index.min()) if hasattr(X.index, 'min') else None,
    'date_max': str(X.index.max()) if hasattr(X.index, 'max') else None,
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'num_features': num_features,
    'cat_features': cat_features,
}
with open(os.path.join(RUN_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

# %% Model Definitions and Hyperparameter Grids (LR / RF / LGBM)

model_configs = {
    'Logistic_Regression': {
        'model': LogisticRegression(class_weight='balanced', max_iter=2000, solver='saga', random_state=SEED),
        'params': {
            'model__C': np.logspace(-3, 2, 20),
            'model__penalty': ['l1', 'l2'],
        }
    },
    'Random_Forest': {
        'model': RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=SEED),
        'params': {
            'model__n_estimators': np.arange(200, 801, 100),
            'model__max_depth': [None] + list(np.arange(4, 21, 4)),
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
}

if HAS_LGBM:
    model_configs['LightGBM'] = {
        'model': LGBMClassifier(
            n_estimators=600,
            objective='binary',
            random_state=SEED,
            verbose=-1,
        ),
        'params': {
            'model__learning_rate': np.logspace(-3, -1, 10),
            'model__num_leaves': np.arange(16, 64),
            'model__max_depth': [-1, 4, 6, 8, 10],
            'model__subsample': np.linspace(0.6, 1.0, 5),
            'model__colsample_bytree': np.linspace(0.6, 1.0, 5),
            'model__reg_lambda': np.logspace(-3, 1, 8),
            'model__min_child_samples': np.arange(10, 51, 5),
        }
    }

# %% Persistence helpers

def _json_default(o):
    import numpy as _np
    import pandas as _pd
    from datetime import datetime as _dt
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    if isinstance(o, (_pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (_pd.Series,)):
        return o.tolist()
    if isinstance(o, (_pd.DataFrame,)):
        return o.to_dict(orient='list')
    return str(o)


def _clean_params(d: dict) -> dict:
    """Convert numpy scalars/arrays in best_params to plain Python types."""
    out = {}
    for k, v in d.items():
        if hasattr(v, 'item'):
            try:
                out[k] = v.item()
                continue
            except Exception:
                pass
        if hasattr(v, 'tolist'):
            try:
                out[k] = v.tolist()
                continue
            except Exception:
                pass
        out[k] = v
    return out


def save_model(model, model_name, run_dir=RUN_DIR):
    path = os.path.join(run_dir, 'models', f"{model_name.lower()}_best.joblib")
    joblib.dump(model, path)
    return path


def save_cv_results(search: RandomizedSearchCV, model_name: str, run_dir=RUN_DIR):
    df = pd.DataFrame(search.cv_results_)
    path = os.path.join(run_dir, 'search', f'{model_name}_cv_results.csv')
    df.to_csv(path, index=False)
    return path


def save_metrics(metrics: dict, model_name: str, run_dir=RUN_DIR):
    path = os.path.join(run_dir, 'metrics', f'{model_name}_metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    return path


def save_report(text: str, model_name: str, run_dir=RUN_DIR):
    path = os.path.join(run_dir, 'reports', f'{model_name}_report.txt')
    with open(path, 'w') as f:
        f.write(text)
    return path


def update_leaderboard(run_dir=RUN_DIR, results: dict | None = None):
    """Append this run to a global CSV leaderboard for quick comparisons."""
    lb_path = os.path.join('experiments', 'leaderboard.csv')
    rows = []
    for m, vals in (results or {}).items():
        rows.append({
            'run_id': RUN_ID,
            'tag': RUN_TAG,
            'model': m,
            'roc_auc': vals.get('ROC-AUC'),
            'pr_auc': vals.get('PR-AUC'),
            'accuracy': vals.get('Accuracy'),
            'f1': vals.get('F1-Score'),
            'run_dir': run_dir,
        })
    df = pd.DataFrame(rows)
    if os.path.exists(lb_path):
        old = pd.read_csv(lb_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(lb_path, index=False)

# %% Train and Evaluate All Models (with artifact saving)

results = {}
cv = TimeSeriesSplit(n_splits=5)

for model_name, config in model_configs.items():
    print(f" Training {model_name}...")  # fixed

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', config['model'])
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=config['params'],
        n_iter=20,
        scoring='roc_auc',
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=SEED,
        refit=True,
    )

    search.fit(X_train, y_train)

    # Predictions on holdout
    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1] if hasattr(best.named_steps['model'], 'predict_proba') else None

    # Metrics
    pr_auc = None
    roc = None
    if y_prob is not None:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)  # auc(x=recall, y=precision)
        roc = roc_auc_score(y_test, y_prob)

    cleaned_params = _clean_params(search.best_params_)

    result = {
        'Best Params': cleaned_params,
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'F1-Score': float(f1_score(y_test, y_pred, zero_division=0)),
        'ROC-AUC': None if roc is None else float(roc),
        'PR-AUC': None if pr_auc is None else float(pr_auc),
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist(),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
    }

    # Persist artifacts for this model
    model_path = save_model(best, model_name)
    cv_path = save_cv_results(search, model_name)
    metrics_path = save_metrics(result, model_name)
    report_text = classification_report(y_test, y_pred, digits=4)
    report_path = save_report(report_text, model_name)

    # Log locations
    result['Artifacts'] = {
        'model': model_path,
        'cv_results': cv_path,
        'metrics': metrics_path,
        'report': report_path,
    }

    results[model_name] = result

# Save a compact results JSON for the whole run
with open(os.path.join(RUN_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=_json_default)

# Update global leaderboard
update_leaderboard(RUN_DIR, results)

# %% Display Results
for model_name, metrics in results.items():
    print(f"{model_name} Results:")  # fixed
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

print(f"Artifacts saved under: {RUN_DIR}")  # fixed

# %% Best model loading helpers
from typing import Optional, Dict, Tuple

LEADERBOARD_PATH = os.path.join('experiments', 'leaderboard.csv')


def _read_leaderboard(path: str = LEADERBOARD_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Leaderboard not found at {path}. Run a training job first.")
    df = pd.read_csv(path)
    # Ensure numeric types
    for c in ['roc_auc', 'pr_auc', 'accuracy', 'f1']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _composite_score(row: pd.Series, weights: Optional[Dict[str, float]]) -> float:
    if not weights:
        # default to ROC-AUC, fallback to PR-AUC, then F1, then Accuracy
        for m in ['roc_auc', 'pr_auc', 'f1', 'accuracy']:
            val = row.get(m, np.nan)
            if pd.notna(val):
                return float(val)
        return -np.inf
    score = 0.0
    for k, w in weights.items():
        val = row.get(k, np.nan)
        if pd.isna(val):
            continue
        score += w * float(val)
    return score


def select_best(
    model: Optional[str] = None,
    primary: Optional[str] = 'roc_auc',
    weights: Optional[Dict[str, float]] = None,
    tag: Optional[str] = None,
    prefer_recent: bool = True,
    since: Optional[str] = None,  # 'YYYYMMDD' filter on run_id date
    path: str = LEADERBOARD_PATH,
) -> pd.Series:
    """Pick the best row from leaderboard with simple, explainable rules.

    Args:
        model: restrict to a single model family (e.g., 'LightGBM').
        primary: if provided, sort by this metric first (higher is better).
        weights: optional composite weights dict, e.g., {'roc_auc':0.6,'pr_auc':0.4}.
        tag: optional run tag filter (e.g., 'daily_sp500_60').
        prefer_recent: tie-break by latest run_id if True.
        since: optional YYYYMMDD; keep rows with run_id date >= since.
    Returns:
        A pandas Series (one row) for the selected best model.
    """
    df = _read_leaderboard(path)

    if model is not None:
        df = df[df['model'] == model]
    if tag is not None and 'tag' in df.columns:
        df = df[df['tag'] == tag]
    if since is not None and 'run_id' in df.columns:
        # run_id like 'YYYYMMDD_HHMMSS'
        df = df[df['run_id'].astype(str).str[:8] >= since]

    if df.empty:
        raise ValueError("No candidates after filtering.")

    # Compute score column
    if weights is not None:
        df = df.copy()
        df['__score'] = df.apply(lambda r: _composite_score(r, weights), axis=1)
        sort_cols = ['__score']
    elif primary is not None and primary in df.columns:
        sort_cols = [primary]
    else:
        sort_cols = ['roc_auc', 'pr_auc', 'f1', 'accuracy']
        sort_cols = [c for c in sort_cols if c in df.columns]

    # Build tie-breaker order
    tie_breakers = []
    for c in ['pr_auc', 'f1', 'accuracy']:
        if c != (primary or '') and c in df.columns:
            tie_breakers.append(c)
    if 'run_id' in df.columns:
        tie_breakers.append('run_id')  # later is better

    sort_by = sort_cols + tie_breakers
    ascending = [False] * len(sort_cols) + [False] * len(tie_breakers)

    best_row = df.sort_values(by=sort_by, ascending=ascending).iloc[0]
    return best_row


def _resolve_model_path(best_row: pd.Series) -> str:
    # Prefer results.json -> Artifacts.model path to avoid naming drift
    run_dir = best_row['run_dir']
    model_name = best_row['model']
    results_path = os.path.join(run_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        if model_name in results and 'Artifacts' in results[model_name]:
            art = results[model_name]['Artifacts']
            if isinstance(art, dict) and 'model' in art:
                return art['model']
    # Fallback to conventional name
    guess = os.path.join(run_dir, 'models', f"{model_name.lower()}_best.joblib")
    if not os.path.exists(guess):
        raise FileNotFoundError(f"Could not locate model artifact. Looked for {guess}")
    return guess


def load_best_model(
    model: Optional[str] = None,
    primary: Optional[str] = 'roc_auc',
    weights: Optional[Dict[str, float]] = None,
    tag: Optional[str] = None,
    prefer_recent: bool = True,
    since: Optional[str] = None,
    leaderboard_path: str = LEADERBOARD_PATH,
) -> Tuple[object, Dict]:
    """Load the best-performing model (sklearn pipeline) and return (pipeline, info dict).

    The info dict includes: run_dir, model_name, metrics row, and artifact path.
    """
    row = select_best(
        model=model,
        primary=primary,
        weights=weights,
        tag=tag,
        prefer_recent=prefer_recent,
        since=since,
        path=leaderboard_path,
    )
    model_path = _resolve_model_path(row)
    pipe = joblib.load(model_path)

    info = {
        'model_name': row['model'],
        'run_dir': row['run_dir'],
        'artifact_path': model_path,
        'metrics': {
            m: (None if m not in row or pd.isna(row[m]) else float(row[m]))
            for m in ['roc_auc', 'pr_auc', 'accuracy', 'f1']
        },
        'row': row.to_dict(),
    }
    return pipe, info

# %% Example (commented usage)
# best_pipe, info = load_best_model(primary='roc_auc')
# y_pred = best_pipe.predict(X_test)
# y_prob = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe, 'predict_proba') else None
# print(info)

# %% ml_process (returns DataFrame only)
import numpy as np
import pandas as pd

def ml_process(
    self,
    sma_window: int,
    ema_window: int,
    *,
    scaling_method: str | None = None,   # "standard" | "minmax" | None
    pca_components: int = 5,
    denoise_cols: list[str] | None = None,
    include_scaled: bool = True,
    include_pca: bool = False,
) -> pd.DataFrame:
    """
    Clean -> indicators -> engineered -> (optional) denoise -> combine -> optional scale/PCA (numeric only)

    ALWAYS returns a single DataFrame. Scaler/PCA objects are **not** returned.
    Notes:
      - To avoid data leakage in ML, prefer handling scaling/PCA **inside** your sklearn Pipeline.
        In that case call with `include_scaled=False, include_pca=False`.
    """
    # --- 1) load/prepare base frames with a consistent DateTimeIndex ---
    base = self.clean_stock_data().copy()           # must contain 'Date'
    if "Date" in base.columns:
        base["Date"] = pd.to_datetime(base["Date"], errors='coerce')
        base = base.set_index("Date", drop=True)
    base = base[~base.index.duplicated(keep='first')]

    if denoise_cols:
        cols = [c for c in denoise_cols if c in base.columns]
        if cols:
            base.loc[:, cols] = self.apply_signal_processing(base.loc[:, cols].copy(), cols)

    inds = self.apply_indicators(sma_window=sma_window, ema_window=ema_window).copy()
    if "Date" in inds.columns:
        inds["Date"] = pd.to_datetime(inds["Date"], errors='coerce')
        inds = inds.set_index("Date", drop=True)
    inds = inds.drop(columns=[c for c in ("Open","High","Low","Close","Volume","Date") if c in inds.columns], errors="ignore")

    eng = self.feature_engineering().copy()
    if "Date" in eng.columns:
        eng["Date"] = pd.to_datetime(eng["Date"], errors='coerce')
        eng = eng.set_index("Date", drop=True)

    # --- 2) align & combine on index (inner join keeps common timestamps) ---
    combined = base.join([inds, eng], how="inner").sort_index()
    combined = combined.loc[:, ~combined.columns.duplicated()].copy()

    # Replace infs that could break scalers/PCA
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- 3) numeric matrix for transforms ---
    num = combined.select_dtypes(include=[np.number]).copy()

    # Bound PCA components by numeric dimensionality and samples
    max_pca = 0
    if include_pca and pca_components and not num.empty:
        max_pca = max(0, min(pca_components, num.shape[1], max(1, num.shape[0] - 1)))

    # --- 4) scaling (optional; numeric only) ---
    scaled_df = None
    if scaling_method:
        if scaling_method.lower() in ("standard", "z", "zscore"):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scaling_method.lower() in ("minmax", "min_max"):
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        scaled_vals = scaler.fit_transform(num.values)
        scaled_df = pd.DataFrame(scaled_vals, index=num.index, columns=[f"{c}__scaled" for c in num.columns])

    # --- 5) PCA (optional; on scaled if available, else on standardized copy) ---
    pca_df = None
    if include_pca and max_pca > 0:
        from sklearn.decomposition import PCA
        if scaled_df is not None:
            X_for_pca = scaled_df.values
        else:
            # zero-center before PCA to avoid scale dominance
            from sklearn.preprocessing import StandardScaler
            X_for_pca = StandardScaler().fit_transform(num.values)
        pca = PCA(n_components=max_pca, svd_solver="auto", whiten=False)
        pcs = pca.fit_transform(X_for_pca)
        pca_df = pd.DataFrame(pcs, index=num.index, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])

    # --- 6) assemble final frame ---
    out = combined.copy()
    if include_scaled and scaled_df is not None:
        out = out.join(scaled_df, how="left")
    if include_pca and pca_df is not None:
        out = out.join(pca_df, how="left")

    # Reset index to keep Date visible as a column
    out = out.reset_index().rename(columns={"index": "Date"})

    # Optional: drop rows with any remaining NaNs created by transforms (comment out if undesired)
    # out = out.dropna()

    self.logger.info(f"ML data processed for {self.stock} | rows={len(out)} | cols={len(out.columns)}")
    return out
