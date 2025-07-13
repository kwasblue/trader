# %% Imports
import json
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, precision_recall_curve,
    auc, confusion_matrix
)

from utils.datautils import load_stock_Data
from sp500 import sp500_tickers

# %% Load and Preprocess
def preprocess_data(df):
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    X = df.drop(columns=['Target'])
    y = df['Target']
    return X, y

length = 60
data = load_stock_Data(sp500_tickers[:length])

dfs = []
for ticker in sp500_tickers[:length]:
    try:
        df = data.get_dataframe(ticker)
        if not df.empty:
            df['Ticker'] = ticker
            dfs.append(df)
    except KeyError:
        print(f"Skipping {ticker}: No DataFrame found.")

combined_df = pd.concat(dfs, ignore_index=True)
combined_df['Date'] = pd.to_datetime(combined_df['Date'])
combined_df = combined_df.sort_values(by='Date')

X, y = preprocess_data(combined_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# %% Preprocessing
num_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

if 'Ticker' not in cat_features:
    cat_features.append('Ticker')

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# %% Model Definitions and Hyperparameter Grids
model_configs = {
    'Logistic_Regression': {
        'model': LogisticRegression(class_weight='balanced', max_iter=1000),
        'params': {
            'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear', 'saga'],
            'model__max_iter': [1000, 2000]
        }
    },
    'Random_Forest': {
        'model': RandomForestClassifier(class_weight='balanced', n_jobs=-1),
        'params': {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5, 10]
        }
    },
    'Ridge_Classifier': {
        'model': RidgeClassifier(),
        'params': {
            'model__alpha': [0.1, 1.0, 10.0]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    }
}

# %% Train and Evaluate All Models
def save_model(model, model_name, directory="saved_models"):
    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}/{model_name.lower()}_best.pkl"
    joblib.dump(model, filename)
    print(f"✅ Saved {model_name} model to: {filename}")

results = {}

for model_name, config in model_configs.items():
    print(f"\n🚀 Training {model_name}...\n")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', config['model'])
    ])
    
    search = RandomizedSearchCV(
        pipeline, param_distributions=config['params'],
        n_iter=10, scoring='accuracy', cv=5, verbose=2, n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    y_pred = search.best_estimator_.predict(X_test)
    y_pred_prob = search.best_estimator_.predict_proba(X_test)[:, 1] if hasattr(search.best_estimator_['model'], "predict_proba") else None
    
    result = {
        'Best Params': search.best_params_,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None,
        'Precision-Recall AUC': auc(*precision_recall_curve(y_test, y_pred_prob)[:2]) if y_pred_prob is not None else None,
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    
    results[model_name] = result
    save_model(search.best_estimator_, model_name)

# %% Display Results
for model_name, metrics in results.items():
    print(f"\n📊 {model_name} Results:")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}:\n{v}")

# %%
