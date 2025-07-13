# %% Imports and Setup
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

from utils.datautils import load_stock_Data
from sp500 import sp500_tickers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# %% Custom Wrapper for LinearRegression to behave like a Classifier
class LinearRegressorAsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        proba = self.model.predict(X)
        proba = np.clip(proba, 0, 1)
        return np.vstack([1 - proba, proba]).T

# %% Load and preprocess data
def preprocess_data(df):
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = df.drop(columns=['Target'])
    y = df['Target']
    return X, y

length = 100
data = load_stock_Data(sp500_tickers[:length])

dfs = []
for ticker in sp500_tickers[:length]:
    try:
        df = data.get_dataframe(ticker)
        if not df.empty:
            dfs.append(df.assign(Ticker=ticker))
    except KeyError:
        print(f"Skipping {ticker}: No DataFrame found.")

combined_df = pd.concat(dfs[:150], ignore_index=True)
combined_df['Date'] = pd.to_datetime(combined_df['Date'])
combined_df = combined_df.sort_values(by='Date')

X, y = preprocess_data(combined_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# %% Pipeline and Preprocessing
num_features = X.select_dtypes(include=['float64', 'int64']).columns
cat_features = X.select_dtypes(include=['object']).columns

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
rf = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
# %% Ensemble Classifier
ensemble_model = VotingClassifier(estimators=[
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('linreg_wrap', LinearRegressorAsClassifier()),
    ('rf', rf),
    ('gb', gb),
    ('knn', knn)
], voting='soft')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('model', ensemble_model)
])

# %% Train and Evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
print("Precision-Recall AUC:", auc(recall, precision))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# %%
