#%% tewsting predictive models
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load dataset (assuming it's a CSV file)
df = pd.read_json(r'C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\proc_data\proc_AAL_file.json')

# Define Features (X) - Exclude non-numeric and irrelevant columns
features = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA 200", "EMA 50", "MACD", "Signal Line", "RSI",
    "ATR", "VWAP", "OBV", "Momentum", "ROC", "SAR",
    "Price Change", "Percent Change", "Price_Range",
    "Daily_Return", "Rolling_Mean_10", "Rolling_Std_10",
    "Close_Open_Diff", "Lag_Close_1", "Lag_Close_2",
    "Lag_Close_3", "Lag_Close_4", "Lag_Close_5"
]
#%%
# Drop rows with NaNs (optional: use imputation instead)
df = df.dropna(subset=features)

# Define target variable (Y) - Label trades as Buy (1) or Sell (0)
df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

# Define features and target variable
X = df[features]
y = df["Target"]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#%%
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Load JSON data
def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def preprocess_data(df: pd.DataFrame):
    df = df.sort_values(by='Date')  # Ensure chronological order
    df.set_index('Date', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)  # Remove entirely NaN columns
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Define features and target
    feature_cols = [col for col in df.columns if col not in ['Target']]
    X = df[feature_cols]
    y = df['Target']  # Define your target variable
    
    return X, y

# Profitability (returns from trading) function
def calculate_profitability(y_true, y_pred, df):
    # Store predicted signals
    df['Predicted_Signal'] = y_pred
    df['Actual_Signal'] = y_true
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Strategy: Buy if the model predicts 1 (up), sell if 0 (down)
    df['Strategy_Return'] = df['Predicted_Signal'] * df['Returns']
    
    # Calculate profitability
    df['Cumulative_Return'] = (1 + df['Returns']).cumprod() - 1
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1
    
    # Compare correct vs incorrect predictions
    correct_predictions = (df['Predicted_Signal'] == df['Actual_Signal']).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions
    
    print(f'Accuracy of Strategy: {accuracy * 100:.2f}%')
    
    # Return cumulative returns
    return df['Cumulative_Strategy_Return'].iloc[-1], df['Cumulative_Return'].iloc[-1]

# Load and preprocess data
df = load_data(r'C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\proc_data\proc_A_file.json')
X, y = preprocess_data(df)

# Train-test split for evaluating model performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define pipeline with imputer and scaler
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Corrected parameter grid
param_dist = {
    'rf__n_estimators': [100, 200, 300, 500],  # More options for number of trees
    'rf__max_depth': [5, 10, 20, None],  # Explore deeper trees and no limit on depth
    'rf__min_samples_split': [2, 5, 10],  # Try different splits for tree nodes
    'rf__min_samples_leaf': [1, 2, 4],  # Vary the number of samples in the leaf node
    'rf__max_features': ['sqrt', 'log2', None],  # Correct max_features values
    'rf__bootstrap': [True, False],  # Use or not use bootstrap samples for trees
    'rf__oob_score': [True],  # OOB can only be True when bootstrap=True
    'rf__warm_start': [True, False],  # Whether to reuse the solution of the previous call
    'rf__class_weight': [None, 'balanced'],  # Handle class imbalance
}

# Ensure correct settings during grid search for oob_score with bootstrap
def fix_oob_param(grid_params):
    # Ensure oob_score=True only if bootstrap=True
    if grid_params['rf__bootstrap'] == False:
        grid_params['rf__oob_score'] = [False]
    return grid_params

param_dist = fix_oob_param(param_dist)

# Time-based cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Run GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_dist, cv=tscv, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters after grid search
print("Best parameters:", grid_search.best_params_)

# Predict on the test data
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate model performance on test data
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Calculate profitability (strategy returns vs. market returns)
strategy_return, market_return = calculate_profitability(y_test, y_pred, df.loc[X_test.index])
print(f"Cumulative Strategy Return: {strategy_return}")
print(f"Cumulative Market Return: {market_return}")

# Evaluate the model and adjust parameters based on results
# (You can create a loop here to adjust parameters if necessary)

# %%
