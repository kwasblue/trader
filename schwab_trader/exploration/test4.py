#%% evaluates model perfoance vs amount of test data relative to total data 
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from data.datautils import load_stock_Data
from sp500 import sp500_tickers
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, 
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, 
    mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

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

# Load and preprocess data
data = load_stock_Data(sp500_tickers[:10])  # Load the data for the first 30 tickers

# Using list comprehension to collect dataframes and add 'Ticker' column
dfs = []
for ticker in sp500_tickers[:5]:
    try:
        df = data.get_dataframe(ticker)
        if not df.empty:
            dfs.append(df.assign(Ticker=ticker))
    except KeyError:
        print(f"Skipping {ticker}: No DataFrame found.")  # Optional logging



# Concatenate all dataframes into one large dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Make sure the data is sorted by Date and set it as the index if necessary
combined_df['Date'] = pd.to_datetime(combined_df['Date'])
combined_df = combined_df.sort_values(by='Date')

# Optionally, set 'Date' as the index
#combined_df.set_index('Date', inplace=True)

X, y = preprocess_data(combined_df)

# Train-test split for evaluating model performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define pipelines for different models
models = {
    'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=1000), True),
    'Support Vector Classifier (SVC)': (SVC(probability=True, class_weight='balanced'), True),
    'K-Nearest Neighbors Classifier': (KNeighborsClassifier(), True),
    'Random Forest Classifier': (RandomForestClassifier(class_weight='balanced'), True),

    'Linear Regression': (LinearRegression(), False),
    'Ridge Regression': (Ridge(), False),
    'Lasso Regression': (Lasso(), False),
    'Elastic Net Regression': (ElasticNet(), False),
    'Decision Tree Regression': (DecisionTreeRegressor(), False),
    'Random Forest Regression': (RandomForestRegressor(), False),
    'Gradient Boosting Regression': (GradientBoostingRegressor(), False),
    'Support Vector Regression (SVR)': (SVR(), False),
    'K-Nearest Neighbors Regression': (KNeighborsRegressor(), False),
    'MLP Regression': (MLPRegressor(max_iter=1000), False),
}


# Define hyperparameter search space for each model
param_dist = {
    # Regression Models
    'Linear Regression': {
        # No hyperparameters to tune for Linear Regression
    },
    'Ridge Regression': {
        'model__alpha': [0.1, 1, 10, 100],  # Regularization strength
        'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],  # Solver to use
    },
    'Lasso Regression': {
        'model__alpha': [0.1, 1, 10, 100],  # Regularization strength
        'model__max_iter': [1000, 2000, 5000],  # Maximum number of iterations
    },
    'Elastic Net Regression': {
        'model__alpha': [0.1, 1, 10, 100],  # Regularization strength
        'model__l1_ratio': [0.1, 0.5, 0.9],  # The balance between Lasso and Ridge
        'model__max_iter': [1000, 2000, 5000],  # Maximum number of iterations
    },
    'Decision Tree Regression': {
        'model__max_depth': [5, 10, 20, None],  # Maximum depth of tree
        'model__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
        'model__min_samples_leaf': [1, 2, 5],  # Minimum samples required at a leaf node
        'model__max_features': ['auto', 'sqrt', 'log2', None],  # Number of features to consider for splitting
    },
    'Random Forest Regression': {
        'model__n_estimators': [100, 200, 300, 500],  # Number of trees in the forest
        'model__max_depth': [5, 10, None],  # Maximum depth of each tree
        'model__min_samples_split': [2, 5],  # Minimum samples required to split a node
        'model__min_samples_leaf': [1, 2],  # Minimum samples required at a leaf node
        'model__max_features': ['sqrt', 'log2', None],  # Number of features to consider for splitting
        'model__bootstrap': [True, False],  # Whether bootstrap samples are used
    },
    'Gradient Boosting Regression': {
        'model__n_estimators': [100, 200, 300, 500],  # Number of boosting stages to be run
        'model__learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
        'model__max_depth': [3, 5, 7],  # Maximum depth of individual trees
        'model__subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for each boosting iteration
        'model__min_samples_split': [2, 5],  # Minimum samples required to split a node
    },
    'Support Vector Regression (SVR)': {
        'model__C': [0.1, 1, 10, 100],  # Regularization parameter
        'model__kernel': ['linear', 'rbf'],  # Kernel to use
        'model__gamma': ['scale', 'auto'],  # Kernel coefficient
        'model__epsilon': [0.1, 0.2, 0.5],  # Epsilon parameter (tolerance for errors)
    },
    'K-Nearest Neighbors Regression': {
        'model__n_neighbors': [3, 5, 7, 10],  # Number of neighbors to use
        'model__weights': ['uniform', 'distance'],  # Weight function used in prediction
        'model__metric': ['euclidean', 'manhattan'],  # Distance metric
    },
    'MLP Regression': {
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Number of neurons in each layer
        'model__activation': ['relu', 'tanh'],  # Activation function
        'model__alpha': [0.0001, 0.001, 0.01],  # Regularization strength
        'model__learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
        'model__max_iter': [1000, 2000, 5000],  # Maximum number of iterations
    },
}


# Function to evaluate a model
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model_name, model, param_dist, X_train, y_train, X_test, y_test, is_classification):
    print(f"Evaluating {model_name}...\n")
    
    num_features = X_train.select_dtypes(include=['float64', 'int64']).columns
    cat_features = X_train.select_dtypes(include=['object']).columns
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
    random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=5, scoring=scoring, verbose=1, n_jobs=-1)
    
    # Train model on increasing fractions of the dataset
    train_sizes = np.linspace(0.1, 1.0, 10)
    performance_metrics = {
        'train_size': [],
        'accuracy': [],
        'f1_score': [],
        'r2_score': [],
        'rmse': []
    }
    
    for size in train_sizes:
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train, y_train, train_size=size if size < 1.0 else 0.99, stratify=y_train if is_classification else None, random_state=42
        )
        
        random_search.fit(X_train_sample, y_train_sample)
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Store performance metrics
        performance_metrics['train_size'].append(size * 100)
        
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            performance_metrics['accuracy'].append(accuracy)
            performance_metrics['f1_score'].append(f1)
            print(f"Train Size: {size*100:.1f}%, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        else:
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            performance_metrics['r2_score'].append(r2)
            performance_metrics['rmse'].append(rmse)
            print(f"Train Size: {size*100:.1f}%, R^2: {r2:.4f}, RMSE: {rmse:.4f}")
    
    # Plot performance vs. training data size
    plt.figure(figsize=(8, 5))
    
    if is_classification:
        plt.plot(performance_metrics['train_size'], performance_metrics['accuracy'], label="Accuracy", marker="o")
        plt.plot(performance_metrics['train_size'], performance_metrics['f1_score'], label="F1 Score", marker="s")
    else:
        plt.plot(performance_metrics['train_size'], performance_metrics['r2_score'], label="R^2 Score", marker="o")
        plt.plot(performance_metrics['train_size'], performance_metrics['rmse'], label="RMSE", marker="s")
    
    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Performance Score")
    plt.title(f"Model Performance vs. Training Data Size: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
#%% Run models with learning curve analysis
for model_name, (model, is_classification) in models.items():
    evaluate_model(model_name, model, param_dist.get(model_name, {}), X_train, y_train, X_test, y_test, is_classification)

# %%
