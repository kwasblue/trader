#%% creates models and evaluates performance
import json
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from data.datautils import load_stock_Data
from sp500 import sp500_tickers
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, 
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, 
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB


def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def preprocess_data(df: pd.DataFrame):
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    feature_cols = [col for col in df.columns if col not in ['Target']]
    X = df[feature_cols]
    y = df['Target']
    return X, y

length = 20
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

models = {
    'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=5000), True),
    #'Support Vector Classifier (SVC)': (SVC(probability=True, class_weight='balanced'), True),
    #'K-Nearest Neighbors Classifier': (KNeighborsClassifier(), True),
    #'Random Forest Classifier': (RandomForestClassifier(class_weight='balanced'), True),

    #'Linear Regression': (LinearRegression(), False),
    #'Ridge Regression': (Ridge(), False),
    #'Lasso Regression': (Lasso(), False),
    #'Elastic Net Regression': (ElasticNet(), False),
    #'Decision Tree Regression': (DecisionTreeRegressor(), False),
    #'Random Forest Regression': (RandomForestRegressor(), False),
    #'Gradient Boosting Regression': (GradientBoostingRegressor(), False),
    #'Support Vector Regression (SVR)': (SVR(), False),
    #'K-Nearest Neighbors Regression': (KNeighborsRegressor(), False),
    #'MLP Regression': (MLPRegressor(max_iter=1000), False),
}

param_dist = {
    # Classification Models
    'Logistic Regression': {
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    },
    # Regression Models
    'Linear Regression': {},
    'Ridge Regression': {
        'model__alpha': [0.1, 1, 10, 100],
        'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
    },
    'Lasso Regression': {
        'model__alpha': [0.1, 1, 10, 100],
        'model__max_iter': [1000, 2000, 5000],
    },
    'Elastic Net Regression': {
        'model__alpha': [0.1, 1, 10, 100],
        'model__l1_ratio': [0.1, 0.5, 0.9],
        'model__max_iter': [1000, 2000, 5000],
    },
    'Decision Tree Regression': {
        'model__max_depth': [5, 10, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 5],
        'model__max_features': ['auto', 'sqrt', 'log2', None],
    },
    'Random Forest Regression': {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__max_features': ['sqrt', 'log2', None],
        'model__bootstrap': [True, False],
    },
    'Gradient Boosting Regression': {
        'model__n_estimators': [100, 200, 300, 500],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__min_samples_split': [2, 5],
    },
    'Support Vector Regression (SVR)': {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto'],
        'model__epsilon': [0.1, 0.2, 0.5],
    },
    'K-Nearest Neighbors Regression': {
        'model__n_neighbors': [3, 5, 7, 10],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan'],
    },
    'MLP Regression': {
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__learning_rate': ['constant', 'adaptive'],
        'model__max_iter': [1000, 2000, 5000],
    },
}


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

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
    random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=5, scoring=scoring, verbose=1, n_jobs=-1)
    random_search.fit(X_train, y_train)

    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    y_pred = random_search.best_estimator_.predict(X_test)

    if is_classification:
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1-Score:", f1_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        y_pred_prob = random_search.best_estimator_.predict_proba(X_test)[:, 1] if hasattr(random_search.best_estimator_, 'predict_proba') else None
        if y_pred_prob is not None:
            print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
            precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
            print("Precision-Recall AUC:", auc(recall, precision))

        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    else:
        print("R^2 Score:", r2_score(y_test, y_pred))
        print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False))

    print("-" * 50)

#%%
for model_name, (model, is_classification) in models.items():
    evaluate_model(model_name, model, param_dist.get(model_name, {}), X_train, y_train, X_test, y_test, is_classification)

# %%