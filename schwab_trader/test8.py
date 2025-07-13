#%% Re-import necessary packages after kernel reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import re

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import re
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def run_multivariate_access_regression(
    df,
    predictors,
    access_col="Daily Return",
    output_path="access_regression_summary.txt"
):
    """
    Runs a multivariate OLS regression with column name sanitization for compatibility.
    """

    def sanitize(col):
        # Replace all non-alphanumeric characters with underscore
        return re.sub(r'\W+', '_', col)

    # Sanitize all column names
    rename_map = {col: sanitize(col) for col in [access_col] + predictors}
    df_model = df[[access_col] + predictors].dropna().rename(columns=rename_map)

    # Rebuild formula
    safe_access_col = rename_map[access_col]
    safe_predictors = [rename_map[col] for col in predictors]
    formula = f"{safe_access_col} ~ " + " + ".join(safe_predictors)

    # Fit and summarize
    model = smf.ols(formula=formula, data=df_model).fit()
    summary = model.summary()
    print(summary)

    # Save summary to file
    with open(output_path, "w") as f:
        f.write(str(summary))

    print(f"\n✅ Regression summary saved to: {output_path}")
    return model

# Define the ridge regression analysis function again
def run_ridge_regression_analysis(df, features, target, alphas=np.logspace(-3, 3, 100)):
    """
    Run Ridge regression on a stock dataset and return model, scaler, and coefficient plot.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing features and target.
        features (list): List of feature column names.
        target (str): Target variable column name.
        alphas (array): Array of alpha values for RidgeCV.

    Returns:
        ridge (RidgeCV): Trained RidgeCV model.
        scaler (StandardScaler): Scaler used to transform features.
        coef_df (pd.DataFrame): DataFrame with feature coefficients.
    """
    # Drop NA
    df_model = df[features + [target]].dropna()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[features])
    y = df_model[target].values

    # Fit Ridge Regression
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_scaled, y)

    # Create DataFrame of coefficients
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': ridge.coef_
    }).sort_values(by='Coefficient', key=np.abs, ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df, y='Feature', x='Coefficient', palette='viridis')
    plt.title(f"Feature Importances for Ridge Regression Predicting {target}")
    plt.tight_layout()
    plt.show()

    return ridge, scaler, coef_df

#%%
test = pd.read_csv("aapl.csv")
features = ['SMA 200','EMA 50','MACD','Signal Line','Bollinger Upper','Bollinger Lower','RSI','ATR','VWAP','OBV','Momentum','ROC','SAR',
            'Rolling_Mean_10','Rolling_Std_10','Lag_Close_1','Lag_Close_2','Lag_Close_3','Lag_Close_4','Lag_Close_5']
target = 'Daily_Return' 
#%%
run_ridge_regression_analysis(test, features=features, target=target)
# %%
run_multivariate_access_regression(test, features, access_col=target)
# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (assuming it's already loaded into `df`)
df = test  # Replace with your file path

# Select features and target

target = "Daily_Return"

X = df[features]
y = df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluate
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"R² (Train): {train_r2:.4f}")
print(f"R² (Test): {test_r2:.4f}")
print(f"MSE (Test): {test_mse:.6f}")

# Feature Importances
feature_importances = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importances, palette="viridis")
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# (Optional) Cross-Validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Average CV R²: {np.mean(cv_scores):.4f}")
# %%
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load your dataset
data = test # Update with your file name

# Select only the important features
features_to_keep = [
    "Price Change",
    "Percent Change",
    "Close_Open_Diff",
    "Rolling_Mean_10",
    "RSI"  # Optional: we can test with and without RSI
]

X = data[features_to_keep]
y = data["Daily_Return"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add constant for OLS intercept
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit OLS
model = sm.OLS(y_train, X_train_sm).fit()

# Print summary
print(model.summary())

# Evaluate on test
y_pred = model.predict(X_test_sm)
test_r2 = model.rsquared
print(f"Test R²: {test_r2:.4f}")

# %%
