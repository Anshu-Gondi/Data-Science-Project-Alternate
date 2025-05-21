import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Paths
data_path = "data/processed/final_data.csv"
split_save_path = "data/models"
model_save_path = "models"  # Folder to save trained models

# Ensure directories exist
os.makedirs(split_save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Define target and features
target_col = 'Price'
numeric_features = ['BHK', 'Bathroom', 'Parking', 'Per_Sqft', 'Area_yards', 'Area']

X = df[numeric_features]
y = df[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Save splits only if not already saved
split_files = {
    "X_train": os.path.join(split_save_path, "X_train.csv"),
    "X_test": os.path.join(split_save_path, "X_test.csv"),
    "y_train": os.path.join(split_save_path, "y_train.csv"),
    "y_test": os.path.join(split_save_path, "y_test.csv"),
}

if not all(os.path.exists(f) for f in split_files.values()):
    print("Saving train-test splits to data/models/ ...")
    pd.DataFrame(X_train, columns=numeric_features).to_csv(
        split_files["X_train"], index=False)
    pd.DataFrame(X_test, columns=numeric_features).to_csv(
        split_files["X_test"], index=False)
    pd.DataFrame(y_train).to_csv(split_files["y_train"], index=False)
    pd.DataFrame(y_test).to_csv(split_files["y_test"], index=False)
else:
    print("Train-test split files already exist. Skipping save.")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models dictionary
models = {
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Evaluation function
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    # Save the trained model
    model_filename = os.path.join(model_save_path, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_filename)
    print(f"Saved {name} model to {model_filename}")

    # Test set metrics
    mae_test = mean_absolute_error(y_test, preds_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
    r2_test = r2_score(y_test, preds_test)

    # Train set metrics
    mae_train = mean_absolute_error(y_train, preds_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
    r2_train = r2_score(y_train, preds_train)

    print(f"\n{name}")
    print("------ Train Performance ------")
    print(f"MAE: {mae_train:.2f}")
    print(f"RMSE: {rmse_train:.2f}")
    print(f"R^2: {r2_train:.2f}")

    print("------ Test Performance -------")
    print(f"MAE: {mae_test:.2f}")
    print(f"RMSE: {rmse_test:.2f}")
    print(f"R^2: {r2_test:.2f}")

    # Comparison Table (sample)
    comparison = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": preds_test
    })
    print("\nSample Actual vs Predicted (Test):")
    print(comparison.head(10))

    # Scatter Plot: Actual vs Predicted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, preds_test, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name} Predictions vs Actual (Scatter)")
    plt.tight_layout()

    scatter_plot_path = os.path.join("results", "images", f"{name.replace(' ', '_').lower()}_scatter.png")
    os.makedirs(os.path.dirname(scatter_plot_path), exist_ok=True)
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Saved scatter plot to {scatter_plot_path}")

    # Distribution Plot: Actual vs Predicted
    plt.figure(figsize=(6, 4))
    sns.histplot(y_test, kde=True, color='blue', label='Actual Price', stat='density', bins=30)
    sns.histplot(preds_test, kde=True, color='orange', label='Predicted Price', stat='density', bins=30)
    plt.title(f"{name} - Distribution of Actual vs Predicted Prices")
    plt.xlabel("Price")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    distplot_path = os.path.join("results", "images", f"{name.replace(' ', '_').lower()}_distplot.png")
    plt.savefig(distplot_path)
    plt.close()
    print(f"Saved distribution plot to {distplot_path}")


# Train and evaluate all models
for name, model in models.items():
    if "Ridge" in name or "Lasso" in name:
        evaluate_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test)
    else:
        evaluate_model(name, model, X_train, y_train, X_test, y_test)
