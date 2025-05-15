# %% [markdown]
# # 🧠 Model Training 
# 
# ## 📦 Modules included 

# %%
# Data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# %% [markdown]
# ## 📈 Statistical Tools and Feature Scaling

# %%
from scipy import stats
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 🔀 Train-Test Split and Cross-Validation

# %%
from sklearn.model_selection import train_test_split, cross_val_score

# %% [markdown]
# ## 📊 Model Evaluation Metrics

# %%
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn import metrics


# %% [markdown]
# ## 🤖 Machine Learning Models

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# ## 📂 Loading The Dataset 

# %%
data = pd.read_csv("../data/processed/processed_data.csv")

data.head()

# %%
# Drop the 'id' column
X = data.drop(columns=['id', 'cardio'])
y = data['cardio']

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits
X_train.to_csv("../data/processed/X_train.csv", index=False)
X_test.to_csv("../data/processed/X_test.csv", index=False)
y_train.to_csv("../data/processed/y_train.csv", index=False)
y_test.to_csv("../data/processed/y_test.csv", index=False)

print("✅ Train-test splits saved to processed data folder!")

# %% [markdown]
# ## Logistic Regression 

# %%
# Initialize Logistic Regression model
logreg_model = LogisticRegression(random_state=42)

# Train the model
logreg_model.fit(X_train, y_train)

# Make predictions on training and test sets
y_pred_train_logreg = logreg_model.predict(X_train)
y_pred_test_logreg = logreg_model.predict(X_test)

# Compute training and test accuracies
train_accuracy_logreg = accuracy_score(y_train, y_pred_train_logreg)
test_accuracy_logreg = accuracy_score(y_test, y_pred_test_logreg)

# Get classification report for detailed metrices 
class_report_logreg = classification_report(y_test, y_pred_test_logreg)

# Display results
print(f"Logistic Regression Training Accuracy: {train_accuracy_logreg:.4f}")
print(f"Logistic Regression Test Accuracy: {test_accuracy_logreg:.4f}")
print(f"Logistic Regression Classification Report:\n{class_report_logreg}")

# %% [markdown]
# ## Random Forest 

# %%
# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on training and test sets
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

# Compute training and test accuracies
train_accuracy_rf = accuracy_score(y_train, y_pred_train_rf)
test_accuracy_rf = accuracy_score(y_test, y_pred_test_rf)

# Get classification report for detailed metrics
class_report_rf = classification_report(y_test, y_pred_test_rf)

# Display results
print(f"Random Forest Training Accuracy: {train_accuracy_rf:.4f}")
print(f"Random Forest Test Accuracy: {test_accuracy_rf:.4f}")
print(f"Random Forest Classification Report:\n{class_report_rf}")

# %% [markdown]
# ## KNeighborsClassifier (KNN)

# %%
# Initialize KNN model
knn_model = KNeighborsClassifier()

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on training and test sets
y_pred_train_knn = knn_model.predict(X_train)
y_pred_test_knn = knn_model.predict(X_test)

# Compute training and test accuracies
train_accuracy_knn = accuracy_score(y_train, y_pred_train_knn)
test_accuracy_knn = accuracy_score(y_test, y_pred_test_knn)

# Get classification report for detailed metrics
class_report_knn = classification_report(y_test, y_pred_test_knn)

# Display results
print(f"K-Nearest Neighbors Training Accuracy: {train_accuracy_knn:.4f}")
print(f"K-Nearest Neighbors Test Accuracy: {test_accuracy_knn:.4f}")
print(f"K-Nearest Neighbors Classification Report:\n{class_report_knn}")

# %% [markdown]
# ## K-Means

# %%
from scipy.stats import mode
import numpy as np
from sklearn.cluster import KMeans

# Fit KMeans on training features (usually after scaling!)
kmeans = KMeans(n_clusters=2, random_state=42)
train_clusters = kmeans.fit_predict(X_train)
test_clusters = kmeans.predict(X_test)

# Find the most common label for each cluster (cluster 0 and cluster 1)
mode_result_0 = mode(y_train[train_clusters == 0])
mode_result_1 = mode(y_train[train_clusters == 1])

# Check the mode result to debug
print("Mode result for cluster 0:", mode_result_0)
print("Mode result for cluster 1:", mode_result_1)

# Safe access to mode value
if isinstance(mode_result_0.mode, np.ndarray):
    cluster_0_mode = mode_result_0.mode[0]
else:
    cluster_0_mode = mode_result_0.mode  # If mode is scalar, take it directly

if isinstance(mode_result_1.mode, np.ndarray):
    cluster_1_mode = mode_result_1.mode[0]
else:
    cluster_1_mode = mode_result_1.mode  # If mode is scalar, take it directly

# Map clusters to actual labels
train_predictions = np.where(
    train_clusters == 0, cluster_0_mode, cluster_1_mode)
test_predictions = np.where(test_clusters == 0, cluster_0_mode, cluster_1_mode)

# Compute training and test accuracies
train_accuracy_kmeans = accuracy_score(y_train, train_predictions)
test_accuracy_kmeans = accuracy_score(y_test, test_predictions)

# Get classification report for detailed metrics
class_report_kmeans = classification_report(y_test, test_predictions)

# Display results
print(f"K-Means Training Accuracy: {train_accuracy_kmeans:.4f}")
print(f"K-Means Test Accuracy: {test_accuracy_kmeans:.4f}")
print(f"K-Means Classification Report:\n{class_report_kmeans}")

# %% [markdown]
# ## SVM

# %%
# Initialize SVM model
svm_model = SVC(kernel='linear', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on training and test sets
y_pred_train_svm = svm_model.predict(X_train)
y_pred_test_svm = svm_model.predict(X_test)

# Compute training and test accuracies
train_accuracy_svm = accuracy_score(y_train, y_pred_train_svm)
test_accuracy_svm = accuracy_score(y_test, y_pred_test_svm)

# Get classification report for detailed metrics
class_report_svm = classification_report(y_test, y_pred_test_svm)

# Display results
print(f"SVM Training Accuracy: {train_accuracy_svm:.4f}")
print(f"SVM Test Accuracy: {test_accuracy_svm:.4f}")
print(f"SVM Classification Report:\n{class_report_svm}")

# %% [markdown]
# # Saving the Models 

# %% [markdown]
# 

# %%
import joblib

# Save the SVM trained model
joblib.dump(svm_model, "../models/svm_model_linear.pkl")
print("✅ SVM model saved to models/svm_model_linear.pkl")

# Save the KMeans model
joblib.dump(kmeans, "../models/kmeans_model.pkl")
print("✅ KMeans model saved to models/kmeans_model.pkl")

# Save the KNN model
joblib.dump(knn_model, "../models/knn_model.pkl")
print("✅ KNN model saved to models/knn_model.pkl")

# Save the Random Forest model
joblib.dump(rf_model, "../models/random_forest_model.pkl")
print("✅ Random Forest model saved to models/random_forest_model.pkl")

# Save the Logistic Regression model
joblib.dump(logreg_model, "../models/logistic_regression_model.pkl")
print("✅ Logistic Regression model saved to models/logistic_regression_model.pkl")

# %% [markdown]
# # ✅ Code to Load All Saved Models

# %%
import os

# Define model paths
model_paths = {
    "svm": "../models/svm_model_linear.pkl",
    "kmeans": "../models/kmeans_model.pkl",
    "knn": "../models/knn_model.pkl",
    "random_forest": "../models/random_forest_model.pkl",
    "logistic_regression": "../models/logistic_regression_model.pkl"
}

# Load models
models = {}

for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"✅ Loaded: {name} model from {path}")
    else:
        print(f"❌ File not found for: {name} model at {path}")

# %%



