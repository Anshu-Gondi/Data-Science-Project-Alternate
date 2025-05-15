# %% [markdown]
# # Feature Engineering

# %% [markdown]
# ## 📌 Import Necessary Libraries for Feature Engineering
# 

# %%
# Suppress warnings  
import warnings
warnings.simplefilter("ignore")

# Data handling and processing  
import pandas as pd  
import numpy as np  
from scipy import stats  

# Data visualization  
import matplotlib.pyplot as plt  
import seaborn as sns  
sns.set()  

# Feature Scaling & Selection  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_val_score  

# %% [markdown]
# ### 📂 Loading the dataset

# %%
data = pd.read_csv("../data/processed/processed_data.csv")

data.head()

# %% [markdown]
# ### Standarization

# %%
from sklearn.preprocessing import StandardScaler
# List of numerical columns to scale
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
# Initialize the scaler
scaler = StandardScaler()
# Fit and transform the numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])
# Check the scaled data
print(data.head())

# %% [markdown]
# # Feature Selection

# %% [markdown]
# ## Chi-Square Method

# %%
from sklearn.feature_selection import SelectKBest, chi2
X = data.drop(columns=['cardio'])  # Features
y = data['cardio']  # Target variable

# Since chi-square works with non-negative values, ensure no negative values are in X
X_abs = X.abs()

# Apply chi-square feature selection
chi2_selector = SelectKBest(chi2, k='all') 
X_new = chi2_selector.fit_transform(X_abs, y)

# Get the scores of each feature
feature_scores = chi2_selector.scores_

# Create a DataFrame to view the feature scores
feature_scores_df = pd.DataFrame({
    'Feature': X.columns,
    'Chi-Square Score': feature_scores
})

# Sort features based on chi-square score
feature_scores_df = feature_scores_df.sort_values(by='Chi-Square Score', ascending=False)

feature_scores_df

# %%
# Drop the 'id' column
X = data.drop(columns=['id', 'cardio'])
y = data['cardio']

# %% [markdown]
# ## Transferring data for model Part

# %%
processed_data = data
# Save preprocessed data
processed_data.to_csv("../data/processed/processed_data.csv", index=False)

print("Preprocessed data saved successfully!")


