# %% [markdown]
# # Data Preprocessing :

# %% [markdown]
# ## Importing Required Libraries

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

# Data preprocessing

from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_val_score


# %% [markdown]
# ## Load Dataset

# %%
data = pd.read_csv("../data/raw/cardio_train.csv", sep=";")

# %% [markdown]
# ### Data Cleaning

# %% [markdown]
# ### Age Column
# Issue: The age was recorded in days, leading to unusually high mean values.
# 
# Solution: Converted age from days to years by dividing the values by 365.
# 
# Technique Used: Unit transformation.

# %%
data['age'] = data['age'] // 365

# %% [markdown]
# ### Blood Pressure (ap_hi, ap_lo)
# Issue: Extreme and erroneous values (e.g., negative values for blood pressure). Solution: Removed rows with systolic blood pressure (ap_hi) outside 90–200 mmHg and diastolic blood pressure (ap_lo) outside 60–120 mmHg.
# 
# Technique Used: Threshold-based filtering which is commonly used to handle outliers, eliminate noise, or select relevant features.

# %%
data = data[(data['ap_hi'] >= 90) & (data['ap_hi'] <= 200)]
data = data[(data['ap_lo'] >= 60) & (data['ap_lo'] <= 120)]

# %% [markdown]
# ### Cholesterol and Glucose (gluc)
# Issue: Potential invalid entries; these columns should only contain categorical values (1, 2, 3).
# 
# Solution: Filtered rows where values in these columns were outside the valid range.
# 
# Technique Used: Value verification.

# %%
valid_categories = [1, 2, 3]
data = data[data['cholesterol'].isin(valid_categories)]
data = data[data['gluc'].isin(valid_categories)]

# %% [markdown]
# ### Binary Variables (smoke, alco, active)
# Issue: Binary variables should only contain 0 and 1; potential for incorrect values.
# 
# Solution: Verified and removed rows where these variables had values other than 0 or 1.
# 
# Technique Used: Value validation for binary features.

# %%
binary_columns = ['smoke', 'alco', 'active']
for col in binary_columns:
    data = data[data[col].isin([0, 1])]

# %% [markdown]
# ### Rechecking After Cleaning

# %%
data.describe()

# %%
data.isnull().sum()

# %%
# Boxplot for cleaned numerical features
data[['age', 'ap_hi', 'ap_lo']].boxplot(figsize=(10, 6))
plt.title("Boxplot of Cleaned Numerical Features")
plt.show()

# %% [markdown]
# ### transfer the data to proprocess one

# %%
processed_data = data

# %%
# Save preprocessed data
processed_data.to_csv("../data/processed/processed_data.csv", index=False)

print("Preprocessed data saved successfully!")

# %%
print(processed_data.head())  # Check if data exists


