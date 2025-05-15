# %% [markdown]
# # Cardivascular Disease Predication
# 
# ## Exploratory Data Analysis (EDA)
# 
# For Exploratory Data Analysis (EDA), we first imported the necessary libraries.
# 
# The main library used for handling data is **`pandas`**, which allows us to create and manipulate dataframes.
# 
# For data visualization, we use **`matplotlib`** and **`seaborn`** to generate informative plots. 
# 
# We also use **`scipy`** and **`numpy`** for statistical analysis and numerical computations.
# 
# ### Importing Required Libraries

# %%
import warnings
warnings.simplefilter("ignore")

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from scipy import stats
import numpy as np

# %% [markdown]
# ### •Initial Data Inspection:

# %%
data = pd.read_csv("../data/raw/cardio_train.csv", sep=";")
data.head(10)

# %% [markdown]
# ### Check data types

# %%
print(data.dtypes)

# %% [markdown]
# ### Null Values

# %%
print(data.isnull().sum())

# %% [markdown]
# ### Check Duplicates

# %%
print(data.duplicated().sum())

# %% [markdown]
# ### Dimensionality

# %%
print(data.shape)

# %% [markdown]
# #### • Descriptive Statistics:

# %%
data.describe()

# %% [markdown]
# **The issues should be addressed to clean the data before proceeding with machine learning models**
# 
# Age Column: The mean age value is unusually high (19468.87), indicating that the age might be in hundredths of a year. This column needs to be converted to actual years.
# 
# Blood Pressure (ap_hi, ap_lo): There are extreme and likely erroneous values, such as -150 for ap_hi and -70 for ap_lo. These need to be corrected or removed.
# 
# Binary Variables (smoke, alco, active): Ensure these binary variables only contain values 0 and 1, as there might be incorrect or inconsistent entries

# %% [markdown]
# ### Skewness

# %%
# Plot histograms for each numerical feature
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 12))

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 4, i)
    sns.histplot(data[feature], kde=True)
    plt.title(f'{feature} - Histogram')

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Interpretation
# Columns with high variance (e.g., id, age, ap_hi, ap_lo) may have a wide spread in values, which could indicate useful features for predictive modeling, but some columns like id should be discarded as they do not hold predictive power.
# 
# Columns with low variance (e.g., gender, smoke, alco, cardio) indicate less variability in the data, which might mean they don’t contribute much in terms of differentiating between observations, especially if they are binary features.

# %% [markdown]
# ### Outliers Detection :

# %% [markdown]
# #### BoxPlot

# %%
sns.boxplot(data=data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']])
plt.show()

# %% [markdown]
# ### IQR Method :

# %%
# Calculate IQR for numerical features
Q1 = data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']].quantile(0.25)
Q3 = data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']].quantile(0.75)
IQR = Q3 - Q1

# Detect outliers
outliers_iqr = ((data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']] < (Q1 - 1.5 * IQR)) |
                (data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']] > (Q3 + 1.5 * IQR)))

print(outliers_iqr)

# %% [markdown]
# ### 📊 Visualization Summary

# %%
# Calculate the correlation matrix
correlation_matrix = data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', linewidth=0.5, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# **From this correlation matrix, we can understand that all data is unique and not strongly correlated with each other.**
# 
# It’s also worth noting that some attributes are highly correlated with the target

# %% [markdown]
# #### Histograms for numerical features

# %%
# Plot histograms for numerical features
data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']].hist(bins=20, figsize=(12, 8))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# %% [markdown]
# #### Scatterplots:

# %%
# Scatterplot between age and weight
sns.scatterplot(x='age', y='weight', data=data)
plt.title("Scatterplot: Age vs Weight")
plt.show()

# Scatterplot between ap_hi and ap_lo
sns.scatterplot(x='ap_hi', y='ap_lo', data=data)
plt.title("Scatterplot: ap_hi vs ap_lo")
plt.show()

# %% [markdown]
# #### Pairplot

# %%
# Pairplot for numerical features
sns.pairplot(data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']])
plt.suptitle("Pairplot of Numerical Features", y=1.02)
plt.show()

# %% [markdown]
# #### Countplot

# %%
# Countplot for gender
sns.countplot(x='gender', data=data)
plt.title("Countplot: Gender")
plt.show()

# Countplot for cholesterol
sns.countplot(x='cholesterol', data=data)
plt.title("Countplot: Cholesterol")
plt.show()

# Countplot for cardio (target variable)
sns.countplot(x='cardio', data=data)
plt.title("Countplot: Cardio")
plt.show()

# %% [markdown]
# #### Barcharts

# %%
# Bar chart for relationship between cardio and smoking
sns.barplot(x='smoke', y='cardio', data=data)
plt.title("Bar Chart: Cardio vs Smoking")
plt.show()

# Bar chart for relationship between cardio and activity level
sns.barplot(x='active', y='cardio', data=data)
plt.title("Bar Chart: Cardio vs Activity")
plt.show()

# %% [markdown]
# ### 📊 **Summary of Visualizations**  
# 
# 🔹 **Correlation Heatmap** – Examines relationships between numerical variables.  
# 
# 🔹 **Scatterplots** – Explores relationships between pairs of numerical features.  
# 
# 🔹 **Pairplot** – Provides insights into multiple feature relationships simultaneously.  
# 
# 🔹 **Count Plot** – Displays the frequency distribution of categorical features.  
# 
# 🔹 **Bar Charts** – Analyzes relationships between categorical variables and the target variable.  
# 
# 💡 *These visualizations help in understanding patterns, correlations, and feature importance for better model performance!* 🚀


