import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load your data (update the path as needed)
df = pd.read_csv('data/interim/cleaned_data.csv')

# ----- LABEL ENCODING -----
le = LabelEncoder()
label_cols = ['Furnishing', 'Locality', 'Status', 'Transaction', 'Type']

for col in label_cols:
    df[col] = le.fit_transform(df[col])
    print(f"{col}: {df[col].unique()}")

# ----- NORMALIZATION -----
scaler = MinMaxScaler()
scale_cols = ['Area', 'Price', 'Per_Sqft', 'Area_yards']

df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Save the processed data
df.to_csv('data/processed/final_data.csv', index=False)
print("Feature engineering completed and saved to data/processed/final_data.csv")
