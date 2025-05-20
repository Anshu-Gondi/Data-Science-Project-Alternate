import pandas as pd
import numpy as np
import os
from scipy import stats


def load_raw_data(path):
    return pd.read_csv(path)


def grp_local(locality):
    locality = locality.lower()  # avoid case sensitive
    if 'rohini' in locality:
        return 'Rohini Sector'
    elif 'dwarka' in locality:
        return 'Dwarka Sector'
    elif 'shahdara' in locality:
        return 'Shahdara'
    elif 'vasant' in locality:
        return 'Vasant kunj'
    elif 'paschim' in locality:
        return 'Paschim Vihar'
    elif 'alaknada' in locality:
        return 'Alaknanda'
    elif 'vasundhar' in locality:
        return 'Vasundhara Enclave'
    elif 'punjabi' in locality:
        return 'Punjabi Bagh'
    elif 'kalkaji' in locality:
        return 'KalKaji'
    elif 'lajpat' in locality:
        return 'Lajpat Nagar'
    else:
        return 'Others'


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    df['Price'] = df['Price'].replace(r'[^\d.]', '', regex=True).astype(float)
    df['Parking'] = df['Parking'].astype('int64')
    df['Bathroom'] = df['Bathroom'].astype('int64')
    return df


def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")


if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "MagicBricks.csv")
    processed_data_path = os.path.join("data", "interim", "cleaned_data.csv")

    # Load the raw data
    df = load_raw_data(raw_data_path)

    print("First few rows of the raw data:")
    print(df.head())

    print("\nStatistical summary of the numeric columns:")
    print(df.describe())

    print("\nCount of null values for each column:")
    print(df.isnull().sum())

    # Step 1: Clean data
    df_clean = clean_data(df)

    # Step 2: Remove outliers using Z-score
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
    df_clean = df_clean[(z_scores < 3).all(axis=1)]

    print("\nCount of null values after cleaning & outlier removal:")
    print(df_clean.isnull().sum())

    print("\nData types of each column:")
    print(df_clean.dtypes)

    print("\nCount of unique values after cleaning:")
    print(df_clean.nunique())

    print("\nCount of each unique value in each column after cleaning:")
    for column in df_clean.columns:
        print(f"\n{column} value counts:")
        print(df_clean[column].value_counts())

    # Step 3: Group the Locality
    df_clean['Locality'] = df_clean['Locality'].apply(grp_local)

    # Step 4: Add Area_yards column
    df_clean['Area_yards'] = df_clean['Area'] / 9

    # Step 5: Grouping insights
    unique_counts_by_locality = df_clean.groupby('Locality').nunique()
    print("\nUnique value counts by Locality:")
    print(unique_counts_by_locality)

    print("\nNo of values in each grouped locality:")
    print(df_clean['Locality'].value_counts())
    
    # Step 6: Checking descriptive satistics of the data
    print("\n📊 Final summary after outlier removal:")
    print(df_clean.describe())
    
    # Step 7: Save processed data
    save_processed_data(df_clean, processed_data_path)
