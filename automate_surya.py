import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan.")
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()
    print("Preprocessing data...")
    
    # 1. Handling Missing Values
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")

    # 2. Encoding
    ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
    ordinal_cols = ['Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = df[col].map(ordinal_map)
    
    binary_map = {'No': 0, 'Yes': 1}
    binary_cols = ['Smoking', 'Family Heart Disease', 'Diabetes', 'High Blood Pressure', 
                   'Low HDL Cholesterol', 'High LDL Cholesterol', 'Heart Disease Status']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # 3. Scaling
    scaler = StandardScaler()
    target_col = 'Heart Disease Status'
    feature_cols = [c for c in df.columns if c != target_col]
    
    if feature_cols:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
    return df

def split_and_save(df, output_dir):
    print("Splitting data...")
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to raw dataset')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    raw_df = load_data(args.input)
    clean_df = preprocess_data(raw_df)
    split_and_save(clean_df, args.output)