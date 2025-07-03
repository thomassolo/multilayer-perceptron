import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_raw_data(filepath="data.csv", num_features=30):
    """Load raw data and assign appropriate column names."""
    column_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(num_features)]
    df = pd.read_csv(filepath, header=None, names=column_names)
    
    print("Raw data overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def clean_data(df):
    """Clean the data by removing ID column and encoding diagnosis."""
    # Remove ID column
    df_cleaned = df.drop(columns=['id']).copy()
    
    # Convert diagnosis column: B -> 0 (Benign), M -> 1 (Malignant)
    df_cleaned['diagnosis'] = df_cleaned['diagnosis'].map({'B': 0, 'M': 1})
    
    # Check for any missing values after mapping
    if df_cleaned['diagnosis'].isna().sum() > 0:
        print("Warning: Some diagnosis values could not be mapped!")
    
    print(f"Diagnosis distribution:")
    print(f"Benign (0): {(df_cleaned['diagnosis'] == 0).sum()}")
    print(f"Malignant (1): {(df_cleaned['diagnosis'] == 1).sum()}")
    
    return df_cleaned


def scale_features(df):
    """Scale features using MinMaxScaler and return scaled dataframe."""
    # Split features and target
    y = df['diagnosis']
    X = df.drop(columns=['diagnosis'])
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create scaled dataframe
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled.insert(0, 'diagnosis', y)
    
    print(f"Features scaled using MinMaxScaler")
    print(f"Feature range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    
    return df_scaled, scaler


def split_dataset(df, test_size=0.2, random_state=42):
    """Split dataset into training and validation sets."""
    train_df, valid_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['diagnosis']
    )
    
    print(f"Dataset split:")
    print(f"Training set: {train_df.shape[0]} samples")
    print(f"Validation set: {valid_df.shape[0]} samples")
    print(f"Test size: {test_size * 100}%")
    
    # Show class distribution in both sets
    print(f"\nTraining set distribution:")
    print(f"Benign (0): {(train_df['diagnosis'] == 0).sum()}")
    print(f"Malignant (1): {(train_df['diagnosis'] == 1).sum()}")
    
    print(f"\nValidation set distribution:")
    print(f"Benign (0): {(valid_df['diagnosis'] == 0).sum()}")
    print(f"Malignant (1): {(valid_df['diagnosis'] == 1).sum()}")
    
    return train_df, valid_df


def save_datasets(train_df, valid_df, train_file="data_train.csv", valid_file="data_valid.csv"):
    """Save training and validation datasets to CSV files."""
    train_df.to_csv(train_file, index=False)
    valid_df.to_csv(valid_file, index=False)
    
    print(f"\n✅ Datasets saved successfully:")
    print(f" - {train_file}")
    print(f" - {valid_file}")


def process_data(input_file="data.csv", num_features=30, test_size=0.2, random_state=42):
    """Main function to process the entire data pipeline."""
    print("Starting data processing pipeline...")
    
    # Load raw data
    df = load_raw_data(input_file, num_features)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Scale features
    df_scaled, scaler = scale_features(df_cleaned)
    
    # Split dataset
    train_df, valid_df = split_dataset(df_scaled, test_size, random_state)
    
    # Save datasets
    save_datasets(train_df, valid_df)
    
    return train_df, valid_df, scaler


if __name__ == "__main__":
    # Run the complete data processing pipeline
    train_df, valid_df, scaler = process_data()
