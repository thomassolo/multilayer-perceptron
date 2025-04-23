import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read the CSV without headers and assign column names
# Assuming 2nd column is diagnosis and 1st is ID
column_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(30)]
df = pd.read_csv("data.csv", header=None, names=column_names)

print("Aperçu des données :")
print("\nNoms des colonnes assignées:")
print(df.columns.tolist())

# Remove ID column
df.drop(columns=['id'], inplace=True)

# Convert diagnosis column: B -> 0, M -> 1
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Split features and target
y = df['diagnosis']
X = df.drop(columns=['diagnosis'])

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create cleaned dataframe
df_cleaned = pd.DataFrame(X_scaled, columns=X.columns)
df_cleaned.insert(0, 'diagnosis', y)

# Split into train and validation sets
train_df, valid_df = train_test_split(
    df_cleaned, test_size=0.2, random_state=42
)

# Save datasets
train_df.to_csv("data_train.csv", index=False)
valid_df.to_csv("data_valid.csv", index=False)

print("✅ Données préparées et sauvegardées :")
print(" - data_train.csv")
print(" - data_valid.csv")
