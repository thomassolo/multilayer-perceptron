import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data.csv")

print("Aperçu des données :")
print(df.head())
print("\nNoms des colonnes disponibles:")
print(df.columns.tolist())

diagnosis_column = None
for possible_name in ['diagnosis', 'label', 'target', 'class', 'result']:
    if possible_name in df.columns:
        diagnosis_column = possible_name
        break

if diagnosis_column:
    print(f"Colonne de diagnostic trouvée: '{diagnosis_column}'")
    
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # 4. Convertir la colonne de diagnostic en 0 (Bénin) et 1 (Malin)
    df[diagnosis_column] = df[diagnosis_column].map({'B': 0, 'M': 1})

    y = df[diagnosis_column]
    X = df.drop(columns=[diagnosis_column])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    df_cleaned = pd.DataFrame(X_scaled, columns=X.columns)
    df_cleaned.insert(0, 'diagnosis', y)

    train_df, valid_df = train_test_split(
        df_cleaned, test_size=0.2, random_state=42
    )

    train_df.to_csv("data_train.csv", index=False)
    valid_df.to_csv("data_valid.csv", index=False)

    print("✅ Données préparées et sauvegardées :")
    print(" - data_train.csv")
    print(" - data_valid.csv")
else:
    print("⚠️ Aucune colonne de diagnostic trouvée!")
    print("Colonnes disponibles:", df.columns.tolist())
    print("Veuillez vérifier le fichier data.csv et identifier manuellement la colonne représentant le diagnostic.")
