# src/feature_engineering/feature_engineering.py

def add_features(df):
    """
    Fügt zeitabhängige Features hinzu, wie Jahr, Monat, Tag, Wochentag und Saison.
    """
    df['Year'] = df['Datum von'].dt.year
    df['Month'] = df['Datum von'].dt.month
    df['Day'] = df['Datum von'].dt.day
    df['Weekday'] = df['Datum von'].dt.weekday
    df['Is_Weekend'] = df['Weekday'] >= 5
    df['Season'] = df['Month'].apply(lambda x: (x % 12 + 3) // 3)  # Jahreszeiten
    return df
