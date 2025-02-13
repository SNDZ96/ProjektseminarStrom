import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Funktion: Datenbereinigung und Feature Engineering
def preprocess_data(df, is_realized=False):
    df['Datum'] = pd.to_datetime(df['Datum von'], format='%d.%m.%Y %H:%M', errors='coerce')
    if 'Datum bis' in df.columns:
        df.drop(columns=['Datum bis'], inplace=True)

    for col in df.columns:
        if 'MWh' in col:
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).replace('-', np.nan).astype(float)

    if is_realized:
        df.rename(columns={'Gesamt (Netzlast) [MWh] Berechnete Auflösungen': 'Netzlast'}, inplace=True)
    else:
        df.rename(columns={'Gesamt [MWh] Originalauflösungen': 'Prognose'}, inplace=True)

    return df

def create_model(file_paths):
    # Lade die Daten
    sunshine_data = pd.read_csv(file_paths['Sonnenscheindauer'], sep=';', decimal=',')
    sunshine_data['Datum'] = pd.to_datetime(sunshine_data['Datum'], format='%Y-%m-%d', errors='coerce')
    sunshine_data['Sunshine_Minutes'] = sunshine_data['Arithmetisches Mittel\nSonnenschein in Minuten']
    sunshine_daily = sunshine_data.groupby('Datum', as_index=False)['Sunshine_Minutes'].sum()
    sunshine_daily['Sunshine_Hours'] = sunshine_daily['Sunshine_Minutes'] / 60

    # Feiertage-Daten laden
    holidays_data = pd.read_csv(file_paths['Feiertage'], sep=';', decimal=',')
    holidays_data['Feiertag'] = pd.to_datetime(holidays_data['Feiertag'], format='%d.%m.%Y', errors='coerce')

    # Laden der restlichen Daten
    prognostizierte_stunde = preprocess_data(pd.read_csv(file_paths['Prognostizierte_Stunde'], sep=';', decimal=','))
    realisierte_stunde = preprocess_data(pd.read_csv(file_paths['Realisierter_Stunde'], sep=';', decimal=','), is_realized=True)

    # Kombinieren der Daten
    combined_data = pd.merge(realisierter_stunde[['Datum', 'Netzlast']],
                             prognostizierte_stunde[['Datum', 'Prognose', 'Photovoltaik und Wind [MWh] Berechnete Auflösungen',
                                                     'Wind Offshore [MWh] Berechnete Auflösungen', 'Wind Onshore [MWh] Berechnete Auflösungen',
                                                     'Photovoltaik [MWh] Berechnete Auflösungen']],
                             on='Datum', how='inner')

    combined_data.rename(columns={'Photovoltaik und Wind [MWh] Berechnete Auflösungen': 'PV_Wind',
                                  'Wind Offshore [MWh] Berechnete Auflösungen': 'Wind_Offshore',
                                  'Wind Onshore [MWh] Berechnete Auflösungen': 'Wind_Onshore',
                                  'Photovoltaik [MWh] Berechnete Auflösungen': 'Photovoltaik'}, inplace=True)

    combined_data = pd.merge(combined_data, sunshine_daily[['Datum', 'Sunshine_Hours']], left_on='Datum', right_on='Datum', how='left')
    combined_data.dropna(inplace=True)

    # Features definieren
    features = ['Prognose', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik', 'Sunshine_Hours']
    X = combined_data[features]
    y = combined_data['Netzlast']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modell erstellen
    xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_scaled, y)
    
    return xgb_model, X_scaled, y
