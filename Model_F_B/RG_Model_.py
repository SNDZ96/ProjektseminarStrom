import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Streamlit-Einstellungen
st.set_page_config(page_title="Stromprognose", layout="wide", initial_sidebar_state="expanded")

# Dummy-Datenpfade
file_paths = {
    "Prognostizierte_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Prognostizierte_Erzeugung_Day-Ahead_201701010000_202301010000_Stunde.csv",
    "Realisierter_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Realisierter_Stromverbrauch_201701010000_202301010000_Stunde.csv",
    "Sonnenscheindauer": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Sonnenscheindauer_Deutschland_neu.csv",
}

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

# Lade und bereinige die Daten
prognostizierte_stunde = preprocess_data(pd.read_csv(file_paths['Prognostizierte_Stunde'], sep=';', decimal=','))
realisierter_stunde = preprocess_data(pd.read_csv(file_paths['Realisierter_Stunde'], sep=';', decimal=','), is_realized=True)

# Kombinieren der Daten
combined_data = pd.merge(
    realisierter_stunde[['Datum', 'Netzlast']],
    prognostizierte_stunde[['Datum', 'Prognose', 'Photovoltaik und Wind [MWh] Berechnete Auflösungen',
                            'Wind Offshore [MWh] Berechnete Auflösungen', 'Wind Onshore [MWh] Berechnete Auflösungen',
                            'Photovoltaik [MWh] Berechnete Auflösungen']],
    on='Datum',
    how='inner'
)

combined_data.rename(columns={
    'Photovoltaik und Wind [MWh] Berechnete Auflösungen': 'PV_Wind',
    'Wind Offshore [MWh] Berechnete Auflösungen': 'Wind_Offshore',
    'Wind Onshore [MWh] Berechnete Auflösungen': 'Wind_Onshore',
    'Photovoltaik [MWh] Berechnete Auflösungen': 'Photovoltaik'
}, inplace=True)

# Feature Engineering
def add_features(df):
    df['Year'] = df['Datum'].dt.year
    df['Month'] = df['Datum'].dt.month
    df['Day'] = df['Datum'].dt.day
    df['Weekday'] = df['Datum'].dt.weekday
    df['Is_Weekend'] = df['Weekday'] >= 5
    df['Season'] = df['Month'].apply(lambda x: (x % 12 + 3) // 3)  # Jahreszeiten
    return df

combined_data = add_features(combined_data)
combined_data.dropna(inplace=True)

# Feature und Ziel definieren
scaler = StandardScaler()
features = ['Prognose', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik', 'Season', 'Is_Weekend', 'Month']
X = combined_data[features]
y = combined_data['Netzlast']
X_scaled = scaler.fit_transform(X)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelle erstellen
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
}

# Metriken
metrics = {}

# Modelle trainieren und evaluieren
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics[model_name] = {'MSE': mse, 'R2': r2, 'MAPE': mape}

# Anzeige der Modellmetriken
st.sidebar.title("Modell-Auswahl")
pages = list(models.keys())
page = st.sidebar.selectbox("Wählen Sie ein Modell:", pages)

# Die Metriken des ausgewählten Modells anzeigen
if page:
    st.write(f"### {page} Modellmetriken")
    st.write(f"**MSE:** {metrics[page]['MSE']:.2f}")
    st.write(f"**R²:** {metrics[page]['R2']:.2f}")
    st.write(f"**MAPE:** {metrics[page]['MAPE']:.2f}%")
