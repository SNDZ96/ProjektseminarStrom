import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Grundlegende Einstellungen für die Streamlit-App
st.set_page_config(page_title="Stromprognose", layout="wide", initial_sidebar_state="expanded")
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="deep")

# Datei-Pfade für die verwendeten Datensätze 
file_paths = {
    "Prognostizierte_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_Finale/DATA/Prognostizierte_Erzeugung_Day-Ahead_201701010000_202301010000_Stunde.csv",
    "Realisierter_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_Finale/DATA/Realisierter_Stromverbrauch_201701010000_202301010000_Stunde.csv",
    "Sonnenscheindauer": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_Finale/DATA/Sonnenscheindauer_Deutschland_neu.csv",
    "Feiertage": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_Finale/DATA/FeiertageDE.csv",
    "Holidays": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_Finale/DATA/bevölkerung_DE.csv"
}

# Funktion zur Datenbereinigung und Umwandlung von Zeitstempeln
# Hier sorge ich dafür, dass die Zeitangaben und Zahlen richtig Formatiert sind
def preprocess_data(df, is_realized=False):
    df['Datum'] = pd.to_datetime(df['Datum von'], format='%d.%m.%Y %H:%M', errors='coerce')
    if 'Datum bis' in df.columns:
        df.drop(columns=['Datum bis'], inplace=True)

    for col in df.columns:
        if 'MWh' in col:
            df[col] = (
                df[col].str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .replace('-', np.nan)
                .astype(float)
            )

    if is_realized:
        df.rename(columns={'Gesamt (Netzlast) [MWh] Berechnete Auflösungen': 'Netzlast'}, inplace=True)
    else:
        df.rename(columns={'Gesamt [MWh] Originalauflösungen': 'Prognose'}, inplace=True)

    return df

# Einlesen der Feiertage
feiertage_data = pd.read_csv(file_paths['Feiertage'], sep=';', decimal=',')
feiertage_data['Feiertag'] = pd.to_datetime(feiertage_data['Feiertag'], format='%d.%m.%Y', errors='coerce')

# Einlesen der Bevölkerungsdaten
bv_data = pd.read_csv(file_paths['Holidays'], sep=';', decimal=',')
bv_data['Datum'] = pd.to_datetime(bv_data['Datum'], format='%Y-%m-%d', errors='coerce')

# Überprüfen, ob die Bevölkerungsdaten korrekt benannt sind
print(bv_data.columns)

# Umwandlung der Bevölkerungswerte in numerisches Format
bv_data['BV'] = bv_data['Bevölkerung'].str.replace('.', '', regex=False).astype(float)
bv_data['Jahr'] = bv_data['Datum'].dt.year
bv_data = bv_data[['Jahr', 'BV']]

# Sonnenscheindauer-Daten einlesen
sunshine_data = pd.read_csv(file_paths['Sonnenscheindauer'], sep=';', decimal=',')
sunshine_data['Datum'] = pd.to_datetime(sunshine_data['Datum'], format='%Y-%m-%d', errors='coerce')
sunshine_data['Datum von'] = sunshine_data['Datum']
sunshine_data['Sunshine_Minutes'] = sunshine_data['Arithmetisches Mittel\nSonnenschein in Minuten']
sunshine_daily = sunshine_data.groupby('Datum von', as_index=False)['Sunshine_Minutes'].sum()
sunshine_daily['Sunshine_Hours'] = sunshine_daily['Sunshine_Minutes'] / 60

# Laden und Bereinigen der anderen Daten
prognostizierte_stunde = preprocess_data(pd.read_csv(file_paths['Prognostizierte_Stunde'], sep=';', decimal=','))
realisierter_stunde = preprocess_data(pd.read_csv(file_paths['Realisierter_Stunde'], sep=';', decimal=','), is_realized=True)

# Zusammenführen der Daten für das Modell
combined_data = pd.merge(
    realisierter_stunde[['Datum', 'Netzlast']],
    prognostizierte_stunde[['Datum', 'Prognose', 'Photovoltaik und Wind [MWh] Berechnete Auflösungen',
                            'Wind Offshore [MWh] Berechnete Auflösungen', 'Wind Onshore [MWh] Berechnete Auflösungen',
                            'Photovoltaik [MWh] Berechnete Auflösungen']],
    on='Datum',
    how='inner'
)

# Umbenennen für mehr Klarheit im modell 
combined_data.rename(columns={
    'Photovoltaik und Wind [MWh] Berechnete Auflösungen': 'PV_Wind',
    'Wind Offshore [MWh] Berechnete Auflösungen': 'Wind_Offshore',
    'Wind Onshore [MWh] Berechnete Auflösungen': 'Wind_Onshore',
    'Photovoltaik [MWh] Berechnete Auflösungen': 'Photovoltaik'
}, inplace=True)

# Einfügen der Sonnenscheindauer-Daten
combined_data = pd.merge(
    combined_data,
    sunshine_daily[['Datum von', 'Sunshine_Hours']],
    left_on='Datum',
    right_on='Datum von',
    how='left'
)

# Feature: Feiertage einfügen 
combined_data['Is_Holiday'] = combined_data['Datum'].isin(feiertage_data['Feiertag'])

# Hinzufügen von Zeit-Features für das Modell
def add_features(df):
    df['Year'] = df['Datum'].dt.year
    df['Month'] = df['Datum'].dt.month
    df['Day'] = df['Datum'].dt.day
    df['Weekday'] = df['Datum'].dt.weekday
    df['Is_Weekend'] = df['Weekday'] >= 5
    df['Season'] = df['Month'].apply(lambda x: (x % 12 + 3) // 3)  # Jahreszeiten berechnen
    return df

combined_data = add_features(combined_data)
combined_data.dropna(inplace=True)

# Bevölkerungswachstumsdaten hinzufügen
combined_data = pd.merge(combined_data, bv_data, left_on='Year', right_on='Jahr', how='left')
combined_data['Population_Growth'] = combined_data['BV']

# Features für das Modell definieren
features = ['Prognose', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik', 
            'Sunshine_Hours', 'Month', 'Is_Weekend', 'Season', 'Is_Holiday', 'Population_Growth']
target = 'Netzlast'

# Standardisierung der Daten
scaler = StandardScaler()
X = combined_data[features]
y = combined_data[target]
X_scaled = scaler.fit_transform(X)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modell initialisieren und trainieren
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Vorhersagen für das Testset
xgb_pred = xgb_model.predict(X_test)

# Fehlerbewertung
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred)

# Vorhersagen für das gesamte Model
combined_data['Prediction_XGB'] = xgb_model.predict(scaler.transform(combined_data[features]))

# Dynamische Prognose für das Jahr 2023
def forecast_2023_dynamic(model, scaler, features, historical_data, start_date, days=365):
    forecast_dates = pd.date_range(start=start_date, periods=days)
    forecast_df = pd.DataFrame({'Datum von': forecast_dates})

    # Zeitabhängige Features
    forecast_df['Year'] = forecast_df['Datum von'].dt.year
    forecast_df['Month'] = forecast_df['Datum von'].dt.month
    forecast_df['Day'] = forecast_df['Datum von'].dt.day
    forecast_df['Weekday'] = forecast_df['Datum von'].dt.weekday
    forecast_df['Is_Weekend'] = forecast_df['Weekday'] >= 5
    forecast_df['Season'] = forecast_df['Month'].apply(lambda x: (x % 12 + 3) // 3)

    # Historische Mittelwerte für dynamische Features
    monthly_means = historical_data.groupby('Month').mean()

    for feature in ['Prognose', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik', 'Sunshine_Hours', 'Population_Growth']:
        forecast_df[feature] = forecast_df['Month'].map(monthly_means[feature])

    forecast_df['Is_Holiday'] = forecast_df['Month'].isin(feiertage_data['Feiertag'].dt.month)

    # Skalierung und Vorhersage
    forecast_features = scaler.transform(forecast_df[features])
    forecast_df['Prediction_XGB'] = model.predict(forecast_features)

    return forecast_df

forecast_2023_df = forecast_2023_dynamic(xgb_model, scaler, features, combined_data, start_date="2023-01-01")

# Streamlit App und seine wichtigen aplikationen 
st.sidebar.title("Navigation")
pages = ["Vergangene Jahre", "Prognose 2023"] + [f"2023 - Monat {i}" for i in range(1, 13)]
page = st.sidebar.selectbox("Wählen Sie eine Seite", pages)

if page == "Vergangene Jahre":
    st.title("XGBoost: Vergangene Jahre")
    st.write(f"**MSE:** {xgb_mse:.2f}")
    st.write(f"**R²-Wert:** {xgb_r2:.2f}")
    st.write(f"**MAPE:** {xgb_mape:.2f}%")

    for year in combined_data['Year'].unique():
        yearly_data = combined_data[combined_data['Year'] == year]
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(yearly_data['Datum'], yearly_data['Netzlast'], label='Tatsächliche Netzlast', color='#1f77b4')
        ax.plot(yearly_data['Datum'], yearly_data['Prediction_XGB'], label='Vorhergesagte Netzlast', color='#ff7f0e')
        ax.set_xlabel('Datum')
        ax.set_ylabel('Netzlast (MWh)')
        ax.set_title(f'Tägliche Netzlast: Tatsächlich vs. Vorhergesagt für {year}')
        ax.legend()
        st.pyplot(fig)

elif page == "Prognose 2023":
    st.title("Dynamische Prognose für 2023")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(forecast_2023_df['Datum von'], 
                 forecast_2023_df['PV_Wind'], 
                 forecast_2023_df['Prediction_XGB'] - forecast_2023_df['PV_Wind'], 
                 labels=['Erneuerbare Energien', 'Restliche Netzlast'], 
                 colors=['#76c7c0', '#ffcc00'])
    ax.set_xlabel('Datum')
    ax.set_ylabel('Netzlast (MWh)')
    ax.set_title('Prognose der Netzlast für 2023 (gestapelte Darstellung)')
    ax.legend()
    st.pyplot(fig)

else:
    month = int(page.split()[-1])
    st.title(f"Prognose für Monat {month} im Jahr 2023")
    monthly_data = forecast_2023_df[forecast_2023_df['Datum von'].dt.month == month]
    st.write("### Tabellarische Daten")
    st.write(monthly_data[['Datum von', 'Prediction_XGB', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik']])

    # Kreisdiagramm für die Monate der Vorhersage
    total_energy = monthly_data['Prediction_XGB'].sum()
    renewable_energy = monthly_data['PV_Wind'].sum()
    other_energy = total_energy - renewable_energy

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        [renewable_energy, other_energy],
        labels=['Erneuerbare Energien', 'Andere Netzlast'],
        autopct='%1.1f%%',
        colors=['#76c7c0', '#ffcc00'],
        startangle=90
    )
    ax.set_title(f'Anteile der Netzlast für Monat {month} (2023)')
    st.pyplot(fig)
