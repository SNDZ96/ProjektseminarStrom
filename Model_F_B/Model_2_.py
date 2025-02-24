
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit-Einstellungen
st.set_page_config(page_title="Stromprognose", layout="wide", initial_sidebar_state="expanded")
# Entferne dunkles Hintergrunddesign
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="deep")

# Dummy-Datenpfade (ersetze sie mit deinen echten Pfaden)
file_paths = {
    "Prognostizierte_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Prognostizierte_Erzeugung_Day-Ahead_201701010000_202301010000_Stunde.csv",
    "Realisierter_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Realisierter_Stromverbrauch_201701010000_202301010000_Stunde.csv",
    "Sonnenscheindauer": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Sonnenscheindauer_Deutschland_neu.csv",
    "Feiertage": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/DeutscheFE.csv",
    "Holidays": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/BV17-22.csv"
}

# Funktion: Datenbereinigung und Feature Engineering
def preprocess_data(df, is_realized=False):
    """
    Cleans and preprocesses the input data.

    Parameters:
        df (DataFrame): The input data to be processed.
        is_realized (bool): Flag to indicate if the data represents realized values. Default is False.

    Returns:
        DataFrame: The cleaned and preprocessed data.
    """
    # Verwende 'Datum von' anstelle von 'Datum'
    df['Datum'] = pd.to_datetime(df['Datum von'], format='%d.%m.%Y %H:%M', errors='coerce')
    if 'Datum bis' in df.columns:
        df.drop(columns=['Datum bis'], inplace=True)

    # MWh-Werte bereinigen
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

# Sonnenscheindauer-Daten laden
sunshine_data = pd.read_csv(file_paths['Sonnenscheindauer'], sep=';', decimal=',')
sunshine_data['Datum'] = pd.to_datetime(sunshine_data['Datum'], format='%Y-%m-%d', errors='coerce')
sunshine_data['Datum von'] = sunshine_data['Datum']
sunshine_data['Sunshine_Minutes'] = sunshine_data['Arithmetisches Mittel\nSonnenschein in Minuten']
sunshine_daily = sunshine_data.groupby('Datum von', as_index=False)['Sunshine_Minutes'].sum()
sunshine_daily['Sunshine_Hours'] = sunshine_daily['Sunshine_Minutes'] / 60

# Feiertage-Daten laden und bereinigen
holidays_data = pd.read_csv(file_paths['Feiertage'], sep=';', decimal=',')
holidays_data['Feiertag'] = pd.to_datetime(holidays_data['Feiertag'], format='%d.%m.%Y', errors='coerce')

# Feiertage-Daten aus BV17-22 laden und bereinigen
bv_data = pd.read_csv(file_paths['Holidays'], sep=';', decimal=',')
bv_data['Datum'] = pd.to_datetime(bv_data['Datum'], format='%Y-%m-%d', errors='coerce')

# Laden und Bereinigen der anderen Daten
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

combined_data = pd.merge(
    combined_data,
    sunshine_daily[['Datum von', 'Sunshine_Hours']],
    left_on='Datum',
    right_on='Datum von',
    how='left'
)

# Robustere Funktion: Zeit- und aggregierte Features
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

# Standardisierung der Daten
scaler = StandardScaler()
features = ['Prognose', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik', 'Sunshine_Hours',
            'Month', 'Is_Weekend', 'Season']
X = combined_data[features]
y = combined_data['Netzlast']
X_scaled = scaler.fit_transform(X)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost-Modell
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Metriken
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# MAPE-Berechnung
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred)

# Vorhersagen für das gesamte Dataset
combined_data['Prediction_XGB'] = xgb_model.predict(scaler.transform(combined_data[features]))

# Dynamische Prognose für 2023
def forecast_2023_dynamic(model, scaler, features, historical_data, start_date, days=365):
    forecast_dates = pd.date_range(start=start_date, periods=days)
    forecast_df = pd.DataFrame({'Datum von': forecast_dates})

    # Füge Basiszeit-Features hinzu
    forecast_df['Year'] = forecast_df['Datum von'].dt.year
    forecast_df['Month'] = forecast_df['Datum von'].dt.month
    forecast_df['Day'] = forecast_df['Datum von'].dt.day
    forecast_df['Weekday'] = forecast_df['Datum von'].dt.weekday
    forecast_df['Is_Weekend'] = forecast_df['Weekday'] >= 5
    forecast_df['Season'] = forecast_df['Month'].apply(lambda x: (x % 12 + 3) // 3)  # Jahreszeiten

    # Dynamische Anpassungen basierend auf historischen Mittelwerten pro Monat
    monthly_means = historical_data.groupby('Month').mean()

    forecast_df['Prognose'] = forecast_df['Month'].map(monthly_means['Prognose'])
    forecast_df['PV_Wind'] = forecast_df['Month'].map(monthly_means['PV_Wind'])
    forecast_df['Wind_Offshore'] = forecast_df['Month'].map(monthly_means['Wind_Offshore'])
    forecast_df['Wind_Onshore'] = forecast_df['Month'].map(monthly_means['Wind_Onshore'])
    forecast_df['Photovoltaik'] = forecast_df['Month'].map(monthly_means['Photovoltaik'])
    
    forecast_df['Sunshine_Hours'] = forecast_df['Month'].map(monthly_means['Sunshine_Hours'])

    forecast_features = forecast_df[features]
    forecast_scaled = scaler.transform(forecast_features)
    forecast_df['Prediction_XGB'] = model.predict(forecast_scaled)
    return forecast_df

forecast_2023_df = forecast_2023_dynamic(
    model=xgb_model,
    scaler=scaler,
    features=features,
    historical_data=combined_data,
    start_date="2023-01-01"
)

# Streamlit App
st.sidebar.title("Navigation")
pages = ["Vergangene Jahre", "Prognose 2023"] + [f"2023 - Monat {i}" for i in range(1, 13)]
try:
    page = st.sidebar.selectbox("Wählen Sie eine Seite", pages)
except Exception as e:
    st.error(f"Fehler bei der Navigation: {e}")
    page = "Vergangene Jahre"

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

    # Kreisdiagramm für den Monat
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
