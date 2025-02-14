import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Setup ---
# Konfiguration der Streamlit-Seite, die breites Layout und erweiterte Seitenleiste verwendet
st.set_page_config(page_title="Stromprognose", layout="wide", initial_sidebar_state="expanded")

# Setze den Stil für die Visualisierungen auf default und wähle das "whitegrid"-Theme von seaborn
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="deep")

# --- Datenpfade ---
# Definiere die Pfade zu den CSV-Dateien, die im Projekt verwendet werden
file_paths = {
    "Prognostizierte_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Prognostizierte_Erzeugung_Day-Ahead_201701010000_202301010000_Stunde.csv",
    "Realisierter_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Realisierter_Stromverbrauch_201701010000_202301010000_Stunde.csv",
    "Sonnenscheindauer": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Sonnenscheindauer_Deutschland_neu.csv",
    "Feiertage": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/FeiertageDE.csv",
    "Holidays": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/bevölkerung_DE.csv",
    "Windgeschwindigkeit": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Windgeschwindigkeit1.csv"  # Neue Datei mit Windgeschwindigkeitsdaten
}

# --- Funktion: Daten bereinigen und für das Modell vorbereiten ---
def preprocess_data(df, is_realized=False):
    """
    Diese Funktion bereinigt die Daten:
    - Wandelt das Datum in das richtige Format um
    - Entfernt unnötige Spalten wie 'Datum bis'
    - Bereinigt alle MWh-Werte (entfernt Punkte und ersetzt Kommas)
    """
    # Datum umwandeln
    df['Datum'] = pd.to_datetime(df['Datum von'], format='%d.%m.%Y %H:%M', errors='coerce')

    # Entfernen der Spalte 'Datum bis', falls vorhanden
    if 'Datum bis' in df.columns:
        df.drop(columns=['Datum bis'], inplace=True)

    # Bereinigung der MWh-Werte (Punkte entfernen und Kommas umwandeln)
    for col in df.columns:
        if 'MWh' in col:
            df[col] = (
                df[col].str.replace('.', '', regex=False)  # Entfernt Punkte
                .str.replace(',', '.', regex=False)  # Wandelt Kommas in Punkte um
                .replace('-', np.nan)  # Negative Werte durch NaN ersetzen
                .astype(float)  # Umwandeln in Fließkommazahlen
            )

    # Umbenennung der Spalten je nach Art der Daten (ob realisierte oder prognostizierte Daten)
    if is_realized:
        df.rename(columns={'Gesamt (Netzlast) [MWh] Berechnete Auflösungen': 'Netzlast'}, inplace=True)
    else:
        df.rename(columns={'Gesamt [MWh] Originalauflösungen': 'Prognose'}, inplace=True)

    return df

# --- Feiertage-Daten einlesen ---
# Laden der Feiertage-Daten und Umwandlung der 'Feiertag'-Spalte in Datetime-Format
feiertage_data = pd.read_csv(file_paths['Feiertage'], sep=';', decimal=',')
feiertage_data['Feiertag'] = pd.to_datetime(feiertage_data['Feiertag'], format='%d.%m.%Y', errors='coerce')

# --- Bevölkerungswachstumsdaten einlesen ---
# Laden der Bevölkerungsdaten und Umwandlung der 'Datum'-Spalte in Datetime-Format
bv_data = pd.read_csv(file_paths['Holidays'], sep=';', decimal=',')
bv_data['Datum'] = pd.to_datetime(bv_data['Datum'], errors='coerce')  # Umwandlung in das Datetime-Format

# Überprüfen der tatsächlichen Spaltennamen in der Datei
print(bv_data.columns)

# Bereinigung der 'Bevölkerung'-Spalte: Entfernen von Tausendertrennzeichen und Umwandlung in float
bv_data['Bevölkerung'] = bv_data['Bevölkerung'].str.replace('.', '', regex=False).astype(float)  # Tausendertrennzeichen entfernen

# Jahr extrahieren und Bevölkerungsdaten nach Jahr gruppieren
# Falls die 'Datum'-Spalte nicht als Datetime umgewandelt wurde, müssen wir sicherstellen, dass sie korrekt ist
if not pd.api.types.is_datetime64_any_dtype(bv_data['Datum']):
    bv_data['Datum'] = pd.to_datetime(bv_data['Datum'], errors='coerce')

# Extrahieren des Jahres aus der 'Datum'-Spalte und Erstellen der 'Jahr'-Spalte
bv_data['Jahr'] = bv_data['Datum'].dt.year

# --- Windgeschwindigkeitsdaten einlesen ---
# Laden der Windgeschwindigkeitsdaten und Umwandlung der 'Datum'-Spalte in Datetime-Format
wind_data = pd.read_csv(file_paths['Windgeschwindigkeit'], sep=';', decimal=',')
wind_data['Datum'] = pd.to_datetime(wind_data['Datum'], errors='coerce')  # Umwandlung der 'Datum'-Spalte in Datetime

# Überprüfen der Windgeschwindigkeitsdaten
print(wind_data.columns)

# Hier nehmen wir an, dass die Spalte 'Average Wind-\ngeschwindigkeit in m/s' die Windgeschwindigkeit enthält
wind_data['Windgeschwindigkeit'] = wind_data['Average Wind-\ngeschwindigkeit in m/s'].astype(float)

# Jahr extrahieren und Windgeschwindigkeitsdaten nach Jahr gruppieren
wind_data['Jahr'] = wind_data['Datum'].dt.year
wind_data = wind_data[['Jahr', 'Windgeschwindigkeit']]  # Wir behalten nur Jahr und Windgeschwindigkeit

# --- Sonnenscheindauer-Daten einlesen ---
# Laden der Sonnenscheindauer-Daten und Umwandlung der 'Datum'-Spalte in Datetime-Format
sunshine_data = pd.read_csv(file_paths['Sonnenscheindauer'], sep=';', decimal=',')
sunshine_data['Datum'] = pd.to_datetime(sunshine_data['Datum'], format='%Y-%m-%d', errors='coerce')
sunshine_data['Datum von'] = sunshine_data['Datum']
sunshine_data['Sunshine_Minutes'] = sunshine_data['Arithmetisches Mittel\nSonnenschein in Minuten']
sunshine_daily = sunshine_data.groupby('Datum von', as_index=False)['Sunshine_Minutes'].sum()
sunshine_daily['Sunshine_Hours'] = sunshine_daily['Sunshine_Minutes'] / 60

# --- Lade und bereinige die Daten für Prognosen und tatsächliche Werte ---
prognostizierte_stunde = preprocess_data(pd.read_csv(file_paths['Prognostizierte_Stunde'], sep=';', decimal=','))
realisierter_stunde = preprocess_data(pd.read_csv(file_paths['Realisierter_Stunde'], sep=';', decimal=','), is_realized=True)

# --- Kombiniere die Daten ---
# Kombiniere die realisierten und prognostizierten Daten nach 'Datum'
combined_data = pd.merge(
    realisierter_stunde[['Datum', 'Netzlast']],
    prognostizierte_stunde[['Datum', 'Prognose', 'Photovoltaik und Wind [MWh] Berechnete Auflösungen',
                            'Wind Offshore [MWh] Berechnete Auflösungen', 'Wind Onshore [MWh] Berechnete Auflösungen',
                            'Photovoltaik [MWh] Berechnete Auflösungen']],
    on='Datum',
    how='inner'
)

# --- Spaltennamen für bessere Lesbarkeit umbenennen ---
combined_data.rename(columns={
    'Photovoltaik und Wind [MWh] Berechnete Auflösungen': 'PV_Wind',
    'Wind Offshore [MWh] Berechnete Auflösungen': 'Wind_Offshore',
    'Wind Onshore [MWh] Berechnete Auflösungen': 'Wind_Onshore',
    'Photovoltaik [MWh] Berechnete Auflösungen': 'Photovoltaik'
}, inplace=True)

# --- Sonnenscheindauer-Daten zusammenführen ---
# Füge die Sonnenscheindauer-Daten zu den kombinierten Daten hinzu
combined_data = pd.merge(
    combined_data,
    sunshine_daily[['Datum von', 'Sunshine_Hours']],
    left_on='Datum',
    right_on='Datum von',
    how='left'
)

# --- Windgeschwindigkeitsdaten zusammenführen ---
# Füge die Windgeschwindigkeitsdaten zu den kombinierten Daten hinzu
combined_data = pd.merge(
    combined_data,
    wind_data[['Jahr', 'Windgeschwindigkeit']],
    on='Jahr',
    how='left'
)

# --- Feiertage-Feature hinzufügen ---
# Füge ein Feature hinzu, das anzeigt, ob ein Tag ein Feiertag ist
combined_data['Is_Holiday'] = combined_data['Datum'].isin(feiertage_data['Feiertag'])

# --- Feature Engineering: Zeit- und saisonale Merkmale ---
def add_features(df):
    """
    Diese Funktion fügt Zeit- und saisonale Merkmale wie Jahr, Monat, Wochentag, Wochenende und Saison hinzu.
    """
    # Extrahiert das Jahr, den Monat und den Tag aus der 'Datum'-Spalte
    df['Year'] = df['Datum'].dt.year
    df['Month'] = df['Datum'].dt.month
    df['Day'] = df['Datum'].dt.day
    df['Weekday'] = df['Datum'].dt.weekday  # Wochentag (0 = Montag, 6 = Sonntag)
    
    # Feiertage werden als Wochenenden betrachtet (Is_Weekend = True) für Samstag (5) und Sonntag (6)
    df['Is_Weekend'] = df['Weekday'] >= 5

  # Bestimmt die Jahreszeit (Frühling = 1, Sommer = 2, Herbst = 3, Winter = 4)
    df['Season'] = df['Month'].apply(lambda x: (x % 12 + 3) // 3)
    
    return df

# Anwenden der Feature-Engineering-Funktion auf die kombinierten Daten
combined_data = add_features(combined_data)

# Entfernen von NaN-Werten, die durch das Mergen entstehen könnten
combined_data.dropna(inplace=True)

# --- Bevölkerungswachstums-Feature hinzufügen ---
# Merging mit den Bevölkerungsdaten, um die Bevölkerungszahl für jedes Jahr hinzuzufügen
combined_data = pd.merge(combined_data, bv_data[['Jahr', 'Bevölkerung']], on='Jahr', how='left')

# Das Bevölkerungswachstum als zusätzliches Feature verwenden
combined_data['Population_Growth'] = combined_data['Bevölkerung'] 

# --- Vorhersage für 2023 erstellen ---
def forecast_2023_dynamic(model, scaler, features, historical_data, start_date, days=365):
    """
    Diese Funktion erstellt eine Vorhersage für das Jahr 2023. Die Daten für die Vorhersage werden dynamisch generiert.
    Es werden Monatsdurchschnittswerte für die Merkmale verwendet.
    """
    # Erstellen eines Zeitrahmens von 365 Tagen ab dem angegebenen Startdatum
    forecast_dates = pd.date_range(start=start_date, periods=days)
    forecast_df = pd.DataFrame({'Datum von': forecast_dates})

    # Füge Zeit-Features zu den Vorhersagedaten hinzu (Jahr, Monat, Wochentag, Wochenende, Saison)
    forecast_df['Year'] = forecast_df['Datum von'].dt.year
    forecast_df['Month'] = forecast_df['Datum von'].dt.month
    forecast_df['Day'] = forecast_df['Datum von'].dt.day
    forecast_df['Weekday'] = forecast_df['Datum von'].dt.weekday
    forecast_df['Is_Weekend'] = forecast_df['Weekday'] >= 5
    forecast_df['Season'] = forecast_df['Month'].apply(lambda x: (x % 12 + 3) // 3)  # Jahreszeiten

    # Berechne die Monatsdurchschnittswerte für die Merkmale, um diese in der Vorhersage zu verwenden
    monthly_means = historical_data.groupby('Month').mean()

    # Weise den Vorhersagedaten die monatlichen Mittelwerte zu
    forecast_df['Prognose'] = forecast_df['Month'].map(monthly_means['Prognose'])
    forecast_df['PV_Wind'] = forecast_df['Month'].map(monthly_means['PV_Wind'])
    forecast_df['Wind_Offshore'] = forecast_df['Month'].map(monthly_means['Wind_Offshore'])
    forecast_df['Wind_Onshore'] = forecast_df['Month'].map(monthly_means['Wind_Onshore'])
    forecast_df['Photovoltaik'] = forecast_df['Month'].map(monthly_means['Photovoltaik'])
    forecast_df['Sunshine_Hours'] = forecast_df['Month'].map(monthly_means['Sunshine_Hours'])

    # Füge Feiertags- und Bevölkerungswachstums-Features hinzu
    forecast_df['Is_Holiday'] = forecast_df['Month'].isin(feiertage_data['Feiertag'].dt.month)
    forecast_df['Population_Growth'] = forecast_df['Month'].map(monthly_means['Population_Growth'])

    # Bereite die Vorhersage-Features vor und skaliere sie
    forecast_features = forecast_df[features]
    forecast_scaled = scaler.transform(forecast_features)

    # Berechne die Vorhersage mit dem XGBoost-Modell
    forecast_df['Prediction_XGB'] = model.predict(forecast_scaled)
    
    return forecast_df

# --- Initialisiere das Modell (XGBoost) und den Scaler ---
xgb_model = XGBRegressor(random_state=42)
scaler = StandardScaler()

# --- Fitte den Scaler mit den Trainingsdaten ---
scaler.fit(combined_data[['Prognose', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik', 'Sunshine_Hours',
                          'Month', 'Is_Weekend', 'Season', 'Is_Holiday', 'Population_Growth']])

# --- Training und Testdaten ---
# Wähle die Merkmale für das Training und die Zielvariable (Netzlast)
X = combined_data[['Prognose', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik', 'Sunshine_Hours',
                   'Month', 'Is_Weekend', 'Season', 'Is_Holiday', 'Population_Growth']]
y = combined_data['Netzlast']

# Skaliere die Eingabedaten
X_scaled = scaler.transform(X)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Modell trainieren ---
# Trainiere das XGBoost-Modell mit den Trainingsdaten
xgb_model.fit(X_train, y_train)

# --- Vorhersagen und Berechnung von Metriken ---
# Mache Vorhersagen für das Testset
xgb_pred = xgb_model.predict(X_test)

# Berechne MSE (Mean Squared Error) und R² (Bestimmtheitsmaß)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# MAPE-Berechnung (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred)

# --- Vorhersagen für das gesamte Dataset ---
# Berechne Vorhersagen für alle Daten im kombinierten Datensatz
combined_data['Prediction_XGB'] = xgb_model.predict(scaler.transform(combined_data[['Prognose', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik', 'Sunshine_Hours',
                                                                                        'Month', 'Is_Weekend', 'Season', 'Is_Holiday', 'Population_Growth']]))

# --- Streamlit App ---
# Erstelle eine Navigationsleiste in der Seitenleiste von Streamlit
st.sidebar.title("Navigation")
pages = ["Vergangene Jahre", "Prognose 2023"] + [f"2023 - Monat {i}" for i in range(1, 13)]
page = st.sidebar.selectbox("Wählen Sie eine Seite", pages)

# Vorhersage für 2023 erstellen
forecast_2023_df = forecast_2023_dynamic(
    model=xgb_model,
    scaler=scaler,  # Verwende den bereits gefitteten scaler
    features=['Prognose', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik', 'Sunshine_Hours',
              'Month', 'Is_Weekend', 'Season', 'Is_Holiday', 'Population_Growth'],
    historical_data=combined_data,
    start_date="2023-01-01"
)

# --- Streamlit Seiteninhalt ---
# Seite für die Visualisierung von "Vergangene Jahre"
if page == "Vergangene Jahre":
    st.title("XGBoost: Vergangene Jahre")
    st.write(f"**MSE:** {xgb_mse:.2f}")
    st.write(f"**R²-Wert:** {xgb_r2:.2f}")
    st.write(f"**MAPE:** {xgb_mape:.2f}%")
    
    # Visualisierung der täglichen Netzlast
    for year in combined_data['Year'].unique():
        yearly_data = combined_data[combined_data['Year'] == year]
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(yearly_data['Datum'], yearly_data['Netzlast'], label='Tatsächliche Netzlast')
        ax.plot(yearly_data['Datum'], yearly_data['Prediction_XGB'], label='Vorhergesagte Netzlast')
        ax.set_xlabel('Datum')
        ax.set_ylabel('Netzlast (MWh)')
        ax.set_title(f'Tägliche Netzlast: Tatsächlich vs. Vorhergesagt für {year}')
        ax.legend()
        st.pyplot(fig)

# Seite für die "Prognose 2023"
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

# Seite für die "Prognose pro Monat 2023"
else:
    # Extrahiert den Monat aus der Seitenwahl und filtert die Daten für diesen Monat
    month = int(page.split()[-1])
    st.title(f"Prognose für Monat {month} im Jahr 2023")
    monthly_data = forecast_2023_df[forecast_2023_df['Datum von'].dt.month == month]
    
    # Zeigt die Daten für den Monat in einer Tabelle an
    st.write("### Tabellarische Daten")
    st.write(monthly_data[['Datum von', 'Prediction_XGB', 'PV_Wind', 'Wind_Offshore', 'Wind_Onshore', 'Photovoltaik']])

    # Visualisiere die Anteile der Netzlast mit einem Kreisdiagramm
    total_energy = monthly_data['Prediction_XGB'].sum()
    renewable_energy = monthly_data['PV_Wind'].sum()
    other_energy = total_energy - renewable_energy

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        [renewable_energy, other_energy],
        labels=['Erneuerbare Energien', 'Andere Netzlast'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title(f'Anteile der Netzlast für Monat {month} (2023)')
    st.pyplot(fig)

