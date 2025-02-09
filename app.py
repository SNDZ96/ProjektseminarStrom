# app.py

import sys
import os
import streamlit as st
import matplotlib.pyplot as plt
from data_preprocessing.preprocess_data import load_and_preprocess_data
from feature_engineering.feature_engineering import add_features
from models.xgboost_model import train_xgboost_model
from evaluation.evaluation import evaluate_model, plot_predictions_vs_actual
from utils.utils import save_model, load_model

# Lade und verarbeite die Daten
realisierte_erzeugung_clean, realisierter_stromverbrauch_clean = load_and_preprocess_data()

# Feature Engineering anwenden
combined_data = add_features(realisierte_erzeugung_clean)

# **Wichtige Änderung: Greife auf eine andere Zielspalte zu**
y = combined_data['Pumpspeicher [MWh] Berechnete Auflösungen']  # Ersetze 'Netzlast' mit einer anderen Zielspalte

# Train-Test-Split durchführen
from sklearn.model_selection import train_test_split
X = combined_data[['Photovoltaik', 'Wasserkraft', 'Wind_Offshore', 'Wind_Onshore', 'Sonstige_Erneuerbare', 
                   'Kernenergie', 'Braunkohle', 'Steinkohle', 'Erdgas', 'Pumpspeicher', 'Sonstige_Konventionelle', 
                   'Month', 'Day', 'Year', 'Is_Weekend', 'Season']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trainiere das Modell
xgb_model, xgb_mse, xgb_r2 = train_xgboost_model(X_train, y_train, X_test, y_test)

# Evaluierung
metrics = evaluate_model(y_test, xgb_model.predict(X_test))

# Zeige die Metriken in Streamlit
st.write(f"MAPE: {metrics['MAPE'] * 100:.2f}%")
st.write(f"MSE: {metrics['MSE']:.2f}")
st.write(f"R²: {metrics['R²']:.2f}")

# Visualisierung der tatsächlichen vs. vorhergesagten Werte
plot_predictions_vs_actual(y_test, xgb_model.predict(X_test))

# Visualisierung der Netzlastvorhersage
st.title("Prognose der Netzlast für 2023")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(combined_data['Datum von'], combined_data['Pumpspeicher [MWh] Berechnete Auflösungen'], label='Tatsächlicher Pumpspeicher', color='#1f77b4')
ax.plot(combined_data['Datum von'], xgb_model.predict(X), label='Vorhergesagter Pumpspeicher', color='#ff7f0e')
ax.set_xlabel('Datum')
ax.set_ylabel('Pumpspeicher (MWh)')
ax.set_title('Täglicher Pumpspeicher: Tatsächlich vs. Vorhergesagt')
ax.legend()
st.pyplot(fig)
