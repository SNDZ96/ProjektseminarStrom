# src/evaluation/evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

def evaluate_model(y_true, y_pred):
    """
    Berechnet die Evaluierungsmetriken (MSE, R², MAPE) für das Modell.

    Parameters:
        y_true (array): Tatsächliche Werte
        y_pred (array): Vorhergesagte Werte

    Returns:
        dict: Metriken (MAPE, MSE, R²)
    """
    # MAPE berechnen
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # MSE und R² berechnen
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    return {'MAPE': mape, 'MSE': mse, 'R²': r2}

def plot_predictions_vs_actual(y_true, y_pred):
    """
    Plottet die tatsächlichen vs. vorhergesagten Werte.

    Parameters:
        y_true (array): Tatsächliche Werte
        y_pred (array): Vorhergesagte Werte
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Tatsächliche Werte')
    plt.plot(y_pred, label='Vorhergesagte Werte')
    plt.xlabel('Zeit')
    plt.ylabel('Netzlast (MWh)')
    plt.title('Tatsächliche vs. Vorhergesagte Netzlast')
    plt.legend()
    plt.show()
