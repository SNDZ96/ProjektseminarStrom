# src/utils/utils.py

import joblib

def save_model(model, filename):
    """
    Speichert das trainierte Modell als Datei.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Lädt das gespeicherte Modell.
    """
    return joblib.load(filename)
