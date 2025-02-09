# src/data_preprocessing/preprocess_data.py

import pandas as pd
import numpy as np

def load_and_preprocess_data():
    """
    Lädt und verarbeitet die benötigten Daten:
    - `Realisierte_Erzeugung_2017_2023_Tag.csv`
    - `Realisierter_Stromverbrauch_2017_2023Tag.csv`

    Returns:
        Tuple: Bereinigte Daten (realisierte Erzeugung, realisierter Stromverbrauch)
    """
    
    # Lade die CSV-Daten
    realisierte_erzeugung = pd.read_csv('/Users/sauanmahmud/Desktop/ProjektseminarStrom/data/Realisierte_Erzeugung_2017_2023_Tag.csv', sep=';', decimal=',')
    realisierter_stromverbrauch = pd.read_csv('/Users/sauanmahmud/Desktop/ProjektseminarStrom/data/Realisierter_Stromverbrauch_2017_2023Tag.csv', sep=';', decimal=',')
    
    # Bereinige und preprocessiere die Daten
    realisierte_erzeugung_clean = preprocess_data(realisierte_erzeugung)
    realisierter_stromverbrauch_clean = preprocess_data(realisierter_stromverbrauch, is_realized=True)
    
    return realisierte_erzeugung_clean, realisierter_stromverbrauch_clean


def preprocess_data(df, is_realized=False):
    """
    Bereinigt und verarbeitet die Eingabedaten.

    Parameters:
        df (DataFrame): Die zu verarbeitenden Daten.
        is_realized (bool): Flag, das angibt, ob es sich um realisierte Werte handelt. Standard ist False.

    Returns:
        DataFrame: Die bereinigten und verarbeiteten Daten.
    """
    # Umwandeln von Datum
    df['Datum von'] = pd.to_datetime(df['Datum von'], format='%d.%m.%Y %H:%M', errors='coerce')
    if 'Datum bis' in df.columns:
        df.drop(columns=['Datum bis'], inplace=True)

    # Umwandlung der 'MWh' Spalten zu numerischen Werten
    for col in df.columns:
        if 'MWh' in col:
            df[col] = (
                df[col].str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .replace('-', np.nan)
                .astype(float)
            )

    # Wenn es sich um realisierte Daten handelt, spaltenbezogen auf 'Netzlast' zugreifen
    if is_realized:
        # Nutze den genauen Spaltennamen, z.B. 'Gesamt (Netzlast) [MWh] Berechnete Auflösungen'
        if 'Gesamt (Netzlast) [MWh] Berechnete Auflösungen' in df.columns:
            df.rename(columns={'Gesamt (Netzlast) [MWh] Berechnete Auflösungen': 'Netzlast'}, inplace=True)
        else:
            print("Warnung: 'Gesamt (Netzlast) [MWh] Berechnete Auflösungen' wurde nicht gefunden.")
    else:
        # Falls die Daten prognostiziert sind, die relevanten Spalten ansprechen
        df.rename(columns={
            'Biomasse [MWh] Berechnete Auflösungen': 'Biomasse',
            'Wasserkraft [MWh] Berechnete Auflösungen': 'Wasserkraft',
            'Wind Offshore [MWh] Berechnete Auflösungen': 'Wind_Offshore',
            'Wind Onshore [MWh] Berechnete Auflösungen': 'Wind_Onshore',
            'Photovoltaik [MWh] Berechnete Auflösungen': 'Photovoltaik',
            'Sonstige Erneuerbare [MWh] Berechnete Auflösungen': 'Sonstige_Erneuerbare',
            'Kernenergie [MWh] Berechnete Auflösungen': 'Kernenergie',
            'Braunkohle [MWh] Berechnete Auflösungen': 'Braunkohle',
            'Steinkohle [MWh] Berechnete Auflösungen': 'Steinkohle',
            'Erdgas [MWh] Berechnete Auflösungen': 'Erdgas',
            'Pumpspeicher [MWh] Berechnete Auflösungen': 'Pumpspeicher',
            'Sonstige Konventionelle [MWh] Berechnete Auflösungen': 'Sonstige_Konventionelle'
        }, inplace=True)

    return df
