# Projektseminar Strom

Dieses Projekt verwendet maschinelles Lernen zur Prognose von Stromerzeugung und -verbrauch auf Basis historischer Daten. Das Ziel des Modells ist es, eine Vorhersage für zukünftige Stromerzeugung und -verbrauch zu treffen, um eine effiziente Planung von Stromnetzen zu unterstützen.

## Funktionen
- Vorverarbeitung von Daten
- Modelltraining und -bewertung
- Prognose von Stromverbrauch und Erzeugung
- Modellvergleich

## Installation

1. Klone das Repository:
    ```bash
    git clone https://github.com/SNDZ96/ProjektseminarStrom.git
    ```

2. Wechsle in das Projektverzeichnis:
    ```bash
    cd ProjektseminarStrom
    ```

3. Erstelle und aktiviere ein virtuelles Umfeld (optional, aber empfohlen):
    ```bash
    python -m venv venv
    source venv/bin/activate  # für MacOS/Linux
    venv\Scripts\activate     # für Windows
    ```

4. Installiere alle notwendigen Python-Pakete:
    ```bash
    pip install -r requirements.txt
    ```

## Verwendung

1. Stelle sicher, dass alle Daten vorhanden sind:
   - /DATA/Prognostizierte_Erzeugung_Day-Ahead_201701010000_202301010000_Stunde.csv
   - /DATA/Realisierter_Stromverbrauch_201701010000_202301010000_Stunde.csv
   - /DATA/Sonnenscheindauer_Deutschland_neu.csv

2. Führe das Modell aus:
    ```bash
    python Model Finale/compare_models.py
    ```

3. Weitere Skripte zum Datenvorverarbeiten und Modellbewertung findest du in den entsprechenden Unterverzeichnissen (`/src/data_preprocessing`, `/src/evaluation`, usw.).

## Projektstruktur

