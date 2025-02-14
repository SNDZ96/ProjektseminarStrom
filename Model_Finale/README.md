Projektseminar Strom

Dieses Projekt verwendet maschinelles Lernen zur Prognose von Stromerzeugung und -verbrauch mit besonderem Fokus auf erneuerbare Energien wie Photovoltaik und Windkraft. Das Ziel des Modells ist es, Vorhersagen für zukünftige Stromerzeugung und -verbrauch zu treffen, um eine effiziente Planung von Stromnetzen zu unterstützen und die Integration erneuerbarer Energien zu optimieren.
Funktionen

Vorverarbeitung von Daten (Datenbereinigung und Feature Engineering)
Modelltraining und -bewertung (XGBoost-Regressor)
Prognose von Stromverbrauch und -erzeugung für das Jahr 2023, mit einem speziellen Fokus auf erneuerbare Energiequellen
Modellvergleich (Vergleich der tatsächlichen und prognostizierten Netzlast, insbesondere in Bezug auf erneuerbare Energien)
Installation

Klone das Repository:
git clone https://github.com/SNDZ96/ProjektseminarStrom.git
Wechsle in das Projektverzeichnis:
cd ProjektseminarStrom
Erstelle und aktiviere ein virtuelles Umfeld (optional, aber empfohlen):
python -m venv venv
source venv/bin/activate  # für MacOS/Linux
venv\Scripts\activate     # für Windows
Installiere alle notwendigen Python-Pakete:
pip install -r requirements.txt
Verwendung

Stelle sicher, dass alle Daten vorhanden sind:
/DATA/Prognostizierte_Erzeugung_Day-Ahead_201701010000_202301010000_Stunde.csv
/DATA/Realisierter_Stromverbrauch_201701010000_202301010000_Stunde.csv
/DATA/Sonnenscheindauer_Deutschland_neu.csv
/DATA/FeiertageDE.csv
/DATA/bevölkerung_DE.csv
Führe das Modell aus:
Um das Modell auszuführen, benutze den folgenden Befehl:
python Model_Finale/compare_models.py
Dieses Skript führt das Modelltraining durch, bewertet das Modell und gibt die Prognosen für die Netzlast für das Jahr 2023 aus. Dabei wird ein besonderer Fokus auf erneuerbare Energien (Photovoltaik und Windkraft) gelegt.
Weitere Skripte zum Datenvorverarbeiten und Modellbewertung findest du in den entsprechenden Unterverzeichnissen:
/src/data_preprocessing: Für die Vorverarbeitung der Daten
/src/evaluation: Für die Modellbewertung (z.B. MSE, R²-Wert, MAPE)
Projektstruktur

ProjektseminarStrom/
│
├── Model_Finale/
│   ├── DATA/                       # Datenordner
│   │   ├── Prognostizierte_Erzeugung_Day-Ahead_201701010000_202301010000_Stunde.csv
│   │   ├── Realisierter_Stromverbrauch_201701010000_202301010000_Stunde.csv
│   │   ├── Sonnenscheindauer_Deutschland_neu.csv
│   │   ├── FeiertageDE.csv
│   │   ├── bevölkerung_DE.csv
│   ├── Finales_model.py            # Hauptmodellcode
│   ├── compare_models.py           # Modellvergleich
│   ├── requirements.txt           # Benötigte Python-Pakete
│   ├── README.md                  # Diese Datei
