Stromprognose mit XGBoost

Dieses Projekt verwendet maschinelles Lernen, um die Netzlast von Stromversorgungsnetzen auf Basis von historischen und prognostizierten Daten vorherzusagen. Der XGBoost-Regressor wird verwendet, um die Netzlast zu schätzen und dynamische Prognosen für das Jahr 2023 zu erstellen.
Anforderungen

Python 3.x
pandas
numpy
scikit-learn
xgboost
streamlit
matplotlib
seaborn
Die benötigten Bibliotheken können mit folgendem Befehl installiert werden:
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn
Daten

Das Projekt verwendet mehrere CSV-Dateien, die für die Modellbildung und -vorhersage wichtig sind:
Prognostizierte_Stunde.csv: Prognostizierte Stromerzeugung für die Stunde.
Realisierter_Stunde.csv: Tatsächlicher Stromverbrauch für die Stunde.
Sonnenscheindauer_Deutschland_neu.csv: Daten zur Sonnenscheindauer in Deutschland.
FeiertageDE.csv: Feiertage in Deutschland.
bevölkerung_DE.csv: Bevölkerungsdaten, die für das Wachstum der Bevölkerung verwendet werden.
Die Datei-Pfade sind in der Python-Datei festgelegt, daher müssen die entsprechenden Dateien auf dem angegebenen Pfad abgelegt werden.
Anwendung

Die Anwendung bietet eine Streamlit-basierte Benutzeroberfläche, mit der du verschiedene Seiten zur Analyse der Vorhersagen und der Modellgüte aufrufen kannst. Folgende Seiten sind verfügbar:
1. Vergangene Jahre
Diese Seite zeigt eine grafische Darstellung der tatsächlichen und vorhergesagten Netzlast für die vergangenen Jahre. Zudem werden die Metriken wie Mean Squared Error (MSE), R² und Mean Absolute Percentage Error (MAPE) angezeigt.
2. Prognose 2023
Diese Seite zeigt eine gestapelte Darstellung der Netzlastprognose für 2023. Die Vorhersagen basieren auf historischen Daten, und es wird die Aufteilung zwischen erneuerbaren Energien (PV und Wind) und der restlichen Netzlast gezeigt.
3. Prognose für einen bestimmten Monat 2023
Auf dieser Seite können die Vorhersagen für jeden Monat im Jahr 2023 angezeigt werden, einschließlich einer tabellarischen Darstellung und einer grafischen Analyse der Netzlastanteile von erneuerbaren Energien und anderen Quellen.
Funktionsweise

Datenvorverarbeitung: Die Daten werden aus CSV-Dateien eingelesen und bereinigt. Zeitstempel werden in das richtige Format konvertiert, und nicht benötigte Spalten werden entfernt.
Modelltraining: Ein XGBoost-Modell wird mit den Vorhersage- und historischen Daten trainiert. Die Eingabevariablen beinhalten sowohl historische Wetterdaten als auch andere relevante Merkmale wie Feiertage und Bevölkerungswachstum.
Prognose: Das Modell wird verwendet, um Netzlastprognosen zu erstellen. Die Ergebnisse werden sowohl für vergangene Jahre als auch für das Jahr 2023 angezeigt.
Benutzeroberfläche: Streamlit stellt eine interaktive Web-Oberfläche zur Verfügung, mit der die Ergebnisse einfach angezeigt und analysiert werden können.
Verwendung

Lade die entsprechenden CSV-Dateien in die angegebenen Verzeichnisse auf deinem Computer.
Führe das Python-Skript aus:
streamlit run dein_script.py
Greife über deinen Webbrowser auf die Streamlit-Anwendung zu und navigiere durch die verschiedenen Seiten, um die Modellprognosen und die Analyse zu sehen.
Weiterentwicklung

Weitere Modelle und Algorithmen könnten getestet werden, um die Prognosegenauigkeit zu verbessern.
Zusätzliche Datenquellen, wie z. B. Wettervorhersagen, könnten integriert werden, um die Vorhersagequalität weiter zu steigern.
