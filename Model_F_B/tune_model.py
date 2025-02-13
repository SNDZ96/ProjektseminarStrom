import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from model import create_model

# Lade das Modell und die Daten
file_paths = {
    "Prognostizierte_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Prognostizierte_Erzeugung_Day-Ahead_201701010000_202301010000_Stunde.csv",
    "Realisierter_Stunde": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Realisierter_Stromverbrauch_201701010000_202301010000_Stunde.csv",
    "Sonnenscheindauer": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/Sonnenscheindauer_Deutschland_neu.csv",
    "Feiertage": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/DeutscheFE.csv",
    "Holidays": "/Users/sauanmahmud/Desktop/ProjektseminarStrom/Model_F_B/DATA_2_weitere_test/BV17-22.csv"
}

# Rufe das Basismodell auf
xgb_model, X_scaled, y = create_model(file_paths)

# Hyperparameter-Tuning mit GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=XGBRegressor(random_state=42), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y)

# Beste Parameter
best_params = grid_search.best_params_
print(f"Beste Parameter: {best_params}")

# Modell mit den besten Parametern erneut trainieren
best_model = grid_search.best_estimator_

# Vorhersagen
y_pred = best_model.predict(X_scaled)

# Metriken
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Bestes Modell - MSE: {mse:.2f}, R²: {r2:.2f}")
