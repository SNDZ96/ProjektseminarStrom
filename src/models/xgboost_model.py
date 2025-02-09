# src/models/xgboost_model.py

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Trainiert das XGBoost-Modell und berechnet die Metriken.
    """
    # XGBoost Modell erstellen
    xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Vorhersage
    xgb_pred = xgb_model.predict(X_test)

    # Berechne Metriken
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)

    return xgb_model, xgb_mse, xgb_r2
