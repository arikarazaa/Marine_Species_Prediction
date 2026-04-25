import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=4,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred_rf_log = rf.predict(X_test)
    y_pred_rf     = np.expm1(y_pred_rf_log)
    y_true        = np.expm1(y_test)

    rf_mae  = mean_absolute_error(y_true, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_true, y_pred_rf))
    rf_r2   = r2_score(y_test, y_pred_rf_log)

    print('── Random Forest ───────────────────────────')
    print(f'  MAE  : {rf_mae:>12,.0f} tonnes')
    print(f'  RMSE : {rf_rmse:>12,.0f} tonnes')
    print(f'  R²   : {rf_r2:>12.4f}  (log-space)')

    return rf, y_pred_rf_log