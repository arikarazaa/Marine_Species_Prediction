import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_model(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_lr_log = lr.predict(X_test)

    # Back-transform from log space
    y_pred_lr = np.expm1(y_pred_lr_log)
    y_true    = np.expm1(y_test)

    lr_mae  = mean_absolute_error(y_true, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_true, y_pred_lr))
    lr_r2   = r2_score(y_test, y_pred_lr_log)   # R² in log space

    print('── Linear Regression ──────────────────────')
    print(f'  MAE  : {lr_mae:>12,.0f} tonnes')
    print(f'  RMSE : {lr_rmse:>12,.0f} tonnes')
    print(f'  R²   : {lr_r2:>12.4f}  (log-space)')

    return lr, y_pred_lr_log