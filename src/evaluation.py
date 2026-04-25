import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.config import ACCENT1, ACCENT3, TEXT_CLR, PLOTLY_LAYOUT

def evaluate_models(y_test, y_pred_lr_log, y_pred_rf_log):
    y_true    = np.expm1(y_test)
    y_pred_lr = np.expm1(y_pred_lr_log)
    y_pred_rf = np.expm1(y_pred_rf_log)

    lr_mae  = mean_absolute_error(y_true, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_true, y_pred_lr))
    lr_r2   = r2_score(y_test, y_pred_lr_log)

    rf_mae  = mean_absolute_error(y_true, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_true, y_pred_rf))
    rf_r2   = r2_score(y_test, y_pred_rf_log)

    results = pd.DataFrame({
        'Model' : ['Linear Regression', 'Random Forest'],
        'MAE'   : [lr_mae,  rf_mae],
        'RMSE'  : [lr_rmse, rf_rmse],
        'R²'    : [lr_r2,   rf_r2],
    })
    print(results.to_string(index=False))

    # ── Bar chart comparing MAE & R² ──────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Mean Absolute Error (lower = better)',
                        'R² Score (higher = better)')
    )

    colors = [ACCENT3, ACCENT1]
    for i, row in results.iterrows():
        fig.add_trace(go.Bar(name=row['Model'], x=[row['Model']], y=[row['MAE']],
                             marker_color=colors[i], showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(name=row['Model'], x=[row['Model']], y=[row['R²']],
                             marker_color=colors[i], showlegend=False), row=1, col=2)

    fig.update_layout(
        title='Model Performance Comparison (Test Set: 2016–2023)',
        height=400,
        **PLOTLY_LAYOUT
    )
    fig.update_annotations(font=dict(color=TEXT_CLR))
    fig.show()

    return results