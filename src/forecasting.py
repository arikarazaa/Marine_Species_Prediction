import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import (ACCENT4, MUTED_CLR, TEXT_CLR, PLOTLY_LAYOUT,
                        FEATURES, CASE_SPECIES, CASE_COLORS)

def forecast_species(species_name, model, model_df, horizon=7):
    """
    Iteratively forecast catch for a given species from 2024 to 2023+horizon.
    At each step, the prediction becomes the lag for the next step.
    """
    sp_hist = (
        model_df[model_df['Species'] == species_name]
        .groupby('Year')[['Tonnes', 'Log_Tonnes', 'Year_Since_1950',
                           'Region_Enc', 'Major_Enc', 'Group_Enc',
                           'Log_Lag1', 'Log_Lag2', 'Log_Rolling5', 'YoY_Change']]
        .mean().reset_index()
    )
    if sp_hist.empty:
        return None, None

    last = sp_hist.iloc[-1].copy()
    forecast_rows = []

    lag1     = last['Log_Tonnes']
    lag2     = last['Log_Lag1']
    rolling5 = last['Log_Rolling5']

    for step in range(1, horizon + 1):
        year = int(last['Year']) + step
        row = {
            'Year_Since_1950': year - 1950,
            'Region_Enc'     : last['Region_Enc'],
            'Major_Enc'      : last['Major_Enc'],
            'Group_Enc'      : last['Group_Enc'],
            'Log_Lag1'       : lag1,
            'Log_Lag2'       : lag2,
            'Log_Rolling5'   : rolling5,
            'YoY_Change'     : last['YoY_Change'],
        }
        pred_log = model.predict(pd.DataFrame([row]))[0]
        pred_log = max(0, pred_log)

        forecast_rows.append({'Year': year, 'Predicted_Mt': np.expm1(pred_log) / 1e6})

        lag2     = lag1
        lag1     = pred_log
        rolling5 = (rolling5 * 4 + pred_log) / 5

    return sp_hist, pd.DataFrame(forecast_rows)


def plot_forecasts(model, model_df):
    # ── Plot forecasts for case species ──────────────────────────────────────────
    fig = make_subplots(rows=1, cols=3, subplot_titles=CASE_SPECIES)

    for col_idx, (sp_name, color) in enumerate(zip(CASE_SPECIES, CASE_COLORS), start=1):
        hist, fcast = forecast_species(sp_name, model, model_df)
        if hist is None:
            continue

        hist['Tonnes_Mt'] = hist['Tonnes'] / 1e6

        fig.add_trace(go.Scatter(
            x=hist['Year'], y=hist['Tonnes_Mt'],
            mode='lines', line=dict(color=color, width=2),
            name='Historical', showlegend=(col_idx == 1)
        ), row=1, col=col_idx)

        bridge_x = [hist['Year'].iloc[-1], fcast['Year'].iloc[0]]
        bridge_y = [hist['Tonnes_Mt'].iloc[-1], fcast['Predicted_Mt'].iloc[0]]

        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            mode='lines', line=dict(color=ACCENT4, width=1.5, dash='dot'),
            showlegend=False
        ), row=1, col=col_idx)

        fig.add_trace(go.Scatter(
            x=fcast['Year'], y=fcast['Predicted_Mt'],
            mode='lines+markers',
            line=dict(color=ACCENT4, width=2, dash='dot'),
            marker=dict(size=6, color=ACCENT4, symbol='diamond'),
            name='Forecast (2024–2030)', showlegend=(col_idx == 1)
        ), row=1, col=col_idx)

        fig.add_vline(x=2023, line_color=MUTED_CLR, line_dash='dash',
                      line_width=1, row=1, col=col_idx)

    fig.update_layout(
        title='Catch Forecast 2024–2030 — Random Forest Projection (Million Tonnes)',
        height=430,
        **PLOTLY_LAYOUT
    )
    fig.update_annotations(font=dict(color=TEXT_CLR))
    fig.show()