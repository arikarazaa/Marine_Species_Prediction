import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import (ACCENT1, ACCENT2, ACCENT3, ACCENT4, MUTED_CLR,
                        TEXT_CLR, PLOTLY_LAYOUT, FEATURES, SPLIT_YEAR,
                        CASE_SPECIES, CASE_COLORS)

def plot_predictions(y_test, y_true, y_pred_lr, y_pred_rf):
    # Sample for readability (plotting 100k+ points is slow)
    plot_idx    = np.random.choice(len(y_test), size=min(5000, len(y_test)), replace=False)
    y_true_s    = np.array(y_true)[plot_idx]
    y_pred_rf_s = np.array(y_pred_rf)[plot_idx]
    y_pred_lr_s = np.array(y_pred_lr)[plot_idx]

    fig = go.Figure()

    # Perfect prediction line
    lim = max(y_true_s.max(), y_pred_rf_s.max())
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim],
        mode='lines', line=dict(color=MUTED_CLR, dash='dash', width=1.5),
        name='Perfect Prediction'
    ))

    fig.add_trace(go.Scatter(
        x=y_true_s, y=y_pred_lr_s,
        mode='markers',
        marker=dict(color=ACCENT3, opacity=0.25, size=4),
        name='Linear Regression'
    ))

    fig.add_trace(go.Scatter(
        x=y_true_s, y=y_pred_rf_s,
        mode='markers',
        marker=dict(color=ACCENT1, opacity=0.3, size=4),
        name='Random Forest'
    ))

    fig.update_layout(
        title='Predicted vs Actual Catch Volume (Tonnes) — Test Set Sample',
        xaxis_title='Actual Catch (tonnes)',
        yaxis_title='Predicted Catch (tonnes)',
        height=500,
        **PLOTLY_LAYOUT
    )
    fig.show()


def plot_feature_importance(rf):
    feat_imp = pd.DataFrame({
        'Feature'   : FEATURES,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)

    name_map = {
        'Log_Lag1'       : 'Last Year Catch (log)',
        'Log_Lag2'       : '2-Year Lag Catch (log)',
        'Log_Rolling5'   : '5-Year Rolling Mean (log)',
        'YoY_Change'     : 'Year-on-Year Change (%)',
        'Year_Since_1950': 'Year (since 1950)',
        'Region_Enc'     : 'Ocean Region',
        'Group_Enc'      : 'ISSCAAP Species Group',
        'Major_Enc'      : 'Major Biological Group',
    }
    feat_imp['Feature'] = feat_imp['Feature'].map(name_map)

    fig = px.bar(
        feat_imp, x='Importance', y='Feature', orientation='h',
        color='Importance',
        color_continuous_scale=[[0, MUTED_CLR], [1, ACCENT1]],
        title='Random Forest — Feature Importances',
        labels={'Importance': 'Mean Decrease in Impurity'}
    )
    fig.update_layout(height=420, coloraxis_showscale=False, **PLOTLY_LAYOUT)
    fig.show()


def plot_species_analysis(model_df, rf):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=CASE_SPECIES,
        shared_yaxes=False
    )

    for col_idx, (sp_name, color) in enumerate(zip(CASE_SPECIES, CASE_COLORS), start=1):
        sp_data = model_df[model_df['Species'] == sp_name].groupby('Year')[
            ['Tonnes', 'Log_Tonnes', 'Year_Since_1950',
             'Region_Enc', 'Major_Enc', 'Group_Enc',
             'Log_Lag1', 'Log_Lag2', 'Log_Rolling5', 'YoY_Change']
        ].mean().reset_index()
        if sp_data.empty:
            continue

        sp_pred_log = rf.predict(sp_data[FEATURES].fillna(0))
        sp_data['Predicted'] = np.expm1(sp_pred_log)
        sp_data['Tonnes_Mt'] = sp_data['Tonnes'] / 1e6
        sp_data['Pred_Mt']   = sp_data['Predicted'] / 1e6

        test_mask = sp_data['Year'] > SPLIT_YEAR

        fig.add_trace(go.Scatter(
            x=sp_data['Year'], y=sp_data['Tonnes_Mt'],
            mode='lines', line=dict(color=color, width=2),
            name='Actual', showlegend=(col_idx == 1)
        ), row=1, col=col_idx)

        sp_test = sp_data[test_mask]
        fig.add_trace(go.Scatter(
            x=sp_test['Year'], y=sp_test['Pred_Mt'],
            mode='lines+markers',
            line=dict(color=MUTED_CLR, width=1.5, dash='dot'),
            marker=dict(size=5, color=MUTED_CLR),
            name='RF Prediction', showlegend=(col_idx == 1)
        ), row=1, col=col_idx)

        fig.add_vline(x=SPLIT_YEAR, line_color=MUTED_CLR, line_dash='dash',
                      line_width=1, row=1, col=col_idx)

    fig.update_layout(
        title='Catch History + Random Forest Predictions — Three Iconic Species (Million Tonnes)',
        height=420,
        **PLOTLY_LAYOUT
    )
    fig.update_annotations(font=dict(color=TEXT_CLR))
    fig.show()