import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

pio.renderers.default = 'browser'

# THEME
DARK_BG   = '#0d1117'
PANEL_BG  = '#161b22'
TEXT_CLR  = '#c9d1d9'
MUTED_CLR = '#8b949e'
ACCENT1   = '#3fb950'
ACCENT2   = '#f78166'
ACCENT3   = '#79c0ff'
ACCENT4   = '#d2a8ff'

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_CLR, size=12),
    title_font_size=14,
    xaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
)

# ML CONFIGURATION
FEATURES = [
    'Year_Since_1950',
    'Region_Enc',
    'Major_Enc',
    'Group_Enc',
    'Log_Lag1',
    'Log_Lag2',
    'Log_Rolling5',
    'YoY_Change',
]

TARGET = 'Log_Tonnes'

SPLIT_YEAR = 2015

CASE_SPECIES = ['Atlantic cod', 'Anchoveta(=Peruvian anchovy)', 'Skipjack tuna']
CASE_COLORS  = [ACCENT2, ACCENT4, ACCENT1]

print('All libraries loaded successfully ✔')