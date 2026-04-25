import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_features(agg):
    # ── Step 2: Build lag and rolling features ────────────────────────────────────
    agg.sort_values(['Species', 'InlandMarine_Group_En', 'Year'], inplace=True)

    group_keys = ['Species', 'InlandMarine_Group_En']

    agg['Lag_1']            = agg.groupby(group_keys)['Tonnes'].shift(1)
    agg['Lag_2']            = agg.groupby(group_keys)['Tonnes'].shift(2)
    agg['Rolling_5yr_Mean'] = agg.groupby(group_keys)['Tonnes'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )
    agg['YoY_Change'] = agg.groupby(group_keys)['Tonnes'].pct_change()

    # ── Step 3: Derived year features ────────────────────────────────────────────
    agg['Year_Since_1950'] = agg['Year'] - 1950

    # ── Step 4: Label encoding ──────────────────────────────────────────────────
    le_region = LabelEncoder()
    le_group  = LabelEncoder()
    le_major  = LabelEncoder()

    agg['Region_Enc'] = le_region.fit_transform(agg['InlandMarine_Group_En'].astype(str))
    agg['Group_Enc']  = le_group.fit_transform(agg['ISSCAAP_Group_En'].astype(str))
    agg['Major_Enc']  = le_major.fit_transform(agg['Major_Group'].astype(str))

    # ── Step 5: Final cleaning ──────────────────────────────────────────────────
    model_df = agg.dropna(subset=['Lag_1', 'Lag_2', 'Rolling_5yr_Mean', 'YoY_Change']).copy()

    cap = model_df['Tonnes'].quantile(0.999)
    model_df = model_df[model_df['Tonnes'] <= cap]

    model_df['Log_Tonnes']   = np.log1p(model_df['Tonnes'])
    model_df['Log_Lag1']     = np.log1p(model_df['Lag_1'])
    model_df['Log_Lag2']     = np.log1p(model_df['Lag_2'])
    model_df['Log_Rolling5'] = np.log1p(model_df['Rolling_5yr_Mean'])

    print(f"Model-ready rows : {len(model_df):,}")
    print(f"Unique species   : {model_df['Species'].nunique():,}")
    print(f"Year range       : {model_df['Year'].min()} – {model_df['Year'].max()}")

    return model_df