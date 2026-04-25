from src.config import *
from src.data_loader import load_data
from src.preprocessing import preprocess_data, aggregate_data
from src.feature_engineering import create_features
from src.split import split_data
from src.models.linear_model import train_linear_model
from src.models.random_forest import train_random_forest
from src.evaluation import evaluate_models
from src.visualization import plot_predictions, plot_feature_importance, plot_species_analysis
from src.forecasting import plot_forecasts

import numpy as np

def main():
    df_raw, df_species, df_country, df_area = load_data()

    df      = preprocess_data(df_raw, df_species, df_country, df_area)
    agg     = aggregate_data(df)
    model_df = create_features(agg)

    X_train, X_test, y_train, y_test = split_data(model_df)

    lr_model, lr_pred_log = train_linear_model(X_train, y_train, X_test, y_test)
    rf_model, rf_pred_log = train_random_forest(X_train, y_train, X_test, y_test)

    results = evaluate_models(y_test, lr_pred_log, rf_pred_log)
    print(results)

    y_true    = np.expm1(y_test)
    y_pred_lr = np.expm1(lr_pred_log)
    y_pred_rf = np.expm1(rf_pred_log)

    plot_predictions(y_test, y_true, y_pred_lr, y_pred_rf)
    plot_feature_importance(rf_model)
    plot_species_analysis(model_df, rf_model)
    plot_forecasts(rf_model, model_df)

if __name__ == "__main__":
    main()