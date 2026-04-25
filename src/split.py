from src.config import FEATURES, TARGET, SPLIT_YEAR

def split_data(model_df):
    train = model_df[model_df['Year'] <= SPLIT_YEAR]
    test  = model_df[model_df['Year'] >  SPLIT_YEAR]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    print(f"Training set  : {len(X_train):,} rows  ({train['Year'].min()}–{train['Year'].max()})")
    print(f"Test set      : {len(X_test):,} rows  ({test['Year'].min()}–{test['Year'].max()})")
    print(f"Features used : {FEATURES}")

    return X_train, X_test, y_train, y_test