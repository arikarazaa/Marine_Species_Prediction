import pandas as pd

def load_data():
    df_raw     = pd.read_csv('data/Capture_Quantity.csv')
    df_species = pd.read_csv('data/CL_FI_SPECIES_GROUPS.csv')
    df_country = pd.read_csv('data/CL_FI_COUNTRY_GROUPS.csv')
    df_area    = pd.read_csv('data/CL_FI_WATERAREA_GROUPS.csv')

    print(f"Raw records      : {len(df_raw):,}")
    print(f"Species entries  : {len(df_species):,}")
    print(f"Country entries  : {len(df_country):,}")
    print(f"Fishing areas    : {len(df_area):,}")
    print(f"\nYear range: {df_raw['PERIOD'].min()} – {df_raw['PERIOD'].max()}")
    df_raw.head(3)

    return df_raw, df_species, df_country, df_area