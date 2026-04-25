def preprocess_data(df_raw, df_species, df_country, df_area):
    df = df_raw[df_raw['MEASURE'] == 'Q_tlw'].copy()
    df = df[df['STATUS'] != 'Q']

    df.rename(columns={
        'PERIOD': 'Year',
        'VALUE': 'Tonnes',
        'STATUS': 'DataStatus'
    }, inplace=True)

    # ── Merge species info ────────────────────────────────────────────────────────
    sp_cols = ['3A_Code', 'Name_En', 'Major_Group', 'ISSCAAP_Group_En', 'Scientific_Name']
    df = df.merge(
        df_species[sp_cols].rename(columns={'Name_En': 'Species'}),
        left_on='SPECIES.ALPHA_3_CODE', right_on='3A_Code', how='left'
    )

    # ── Merge country info ────────────────────────────────────────────────────────
    co_cols = ['UN_Code', 'ISO3_Code', 'Name_En', 'Continent_Group_En', 'GeoRegion_Group_En', 'EcoClass_Group_En']
    df = df.merge(
        df_country[co_cols].rename(columns={'Name_En': 'Country'}),
        left_on='COUNTRY.UN_CODE', right_on='UN_Code', how='left'
    )

    # ── Merge fishing area info ───────────────────────────────────────────────────
    area_cols = ['Code', 'Name_En', 'Ocean_Group_En', 'InlandMarine_Group_En']
    df = df.merge(
        df_area[area_cols].rename(columns={'Name_En': 'FishingArea', 'Code': 'Area_Code_Ref'}),
        left_on='AREA.CODE', right_on='Area_Code_Ref', how='left'
    )

    print(f"Merged dataset shape: {df.shape}")
    print(f"Missing species name : {df['Species'].isna().sum():,} rows")
    print(f"Missing country name : {df['Country'].isna().sum():,} rows")
    return df


def aggregate_data(df):
    agg = (
        df.groupby(['Year', 'Species', 'ISSCAAP_Group_En', 'Major_Group', 'InlandMarine_Group_En'], dropna=False)
        ['Tonnes'].sum()
        .reset_index()
    )

    agg.dropna(subset=['Species', 'ISSCAAP_Group_En'], inplace=True)
    agg = agg[agg['Tonnes'] > 0].copy()
    agg['InlandMarine_Group_En'] = agg['InlandMarine_Group_En'].fillna('Unknown')

    print(f"Aggregated shape: {agg.shape}")
    return agg