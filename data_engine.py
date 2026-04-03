import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.upper()
    df_clean = df.dropna()
    
    columns_to_drop = ['DATE', 'TIME']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns], errors='ignore')
    return df_clean


def run_pca_fusion(df_clean, target_col='ROOM_OCCUPANCY_COUNT'):
    temp_cols = ['S1_TEMP', 'S2_TEMP', 'S3_TEMP', 'S4_TEMP']
    light_cols = ['S1_LIGHT', 'S2_LIGHT', 'S3_LIGHT', 'S4_LIGHT']
    required_cols = temp_cols + light_cols + [target_col]

    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_cols)
        )
    
    scaler_pca = StandardScaler()
    pca_features_scaled = scaler_pca.fit_transform(df_clean[temp_cols + light_cols])

    pca = PCA(n_components=3, random_state=42)
    microclimate_pcs = pca.fit_transform(pca_features_scaled)

    df_pcs = pd.DataFrame(microclimate_pcs, columns=['MICROCLIMATE_PC1', 'MICROCLIMATE_PC2', 'MICROCLIMATE_PC3'], index=df_clean.index)
    other_features = df_clean.drop(columns=temp_cols + light_cols)
    df_final = pd.concat([df_pcs, other_features], axis=1)

    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, pca, X.columns