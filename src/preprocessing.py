"""
src/preprocessing.py
--------------------
Pipeline completo de pré-processamento do Airbnb NYC parametrizado via configs.yaml:
carregamento, limpeza, feature engineering, encoding, escalonamento e exportação.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from src.config import carregar_configs

# ---------------------------------------------------------------------------
# Carregamento das configurações do projeto (configs.yaml)
# ---------------------------------------------------------------------------
CFG = carregar_configs()
PREPROC_CFG = CFG["preprocessing"]
DATA_CFG = CFG["data"]

LANDMARK = PREPROC_CFG["reference_landmark"]
TS_LAT = LANDMARK["lat"]
TS_LON = LANDMARK["lon"]
EARTH_RADIUS_KM = LANDMARK["earth_radius_km"]

MIN_PRICE_FILTER = PREPROC_CFG["min_price_filter"]
FILL_NULLS_ZERO_COLS = PREPROC_CFG["fill_nulls_zero_cols"]
COLS_LOG = PREPROC_CFG["log_transform_cols"]
ORDINAL_ROOM_TYPE_MAP = PREPROC_CFG["ordinal_room_type_map"]
COLS_ONEHOT = PREPROC_CFG["onehot_cols"]
COLS_REMOVER = PREPROC_CFG["drop_cols"]
COLS_REMOVER_COLINEARES = PREPROC_CFG["drop_collinear_cols"]
GEO_COLS = PREPROC_CFG["geo_cols"]
COLS_CONTINUAS = PREPROC_CFG["numerical_model_cols"]
LOG_REVERT_COLS = PREPROC_CFG["log_revert_cols"]
CLIP_LOWER_Q = PREPROC_CFG["clipping"]["lower_quantile"]
CLIP_UPPER_Q = PREPROC_CFG["clipping"]["upper_quantile"]


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2=TS_LAT, lon2=TS_LON, r=EARTH_RADIUS_KM):
    """Distância de Haversine em quilômetros entre dois pontos geográficos."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


# ---------------------------------------------------------------------------
# Carregamento e limpeza inicial
# ---------------------------------------------------------------------------
def carregar_dados_brutos(caminho=None, min_price=None):
    """
    Carrega o CSV bruto do Airbnb NYC e remove registros com price <= min_price_filter.
    """
    if caminho is None:
        caminho = os.path.join(DATA_CFG["raw_dir"], DATA_CFG["raw_filename"])
    if min_price is None:
        min_price = MIN_PRICE_FILTER

    df = pd.read_csv(caminho)
    n_antes = len(df)
    df = df[df["price"] > min_price].copy()
    n_removidos = n_antes - len(df)
    print(
        f"[preprocessing] {n_removidos} registros com price <= {min_price} removidos "
        f"({n_removidos / n_antes * 100:.2f}% da base)."
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering (alinhado ao 01_EDA.ipynb e configs.yaml)
# ---------------------------------------------------------------------------
def aplicar_feature_engineering(df, cfg=None):
    """
    Aplica todo o feature engineering parametrizado pelo configs.yaml:
    1. Imputação de nulos com 0 (reviews_per_month)
    2. Transformação logarítmica (np.log1p) nas variáveis de cauda longa
    3. Ordinal Encoding em room_type (Shared=0, Private=1, Entire=2)
    4. One-Hot Encoding em neighbourhood_group (drop_first=True)
    5. Criação de features derivadas em log:
       - distancia_Times_Square_log (Haversine + log1p)
       - custo_estadia_log (price * minimum_nights + log1p)
       - volume_oferta_dias_anfitriao_log (availability_365 * calculated_host_listings_count + log1p)
       - meses_ativo_log (number_of_reviews / (reviews_per_month + 1e-6) + log1p)
    6. Remoção de colunas brutas/identificadores e colineares
    """
    p_cfg = (cfg or CFG)["preprocessing"]
    df_orig = df.copy()

    # 1. Imputação de nulos
    for col in p_cfg["fill_nulls_zero_cols"]:
        if col in df_orig.columns:
            df_orig[col] = df_orig[col].fillna(0)

    # 2. Transformação logarítmica inicial
    for col in p_cfg["log_transform_cols"]:
        if col in df_orig.columns:
            df_orig[f"{col}_log"] = np.log1p(df_orig[col])

    df_tratado = df_orig.copy()

    # 3. Ordinal Encoding no room_type
    if "room_type" in df_tratado.columns and df_tratado["room_type"].dtype == object:
        df_tratado["room_type"] = df_tratado["room_type"].map(p_cfg["ordinal_room_type_map"])

    # 4. One-Hot Encoding no neighbourhood_group
    cols_oh = [c for c in p_cfg["onehot_cols"] if c in df_tratado.columns]
    if cols_oh:
        df_tratado = pd.get_dummies(
            df_tratado, columns=cols_oh, drop_first=True, dtype=int
        )

    # 5. Remoção de features brutas e identificadores
    cols_drop = [c for c in p_cfg["drop_cols"] if c in df_tratado.columns]
    df_tratado = df_tratado.drop(columns=cols_drop)

    # 6. Criação de features derivadas
    lm = p_cfg["reference_landmark"]
    dist_km = _haversine_km(
        df_tratado["latitude"], df_tratado["longitude"],
        lat2=lm["lat"], lon2=lm["lon"], r=lm["earth_radius_km"]
    )
    df_tratado["distancia_Times_Square_log"] = np.log1p(dist_km)
    df_tratado["custo_estadia_log"] = np.log1p(df_orig["price"] * df_orig["minimum_nights"])
    df_tratado["volume_oferta_dias_anfitriao_log"] = np.log1p(
        df_orig["availability_365"] * df_orig["calculated_host_listings_count"]
    )
    df_tratado["meses_ativo_log"] = np.log1p(
        df_orig["number_of_reviews"] / (df_orig["reviews_per_month"] + 1e-6)
    )

    # 7. Remoção de colunas colineares
    cols_colin = [c for c in p_cfg["drop_collinear_cols"] if c in df_tratado.columns]
    df_tratado = df_tratado.drop(columns=cols_colin)

    return df_tratado


# ---------------------------------------------------------------------------
# Construção das bases X_analise, X_geo, X_modelo e X_tratado
# ---------------------------------------------------------------------------
def construir_base_perfilamento(df_processado, cols_log=None):
    """
    Reverte colunas em log (`_log`) com `np.expm1` para obter valores na escala original,
    gerando `X_analise` para o perfilamento comercial.
    """
    if cols_log is None:
        cols_log = LOG_REVERT_COLS

    X_analise = df_processado.copy()
    for col in cols_log:
        if col in X_analise.columns:
            nome_original = col.replace("_log", "")
            X_analise[nome_original] = np.expm1(X_analise[col])
            X_analise = X_analise.drop(columns=[col])

    return X_analise


def construir_X_modelo(df_processado, geo_cols=None):
    """
    Separa as coordenadas geográficas (`X_geo`) das features explicativas (`X_modelo`).
    """
    if geo_cols is None:
        geo_cols = GEO_COLS

    X_geo = df_processado[geo_cols].copy()
    X_modelo = df_processado.drop(columns=geo_cols, errors="ignore")
    return X_geo, X_modelo


def escalonar_com_column_transformer(
    X_modelo,
    cols_numericas=None,
    lower_quantile=None,
    upper_quantile=None,
    retornar_limites=False,
):
    """
    Aplica clipping nos quantis (lower_quantile, upper_quantile) e RobustScaler
    exclusivamente nas variáveis numéricas contínuas, mantendo categóricas intactas
    e preservando a ordem original das colunas de X_modelo.
    """
    if cols_numericas is None:
        cols_numericas = [c for c in COLS_CONTINUAS if c in X_modelo.columns]
    if lower_quantile is None:
        lower_quantile = CLIP_LOWER_Q
    if upper_quantile is None:
        upper_quantile = CLIP_UPPER_Q

    cols_categoricas = [c for c in X_modelo.columns if c not in cols_numericas]

    X_clipado = X_modelo.copy()
    lim_inf = X_clipado[cols_numericas].quantile(lower_quantile)
    lim_sup = X_clipado[cols_numericas].quantile(upper_quantile)
    X_clipado[cols_numericas] = X_clipado[cols_numericas].clip(
        lower=lim_inf, upper=lim_sup, axis=1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), cols_numericas),
            ("cat", "passthrough", cols_categoricas),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    X_tratado = preprocessor.fit_transform(X_clipado)[X_modelo.columns]

    if retornar_limites:
        return preprocessor, X_tratado, lim_inf, lim_sup
    return preprocessor, X_tratado


def exportar_master(df_processado, diretorio_saida=None, nome_arquivo=None):
    """
    Exporta a base processada para data/processed/master.csv.
    """
    if diretorio_saida is None:
        diretorio_saida = DATA_CFG["processed_dir"]
    if nome_arquivo is None:
        nome_arquivo = DATA_CFG["processed_filename"]

    os.makedirs(diretorio_saida, exist_ok=True)
    caminho = os.path.join(diretorio_saida, nome_arquivo)
    df_processado.to_csv(caminho, index=False)
    print(f"[preprocessing] {nome_arquivo} exportado: {df_processado.shape} -> {caminho}")
