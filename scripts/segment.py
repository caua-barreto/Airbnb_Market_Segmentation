"""
scripts/segment.py
------------------
Script de inferência parametrizado via configs.yaml: recebe um CSV com novos
anúncios no esquema bruto original do Airbnb NYC, carrega o pipeline treinado
e exporta um CSV contendo id, cluster e nome_perfil.

Uso:
    python scripts/segment.py [caminho/para/novos_anuncios.csv]
"""

import sys
import os
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import carregar_configs
from src.preprocessing import (
    carregar_dados_brutos,
    aplicar_feature_engineering,
    construir_X_modelo,
)

CFG = carregar_configs()
DATA_CFG = CFG["data"]
PREPROC_CFG = CFG["preprocessing"]
ARTIFACTS_CFG = CFG["artifacts"]
PROFILES_MAP = {int(k): v for k, v in CFG["profiles_mapping"].items()}


def main():
    if len(sys.argv) >= 2:
        caminho_entrada = sys.argv[1]
    else:
        caminho_entrada = os.path.join(
            PROJECT_ROOT, DATA_CFG["raw_dir"], DATA_CFG["raw_filename"]
        )

    caminho_saida = os.path.join(PROJECT_ROOT, DATA_CFG["predictions_output"])

    # 1. Carregar pipeline (prioriza modelo_hdbscan.joblib ou kmeans_pipeline.joblib)
    hdb_path = os.path.join(
        PROJECT_ROOT, ARTIFACTS_CFG["model_dir"], ARTIFACTS_CFG["model_filename"]
    )
    km_path = os.path.join(
        PROJECT_ROOT, ARTIFACTS_CFG["model_dir"], ARTIFACTS_CFG["kmeans_model_filename"]
    )

    if os.path.exists(hdb_path):
        pipeline_path = hdb_path
    elif os.path.exists(km_path):
        pipeline_path = km_path
    else:
        print(f"[segment] Pipeline nao encontrado em {ARTIFACTS_CFG['model_dir']}/")
        sys.exit(1)

    pipeline = joblib.load(pipeline_path)
    preprocessor = pipeline.get("scaler") or pipeline.get("preprocessor")
    modelo = pipeline.get("modelo") or pipeline.get("kmeans")
    lim_inf = pipeline.get("clip_lower")
    lim_sup = pipeline.get("clip_upper")
    colunas_treino = pipeline.get("colunas_treino")

    # 2. Carregar e filtrar dados de entrada
    df_novos = carregar_dados_brutos(
        caminho_entrada, min_price=PREPROC_CFG["min_price_filter"]
    )
    ids = (
        df_novos["id"].copy()
        if "id" in df_novos.columns
        else pd.Series(range(len(df_novos)))
    )

    # 3. Feature engineering e separação de X_modelo
    df_proc = aplicar_feature_engineering(df_novos, cfg=CFG)
    _, X_modelo = construir_X_modelo(df_proc, PREPROC_CFG["geo_cols"])

    # 4. Clipping e escalonamento com o preprocessor salvo
    X_clip = X_modelo.copy()
    cols_num = [c for c in PREPROC_CFG["numerical_model_cols"] if c in X_clip.columns]
    if lim_inf is not None and lim_sup is not None:
        X_clip[cols_num] = X_clip[cols_num].clip(lower=lim_inf, upper=lim_sup, axis=1)

    X_tratado = preprocessor.transform(X_clip)
    if isinstance(X_tratado, pd.DataFrame) and colunas_treino is not None:
        X_tratado = X_tratado[colunas_treino]

    if hasattr(modelo, "predict"):
        clusters = modelo.predict(X_tratado)
    else:
        clusters = modelo.labels_[: len(X_tratado)]

    # 5. Exportar resultado
    df_saida = pd.DataFrame({
        "id": ids.values,
        "cluster": clusters,
        "perfil_comercial": [
            PROFILES_MAP.get(int(c), f"Cluster {c}") for c in clusters
        ],
    })

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df_saida.to_csv(caminho_saida, index=False)

    print(f"[segment] {len(df_saida):,} anuncios segmentados.")
    print("[segment] Distribuicao por perfil:")
    print(df_saida["perfil_comercial"].value_counts().to_string())
    print(f"[segment] Resultado exportado para: {caminho_saida}")


if __name__ == "__main__":
    main()
