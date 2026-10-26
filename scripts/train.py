"""
scripts/train.py
----------------
Executa o pré-processamento completo e treina os modelos K-Means e HDBSCAN
parametrizados pelo configs.yaml, salvando os artefatos em artifacts/ e
relatórios em reports/metrics/.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import hdbscan

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import carregar_configs
from src.preprocessing import (
    carregar_dados_brutos,
    aplicar_feature_engineering,
    construir_base_perfilamento,
    construir_X_modelo,
    escalonar_com_column_transformer,
    exportar_master,
)
from src.modeling import treinar_kmeans, projetar_pca
from src.profiling import gerar_perfil_clusters
from src.plot_figs import (
    plot_distribuicao_clusters,
    plot_clusters_2d_plotly,
    plot_violino_perfis_hdbscan,
)

CFG = carregar_configs()
DATA_CFG = CFG["data"]
PREPROC_CFG = CFG["preprocessing"]
MOD_CFG = CFG["modeling"]
ARTIFACTS_CFG = CFG["artifacts"]
VIS_CFG = CFG["visualization"]
PROFILES_MAP = {int(k): v for k, v in CFG["profiles_mapping"].items()}

DIR_RAW = os.path.join(PROJECT_ROOT, DATA_CFG["raw_dir"])
DIR_PROCESSED = os.path.join(PROJECT_ROOT, DATA_CFG["processed_dir"])
DIR_ARTIFACTS = os.path.join(PROJECT_ROOT, ARTIFACTS_CFG["model_dir"])
DIR_METRICS = os.path.join(PROJECT_ROOT, ARTIFACTS_CFG["metrics_dir"])

for d in (DIR_PROCESSED, DIR_ARTIFACTS, DIR_METRICS):
    os.makedirs(d, exist_ok=True)

KMEANS_K = MOD_CFG["kmeans"]["best_k"]
RANDOM_STATE = MOD_CFG["random_state"]
BAIRROS_DUMMIES = PREPROC_CFG["bairros_dummies"]
CORES_KMEANS = {int(k): v for k, v in VIS_CFG["kmeans_colors"].items()}


def main():
    print(f"[train] Iniciando pipeline de treinamento (K-Means K={KMEANS_K} & HDBSCAN)...")

    # 1. Pré-processamento
    caminho_raw = os.path.join(DIR_RAW, DATA_CFG["raw_filename"])
    df = carregar_dados_brutos(caminho_raw, min_price=PREPROC_CFG["min_price_filter"])
    df_proc = aplicar_feature_engineering(df, cfg=CFG)
    exportar_master(df_proc, DIR_PROCESSED, DATA_CFG["processed_filename"])

    X_analise = construir_base_perfilamento(df_proc, PREPROC_CFG["log_revert_cols"])
    X_geo, X_modelo = construir_X_modelo(df_proc, PREPROC_CFG["geo_cols"])
    preprocessor, X_tratado, lim_inf, lim_sup = escalonar_com_column_transformer(
        X_modelo,
        cols_numericas=PREPROC_CFG["numerical_model_cols"],
        lower_quantile=PREPROC_CFG["clipping"]["lower_quantile"],
        upper_quantile=PREPROC_CFG["clipping"]["upper_quantile"],
        retornar_limites=True,
    )

    # 2. Treinamento K-Means
    km_final, labels_k8 = treinar_kmeans(
        X_tratado,
        n_clusters=KMEANS_K,
        random_state=RANDOM_STATE,
        n_init=MOD_CFG["kmeans"]["n_init"],
        max_iter=MOD_CFG["kmeans"]["max_iter"],
    )

    plot_distribuicao_clusters(
        labels_k8,
        titulo=f"Distribuicao de Registros por Cluster (K={KMEANS_K})",
        nome_arquivo="02_kmeans_distribuicao.png",
    )

    # 3. Projeção PCA
    X_pca, var_ind, var_acum = projetar_pca(
        X_tratado,
        n_components=MOD_CFG["pca"]["n_components"],
        random_state=RANDOM_STATE,
    )
    print(f"[train] PCA: var. acumulada = {var_acum * 100:.2f}%")

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Cluster"] = labels_k8
    plot_clusters_2d_plotly(
        df_pca,
        "PC1",
        "PC2",
        mapa_cores=CORES_KMEANS,
        titulo=f"Projecao 2D (PCA): Clusters do K-Means (K={KMEANS_K})",
        nome_arquivo="02_kmeans_pca.png",
    )

    # 4. Treinamento HDBSCAN Final
    hdb_cfg = MOD_CFG["hdbscan"]
    hdb_final = hdbscan.HDBSCAN(
        min_cluster_size=hdb_cfg["best_min_cluster_size"],
        min_samples=hdb_cfg["best_min_samples"],
        core_dist_n_jobs=-1,
    )
    labels_hdb = hdb_final.fit_predict(X_tratado)

    # 5. Salvamento dos pipelines (K-Means e HDBSCAN)
    kmeans_path = os.path.join(DIR_ARTIFACTS, ARTIFACTS_CFG["kmeans_model_filename"])
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "kmeans": km_final,
            "clip_lower": lim_inf,
            "clip_upper": lim_sup,
            "colunas_treino": X_tratado.columns.tolist(),
        },
        kmeans_path,
    )

    hdbscan_path = os.path.join(DIR_ARTIFACTS, ARTIFACTS_CFG["model_filename"])
    joblib.dump(
        {
            "modelo": hdb_final,
            "scaler": preprocessor,
            "clip_lower": lim_inf,
            "clip_upper": lim_sup,
            "colunas_treino": X_tratado.columns.tolist(),
        },
        hdbscan_path,
    )
    print(f"[train] Pipelines salvos em: {kmeans_path} e {hdbscan_path}")

    # 6. Perfilamento e exportação de relatórios
    X_analise["Cluster_KMeans"] = labels_k8
    X_analise["Cluster_HDBSCAN"] = labels_hdb

    perfil_km, distritos_km = gerar_perfil_clusters(
        X_analise, "Cluster_KMeans", BAIRROS_DUMMIES
    )
    perfil_hdb, distritos_hdb = gerar_perfil_clusters(
        X_analise, "Cluster_HDBSCAN", BAIRROS_DUMMIES
    )

    perfil_km.to_csv(os.path.join(DIR_METRICS, "perfil_kmeans_k8.csv"))
    distritos_km.to_csv(os.path.join(DIR_METRICS, "composicao_bairros_kmeans_k8.csv"))
    perfil_hdb.to_csv(os.path.join(DIR_METRICS, "perfil_hdbscan.csv"))
    distritos_hdb.to_csv(os.path.join(DIR_METRICS, "composicao_bairros_hdbscan.csv"))

    with open(os.path.join(DIR_ARTIFACTS, "perfis_kmeans.json"), "w") as f:
        json.dump(perfil_km.to_dict(orient="index"), f, indent=2, default=str)

    # 7. Violino dos Perfis HDBSCAN
    plot_violino_perfis_hdbscan(
        X_analise=X_analise,
        mapa_perfis=PROFILES_MAP,
        col_cluster="Cluster_HDBSCAN",
    )

    print("[train] Treinamento concluido.")


if __name__ == "__main__":
    main()
