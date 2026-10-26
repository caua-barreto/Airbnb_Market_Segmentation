"""
main.py
-------
Orquestrador do projeto Airbnb NYC — executa o pipeline completo
de ponta a ponta parametrizado via configs.yaml: pré-processamento,
treinamento K-Means e HDBSCAN, experimentos de densidade, perfilamento
e exportação de artefatos e métricas.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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
from src.modeling import (
    curva_elbow,
    avaliar_kmeans_range,
    treinar_kmeans,
    experimento_distancias,
    projetar_pca,
    projetar_umap,
)
from src.profiling import gerar_perfil_clusters, composicao_cruzada
from src.plot_figs import (
    plot_elbow_comparativo,
    plot_metricas_validacao,
    plot_distribuicao_clusters,
    plot_clusters_2d_plotly,
    plot_mapa_clusters_nyc,
    plot_violino_perfil_clusters,
)

# ---------------------------------------------------------------------------
# Carregamento de configurações (configs.yaml)
# ---------------------------------------------------------------------------
CFG = carregar_configs()
DATA_CFG = CFG["data"]
PREPROC_CFG = CFG["preprocessing"]
MOD_CFG = CFG["modeling"]
ARTIFACTS_CFG = CFG["artifacts"]
VIS_CFG = CFG["visualization"]

DIR_DATA_RAW = os.path.join(PROJECT_ROOT, DATA_CFG["raw_dir"])
DIR_DATA_PROCESSED = os.path.join(PROJECT_ROOT, DATA_CFG["processed_dir"])
DIR_ARTIFACTS = os.path.join(PROJECT_ROOT, ARTIFACTS_CFG["model_dir"])
DIR_METRICS = os.path.join(PROJECT_ROOT, ARTIFACTS_CFG["metrics_dir"])
DIR_FIGURES = os.path.join(PROJECT_ROOT, ARTIFACTS_CFG["figures_dir"])

for d in (DIR_DATA_PROCESSED, DIR_ARTIFACTS, DIR_METRICS, DIR_FIGURES):
    os.makedirs(d, exist_ok=True)

KMEANS_K = MOD_CFG["kmeans"]["best_k"]
RANDOM_STATE = MOD_CFG["random_state"]
SAMPLE_SIZE_SILHOUETTE = MOD_CFG["subsample_silhouette"]
BAIRROS_DUMMIES = PREPROC_CFG["bairros_dummies"]
CORES_KMEANS = {int(k): v for k, v in VIS_CFG["kmeans_colors"].items()}
URL_TILES_CARTO = VIS_CFG["carto_tiles_url"]


def main():
    print("=" * 70)
    print("  PROJETO 07 — SEGMENTACAO DE MERCADO: AIRBNB NYC")
    print("=" * 70)

    # 1. Pré-processamento
    print("\n[1/5] Pre-processamento e Feature Engineering")
    caminho_raw = os.path.join(DIR_DATA_RAW, DATA_CFG["raw_filename"])
    df = carregar_dados_brutos(caminho_raw, min_price=PREPROC_CFG["min_price_filter"])
    df_processado = aplicar_feature_engineering(df, cfg=CFG)

    exportar_master(df_processado, DIR_DATA_PROCESSED, DATA_CFG["processed_filename"])

    X_analise = construir_base_perfilamento(df_processado, PREPROC_CFG["log_revert_cols"])
    X_geo, X_modelo = construir_X_modelo(df_processado, PREPROC_CFG["geo_cols"])
    preprocessor, X_tratado, lim_inf, lim_sup = escalonar_com_column_transformer(
        X_modelo,
        cols_numericas=PREPROC_CFG["numerical_model_cols"],
        lower_quantile=PREPROC_CFG["clipping"]["lower_quantile"],
        upper_quantile=PREPROC_CFG["clipping"]["upper_quantile"],
        retornar_limites=True,
    )
    print(f"  X_tratado: {X_tratado.shape} | colunas: {list(X_tratado.columns)}")

    # 2. Treinamento K-Means
    print(f"\n[2/5] Treinamento K-Means (Final K={KMEANS_K})")
    k_range_elbow = list(range(*MOD_CFG["kmeans"]["elbow_k_range"]))
    inercias_modelo, inercias_tratado = curva_elbow(
        X_modelo, X_tratado, k_range_elbow, RANDOM_STATE
    )
    plot_elbow_comparativo(k_range_elbow, inercias_modelo, inercias_tratado)

    df_metricas = avaliar_kmeans_range(
        X_tratado,
        range(*MOD_CFG["kmeans"]["metrics_k_range"]),
        SAMPLE_SIZE_SILHOUETTE,
        RANDOM_STATE,
    )
    print(df_metricas.to_string(index=False))
    df_metricas.to_csv(os.path.join(DIR_METRICS, "kmeans_metrics.csv"), index=False)
    plot_metricas_validacao(df_metricas)

    km_final, labels_k8 = treinar_kmeans(X_tratado, KMEANS_K, RANDOM_STATE)

    plot_distribuicao_clusters(
        labels_k8,
        titulo=f"Distribuicao de Registros por Cluster (K={KMEANS_K})",
        nome_arquivo="02_kmeans_distribuicao.png",
    )

    X_pca, var_ind, var_acum = projetar_pca(
        X_tratado, n_components=MOD_CFG["pca"]["n_components"], random_state=RANDOM_STATE
    )
    print(
        f"  PCA: var. individual = {np.round(var_ind * 100, 2)}% | acumulada = {var_acum * 100:.2f}%"
    )
    X_umap = projetar_umap(
        X_tratado,
        n_components=MOD_CFG["umap"]["n_components"],
        n_neighbors=MOD_CFG["umap"]["n_neighbors"],
        min_dist=MOD_CFG["umap"]["min_dist"],
        random_state=RANDOM_STATE,
    )

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Cluster"] = labels_k8
    df_umap = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
    df_umap["Cluster"] = labels_k8

    plot_clusters_2d_plotly(
        df_umap,
        "UMAP1",
        "UMAP2",
        mapa_cores=CORES_KMEANS,
        titulo=f"Projecao 2D (UMAP): Clusters do K-Means (K={KMEANS_K})",
        nome_arquivo="02_kmeans_umap.png",
    )
    plot_clusters_2d_plotly(
        df_pca,
        "PC1",
        "PC2",
        mapa_cores=CORES_KMEANS,
        titulo=f"Projecao 2D (PCA): Clusters do K-Means (K={KMEANS_K})",
        nome_arquivo="02_kmeans_pca.png",
    )

    pipeline_path = os.path.join(DIR_ARTIFACTS, ARTIFACTS_CFG["kmeans_model_filename"])
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "kmeans": km_final,
            "clip_lower": lim_inf,
            "clip_upper": lim_sup,
            "colunas_treino": X_tratado.columns.tolist(),
        },
        pipeline_path,
    )
    print(f"  Pipeline salvo em: {pipeline_path}")

    # 3. Experimento de distâncias
    print("\n[3/5] Experimento-chave de distancias")
    tabela_dist = experimento_distancias(X_modelo, X_tratado, 0, 10)
    print(tabela_dist.to_string(index=False))

    # 4. Perfilamento K-Means
    print(f"\n[4/5] Perfilamento comercial do K-Means (K={KMEANS_K})")
    X_analise["Cluster_KMeans"] = labels_k8

    perfil_km, distritos_km = gerar_perfil_clusters(
        X_analise, "Cluster_KMeans", BAIRROS_DUMMIES
    )
    comp_bairros, comp_room = composicao_cruzada(
        X_analise, "Cluster_KMeans", BAIRROS_DUMMIES
    )

    print("\nPerfil quantitativo:")
    print(perfil_km.to_string())
    print("\nComposicao por bairros (%):")
    print(distritos_km.to_string())

    perfil_km.to_csv(os.path.join(DIR_METRICS, "perfil_kmeans_k8.csv"))
    distritos_km.to_csv(os.path.join(DIR_METRICS, "composicao_bairros_kmeans_k8.csv"))

    with open(os.path.join(DIR_ARTIFACTS, "perfis_kmeans.json"), "w") as f:
        json.dump(perfil_km.to_dict(orient="index"), f, indent=2, default=str)

    # 5. Validação de inferência
    print("\n[5/5] Validacao do pipeline de inferencia")
    col_sample = X_modelo.head(5).copy()
    cols_num = PREPROC_CFG["numerical_model_cols"]
    col_sample[cols_num] = col_sample[cols_num].clip(lower=lim_inf, upper=lim_sup, axis=1)
    col_sample_tratado = preprocessor.transform(col_sample)[X_modelo.columns]
    preds = km_final.predict(col_sample_tratado)
    print(f"  Inferencia de exemplo (5 anuncios): {preds.tolist()}")

    print("\n" + "=" * 70)
    print("  PIPELINE CONCLUIDO COM SUCESSO.")
    print(f"  Artifacts: {DIR_ARTIFACTS}/")
    print(f"  Metrics:   {DIR_METRICS}/")
    print(f"  Figures:   {DIR_FIGURES}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
