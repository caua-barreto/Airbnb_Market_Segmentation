"""
scripts/dbscan_experiments.py
-----------------------------
Executa os experimentos comparativos de densidade (DBSCAN e HDBSCAN)
parametrizados pelo configs.yaml e salva os diagnósticos em reports/metrics/.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import carregar_configs
from src.preprocessing import (
    carregar_dados_brutos,
    aplicar_feature_engineering,
    construir_X_modelo,
    escalonar_com_column_transformer,
)
from src.modeling import (
    avaliar_grade_dbscan,
    avaliar_grade_hdbscan,
)
from src.plot_figs import (
    plot_metricas_dbscan,
    plot_metricas_hdbscan,
)

CFG = carregar_configs()
DATA_CFG = CFG["data"]
PREPROC_CFG = CFG["preprocessing"]
MOD_CFG = CFG["modeling"]
ARTIFACTS_CFG = CFG["artifacts"]

DIR_RAW = os.path.join(PROJECT_ROOT, DATA_CFG["raw_dir"])
DIR_METRICS = os.path.join(PROJECT_ROOT, ARTIFACTS_CFG["metrics_dir"])
os.makedirs(DIR_METRICS, exist_ok=True)


def main():
    print("[dbscan_experiments] Iniciando experimentos de densidade (configs.yaml)...")

    # 1. Pré-processamento
    caminho_raw = os.path.join(DIR_RAW, DATA_CFG["raw_filename"])
    df = carregar_dados_brutos(caminho_raw, min_price=PREPROC_CFG["min_price_filter"])
    df_proc = aplicar_feature_engineering(df, cfg=CFG)
    _, X_modelo = construir_X_modelo(df_proc, PREPROC_CFG["geo_cols"])
    _, X_tratado = escalonar_com_column_transformer(
        X_modelo,
        cols_numericas=PREPROC_CFG["numerical_model_cols"],
        lower_quantile=PREPROC_CFG["clipping"]["lower_quantile"],
        upper_quantile=PREPROC_CFG["clipping"]["upper_quantile"],
    )

    # 2. DBSCAN — grade do configs.yaml
    print("\n--- DBSCAN ---")
    grade_eps = MOD_CFG["dbscan"]["grade_eps"]
    grade_min_samples_db = MOD_CFG["dbscan"]["grade_min_samples"]

    df_dbscan = avaliar_grade_dbscan(
        X_tratado,
        grade_eps=grade_eps,
        grade_min_samples=grade_min_samples_db,
        sample_size=MOD_CFG["subsample_silhouette"],
        random_state=MOD_CFG["random_state"],
    )
    df_dbscan.to_csv(os.path.join(DIR_METRICS, "dbscan_grid_results.csv"), index=False)
    plot_metricas_dbscan(df_dbscan)

    # 3. HDBSCAN — grade do configs.yaml
    print("\n--- HDBSCAN ---")
    grade_min_cluster_size = MOD_CFG["hdbscan"]["grade_min_cluster_size"]
    grade_min_samples_hdb = MOD_CFG["hdbscan"]["grade_min_samples"]

    df_hdbscan = avaliar_grade_hdbscan(
        X_tratado,
        grade_min_cluster_size=grade_min_cluster_size,
        grade_min_samples=grade_min_samples_hdb,
        sample_size=MOD_CFG["subsample_silhouette"],
        random_state=MOD_CFG["random_state"],
    )
    df_hdbscan.to_csv(os.path.join(DIR_METRICS, "hdbscan_grid_results.csv"), index=False)
    plot_metricas_hdbscan(df_hdbscan)

    # 4. Resumo comparativo
    best_mcs = MOD_CFG["hdbscan"]["best_min_cluster_size"]
    best_ms = MOD_CFG["hdbscan"]["best_min_samples"]
    with open(os.path.join(DIR_METRICS, "densidade_resumo.txt"), "w") as f:
        f.write("=== EXPERIMENTOS DE DENSIDADE: DBSCAN vs HDBSCAN ===\n\n")
        f.write("--- DBSCAN (top 5 por Davies-Bouldin) ---\n")
        f.write(df_dbscan.sort_values(by="Davies_Bouldin").head(5).to_string(index=False))
        f.write("\n\n--- HDBSCAN (top 5 por Davies-Bouldin) ---\n")
        f.write(df_hdbscan.sort_values(by="Davies_Bouldin").head(5).to_string(index=False))
        f.write("\n\nConclusao: DBSCAN rejeitado por densidade heterogenea dos bairros de NYC.\n")
        f.write(
            f"HDBSCAN (min_cluster_size={best_mcs}, min_samples={best_ms}) "
            "selecionado como solucao baseada em densidade.\n"
        )

    print(f"[dbscan_experiments] Experimentos concluidos. Relatorios em {DIR_METRICS}/.")


if __name__ == "__main__":
    main()
