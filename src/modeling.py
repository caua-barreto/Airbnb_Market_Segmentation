"""
src/modeling.py
---------------
Funções de treinamento e avaliação dos algoritmos de clusterização (K-Means, DBSCAN e HDBSCAN)
e redução de dimensionalidade (PCA e UMAP), parametrizadas via configs.yaml.
"""

import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.config import carregar_configs

# ---------------------------------------------------------------------------
# Carregamento das configurações de modelagem (configs.yaml)
# ---------------------------------------------------------------------------
CFG = carregar_configs()
MOD_CFG = CFG["modeling"]

DEFAULT_SEED = MOD_CFG["random_state"]
DEFAULT_SUBSAMPLE = MOD_CFG["subsample_silhouette"]
KMEANS_CFG = MOD_CFG["kmeans"]
DBSCAN_CFG = MOD_CFG["dbscan"]
HDBSCAN_CFG = MOD_CFG["hdbscan"]
PCA_CFG = MOD_CFG["pca"]
UMAP_CFG = MOD_CFG["umap"]


# ---------------------------------------------------------------------------
# Experimento-chave de distâncias (antes vs depois do escalonamento)
# ---------------------------------------------------------------------------
def experimento_distancias(X_modelo, X_tratado, idx_a=0, idx_b=10):
    """
    Calcula a distância Euclidiana entre dois anúncios em dois cenários:
    A) Encoding + sem escalonamento (X_modelo)
    B) Encoding + com RobustScaler (X_tratado)
    """
    a_raw = X_modelo.iloc[idx_a]
    b_raw = X_modelo.iloc[idx_b]
    a_sca = X_tratado.iloc[idx_a]
    b_sca = X_tratado.iloc[idx_b]

    delta_antes = (a_raw - b_raw).abs()
    delta_depois = (a_sca - b_sca).abs()
    contrib_antes = (a_raw - b_raw) ** 2
    contrib_depois = (a_sca - b_sca) ** 2

    tabela = pd.DataFrame({
        "Feature": X_modelo.columns,
        "|delta|_antes (encoding, sem escala)": delta_antes.values,
        "|delta|_depois (encoding, RobustScaler)": delta_depois.values,
        "Contrib_Quad_antes": contrib_antes.values,
        "Contrib_Quad_depois": contrib_depois.values,
    })

    dist_antes = np.sqrt(contrib_antes.sum())
    dist_depois = np.sqrt(contrib_depois.sum())
    print(f"Distancia Euclidiana (Cenario A, sem escala): {dist_antes:.4f}")
    print(f"Distancia Euclidiana (Cenario B, RobustScaler): {dist_depois:.4f}")
    return tabela.round(4)


# ---------------------------------------------------------------------------
# K-Means — Elbow + Métricas
# ---------------------------------------------------------------------------
def treinar_kmeans(
    X,
    n_clusters=None,
    random_state=DEFAULT_SEED,
    n_init=None,
    max_iter=None,
):
    """Treina e retorna um KMeans com K-Means++."""
    if n_clusters is None:
        n_clusters = KMEANS_CFG["best_k"]
    if n_init is None:
        n_init = KMEANS_CFG["n_init"]
    if max_iter is None:
        max_iter = KMEANS_CFG["max_iter"]

    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = km.fit_predict(X)
    return km, labels


def curva_elbow(
    X_modelo,
    X_tratado,
    k_range=None,
    random_state=DEFAULT_SEED,
    n_init=None,
    max_iter=None,
):
    """Calcula a curva de inércia para dois cenários (com e sem escalonamento)."""
    if k_range is None:
        k_range = range(*KMEANS_CFG["elbow_k_range"])
    if n_init is None:
        n_init = KMEANS_CFG.get("elbow_n_init", 20)
    if max_iter is None:
        max_iter = KMEANS_CFG["max_iter"]

    inercias_modelo = []
    inercias_tratado = []
    for k in k_range:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        km.fit(X_modelo)
        inercias_modelo.append(km.inertia_)
        km.fit(X_tratado)
        inercias_tratado.append(km.inertia_)
    return inercias_modelo, inercias_tratado


def avaliar_kmeans_range(
    X_tratado,
    k_range=None,
    sample_size=DEFAULT_SUBSAMPLE,
    random_state=DEFAULT_SEED,
    n_init=None,
    max_iter=None,
):
    """Avalia K-Means para uma grade de K, retornando DataFrame com métricas."""
    if k_range is None:
        k_range = range(*KMEANS_CFG["metrics_k_range"])
    if n_init is None:
        n_init = KMEANS_CFG["n_init"]
    if max_iter is None:
        max_iter = KMEANS_CFG["max_iter"]

    resultados = []
    for k in k_range:
        km, labels = treinar_kmeans(X_tratado, k, random_state, n_init, max_iter)
        valores, contagens = np.unique(labels, return_counts=True)
        dist_tam = {int(c): int(q) for c, q in zip(valores, contagens)}
        resultados.append({
            "K": k,
            "Inertia": km.inertia_,
            "Silhouette": silhouette_score(
                X_tratado, labels, sample_size=sample_size, random_state=random_state
            ),
            "Davies_Bouldin": davies_bouldin_score(X_tratado, labels),
            "Calinski_Harabasz": calinski_harabasz_score(X_tratado, labels),
            "Tamanho_dos_Clusters": dist_tam,
        })
    df = pd.DataFrame(resultados)
    for col in ["Inertia", "Silhouette", "Davies_Bouldin", "Calinski_Harabasz"]:
        df[col] = df[col].round(2)
    return df


# ---------------------------------------------------------------------------
# DBSCAN — Grade de Hiperparâmetros (excluindo ruído -1 nas métricas)
# ---------------------------------------------------------------------------
def avaliar_grade_dbscan(
    X_tratado,
    grade_eps=None,
    grade_min_samples=None,
    sample_size=DEFAULT_SUBSAMPLE,
    random_state=DEFAULT_SEED,
):
    """
    Varredura de hiperparâmetros para DBSCAN.
    Calcula Silhouette, Davies-Bouldin e Calinski-Harabasz apenas sobre os clusters
    válidos (labels != -1) para não tratar o ruído como um cluster esférico único.
    """
    if grade_eps is None:
        grade_eps = DBSCAN_CFG["grade_eps"]
    if grade_min_samples is None:
        grade_min_samples = DBSCAN_CFG["grade_min_samples"]

    resultados = []
    print("Iniciando varredura de hiperparametros do DBSCAN...\n")
    for eps in grade_eps:
        for min_samples in grade_min_samples:
            modelo = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = modelo.fit_predict(X_tratado)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            qtd_ruido = list(labels).count(-1)
            perc_ruido = (qtd_ruido / len(labels)) * 100

            mask = labels != -1
            X_valido = X_tratado[mask]
            labels_valido = labels[mask]

            if n_clusters > 1 and len(X_valido) > n_clusters:
                sil = silhouette_score(
                    X_valido,
                    labels_valido,
                    sample_size=min(sample_size, len(X_valido)),
                    random_state=random_state,
                )
                db = davies_bouldin_score(X_valido, labels_valido)
                ch = calinski_harabasz_score(X_valido, labels_valido)
            else:
                sil, db, ch = np.nan, np.nan, np.nan

            resultados.append({
                "eps": eps,
                "min_samples": min_samples,
                "N_Clusters": n_clusters,
                "%_Ruido": round(perc_ruido, 2),
                "Silhouette": sil,
                "Davies_Bouldin": db,
                "Calinski_Harabasz": ch,
            })
            print(
                f"Testado: eps={eps:<4} | min_samples={min_samples:<3} "
                f"-> Clusters: {n_clusters:<2} | Ruido: {perc_ruido:.1f}%"
            )
    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# HDBSCAN — Grade de Hiperparâmetros (excluindo ruído -1 nas métricas)
# ---------------------------------------------------------------------------
def avaliar_grade_hdbscan(
    X_tratado,
    grade_min_cluster_size=None,
    grade_min_samples=None,
    sample_size=DEFAULT_SUBSAMPLE,
    random_state=DEFAULT_SEED,
):
    """
    Varredura de hiperparâmetros para HDBSCAN.
    Calcula Silhouette, Davies-Bouldin e Calinski-Harabasz apenas sobre os clusters
    válidos (labels != -1).
    """
    import hdbscan

    if grade_min_cluster_size is None:
        grade_min_cluster_size = HDBSCAN_CFG["grade_min_cluster_size"]
    if grade_min_samples is None:
        grade_min_samples = HDBSCAN_CFG["grade_min_samples"]

    resultados = []
    print("Iniciando varredura de hiperparametros do HDBSCAN...\n")
    for min_cluster_size in grade_min_cluster_size:
        for min_samples in grade_min_samples:
            modelo = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                core_dist_n_jobs=-1,
            )
            labels = modelo.fit_predict(X_tratado)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            qtd_ruido = list(labels).count(-1)
            perc_ruido = (qtd_ruido / len(labels)) * 100

            mask = labels != -1
            X_valido = X_tratado[mask]
            labels_valido = labels[mask]

            if n_clusters > 1 and len(X_valido) > n_clusters:
                sil = silhouette_score(
                    X_valido,
                    labels_valido,
                    sample_size=min(sample_size, len(X_valido)),
                    random_state=random_state,
                )
                db = davies_bouldin_score(X_valido, labels_valido)
                ch = calinski_harabasz_score(X_valido, labels_valido)
            else:
                sil, db, ch = np.nan, np.nan, np.nan

            resultados.append({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "N_Clusters": n_clusters,
                "%_Ruido": round(perc_ruido, 2),
                "Silhouette": sil,
                "Davies_Bouldin": db,
                "Calinski_Harabasz": ch,
            })
            print(
                f"Testado: min_cluster_size={min_cluster_size:<3} "
                f"| min_samples={min_samples:<2} "
                f"-> Clusters: {n_clusters:<2} | Ruido: {perc_ruido:.1f}%"
            )
    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Redução de Dimensionalidade (PCA e UMAP)
# ---------------------------------------------------------------------------
def projetar_pca(X_tratado, n_components=None, random_state=DEFAULT_SEED):
    """Aplica PCA e retorna array projetado + variância explicada."""
    if n_components is None:
        n_components = PCA_CFG["n_components"]
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_tratado)
    return X_pca, pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum()


def projetar_umap(
    X_tratado,
    n_components=None,
    n_neighbors=None,
    min_dist=None,
    random_state=DEFAULT_SEED,
):
    """Aplica UMAP e retorna array projetado em 2D."""
    import umap

    if n_components is None:
        n_components = UMAP_CFG["n_components"]
    if n_neighbors is None:
        n_neighbors = UMAP_CFG["n_neighbors"]
    if min_dist is None:
        min_dist = UMAP_CFG["min_dist"]

    redutor = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return redutor.fit_transform(X_tratado)
