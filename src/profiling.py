"""
src/profiling.py
----------------
Funcoes de perfilamento dos clusters: tabelas executivas por grupo,
composicao percentual por room_type e neighbourhood_group,
e linha de referencia com a composicao global do dataset.
"""

import numpy as np
import pandas as pd
from src.config import carregar_configs

CFG = carregar_configs()
DEFAULT_BAIRROS_DUMMIES = CFG["preprocessing"]["bairros_dummies"]
DEFAULT_COLLINEARITY_LIMIT = CFG["preprocessing"]["collinearity_threshold"]


def gerar_perfil_clusters(X_analise, col_cluster, bairros_dummies=None):
    if bairros_dummies is None:
        bairros_dummies = DEFAULT_BAIRROS_DUMMIES
    """
    Gera a tabela executiva de perfilamento por cluster:
    - Contagem e participacao (%)
    - Mediana e quartis (Q1, Q3) de price
    - Medianas de availability_365, minimum_nights, number_of_reviews,
      reviews_per_month, calculated_host_listings_count,
      distancia_Times_Square_km, custo_estadia, meses_ativos
    - Moda de room_type

    Parametros
    ----------
    X_analise : pd.DataFrame
        Base em escala real com todas as variaveis de negocio.
    col_cluster : str
        Nome da coluna de rotulo do cluster.
    bairros_dummies : list of str
        Lista com nomes das colunas dummy de bairro.

    Retorna
    -------
    tuple of pd.DataFrame
        (perfil_quantitativo, distribuicao_distritos)
    """
    # Mapeamento flexivel para acomodar diferentes convencoes de nome
    c_dist = 'distancia_Times_Square' if 'distancia_Times_Square' in X_analise.columns else 'distancia_Times_Square_km'
    c_meses = 'meses_ativo' if 'meses_ativo' in X_analise.columns else 'meses_ativos'
    c_vol = 'volume_oferta_dias_anfitriao'

    mapeamento_cols = {
        'price': 'price',
        'availability_365': 'availability_365',
        'minimum_nights': 'minimum_nights',
        'number_of_reviews': 'number_of_reviews',
        'reviews_per_month': 'reviews_per_month',
        'calculated_host_listings_count': 'calculated_host_listings_count',
        'distancia_Times_Square_km': c_dist,
        'custo_estadia': 'custo_estadia',
        'meses_ativos': c_meses,
        'volume_oferta_dias_anfitriao': c_vol,
        'room_type': 'room_type',
    }

    agg_dict = {'Tamanho_do_Cluster': ('price', 'count')}

    for nome_col, col_df in mapeamento_cols.items():
        if col_df not in X_analise.columns:
            continue
        if nome_col == 'room_type':
            agg_dict['room_type_Moda'] = (col_df, lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        elif nome_col == 'price':
            agg_dict['Preco_Mediano'] = (col_df, 'median')
            agg_dict['Preco_Q1'] = (col_df, lambda x: x.quantile(0.25))
            agg_dict['Preco_Q3'] = (col_df, lambda x: x.quantile(0.75))
        else:
            nome_exibicao = _nome_exibicao(nome_col)
            agg_dict[nome_exibicao] = (col_df, 'median')

    perfil = X_analise.groupby(col_cluster).agg(**agg_dict).round(2)
    perfil['Perc_Base (%)'] = ((perfil['Tamanho_do_Cluster'] / len(X_analise)) * 100).round(2)

    # Distribuicao percentual por distrito (neighbourhood_group)
    if bairros_dummies and all(c in X_analise.columns for c in bairros_dummies):
        comp_bairros = X_analise.groupby(col_cluster)[bairros_dummies].mean().mul(100).round(1)
        if 'neighbourhood_group_Bronx' not in comp_bairros.columns:
            comp_bairros['neighbourhood_group_Bronx'] = (
                (100.0 - comp_bairros.sum(axis=1)).clip(lower=0.0).round(1)
            )
    else:
        comp_bairros = pd.DataFrame()

    # Percentual de ruido (apenas para modelos baseados em densidade)
    if -1 in perfil.index:
        n_ruido = perfil.loc[-1, 'Tamanho_do_Cluster'] if -1 in perfil.index else 0
        print(f'[profiling] Ruido (cluster -1): {n_ruido} anuncios ({n_ruido / len(X_analise) * 100:.2f}% da base)')

    return perfil, comp_bairros


def composicao_cruzada(X_analise, col_cluster, bairros_dummies=None):
    if bairros_dummies is None:
        bairros_dummies = DEFAULT_BAIRROS_DUMMIES
    """
    Gera tabelas de composicao percentual dos clusters por room_type
    e neighbourhood_group, incluindo linha de referencia com a
    composicao global do dataset.

    Parametros
    ----------
    X_analise : pd.DataFrame
        Base em escala real.
    col_cluster : str
        Nome da coluna de rotulo do cluster.
    bairros_dummies : list of str
        Nomes das colunas dummy de bairro.

    Retorna
    -------
    tuple of pd.DataFrame
        (composicao_bairros, composicao_room_type)
    """
    # Bairros
    if all(c in X_analise.columns for c in bairros_dummies):
        comp_bairros = X_analise.groupby(col_cluster)[bairros_dummies].mean().mul(100).round(1)

        ref_bairros = X_analise[bairros_dummies].mean().mul(100).round(1)
        ref_bairros.name = 'Global (%)'
        comp_bairros = pd.concat([comp_bairros, ref_bairros.to_frame().T])
    else:
        comp_bairros = pd.DataFrame()

    # Room type
    if 'room_type' in X_analise.columns:
        room_dummies = pd.get_dummies(X_analise['room_type'], prefix='room_type', dtype=int)
        room_dummies[col_cluster] = X_analise[col_cluster].values
        comp_room = room_dummies.groupby(col_cluster).mean().mul(100).round(1)

        ref_room = room_dummies.drop(columns=[col_cluster]).mean().mul(100).round(1)
        ref_room.name = 'Global (%)'
        comp_room = pd.concat([comp_room, ref_room.to_frame().T])
    else:
        comp_room = pd.DataFrame()

    return comp_bairros, comp_room


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------
def _nome_exibicao(col):
    """Converte nome interno de coluna para nome de exibicao no perfil."""
    mapa = {
        'availability_365': 'Disponibilidade_Mediana',
        'minimum_nights': 'Estadia_Minima_Mediana',
        'number_of_reviews': 'Reviews_Total_Mediana',
        'reviews_per_month': 'Reviews_Mes_Mediana',
        'calculated_host_listings_count': 'Anuncios_Anfitriao_Mediana',
        'distancia_Times_Square_km': 'Distancia_TS_km_Mediana',
        'custo_estadia': 'Custo_Estadia_Mediano',
        'meses_ativos': 'Meses_Ativos_Mediana',
        'volume_oferta_dias_anfitriao': 'Volume_Oferta_Host',
    }
    return mapa.get(col, f'{col}_Mediana')

# ---------------------------------------------------------------------------
# Estatisticas descritivas para EDA
# ---------------------------------------------------------------------------
def resumo_estatistico(df, coluna='price', monetario=False):
    """
    Calcula as principais estatisticas descritivas de uma variavel numerica,
    incluindo quartis, IQR, assimetria e curtose.
    """
    serie = df[coluna].dropna()
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    fmt = lambda v: f"${v:,.2f}" if monetario else f"{v:,.2f}"
    metricas = {
        'Metrica': [
            'Media', 'Mediana', 'Desvio Padrao', 'Minimo',
            '1o Quartil (Q1 - 25%)', '3o Quartil (Q3 - 75%)',
            'Intervalo Interquartil (IQR)', 'Maximo',
            'Assimetria (Skewness)', 'Curtose (Kurtosis)',
        ],
        'Valor': [
            fmt(serie.mean()), fmt(serie.median()), fmt(serie.std()),
            fmt(serie.min()), fmt(q1), fmt(q3), fmt(iqr), fmt(serie.max()),
            f"{serie.skew():.2f}", f"{serie.kurt():.2f}",
        ],
    }
    return pd.DataFrame(metricas)


def resumo_estatistico_numerico(df):
    """
    Calcula as principais estatisticas descritivas para TODAS as variaveis
    numericas do DataFrame, retornando uma matriz comparativa.
    """
    df_num = df.select_dtypes(include=[np.number])
    resultados = {}
    for coluna in df_num.columns:
        serie = df_num[coluna].dropna()
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        resultados[coluna] = [
            serie.mean(), serie.median(), serie.std(), serie.min(),
            q1, q3, iqr, serie.max(), serie.skew(), serie.kurt(),
        ]
    metricas = [
        'Media', 'Mediana', 'Desvio Padrao', 'Minimo',
        '1o Quartil (Q1 - 25%)', '3o Quartil (Q3 - 75%)',
        'Intervalo Interquartil (IQR)', 'Maximo',
        'Assimetria (Skewness)', 'Curtose (Kurtosis)',
    ]
    df_resumo = pd.DataFrame(resultados, index=metricas)
    if hasattr(df_resumo, 'map'):
        return df_resumo.map(lambda x: f"{x:,.2f}")
    else:
        return df_resumo.applymap(lambda x: f"{x:,.2f}")


def analisar_multicolinearidade_pares(df, limite=None):
    if limite is None:
        limite = DEFAULT_COLLINEARITY_LIMIT
    """
    Identifica pares de variaveis numericas com correlacao linear
    absoluta superior ao limite estabelecido.
    """
    matriz_corr = df.select_dtypes(include=[np.number]).corr().abs()
    triangulo_sup = matriz_corr.where(
        np.triu(np.ones(matriz_corr.shape), k=1).astype(bool)
    )
    pares_altos = [
        (linha, coluna, triangulo_sup.loc[linha, coluna])
        for coluna in triangulo_sup.columns
        for linha in triangulo_sup.index
        if triangulo_sup.loc[linha, coluna] > limite
    ]
    print(f'--- Pares com Correlacao > {limite} ---')
    if not pares_altos:
        print('Nenhum par encontrado acima do limite.')
    for par in sorted(pares_altos, key=lambda x: x[2], reverse=True):
        print(f'{par[0]} & {par[1]}: {par[2]:.3f}')
