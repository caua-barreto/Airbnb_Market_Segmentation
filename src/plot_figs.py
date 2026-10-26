"""
src/plot_figs.py
----------------
Funcoes de visualizacao para EDA, modelagem e perfilamento.
Todas salvam automaticamente em reports/figures/ com dpi=300 e bbox_inches='tight'."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
from typing import Union, List, Optional, Tuple, Dict
from src.config import carregar_configs, PROJECT_ROOT

CFG = carregar_configs()
FIG_DIR = str(PROJECT_ROOT / CFG["artifacts"]["figures_dir"])
os.makedirs(FIG_DIR, exist_ok=True)
DEFAULT_TILES_URL = CFG["visualization"]["carto_tiles_url"]
LANDMARK_CFG = CFG["preprocessing"]["reference_landmark"]


def _salvar_figura(nome_arquivo, dpi=300):
    """Salva a figura atual em reports/figures/ com alta resolucao.
    NAO fecha a figura — o caller (plt.show()) cuida da exibicao."""
    caminho = os.path.join(FIG_DIR, nome_arquivo)
    plt.savefig(caminho, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f'[figures] {nome_arquivo} salvo.')


def _salvar_plotly(fig, nome_arquivo):
    """Salva figura Plotly como HTML e PNG em figures/."""
    caminho_html = os.path.join(FIG_DIR, nome_arquivo.replace('.png', '.html'))
    caminho_png = os.path.join(FIG_DIR, nome_arquivo)
    fig.write_html(caminho_html)
    try:
        fig.write_image(caminho_png, scale=3)
    except Exception:
        pass  # kaleido nao disponivel
    print(f'[figures] {nome_arquivo} salvo (HTML).')


# ---------------------------------------------------------------------------
# EDA — Distribuicoes e Boxplots
# ---------------------------------------------------------------------------
def histograma_simples(
    df, coluna='price', bins='sturges',
    titulo=None, xlabel=None, ylabel='Frequencia',
    color='steelblue', figsize=(10, 6),
    nome_arquivo=None,
):
    dados = df[coluna].dropna()
    media, mediana = dados.mean(), dados.median()
    modas = dados.mode()
    moda = modas[0] if not modas.empty else None

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={'height_ratios': (1, 4)}, figsize=figsize,
    )
    if titulo:
        fig.suptitle(titulo, fontsize=14, fontweight='bold')

    sns.boxplot(x=dados, ax=ax_box, color=color, fliersize=4)
    ax_box.axvline(media, color='darkorange', linestyle='--', linewidth=1.5)
    ax_box.axvline(mediana, color='forestgreen', linestyle='--', linewidth=1.5)
    if moda is not None:
        ax_box.axvline(moda, color='firebrick', linestyle='--', linewidth=1.5)
    ax_box.set(xlabel='')
    ax_box.grid(True, linestyle='--', alpha=0.7)

    sns.histplot(x=dados, bins=bins, kde=True, ax=ax_hist,
                 color=color, edgecolor='black', alpha=0.6)
    ax_hist.axvline(media, color='darkorange', linestyle='--', linewidth=2,
                    label=f'Media: {media:.2f}')
    ax_hist.axvline(mediana, color='forestgreen', linestyle='--', linewidth=2,
                    label=f'Mediana: {mediana:.2f}')
    if moda is not None:
        ax_hist.axvline(moda, color='firebrick', linestyle='--', linewidth=2,
                        label=f'Moda: {moda:.2f}')
    ax_hist.legend(frameon=True, fontsize=10)
    ax_hist.set_xlabel(xlabel if xlabel else coluna, fontsize=12)
    ax_hist.set_ylabel(ylabel, fontsize=12)
    ax_hist.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if titulo:
        plt.subplots_adjust(top=0.92)
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()


def plot_boxplot_outliers(
    df, coluna='price', monetario=False,
    cor='#D65F5F', figsize=(12, 5), nome_arquivo=None,
):
    serie = df[coluna].dropna()
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    limite_superior = q3 + 1.5 * iqr
    limite_inferior = q1 - 1.5 * iqr
    outliers_sup = serie[serie > limite_superior]
    outliers_inf = serie[serie < limite_inferior]
    total_outliers = len(outliers_sup) + len(outliers_inf)

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=serie, color=cor, width=0.4, linewidth=1.5, ax=ax,
                flierprops={'marker': 'o', 'markerfacecolor': 'white',
                            'markeredgecolor': '#4A4A4A', 'alpha': 0.5, 'markersize': 5})
    ax.axvline(limite_superior, color='#2B2B2B', linestyle='--', linewidth=1.5, alpha=0.7)
    offset = (serie.max() - serie.min()) * 0.015
    texto = f'Limite Superior\n${limite_superior:,.2f}' if monetario else f'Limite Superior\n{limite_superior:,.2f}'
    ax.text(limite_superior + offset, -0.28, texto, color='#2B2B2B', fontsize=10, va='center')
    if monetario:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'${x/1000:,.1f}K' if abs(x) >= 1000 else f'${x:,.0f}')
        )
    ax.set_title(f'Distribuicao de {coluna} e Deteccao de Outliers', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(f'{coluna} ($)' if monetario else coluna, fontsize=12, labelpad=10)
    ax.set_yticks([])
    sns.despine(left=True)
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()

    print(f'--- Diagnostico de Outliers: {coluna} ---')
    print(f'Limite Superior (Q3 + 1.5*IQR): {limite_superior:,.2f}')
    print(f'Limite Inferior (Q1 - 1.5*IQR): {limite_inferior:,.2f}')
    print(f'Outliers Superiores: {len(outliers_sup)} | Inferiores: {len(outliers_inf)}')
    print(f'Total: {total_outliers} ({(total_outliers / len(serie)) * 100:.2f}%)')


def plot_distribuicao(
    df, coluna='price', transformacao='original',
    cor=None, bins=50, figsize=(10, 6), nome_arquivo=None,
):
    serie = df[coluna].dropna()
    padrao_cores = {'original': 'royalblue', 'sqrt': 'darkorange',
                    'log': 'forestgreen', 'log1p': 'seagreen', 'cbrt': 'purple'}
    t = (transformacao or 'original').lower()

    if t in ('original', 'none', 'nenhuma'):
        dados, label_trans = serie, 'Original'
    elif t == 'sqrt':
        dados, label_trans = np.sqrt(serie), 'Raiz Quadrada (Sqrt)'
    elif t == 'log':
        dados, label_trans = np.log(serie), 'Logaritmo Natural (log)'
    elif t == 'log1p':
        dados, label_trans = np.log1p(serie), 'Logaritmo Natural (log1p)'
    elif t == 'cbrt':
        dados, label_trans = np.cbrt(serie), 'Raiz Cubica (cbrt)'
    else:
        raise ValueError(f"Transformacao '{transformacao}' invalida.")

    cor_final = cor if cor else padrao_cores.get(t, 'royalblue')

    plt.figure(figsize=figsize)
    sns.histplot(dados, kde=True, stat='density', color=cor_final,
                 bins=bins, edgecolor='black', alpha=0.6, label='Densidade Real (KDE)')
    mu, std = stats.norm.fit(dados)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=2, label=f'Ajuste Normal ($\\mu$={mu:,.2f}, $\\sigma$={std:,.2f})')
    plt.title(f'Distribuicao de {coluna} ({label_trans}) com Ajuste Normal', fontsize=13, fontweight='bold')
    plt.xlabel(f'{coluna} ({label_trans})', fontsize=11)
    plt.ylabel('Densidade', fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()


def plot_matriz_correlacao(
    df, titulo='Matriz de Correlacao', triangulo=True,
    cmap='coolwarm', figsize=(14, 10), annot_size=9, nome_arquivo=None,
):
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr()
    mascara = None
    if triangulo in (True, 'inferior'):
        mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    elif triangulo == 'superior':
        mascara = np.tril(np.ones_like(corr, dtype=bool), k=-1)

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, vmin=-1, vmax=1,
                center=0, square=True, linewidths=.5, cbar_kws={'shrink': .75},
                mask=mascara, annot_kws={'size': annot_size})
    plt.title(titulo, fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()


# ---------------------------------------------------------------------------
# Clusterizacao — Elbow, Metricas, Distribuicao
# ---------------------------------------------------------------------------
def plot_elbow_comparativo(
    k_valores, inercias_orig, inercias_trat,
    titulos=('Metodo Elbow: Dados Originais (X_modelo)', 'Metodo Elbow: Dados Tratados (X_tratado)'),
    cores=('#7F8C8D', '#C00000'), figsize=(15, 6), nome_arquivo='02_kmeans_elbow.png',
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    dados = [(inercias_orig, cores[0], titulos[0]), (inercias_trat, cores[1], titulos[1])]
    for ax, (inercias, cor, titulo) in zip(axes, dados):
        ax.plot(k_valores, inercias, marker='o', markersize=6, color=cor, linewidth=2)
        ax.set_title(titulo, fontsize=13, fontweight='bold', color='#333333', pad=12)
        ax.set_xlabel('Numero de Clusters (K)', fontsize=11, fontweight='bold', color='#4A4A4A')
        if ax == axes[0]:
            ax.set_ylabel('Inercia (WCSS)', fontsize=11, fontweight='bold', color='#4A4A4A')
        ax.set_xticks(k_valores)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#D5D8DC')
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()


def plot_metricas_validacao(
    df_resultados, coluna_k='K',
    metricas=('Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz'),
    titulos=None, cores=('#2E86C1', '#C00000', '#28B463'),
    suptitle='Metricas de Validacao para cada K (Subsample de 10k pontos)',
    figsize=(16, 5), nome_arquivo='02_kmeans_metricas.png',
):
    if titulos is None:
        titulos = metricas
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(suptitle, fontsize=18, fontweight='bold', color='#1C1C1C', y=1.05)
    for i, (metrica, titulo, cor) in enumerate(zip(metricas, titulos, cores)):
        ax = axes[i]
        ax.plot(df_resultados[coluna_k], df_resultados[metrica],
                marker='o', linewidth=2.5, color=cor, markersize=8,
                markerfacecolor='white', markeredgewidth=2)
        ax.set_title(titulo, fontsize=14, fontweight='bold', color='#1C1C1C', pad=15)
        ax.set_xlabel('Numero de Clusters (K)', fontsize=12, fontweight='bold', color='#4A4A4A')
        ax.set_xticks(df_resultados[coluna_k])
        ax.tick_params(axis='both', colors='#4A4A4A', labelsize=11)
        sns.despine(ax=ax)
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()


def plot_distribuicao_clusters(
    labels, titulo='Distribuicao de Registros por Cluster',
    xlabel='ID do Cluster', ylabel='Volume de Registros',
    limiar_destaque=10000, cor_majoritario='#7F8C8D', cor_nicho='#C00000',
    figsize=(10, 5), nome_arquivo=None,
):
    clusters_ids, contagens = np.unique(labels, return_counts=True)
    cores = [cor_nicho if v < limiar_destaque else cor_majoritario for v in contagens]
    fig, ax = plt.subplots(figsize=figsize)
    barras = ax.bar(clusters_ids, contagens, color=cores, edgecolor='none')
    ax.set_title(titulo, fontsize=14, fontweight='bold', color='#1C1C1C', pad=20)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold', color='#4A4A4A')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='#4A4A4A')
    ax.bar_label(barras, fmt='{:.0f}', padding=5, color='#333333', fontweight='bold', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    ax.set_xticks(clusters_ids)
    ax.tick_params(axis='both', which='major', labelsize=10, color='#4A4A4A')
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#D5D8DC')
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()


def plot_clusters_2d_plotly(
    df_proj, col_x, col_y, col_cluster='Cluster',
    mapa_cores=None, n_amostra=5000, titulo='Projecao 2D dos Clusters',
    width=900, height=600, random_state=42, nome_arquivo=None,
):
    df_amostra = df_proj.sample(n=min(n_amostra, len(df_proj)), random_state=random_state).copy()
    df_amostra[col_cluster] = df_amostra[col_cluster].astype(str)
    mapa_str = {str(k): v for k, v in mapa_cores.items()} if mapa_cores else None
    fig = px.scatter(
        df_amostra, x=col_x, y=col_y, color=col_cluster,
        color_discrete_map=mapa_str, opacity=0.7, render_mode='svg',
        title=titulo,
        hover_data={col_x: False, col_y: False, col_cluster: True},
    )
    fig.update_layout(
        plot_bgcolor='white',
        title_font=dict(size=18, color='#1C1C1C', family='Arial'),
        legend_title_text='ID do Cluster',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=12, color='#4A4A4A')),
        width=width, height=height, margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F2F3F4',
                     title_font=dict(size=12, color='#4A4A4A'), zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F2F3F4',
                     title_font=dict(size=12, color='#4A4A4A'), zeroline=False)
    if nome_arquivo:
        _salvar_plotly(fig, nome_arquivo)
    fig.show()



# ---------------------------------------------------------------------------
# DBSCAN / HDBSCAN — Metricas
# ---------------------------------------------------------------------------
def plot_metricas_dbscan(df_dbscan, nome_arquivo='02_dbscan_metricas.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Metricas de Validacao e Taxa de Ruido: DBSCAN (por Epsilon e Min Samples)',
                 fontsize=14, fontweight='bold', color='#1C1C1C', y=1.02)
    metricas = ['Silhouette', 'Davies_Bouldin', '%_Ruido']
    titulos = ['Silhouette Score (Maior - Melhor)', 'Davies-Bouldin Index (Menor - Melhor)',
               'Percentual de Ruido (%) na Base']
    for i, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[i]
        sns.lineplot(data=df_dbscan, x='eps', y=metrica, hue='min_samples',
                     marker='o', ax=ax, palette='Dark2')
        ax.set_title(titulo, fontsize=12, fontweight='bold', color='#1C1C1C', pad=10)
        ax.set_xlabel('eps (Raio de Vizinhanca)', fontsize=10, color='#4A4A4A')
        if metrica == 'Silhouette':
            ax.axhline(y=0, color='#7F8C8D', linestyle='--', linewidth=1)
        if metrica == '%_Ruido':
            ax.set_ylim(0, 100)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='#D5D8DC')
        ax.legend(title='min_samples', fontsize=9, title_fontsize=10)
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()


def plot_metricas_hdbscan(df_hdbscan, nome_arquivo='02_hdbscan_metricas.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Metricas de Validacao e Taxa de Ruido: HDBSCAN (por Min Cluster Size e Min Samples)',
                 fontsize=14, fontweight='bold', color='#1C1C1C', y=1.02)
    metricas = ['Silhouette', 'Davies_Bouldin', '%_Ruido']
    titulos = ['Silhouette Score (Maior - Melhor)', 'Davies-Bouldin Index (Menor - Melhor)',
               'Percentual de Ruido (%) na Base']
    for i, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[i]
        sns.lineplot(data=df_hdbscan, x='min_cluster_size', y=metrica, hue='min_samples',
                     marker='o', ax=ax, palette='Dark2')
        ax.set_title(titulo, fontsize=12, fontweight='bold', color='#1C1C1C', pad=10)
        ax.set_xlabel('min_cluster_size', fontsize=10, color='#4A4A4A')
        if metrica == 'Silhouette':
            ax.axhline(y=0, color='#7F8C8D', linestyle='--', linewidth=1)
        if metrica == '%_Ruido':
            ax.set_ylim(0, 100)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='#D5D8DC')
        ax.legend(title='min_samples', fontsize=9, title_fontsize=10)
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()


# ---------------------------------------------------------------------------
# Geoespacial
# ---------------------------------------------------------------------------
def plot_mapa_clusters_nyc(X_geo, labels, titulo, url_tiles_carto=None,
                           nome_arquivo='05_mapa_hdbscan_nyc.png'):
    import geopandas as gpd
    import contextily as ctx
    from shapely.geometry import Point

    if url_tiles_carto is None:
        url_tiles_carto = DEFAULT_TILES_URL

    df_mapa = X_geo.copy()
    df_mapa['Cluster'] = labels
    gdf = gpd.GeoDataFrame(df_mapa,
                           geometry=gpd.points_from_xy(df_mapa['longitude'], df_mapa['latitude']),
                           crs='EPSG:4326')
    gdf_mercator = gdf.to_crs(epsg=3857)
    clusters_validos = sorted([c for c in df_mapa['Cluster'].unique() if c != -1 and c != '6. Ruído Estruturado'])
    paleta_hex = sns.color_palette('Dark2', n_colors=max(len(clusters_validos), 1)).as_hex()
    mapa_cores = {c: paleta_hex[i] for i, c in enumerate(clusters_validos)}

    fig, ax = plt.subplots(figsize=(14, 11))
    gdf_ruido = gdf_mercator[(gdf_mercator['Cluster'] == -1) | (gdf_mercator['Cluster'] == '6. Ruído Estruturado')]
    if not gdf_ruido.empty:
        gdf_ruido.plot(ax=ax, color='#D5D8DC', markersize=3, alpha=0.3, label='Ruido (-1)')
    for c_id in clusters_validos:
        gdf_c = gdf_mercator[gdf_mercator['Cluster'] == c_id]
        rotulo = f'Cluster {c_id}' if isinstance(c_id, (int, np.integer)) else str(c_id)
        gdf_c.plot(ax=ax, color=mapa_cores[c_id], markersize=8, alpha=0.8, label=rotulo)
    ts_lon, ts_lat = LANDMARK_CFG['lon'], LANDMARK_CFG['lat']
    ts_point = gpd.GeoSeries([Point(ts_lon, ts_lat)], crs='EPSG:4326').to_crs(epsg=3857)
    ts_point.plot(ax=ax, color='black', marker='*', markersize=350, edgecolor='white', zorder=5)
    ax.scatter([], [], color='black', marker='*', s=120, edgecolor='white', label=LANDMARK_CFG['name'])
    ctx.add_basemap(ax, source=url_tiles_carto)
    ax.set_title(titulo, fontsize=18, fontweight='bold', color='#1C1C1C', pad=20)
    ax.set_axis_off()
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='white',
              edgecolor='#BDC3C7', title='Legenda', title_fontproperties={'weight': 'bold'},
              markerscale=1.8, labelspacing=0.8, handletextpad=0.8, borderpad=0.8)
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()


# ---------------------------------------------------------------------------
# Violino pos-perfilamento
# ---------------------------------------------------------------------------
def plot_violino_perfil_clusters(
    X_analise, col_cluster, colunas, nomes_clusters=None,
    titulo='Distribuicao dos Atributos Comerciais por Cluster',
    figsize=None, nome_arquivo='02_perfil_violino_kmeans.png',
):
    n = len(colunas)
    n_cols = min(2, n)
    n_rows = (n + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (7 * n_cols, 5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    for i, col in enumerate(colunas):
        ax = axes[i]
        sns.violinplot(data=X_analise, x=col_cluster, y=col, ax=ax,
                       palette='Dark2', inner='quartile', cut=0)
        ax.set_title(col, fontsize=12, fontweight='bold', color='#333333')
        ax.set_xlabel('')
        if nomes_clusters is not None:
            ax.set_xticklabels(nomes_clusters, rotation=45, ha='right', fontsize=9)
        sns.despine(ax=ax)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(titulo, fontsize=16, fontweight='bold', color='#1C1C1C', y=1.01)
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()

def plot_violino_perfis_hdbscan(
    X_analise,
    mapa_perfis,
    col_cluster='Cluster_HDBSCAN',
    features_principais=(
        ('price', 'Preco da Diaria (US$)', 400),
        ('distancia_Times_Square', 'Distancia ate Times Square (km)', None),
        ('availability_365', 'Disponibilidade Anual (Dias)', None),
    ),
    incluir_ruido=False,
    figsize=(18, 6),
    nome_arquivo='02_perfil_violino_hdbscan.png',
):
    """
    Plota graficos de violino (1x3) comparando os Perfis Comerciais
    nas 3 principais variaveis definidoras de cada perfil.
    """
    df_plot = X_analise.copy()
    df_plot['Perfil_Comercial'] = df_plot[col_cluster].map(
        lambda c: mapa_perfis.get(c, '6. Ruido (-1)')
    )
    if not incluir_ruido:
        df_plot = df_plot[df_plot[col_cluster] != -1].copy()
    ordem_perfis = sorted(df_plot['Perfil_Comercial'].unique())
    paleta_perfis = [
        '#2E86C1', '#C00000', '#28B463', '#F39C12', '#8E44AD', '#7F8C8D'
    ][:len(ordem_perfis)]

    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        'Distribuicao de Densidade (Violin Plot) das 3 Principais Dimensoes por Perfil Comercial (HDBSCAN)',
        fontsize=15, fontweight='bold', color='#1C1C1C', y=1.03,
    )
    for ax, (col, label_y, lim_y) in zip(axes, features_principais):
        df_sub = df_plot.copy()
        if lim_y is not None:
            df_sub = df_sub[df_sub[col] <= lim_y]
        sns.violinplot(
            data=df_sub, x='Perfil_Comercial', y=col,
            order=ordem_perfis, palette=paleta_perfis,
            inner='quartile', cut=0, linewidth=1.2, ax=ax,
        )
        ax.set_title(label_y, fontsize=13, fontweight='bold', color='#34495E', pad=12)
        ax.set_xlabel('')
        ax.set_ylabel(label_y, fontsize=11, fontweight='bold', color='#4A4A4A')
        ax.tick_params(axis='x', rotation=30, labelsize=9.5)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')
            label.set_fontweight('bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    if nome_arquivo:
        _salvar_figura(nome_arquivo)
    plt.show()
