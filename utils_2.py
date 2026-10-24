import math
from typing import Union, List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import plotly.graph_objects as go


def histograma_simples(
    df: pd.DataFrame, 
    coluna: str = 'price', 
    bins: Union[int, str] = 'sturges',
    titulo: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = 'Frequência',
    color: str = 'steelblue',
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Gera um gráfico combinado com Boxplot (topo) e Histograma com KDE (base),
    destacando as linhas de Média, Mediana e Moda.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str, default='price'
        Nome da coluna numérica a ser analisada.
    bins : int ou str, default='sturges'
        Número de compartimentos do histograma ou método de cálculo (ex: 30, 'auto', 'sturges').
    titulo : str, optional
        Título principal do gráfico.
    xlabel : str, optional
        Rótulo do eixo X. Se None, usa o nome da coluna.
    ylabel : str, default='Frequência'
        Rótulo do eixo Y.
    color : str, default='steelblue'
        Cor base para o boxplot e o histograma.
    figsize : tuple, default=(10, 6)
        Dimensões da figura (largura, altura).
    """
    dados = df[coluna].dropna()
    media = dados.mean()
    mediana = dados.median()
    
    modas = dados.mode()
    moda = modas[0] if not modas.empty else None

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, 
        sharex=True, 
        gridspec_kw={"height_ratios": (1, 4)}, 
        figsize=figsize
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

    sns.histplot(
        x=dados, 
        bins=bins, 
        kde=True, 
        ax=ax_hist, 
        color=color, 
        edgecolor='black', 
        alpha=0.6
    )

    ax_hist.axvline(media, color='darkorange', linestyle='--', linewidth=2, label=f'Média: {media:.2f}')
    ax_hist.axvline(mediana, color='forestgreen', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    if moda is not None:
        ax_hist.axvline(moda, color='firebrick', linestyle='--', linewidth=2, label=f'Moda: {moda:.2f}')

    ax_hist.legend(frameon=True, fontsize=10)
    rotulo_x = xlabel if xlabel is not None else coluna
    ax_hist.set_xlabel(rotulo_x, fontsize=12)
    ax_hist.set_ylabel(ylabel, fontsize=12)
    ax_hist.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    
    if titulo:
        plt.subplots_adjust(top=0.92)
        
    plt.show()


def resumo_estatistico(df: pd.DataFrame, coluna: str = 'price', monetario: bool = False) -> pd.DataFrame:
    """
    Calcula as principais estatísticas descritivas de uma variável numérica,
    incluindo quartis, IQR, assimetria e curtose.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str, default='price'
        Nome da coluna numérica a ser analisada.
    monetario : bool, default=False
        Se True, formata os valores numéricos com o prefixo '$'.
        Se False, formata com duas casas decimais simples.

    Retorna:
    -------
    pd.DataFrame
        Tabela com as métricas e seus respectivos valores.
    """
    serie = df[coluna].dropna()
    
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    
    fmt = lambda v: f"${v:,.2f}" if monetario else f"{v:,.2f}"
    
    metricas = {
        'Métrica': [
            'Média', 
            'Mediana', 
            'Desvio Padrão', 
            'Mínimo', 
            '1º Quartil (Q1 - 25%)', 
            '3º Quartil (Q3 - 75%)', 
            'Intervalo Interquartil (IQR)', 
            'Máximo',
            'Assimetria (Skewness)', 
            'Curtose (Kurtosis)'
        ],
        'Valor': [
            fmt(serie.mean()),
            fmt(serie.median()),
            fmt(serie.std()),
            fmt(serie.min()),
            fmt(q1),
            fmt(q3),
            fmt(iqr),
            fmt(serie.max()),
            f"{serie.skew():.2f}",
            f"{serie.kurt():.2f}"
        ]
    }
    
    return pd.DataFrame(metricas)


def plot_distribuicao(
    df: pd.DataFrame, 
    coluna: str = 'price', 
    transformacao: str = 'original', 
    cor: Optional[str] = None, 
    bins: int = 50, 
    figsize: Tuple[int, int] = (10, 6)
) -> pd.DataFrame:
    """
    Plota a distribuição de uma variável com KDE e ajuste normal teórico,
    e retorna as principais métricas estatísticas da distribuição.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str, default='price'
        Coluna numérica a ser analisada.
    transformacao : str, default='original'
        Tipo de transformação: 'original', 'sqrt', 'log', 'log1p', 'cbrt'.
    cor : str, optional
        Cor do histograma. Aceita nomes ou Hex.
    bins : int, default=50
        Número de bins do histograma.
    figsize : tuple, default=(10, 6)
        Dimensões da figura (largura, altura).
        
    Retorna:
    -------
    pd.DataFrame
        DataFrame contendo Média, Mediana, Desvio Padrão, Assimetria e Curtose.
    """
    serie = df[coluna].dropna()

    padrao_cores = {
        'original': 'royalblue',
        'sqrt': 'darkorange',
        'log': 'forestgreen',
        'log1p': 'seagreen',
        'cbrt': 'purple'
    }

    t = (transformacao or 'original').lower()

    if t in ('original', 'none', 'nenhuma'):
        dados = serie
        label_trans = 'Original'
        cor_final = cor if cor else padrao_cores['original']
    elif t == 'sqrt':
        dados = np.sqrt(serie)
        label_trans = 'Raiz Quadrada (Sqrt)'
        cor_final = cor if cor else padrao_cores['sqrt']
    elif t == 'log':
        dados = np.log(serie)
        label_trans = 'Logaritmo Natural (log)'
        cor_final = cor if cor else padrao_cores['log']
    elif t == 'log1p':
        dados = np.log1p(serie)
        label_trans = 'Logaritmo Natural (log1p)'
        cor_final = cor if cor else padrao_cores['log1p']
    elif t == 'cbrt':
        dados = np.cbrt(serie)
        label_trans = 'Raiz Cúbica (cbrt)'
        cor_final = cor if cor else padrao_cores['cbrt']
    else:
        raise ValueError(
            f"Transformação '{transformacao}' inválida. Opções: 'original', 'sqrt', 'log', 'log1p', 'cbrt'."
        )

    plt.figure(figsize=figsize)

    sns.histplot(
        dados, 
        kde=True, 
        stat="density", 
        color=cor_final, 
        bins=bins, 
        edgecolor='black', 
        alpha=0.6, 
        label='Densidade Real (KDE)'
    )

    mu, std = stats.norm.fit(dados)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = stats.norm.pdf(x, mu, std)

    plt.plot(
        x, p, 'r--', linewidth=2, 
        label=f'Ajuste Normal Teórico\n($\\mu$={mu:,.2f}, $\\sigma$={std:,.2f})'
    )

    plt.title(f'Distribuição de {coluna} ({label_trans}) com Ajuste Normal', fontsize=13, fontweight='bold')
    plt.xlabel(f'{coluna} ({label_trans})', fontsize=11)
    plt.ylabel('Densidade', fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

    df_stats = pd.DataFrame({
        'Transformação': [label_trans],
        'Média': [dados.mean()],
        'Mediana': [dados.median()],
        'Desvio Padrão': [dados.std()],
        'Assimetria (Skew)': [dados.skew()],
        'Curtose (Kurt)': [dados.kurt()]
    })
    
    for col in ['Média', 'Mediana', 'Desvio Padrão', 'Assimetria (Skew)', 'Curtose (Kurt)']:
        df_stats[col] = df_stats[col].apply(lambda val: f"{val:.4f}")
        
    return df_stats


def plot_boxplot_outliers(
    df: pd.DataFrame, 
    coluna: str = 'price', 
    monetario: bool = False, 
    cor: str = '#D65F5F', 
    figsize: Tuple[int, int] = (12, 5)
) -> dict:
    """
    Gera um boxplot horizontal destacando os limites de Tukey e outliers.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    coluna : str, default='price'
        Nome da coluna numérica.
    monetario : bool, default=False
        Se True, formata o eixo X com '$' e notação 'K'.
    cor : str, default='#D65F5F'
        Cor de preenchimento do boxplot.
    figsize : tuple, default=(12, 5)
        Dimensões da figura.

    Retorna:
    -------
    dict
        Dicionário com estatísticas dos limites e contagem de outliers.
    """
    serie = df[coluna].dropna()
    
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_superior = q3 + 1.5 * iqr
    limite_inferior = q1 - 1.5 * iqr
    
    outliers_sup = serie[serie > limite_superior]
    outliers_inf = serie[serie < limite_inferior]
    total_outliers = len(outliers_sup) + len(outliers_inf)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        x=serie, 
        color=cor, 
        width=0.4, 
        linewidth=1.5,
        ax=ax,
        flierprops={
            'marker': 'o',
            'markerfacecolor': 'white', 
            'markeredgecolor': '#4A4A4A',
            'alpha': 0.5,
            'markersize': 5
        }
    )

    ax.axvline(limite_superior, color='#2B2B2B', linestyle='--', linewidth=1.5, alpha=0.7)
    
    offset = (serie.max() - serie.min()) * 0.015
    texto_corte = f'Limite Superior\n${limite_superior:,.2f}' if monetario else f'Limite Superior\n{limite_superior:,.2f}'
    
    ax.text(
        limite_superior + offset, -0.28, 
        texto_corte, 
        color='#2B2B2B', fontsize=10, va='center'
    )

    if monetario:
        formatter = ticker.FuncFormatter(lambda x, pos: f'${x/1000:,.1f}K' if abs(x) >= 1000 else f'${x:,.0f}')
        ax.xaxis.set_major_formatter(formatter)

    ax.set_title(f'Distribuição de {coluna} e Detecção de Outliers', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(f'{coluna} ($)' if monetario else coluna, fontsize=12, labelpad=10)
    ax.set_yticks([])
    sns.despine(left=True)

    plt.tight_layout()
    plt.show()

    print(f"--- Diagnóstico de Outliers: {coluna} ---")
    print(f"Limite Superior (Q3 + 1.5*IQR): {limite_superior:,.2f}")
    print(f"Limite Inferior (Q1 - 1.5*IQR): {limite_inferior:,.2f}")
    print(f"Outliers Superiores: {len(outliers_sup)} registros")
    print(f"Outliers Inferiores: {len(outliers_inf)} registros")
    print(f"Total de Outliers: {total_outliers} ({(total_outliers / len(serie)) * 100:.2f}%)")


def analisar_valores_nulos(df: pd.DataFrame, plotar: bool = True, figsize: Tuple[int, int] = (10, 6)) -> pd.DataFrame:
    """
    Analisa os valores nulos do DataFrame, retornando uma tabela com a quantidade 
    e o percentual de dados faltantes por coluna. Opcionalmente, plota um gráfico.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    plotar : bool, default=True
        Se True, gera um gráfico de barras horizontal com o percentual de nulos.
    figsize : tuple, default=(10, 6)
        Dimensões do gráfico gerado.

    Retorna:
    -------
    pd.DataFrame
        Tabela contendo colunas com nulos, ordenadas da maior para a menor quantidade.
    """
    total_nulos = df.isnull().sum()
    percentual_nulos = (total_nulos / len(df)) * 100
    
    df_nulos = pd.DataFrame({
        'Total Nulos': total_nulos, 
        'Percentual (%)': percentual_nulos,
        'Tipo de Dado': df.dtypes
    })
    
    df_nulos = df_nulos[df_nulos['Total Nulos'] > 0].sort_values(by='Total Nulos', ascending=False)
    
    if df_nulos.empty:
        print("Nenhum valor nulo encontrado no DataFrame!")
        return df_nulos

    if plotar:
        plt.figure(figsize=figsize)
        sns.set_theme(style="whitegrid")
        
        ax = sns.barplot(
            x='Percentual (%)', 
            y=df_nulos.index, 
            data=df_nulos, 
            color='#E66C37', 
            edgecolor='black',
            alpha=0.8
        )
        
        for p in ax.patches:
            width = p.get_width()
            plt.text(
                width + 0.5, 
                p.get_y() + p.get_height()/2. + 0.1, 
                f'{width:.1f}%', 
                ha="left", 
                va="center",
                fontsize=10,
                color='black'
            )
            
        plt.title('Percentual de Valores Nulos por Variável', fontsize=14, fontweight='bold')
        plt.xlabel('Percentual (%)', fontsize=12)
        plt.ylabel('Variáveis', fontsize=12)
        plt.xlim(0, max(df_nulos['Percentual (%)']) + 10)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        
    return df_nulos


def plot_dispersao_tendencia(df: pd.DataFrame, x_col: str, y_col: str, figsize: Tuple[int, int] = (8, 5)):
    """
    Gera um gráfico de dispersão com linha de regressão (tendência linear).
    Ideal para avaliar a relação bivariada entre duas variáveis numéricas.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    x_col : str
        Nome da variável preditora (eixo X).
    y_col : str
        Nome da variável resposta (eixo Y).
    figsize : tuple, default=(8, 5)
        Dimensões da figura (largura, altura).
    """
    plt.figure(figsize=figsize)
    sns.regplot(
        data=df, 
        x=x_col, 
        y=y_col,
        scatter_kws={'alpha': 0.4, 'color': 'royalblue', 's': 20},
        line_kws={'color': 'firebrick', 'linewidth': 2}
    )
    plt.title(f'Relação entre {x_col} e {y_col}', fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_categoria_preco(
    df: pd.DataFrame, 
    col_cat: str, 
    col_num: str = 'price', 
    top_n: int = 10, 
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Analisa a distribuição de uma variável numérica dentro das N categorias mais frequentes, 
    ordenadas de forma decrescente pela mediana.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    col_cat : str
        Nome da variável categórica (eixo X).
    col_num : str, default='price'
        Nome da variável numérica (eixo Y).
    top_n : int, default=10
        Quantidade máxima de categorias (as mais frequentes) a serem exibidas.
    figsize : tuple, default=(12, 6)
        Dimensões da figura (largura, altura).
    """
    top_categorias = df[col_cat].value_counts().nlargest(top_n).index
    df_filtrado = df[df[col_cat].isin(top_categorias)]
    
    ordem = df_filtrado.groupby(col_cat)[col_num].median().sort_values(ascending=False).index

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df_filtrado, 
        x=col_cat, 
        y=col_num, 
        order=ordem,
        palette="viridis",
        showfliers=False
    )
    
    plt.title(f'Distribuição de {col_num} por {col_cat} (Top {top_n})', fontsize=14, fontweight='bold')
    plt.xlabel(col_cat, fontsize=12)
    plt.ylabel(col_num, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    plt.show()


def analisar_multicolinearidade_pares(df: pd.DataFrame, limite: float = 0.80):
    """
    Identifica pares de variáveis numéricas com correlação linear absoluta superior ao limite estabelecido.
    Essencial para evitar colinearidade excessiva e distorção de distâncias euclidianas em clusterização.
    """
    matriz_corr = df.select_dtypes(include=[np.number]).corr().abs()
    triangulo_sup = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
    
    pares_altos = [
        (linha, coluna, triangulo_sup.loc[linha, coluna]) 
        for coluna in triangulo_sup.columns 
        for linha in triangulo_sup.index 
        if triangulo_sup.loc[linha, coluna] > limite
    ]
    
    print(f"--- Pares com Correlação > {limite} ---")
    if not pares_altos:
        print("Nenhum par encontrado acima do limite.")
    for par in sorted(pares_altos, key=lambda x: x[2], reverse=True):
        print(f"{par[0]} & {par[1]}: {par[2]:.3f}")


def plot_matriz_correlacao(
    df: pd.DataFrame, 
    titulo: str = 'Matriz de Correlação', 
    triangulo: Union[bool, str] = True,
    cmap: str = 'coolwarm', 
    figsize: Tuple[int, int] = (14, 10), 
    annot_size: int = 9
) -> pd.DataFrame:
    """
    Plota heatmap da matriz de correlação de variáveis numéricas.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame com colunas numéricas.
    titulo : str, default='Matriz de Correlação'
        Título do gráfico.
    triangulo : bool or str, default=True
        - True ou 'inferior': exibe apenas o triângulo inferior.
        - 'superior': exibe apenas o triângulo superior.
        - False ou 'completa': exibe a matriz completa.
    cmap : str, default='coolwarm'
        Paleta de cores do heatmap.
    figsize : tuple, default=(14, 10)
        Dimensões da figura.
    annot_size : int, default=9
        Tamanho da fonte das anotações.

    Retorna:
    -------
    pd.DataFrame
        Matriz de correlação calculada.
    """
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr()

    mascara = None
    if triangulo in (True, 'inferior'):
        mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    elif triangulo == 'superior':
        mascara = np.tril(np.ones_like(corr, dtype=bool), k=-1)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap=cmap,
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=.5, cbar_kws={'shrink': .75},
        mask=mascara, annot_kws={'size': annot_size}
    )
    plt.title(titulo, fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return corr


def plot_grid_distribuicao(
    df: pd.DataFrame,
    features: List[str],
    titulo: str = 'Distribuição das Features',
    cor: str = 'steelblue',
    bins: Union[int, str] = 'sturges',
    figsize: Tuple[int, int] = (16, 12),
):
    """
    Plota um grid 2x2 com boxplot (topo) + histograma com KDE (base)
    para as primeiras 4 features numéricas informadas.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    features : list
        Lista com até 4 colunas numéricas a serem plotadas.
    titulo : str, default='Distribuição das Features'
        Título principal da figura.
    cor : str, default='steelblue'
        Cor dos boxplots e histogramas.
    bins : str ou int, default='sturges'
        Método ou número de bins do histograma.
    figsize : tuple, default=(16, 12)
        Dimensões da figura.
    """
    import matplotlib.gridspec as gridspec

    feats = features[:4]
    n_feats = len(feats)
    if n_feats == 0:
        print("Nenhuma feature fornecida para plotar.")
        return

    n_rows = 2 if n_feats > 2 else 1
    n_cols = 2 if n_feats > 1 else 1

    fig = plt.figure(figsize=figsize)
    outer_gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.2)

    for i, col in enumerate(feats):
        row_idx = i // 2 if n_rows > 1 else 0
        col_idx = i % 2 if n_cols > 1 else 0

        inner_gs = outer_gs[row_idx, col_idx].subgridspec(
            2, 1, height_ratios=[1, 4], hspace=0.05
        )

        ax_box = fig.add_subplot(inner_gs[0])
        ax_hist = fig.add_subplot(inner_gs[1], sharex=ax_box)

        dados = df[col].dropna()
        media = dados.mean()
        mediana = dados.median()
        modas = dados.mode()
        moda = modas[0] if not modas.empty else None

        sns.boxplot(x=dados, ax=ax_box, color=cor, fliersize=4)
        ax_box.axvline(media, color='darkorange', linestyle='--', linewidth=1.5)
        ax_box.axvline(mediana, color='forestgreen', linestyle='--', linewidth=1.5)
        if moda is not None:
            ax_box.axvline(moda, color='firebrick', linestyle='--', linewidth=1.5)

        ax_box.set_title(f'Distribuição: {col}', fontsize=13, fontweight='bold', pad=10)
        ax_box.set_xlabel('')
        ax_box.grid(True, linestyle='--', alpha=0.7)
        plt.setp(ax_box.get_xticklabels(), visible=False)

        sns.histplot(
            x=dados, bins=bins, kde=True, ax=ax_hist,
            color=cor, edgecolor='black', alpha=0.6,
        )

        ax_hist.axvline(media, color='darkorange', linestyle='--', linewidth=2, label=f'Média: {media:.1f}')
        ax_hist.axvline(mediana, color='forestgreen', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.1f}')
        if moda is not None:
            ax_hist.axvline(moda, color='firebrick', linestyle='--', linewidth=2, label=f'Moda: {moda:.1f}')

        ax_hist.legend(frameon=True, fontsize=9, loc='upper right')
        ax_hist.set_xlabel(col, fontsize=11)
        ax_hist.set_ylabel('Frequência', fontsize=11)
        ax_hist.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(titulo, fontsize=18, fontweight='bold', y=0.96)
    plt.show()


def plot_grid_boxplot_alvo(
    df: pd.DataFrame,
    features: List[str],
    alvo: str = 'cluster',
    hue: Optional[str] = None,
    titulo: Optional[str] = None,
    paleta: str = 'Set2',
    labels_x: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 12),
):
    """
    Plota um grid 2x2 de boxplots agrupados por uma variável discreta (ex: clusters formados)
    e opcionalmente coloridos por hue, útil para visualizar a separabilidade das variáveis por cluster.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    features : list
        Lista com até 4 colunas numéricas.
    alvo : str, default='cluster'
        Coluna do eixo X (variável de agrupamento/cluster).
    hue : str, optional
        Coluna adicional de agrupamento por cor.
    titulo : str, optional
        Título principal. Se None, gera automaticamente.
    paleta : str, default='Set2'
        Paleta de cores do seaborn.
    labels_x : list, optional
        Rótulos personalizados para o eixo X.
    figsize : tuple, default=(16, 12)
        Dimensões da figura.
    """
    feats = features[:4]
    n_feats = len(feats)
    if n_feats == 0:
        print("Nenhuma feature fornecida para plotar.")
        return

    n_rows = 2 if n_feats > 2 else 1
    n_cols = 2 if n_feats > 1 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    titulo_final = titulo or f'Distribuição das Features por {alvo}'

    for i, var in enumerate(feats):
        sns.boxplot(
            x=alvo, y=var, hue=hue,
            data=df, ax=axes[i], palette=paleta,
            showmeans=True,
            meanprops={
                'marker': 'D', 'markerfacecolor': 'white',
                'markeredgecolor': 'black', 'markersize': 6,
            },
        )
        axes[i].set_title(f'{var} vs {alvo}', fontsize=13, fontweight='bold', pad=10)
        axes[i].set_xlabel(alvo, fontsize=11)
        axes[i].set_ylabel(var, fontsize=11)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

        if labels_x is not None:
            axes[i].set_xticklabels(labels_x)

        if hue is not None:
            if i == 0:
                axes[i].legend(title=hue, loc='upper left')
            else:
                leg = axes[i].get_legend()
                if leg:
                    leg.remove()

    plt.suptitle(titulo_final, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def teste_significancia_chi2(
    df: pd.DataFrame,
    colunas_nominais: List[str],
    alvo: str = 'cluster',
) -> pd.DataFrame:
    """
    Aplica o teste Qui-Quadrado e calcula o V de Cramér para medir a
    força de associação entre variáveis categóricas e os clusters formados (ou variável categórica).

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    colunas_nominais : list
        Lista de colunas categóricas nominais a serem testadas.
    alvo : str, default='cluster'
        Nome da variável de agrupamento ou categórica de referência.

    Retorna:
    -------
    pd.DataFrame
        Tabela com Feature, Categorias, Chi², p-valor e V de Cramér,
        ordenada de forma decrescente pelo V de Cramér.
    """
    resultados = []

    for col in colunas_nominais:
        sub = df[[col, alvo]].dropna()
        if sub.empty or sub[col].nunique() < 2 or sub[alvo].nunique() < 2:
            continue
        tab = pd.crosstab(sub[col], sub[alvo])

        chi2, p_valor, dof, _ = stats.chi2_contingency(tab)
        n = tab.sum().sum()
        k = min(tab.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 and n > 0 else 0.0

        resultados.append({
            'Feature': col,
            'Categorias': sub[col].nunique(),
            'Chi²': round(chi2, 2),
            'p-valor': f'{p_valor:.4e}',
            'V de Cramér': round(cramers_v, 4),
        })

    return (
        pd.DataFrame(resultados)
        .sort_values('V de Cramér', ascending=False)
        .reset_index(drop=True)
    )

def resumo_estatistico_numerico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula as principais estatísticas descritivas para TODAS as variáveis 
    numéricas do DataFrame, retornando uma matriz comparativa.
    """
    # Seleciona apenas as colunas numéricas
    df_numerico = df.select_dtypes(include=[np.number])
    
    resultados = {}
    
    for coluna in df_numerico.columns:
        serie = df_numerico[coluna].dropna()
        
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        
        # Guardamos os valores crus primeiro
        resultados[coluna] = [
            serie.mean(),
            serie.median(),
            serie.std(),
            serie.min(),
            q1,
            q3,
            iqr,
            serie.max(),
            serie.skew(),
            serie.kurt()
        ]
        
    metricas = [
        'Média', 
        'Mediana', 
        'Desvio Padrão', 
        'Mínimo', 
        '1º Quartil (Q1 - 25%)', 
        '3º Quartil (Q3 - 75%)', 
        'Intervalo Interquartil (IQR)', 
        'Máximo',
        'Assimetria (Skewness)', 
        'Curtose (Kurtosis)'
    ]
    
    # Monta o DataFrame com as métricas como índice
    df_resumo = pd.DataFrame(resultados, index=metricas)
    
    # Aplica a formatação em todas as células (usamos .map ou .applymap dependendo da versão do Pandas)
    if hasattr(df_resumo, 'map'):
        df_formatado = df_resumo.map(lambda x: f"{x:,.2f}")
    else:
        df_formatado = df_resumo.applymap(lambda x: f"{x:,.2f}")
        
    return df_formatado

def pipeline_clusterizacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de pré-processamento para clusterização baseada em distância
    (K-Means, DBSCAN), produzindo um DataFrame normalizado com RobustScaler.

    Etapas executadas na ordem:
    1. Tratamento de nulos e flag de atividade (has_reviews)
    2. Criação de métricas de negócio (ticket_diario, receita_mensal_est, anfitriao)
    3. Transformação logarítmica (np.log1p) em variáveis de cauda longa
    4. One-Hot Encoding de variáveis categóricas
    5. Seleção estrita de features e remoção de colunas originais/identificadores
    6. Normalização robusta (RobustScaler)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame bruto do Airbnb NYC com as 16 colunas originais.

    Retorna
    -------
    pd.DataFrame
        DataFrame normalizado pronto para algoritmos de clusterização (df_cluster_scaled).
    """
    df_copia = df.copy()

    # ---------------------------------------------------------------
    # 1. Tratamento de Nulos e Indicadores de Atividade
    # ---------------------------------------------------------------
    df_copia['reviews_per_month'] = df_copia['reviews_per_month'].fillna(0)
    df_copia['has_reviews'] = df_copia['last_review'].notna().astype(int)
    df_copia.drop(columns=['last_review'], inplace=True)

    # ---------------------------------------------------------------
    # 2. Criação de Novas Métricas de Negócio (escala original)
    # ---------------------------------------------------------------
    epsilon = 1e-5
    df_copia['ticket_diario'] = df_copia['price'] / (df_copia['minimum_nights'] + epsilon)
    df_copia['receita_mensal_est'] = (
        df_copia['reviews_per_month'] * df_copia['minimum_nights'] * df_copia['price']
    )

    # Categorização do anfitrião
    condicoes = [
        df_copia['calculated_host_listings_count'] == 1,
        df_copia['calculated_host_listings_count'].between(2, 5),
        df_copia['calculated_host_listings_count'] > 5,
    ]
    escolhas = [1, 2, 3]
    df_copia['anfitriao'] = np.select(condicoes, escolhas, default=3)

    # ---------------------------------------------------------------
    # 3. Transformação Logarítmica (np.log1p)
    # ---------------------------------------------------------------
    features_para_log = [
        'price', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'ticket_diario', 'receita_mensal_est',
    ]

    for col in features_para_log:
        df_copia[f'{col}_log'] = np.log1p(df_copia[col])

    # ---------------------------------------------------------------
    # 4. Codificação Categórica (One-Hot Encoding)
    # ---------------------------------------------------------------
    df_copia = pd.get_dummies(
        df_copia,
        columns=['neighbourhood_group', 'room_type'],
        drop_first=True,
        dtype=int,
    )

    # ---------------------------------------------------------------
    # 5. Limpeza e Seleção Estrita de Features
    # ---------------------------------------------------------------
    colunas_log = [f'{col}_log' for col in features_para_log]
    colunas_dummies = [
        col for col in df_copia.columns
        if col.startswith('neighbourhood_group_') or col.startswith('room_type_')
    ]
    colunas_escalaveis = ['latitude', 'longitude', 'availability_365', 'anfitriao']

    features_finais = colunas_log + colunas_escalaveis + colunas_dummies + ['has_reviews']

    df_copia = df_copia[features_finais].copy()

    # ---------------------------------------------------------------
    # 6. Normalização Robusta (RobustScaler)
    # ---------------------------------------------------------------
    scaler = RobustScaler()
    array_scaled = scaler.fit_transform(df_copia)
    df_cluster_scaled = pd.DataFrame(array_scaled, columns=df_copia.columns, index=df_copia.index)

    return df_cluster_scaled


def plot_elbow_comparativo(
    k_valores,
    inercias_orig,
    inercias_trat,
    titulos=('Metodo Elbow: Dados Originais (X)', 'Metodo Elbow: Dados Tratados (X_tratado)'),
    cores=('#7F8C8D', '#C00000'),
    figsize=(15, 6)
):
    """
    Plota 1x2 subplots comparando a curva WCSS (inercia) antes e depois
    do escalonamento, com spines limpos e grid suave no eixo Y.

    Parametros
    ----------
    k_valores : iterable
        Valores de K testados (ex: range(1, 21)).
    inercias_orig : list
        Inercias do dataset original (X).
    inercias_trat : list
        Inercias do dataset tratado (X_tratado).
    titulos : tuple of str
        Titulos dos subplots esquerdo e direito.
    cores : tuple of str
        Cores das linhas (original, tratado).
    figsize : tuple
        Dimensoes da figura (largura, altura).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    dados = [
        (inercias_orig, cores[0], titulos[0]),
        (inercias_trat, cores[1], titulos[1]),
    ]

    for ax, (inercias, cor, titulo) in zip(axes, dados):
        ax.plot(k_valores, inercias, marker='o', markersize=6,
                color=cor, linewidth=2)
        ax.set_title(titulo, fontsize=13, fontweight='bold',
                     color='#333333', pad=12)
        ax.set_xlabel('Numero de Clusters (K)', fontsize=11,
                      fontweight='bold', color='#4A4A4A')
        if ax == axes[0]:
            ax.set_ylabel('Inercia (WCSS)', fontsize=11,
                          fontweight='bold', color='#4A4A4A')
        ax.set_xticks(k_valores)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#D5D8DC')

    plt.tight_layout()
    plt.show()


def plot_metricas_validacao(
    df_resultados,
    coluna_k='K',
    metricas=('Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz'),
    titulos=None,
    cores=('#2E86C1', '#C00000', '#28B463'),
    suptitle='Metricas de Validacao para cada K (Subsample de 10k pontos)',
    figsize=(16, 5)
):
    """
    Plota 1x3 subplots com as curvas de evolucao das metricas de
    clusterizacao por K (Silhouette, Davies-Bouldin, Calinski-Harabasz).

    Parametros
    ----------
    df_resultados : pd.DataFrame
        DataFrame com as metricas, contendo uma coluna para K e uma para cada metrica.
    coluna_k : str
        Nome da coluna com os valores de K.
    metricas : tuple of str
        Nomes das colunas de metricas a plotar.
    titulos : tuple of str or None
        Titulos de cada subplot. Se None, usa os nomes das metricas.
    cores : tuple of str
        Cores das linhas para cada metrica.
    suptitle : str
        Titulo principal da figura.
    figsize : tuple
        Dimensoes da figura (largura, altura).
    """
    if titulos is None:
        titulos = metricas

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(suptitle, fontsize=18, fontweight='bold',
                 color='#1C1C1C', y=1.05)

    for i, (metrica, titulo, cor) in enumerate(zip(metricas, titulos, cores)):
        ax = axes[i]
        ax.plot(
            df_resultados[coluna_k], df_resultados[metrica],
            marker='o', linewidth=2.5, color=cor, markersize=8,
            markerfacecolor='white', markeredgewidth=2
        )
        ax.set_title(titulo, fontsize=14, fontweight='bold',
                     color='#1C1C1C', pad=15)
        ax.set_xlabel('Numero de Clusters (K)', fontsize=12,
                      fontweight='bold', color='#4A4A4A')
        ax.set_xticks(df_resultados[coluna_k])
        ax.tick_params(axis='both', colors='#4A4A4A', labelsize=11)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()


def plot_distribuicao_clusters(
    labels,
    titulo='Distribuicao de Registros por Cluster',
    xlabel='ID do Cluster',
    ylabel='Volume de Registros',
    limiar_destaque=10000,
    cor_majoritario='#7F8C8D',
    cor_nicho='#C00000',
    figsize=(10, 5)
):
    """
    Plota a distribuicao de registros por cluster, destacando grupos
    com volume abaixo de um limiar como nichos.

    A contagem e calculada dinamicamente via np.unique, sem dicionario
    hardcoded.

    Parametros
    ----------
    labels : np.ndarray
        Array de labels de cluster (ex: saida de km.labels_ ou fit_predict).
    titulo : str
        Titulo do grafico.
    xlabel : str
        Rotulo do eixo X.
    ylabel : str
        Rotulo do eixo Y.
    limiar_destaque : int
        Clusters com volume menor que este valor recebem cor de destaque.
    cor_majoritario : str
        Cor para clusters acima do limiar.
    cor_nicho : str
        Cor para clusters abaixo do limiar.
    figsize : tuple
        Dimensoes da figura (largura, altura).
    """
    clusters_ids, contagens = np.unique(labels, return_counts=True)

    cores = [
        cor_nicho if v < limiar_destaque else cor_majoritario
        for v in contagens
    ]

    fig, ax = plt.subplots(figsize=figsize)
    barras = ax.bar(clusters_ids, contagens, color=cores, edgecolor='none')

    ax.set_title(titulo, fontsize=14, fontweight='bold',
                 color='#1C1C1C', pad=20)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold', color='#4A4A4A')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='#4A4A4A')

    ax.bar_label(barras, fmt='{:.0f}', padding=5, color='#333333',
                 fontweight='bold', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')

    ax.set_xticks(clusters_ids)
    ax.tick_params(axis='both', which='major', labelsize=10, color='#4A4A4A')
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#D5D8DC')

    plt.tight_layout()
    plt.show()


def plot_clusters_2d_plotly(
    df_proj,
    col_x,
    col_y,
    col_cluster='Cluster',
    mapa_cores=None,
    n_amostra=5000,
    titulo='Projecao 2D dos Clusters',
    width=900,
    height=600,
    random_state=42
):
    """
    Gera uma projecao 2D interativa com Plotly, amostrando os dados
    para performance e aplicando a paleta de cores por cluster.

    Parametros
    ----------
    df_proj : pd.DataFrame
        DataFrame com as colunas de projecao e a coluna de cluster.
    col_x, col_y : str
        Nomes das colunas dos eixos X e Y.
    col_cluster : str
        Nome da coluna de identificacao do cluster.
    mapa_cores : dict or None
        Dicionario mapeando cluster_id -> cor. Se None, usa paleta default.
    n_amostra : int
        Tamanho da amostra para renderizacao.
    titulo : str
        Titulo do grafico.
    width, height : int
        Dimensoes do grafico em pixels.
    random_state : int
        Semente para reproducibilidade da amostragem.

    Retorna
    -------
    plotly.graph_objects.Figure
        Figura Plotly renderizada.
    """
    df_amostra = df_proj.sample(n=min(n_amostra, len(df_proj)),
                                random_state=random_state).copy()
    df_amostra[col_cluster] = df_amostra[col_cluster].astype(str)

    if mapa_cores is not None:
        mapa_str = {str(k): v for k, v in mapa_cores.items()}
    else:
        mapa_str = None

    fig = px.scatter(
        df_amostra, x=col_x, y=col_y, color=col_cluster,
        color_discrete_map=mapa_str, opacity=0.7,
        render_mode='svg', title=titulo,
        hover_data={col_x: False, col_y: False, col_cluster: True}
    )

    fig.update_layout(
        plot_bgcolor='white',
        title_font=dict(size=18, color='#1C1C1C', family='Arial'),
        legend_title_text='ID do Cluster',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    font=dict(size=12, color='#4A4A4A')),
        width=width, height=height,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F2F3F4',
                     title_font=dict(size=12, color='#4A4A4A'), zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F2F3F4',
                     title_font=dict(size=12, color='#4A4A4A'), zeroline=False)

    fig.show()
    return fig


def plot_clusters_3d_matplotlib(
    df_proj,
    cols_xyz=('PC1', 'PC2', 'PC3'),
    labels=None,
    mapa_cores=None,
    titulo='Projecao 3D (PCA): Visao Espacial dos Clusters',
    labels_eixos=('Componente Principal 1', 'Componente Principal 2', 'Componente Principal 3'),
    alpha=0.2,
    s=12,
    elev=20,
    azim=45,
    figsize=(12, 8)
):
    """
    Gera uma projecao 3D dos clusters no Matplotlib, com paineis limpos
    e mapeamento de cores por cluster.

    Parametros
    ----------
    df_proj : pd.DataFrame
        DataFrame com as colunas de projecao e, opcionalmente, a coluna de cluster.
    cols_xyz : tuple of str
        Nomes das colunas para os eixos X, Y, Z.
    labels : np.ndarray or None
        Array de labels de cluster. Se None, usa df_proj['Cluster'].
    mapa_cores : dict or None
        Dicionario mapeando cluster_id -> cor. Se None, usa paleta default.
    titulo : str
        Titulo do grafico.
    labels_eixos : tuple of str
        Rotulos dos eixos X, Y, Z.
    alpha : float
        Transparencia dos pontos.
    s : float
        Tamanho dos pontos.
    elev, azim : float
        Angulos de elevacao e azimute da camera 3D.
    figsize : tuple
        Dimensoes da figura (largura, altura).
    """
    if labels is None:
        labels = df_proj['Cluster'].values if 'Cluster' in df_proj.columns else None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if labels is not None and mapa_cores is not None:
        cores_pontos = [mapa_cores[label] for label in labels]
    else:
        cores_pontos = None

    ax.scatter(
        df_proj[cols_xyz[0]], df_proj[cols_xyz[1]], df_proj[cols_xyz[2]],
        c=cores_pontos, s=s, alpha=alpha, edgecolors='none'
    )

    ax.set_title(titulo, fontsize=16, fontweight='bold',
                 color='#1C1C1C', pad=25)
    ax.set_xlabel(labels_eixos[0], fontweight='bold',
                  color='#4A4A4A', labelpad=10)
    ax.set_ylabel(labels_eixos[1], fontweight='bold',
                  color='#4A4A4A', labelpad=10)
    ax.set_zlabel(labels_eixos[2], fontweight='bold',
                  color='#4A4A4A', labelpad=10)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    ax.grid(True, linestyle='--', alpha=0.3, color='#BDC3C7')
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()
