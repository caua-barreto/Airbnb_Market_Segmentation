"""
src/config.py
-------------
Utilitário centralizado para carregamento do arquivo configs.yaml e variáveis de ambiente.
Garante reprodutibilidade entre notebooks (01_EDA, 02_Modeling), scripts e módulos src/.
"""

import os
from pathlib import Path
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / os.getenv("CONFIG_PATH", "configs.yaml")


def carregar_configs(caminho=None) -> dict:
    """
    Carrega e retorna o dicionário de configurações do projeto a partir do configs.yaml.

    Parâmetros
    ----------
    caminho : str ou Path, optional
        Caminho customizado para o arquivo YAML. Se None, utiliza configs.yaml na raiz.

    Retorna
    -------
    dict
        Dicionário contendo todas as configurações de dados, pré-processamento,
        modelagem, mapeamento de perfis e artefatos.
    """
    config_path = Path(caminho) if caminho is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        # Fallback caso seja chamado a partir de subdiretório (ex: notebooks/)
        alt_path = Path("..") / "configs.yaml"
        if alt_path.exists():
            config_path = alt_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg

