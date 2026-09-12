"""
data_prep.py — Camada de dados (Seções 2.1 e 3.5 do relatório).

Grupo: Efraim Oliveira, Érica Tarsis, Ricardo Moraes, Rodrigo Bernardino, Thiago Galvão
POSTECH DTAT — 2026

PROPÓSITO
    Transforma o arquivo bruto WineQT.csv na base de trabalho confiável:
    carrega, audita (linhas, colunas, nulos), remove duplicatas reais
    (ignorando a coluna técnica 'Id', que mascara repetições) e produz a base
    limpa de 1.018 amostras. Também define a lista de variáveis preditoras, a
    construção do alvo binário por limiar e a divisão estratificada
    treino/teste usada em todo o estudo.

USO
    from src import data_prep as dp
    df = dp.preparar_base()                       # carrega + deduplica
    variaveis = dp.listar_variaveis(df)
    X_tr, X_te, y_tr, y_te = dp.dividir_treino_teste(df, variaveis, corte=6)

REGRA DE GOVERNANÇA
    Todas as estatísticas do projeto usam a base limpa (1.018), nunca a bruta
    (1.143). A binarização >= 6 ("média/alta") equilibra as classes (~1,16:1),
    ao passo que >= 7 gera desbalanceamento severo (~1:6).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def carregar_dados(caminho: str | Path | None = None) -> pd.DataFrame:
    """Lê o CSV bruto. Sem argumento, usa data/WineQT.csv (config.ARQ_DADOS)."""
    caminho = Path(caminho) if caminho is not None else config.ARQ_DADOS
    if not caminho.exists():
        raise FileNotFoundError(
            f"WineQT.csv não encontrado em: {caminho}\n"
            f"Coloque o arquivo em {config.DIR_DADOS} ou informe o caminho."
        )
    df = pd.read_csv(caminho)
    print(f"Base bruta: {df.shape[0]} x {df.shape[1]} | "
          f"nulos: {int(df.isna().sum().sum())}")
    return df


def deduplicar(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas idênticas ignorando 'Id'. Bruto (1.143) -> limpo (1.018)."""
    cc = [c for c in df_bruto.columns if c.lower() != "id"]
    qtd_dup = int(df_bruto.duplicated(subset=cc).sum())
    df = df_bruto.drop_duplicates(subset=cc, keep="first").reset_index(drop=True)
    print(f"Idênticos removidos: {qtd_dup} | Base limpa: {df.shape[0]} amostras")
    return df


def preparar_base(caminho: str | Path | None = None) -> pd.DataFrame:
    """Conveniência: carrega o bruto e devolve a base limpa pronta para uso."""
    return deduplicar(carregar_dados(caminho))


def listar_variaveis(df: pd.DataFrame) -> list[str]:
    """Lista as variáveis preditoras (tudo exceto 'quality' e 'Id')."""
    return [c for c in df.columns if c not in ("quality", "Id")]


def criar_alvo(df: pd.DataFrame, corte: int) -> pd.Series:
    """Cria o alvo binário: 1 (Alta) se quality >= corte, senão 0 (Baixa/Média)."""
    return (df["quality"] >= corte).astype(int)


def dividir_treino_teste(df: pd.DataFrame, features: list[str], corte: int,
                         test_size: float = 0.20):
    """
    Divisão estratificada 80/20 (mantém a proporção de classes) com SEED fixa.
    Retorna (X_treino, X_teste, y_treino, y_teste).
    """
    y = criar_alvo(df, corte)
    return train_test_split(df[features], y, test_size=test_size,
                            random_state=config.SEED, stratify=y)
