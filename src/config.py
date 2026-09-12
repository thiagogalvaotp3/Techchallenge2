"""
config.py — Fundação compartilhada do projeto (Tech Challenge — Fase 2).

Grupo: Efraim Oliveira, Érica Tarsis, Ricardo Moraes, Rodrigo Bernardino, Thiago Galvão
POSTECH DTAT — 2026

PROPÓSITO
    Concentra tudo o que precisa ser idêntico em todo o estudo para garantir
    reprodutibilidade e padronização visual: a semente aleatória (SEED = 43),
    a paleta de cores, o estilo dos gráficos, os caminhos do repositório
    (data/ e results/) e os auxiliares de formatação/anotação no padrão pt-BR.

USO
    É importado por data_prep, modeling, analysis e run_all. Não depende de
    nenhum outro módulo do projeto (evita import circular). Ao ser importado,
    fixa a semente e aplica o estilo Matplotlib/Seaborn.

OBSERVAÇÃO TÉCNICA (para quem não é da área)
    Como estes arquivos rodam "sem tela" (modo script), o Matplotlib é
    configurado no backend 'Agg': em vez de abrir janelas, os gráficos são
    salvos como imagens .png dentro de results/figuras/.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend não interativo: salva imagens em vez de exibir janelas

# Silencia avisos de descontinuação de bibliotecas (mesma política do notebook),
# mantendo o log limpo e focado nos resultados.
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# --------------------------------------------------------------------------- #
# 1. Reprodutibilidade
# --------------------------------------------------------------------------- #
SEED = 43  # 43 é primo (decisão de grupo); substitui o difundido 42
np.random.seed(SEED)

# --------------------------------------------------------------------------- #
# 2. Caminhos do repositório (resolvidos a partir da posição deste arquivo)
#    src/config.py  ->  parents[1] = raiz do repositório
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parents[1]
DIR_DADOS = RAIZ / "data"
DIR_RESULTS = RAIZ / "results"
DIR_FIGURAS = DIR_RESULTS / "figuras"
DIR_TABELAS = DIR_RESULTS / "tabelas"
ARQ_DADOS = DIR_DADOS / "WineQT.csv"

# --------------------------------------------------------------------------- #
# 3. Paleta e estilo
# --------------------------------------------------------------------------- #
LARANJA, AZUL = "#C55A11", "#2F5496"  # 0 = Baixa/Média | 1 = Alta
PALETA = ["#2F5496", "#C55A11", "#2E8B6F", "#7B5EA7", "#BF8F00", "#595959", "#A6342B"]


def aplicar_estilo() -> None:
    """Aplica o tema visual padronizado (chamado no import)."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# --------------------------------------------------------------------------- #
# 4. Formatação pt-BR e auxiliares de anotação de gráficos
# --------------------------------------------------------------------------- #
def br(x, dec: int = 3) -> str:
    """Formata número no padrão pt-BR (vírgula decimal)."""
    return f"{x:.{dec}f}".replace(".", ",")


def eixo_virgula(ax, eixo: str = "x", dec: int = 2) -> None:
    """Troca o separador decimal do eixo para vírgula."""
    f = mticker.FuncFormatter(lambda v, _: f"{v:.{dec}f}".replace(".", ","))
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(f)


def anota_vertical(ax, vals, total=None, folga: float = 0.32) -> None:
    """Rótulos sobre barras verticais, com folga para não colidir com o título."""
    ymax = max(vals) if len(vals) else 1
    ax.set_ylim(0, ymax * (1 + folga))
    for i, v in enumerate(vals):
        t = (f"{v:,.0f}".replace(",", ".") if total is None
             else f"{v:,.0f}\n({v/total:.0%})".replace(",", "."))
        ax.text(i, v + ymax * 0.02, t, ha="center", va="bottom",
                fontsize=10, fontweight="bold")


def anota_h_pos(ax, vals, folga: float = 0.18, dec: int = 3) -> None:
    """Rótulos em barras horizontais estritamente positivas."""
    xmax = max(vals) if len(vals) else 1
    ax.set_xlim(0, xmax * (1 + folga))
    for i, v in enumerate(vals):
        ax.text(v + xmax * 0.01, i, br(v, dec), va="center",
                fontsize=10, fontweight="bold")


def barh_signed(ax, labels, vals, cor_pos=AZUL, cor_neg=LARANJA,
                dec: int = 3, folga: float = 0.16) -> None:
    """Barras horizontais COM SINAL: rótulo do lado da ponta + xlim simétrico."""
    ax.barh(labels, vals, color=[cor_pos if v >= 0 else cor_neg for v in vals],
            edgecolor="white")
    ax.axvline(0, color="0.4", lw=0.8)
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or 1
    ax.set_xlim(vmin - span * folga * 2.2, vmax + span * folga * 2.2)
    for i, v in enumerate(vals):
        ax.text(v + (span * 0.015 if v >= 0 else -span * 0.015), i, br(v, dec),
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9.5, fontweight="bold")


# --------------------------------------------------------------------------- #
# 5. Persistência de artefatos em results/
# --------------------------------------------------------------------------- #
def garantir_dirs() -> None:
    """Cria results/figuras e results/tabelas se ainda não existirem."""
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)


def salvar_figura(fig, nome: str, dpi: int = 150) -> Path:
    """Salva a figura em results/figuras/<nome>.png e a fecha."""
    garantir_dirs()
    caminho = DIR_FIGURAS / f"{nome}.png"
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return caminho


def salvar_tabela(df, nome: str) -> Path:
    """Salva a tabela em results/tabelas/<nome>.csv no padrão pt-BR (sep=';')."""
    garantir_dirs()
    caminho = DIR_TABELAS / f"{nome}.csv"
    df.to_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig")
    return caminho


# Aplica o estilo assim que o módulo é carregado.
aplicar_estilo()
