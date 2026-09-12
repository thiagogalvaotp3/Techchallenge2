"""
run_all.py — Orquestrador do estudo (reprodução ponta a ponta sem o notebook).

Grupo: Efraim Oliveira, Érica Tarsis, Ricardo Moraes, Rodrigo Bernardino, Thiago Galvão
POSTECH DTAT — 2026

PROPÓSITO
    Executa todo o pipeline na MESMA ordem do relatório (Seções 2 a 7),
    importando os módulos do pacote src, e grava cada figura e tabela em
    results/. Existe como rede de segurança: se o notebook falhar na banca, os
    avaliadores reproduzem os resultados rodando apenas este arquivo.

COMO EXECUTAR (a partir da raiz do repositório wine-quality-classification/)
    python src/run_all.py
        ou
    python -m src.run_all

REQUISITOS
    data/WineQT.csv presente e dependências de requirements.txt instaladas.
    Saídas: results/figuras/*.png e results/tabelas/*.csv

LEI DE MURPHY (o que isto previne)
    Estado implícito entre células de notebook é uma fonte clássica de erro.
    Aqui o fluxo é explícito e determinístico (SEED = 43): cada etapa recebe
    exatamente os objetos de que depende.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Permite rodar tanto como script (python src/run_all.py) quanto como módulo.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import f1_score

from src import config
from src import data_prep as dp
from src import modeling as ml
from src import analysis as an


def secao(titulo: str) -> None:
    print("\n" + "=" * 78 + f"\n{titulo}\n" + "=" * 78)


def main() -> None:
    secao("Tech Challenge — Fase 2 | Reprodução completa | SEED = "
          f"{config.SEED}")
    config.garantir_dirs()

    # ---- Seção 2.1 — base de dados --------------------------------------- #
    secao("Seção 2 — Análise Exploratória de Dados")
    df = dp.preparar_base()
    variaveis = dp.listar_variaveis(df)
    cv = ml.criar_cv()
    y6 = dp.criar_alvo(df, 6)
    X_base = df[variaveis]

    an.fig01_distribuicao_notas(df)
    an.fig02_balanceamento_limiar(df)
    an.fig03_histogramas(df, variaveis)
    an.fig04_outliers(df, variaveis)
    an.fig05_06_correlacoes(df, variaveis)
    an.figA_boxplots_classe(df)
    an.tab_vif(df, variaveis)

    # ---- Seção 3 — pré-processamento e engenharia de variáveis ----------- #
    secao("Seção 3 — Pré-Processamento e Engenharia de Variáveis")
    an.tab02_so2_ratio(df, X_base, y6, cv)
    an.tab02b_interacoes(df, X_base, y6, cv)
    an.tablog_transformacao_log(df, X_base, y6, cv)

    X_tr6, X_te6, y_tr6, y_te6 = dp.dividir_treino_teste(df, variaveis, corte=6)
    X_tr7, X_te7, y_tr7, y_te7 = dp.dividir_treino_teste(df, variaveis, corte=7)
    print(f"Corte ≥ 6 → treino {len(X_tr6)} | teste {len(X_te6)} | "
          f"%Alta teste = {y_te6.mean():.3f}")

    # ---- Seção 4 — desenvolvimento e otimização -------------------------- #
    secao("Seção 4 — Desenvolvimento e Otimização de Modelos")
    print("Modelos:", ", ".join(ml.construir().keys()))
    grade_lr, grade_rf = ml.otimizar(X_tr6, y_tr6, cv)
    rf_tunado = grade_rf.best_estimator_
    print(f"Regressão Logística → {grade_lr.best_params_} | "
          f"F1(CV) = {config.br(grade_lr.best_score_, 4)}")
    print(f"Random Forest       → {grade_rf.best_params_} | "
          f"F1(CV) = {config.br(grade_rf.best_score_, 4)}")
    f1_rf_teste = f1_score(y_te6, rf_tunado.fit(X_tr6, y_tr6).predict(X_te6))
    print(f"Random Forest otimizada — F1(teste) = {config.br(f1_rf_teste, 4)}")

    # ---- Seção 5 — Parte I (corte ≥ 7, desbalanceado) -------------------- #
    secao("Seção 5 — Avaliação Parte I (corte ≥ 7)")
    tab7 = ml.avaliar(ml.construir(False), X_tr7, y_tr7, X_te7, y_te7)
    print(tab7.to_string())
    config.salvar_tabela(tab7, "tab04_parteI_sem_correcao")
    an.grafico_f1(tab7, "Parte I (corte ≥ 7) — F1 por modelo, SEM correção",
                  "fig07_parteI_sem_correcao")

    tab7b = ml.avaliar(ml.construir(True), X_tr7, y_tr7, X_te7, y_te7)
    print(tab7b.to_string())
    config.salvar_tabela(tab7b, "tab05_parteI_balanced")
    an.grafico_f1(tab7b, "Parte I (corte ≥ 7) — F1 por modelo, COM class_weight='balanced'",
                  "fig08_parteI_balanced")

    # ---- Seção 6 — Parte II (corte ≥ 6, equilibrado) --------------------- #
    secao("Seção 6 — Avaliação Parte II (corte ≥ 6)")
    tab6 = ml.avaliar(ml.construir(False), X_tr6, y_tr6, X_te6, y_te6)
    print(tab6.to_string())
    config.salvar_tabela(tab6, "tab06_parteII")
    an.grafico_f1(tab6, "Parte II (corte ≥ 6) — F1 por modelo (teste)", "fig09_parteII")

    estab = ml.estabilidade_cv(ml.construir(False), X_tr6, y_tr6, cv)
    print("Tabela 7 — estabilidade (F1 em validação cruzada, 5 dobras):")
    print(estab.to_string())
    config.salvar_tabela(estab, "tab07_estabilidade_cv")

    an.fig10_matrizes_confusao(rf_tunado, X_tr6, y_tr6, X_te6, y_te6)
    an.fig11_roc(X_tr6, y_tr6, X_te6, y_te6)
    an.figB_pr_e_testes(X_tr6, y_tr6, X_te6, y_te6, cv)

    # ---- Seção 7 — interpretabilidade ------------------------------------ #
    secao("Seção 7 — Interpretação dos Resultados")
    rf_interpret, melhores = ml.rf_para_interpretacao(grade_rf, X_tr6, y_tr6)
    print("Random Forest (interpretabilidade):", melhores)
    an.fig12_importancia(rf_interpret, variaveis)
    an.fig13_shap(rf_interpret, X_te6)
    an.fig14_permutacao(rf_interpret, X_te6, y_te6, variaveis)

    # ---- Manifesto final ------------------------------------------------- #
    secao("Concluído — artefatos gravados em results/")
    figs = sorted(config.DIR_FIGURAS.glob("*.png"))
    tabs = sorted(config.DIR_TABELAS.glob("*.csv"))
    print(f"Figuras ({len(figs)}):")
    for f in figs:
        print(f"  results/figuras/{f.name}")
    print(f"Tabelas ({len(tabs)}):")
    for t in tabs:
        print(f"  results/tabelas/{t.name}")


if __name__ == "__main__":
    main()
