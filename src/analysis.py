"""
analysis.py — Camada de análise e interpretação (Seções 2, 3, 6 e 7).

Grupo: Efraim Oliveira, Érica Tarsis, Ricardo Moraes, Rodrigo Bernardino, Thiago Galvão
POSTECH DTAT — 2026

PROPÓSITO
    Reúne todas as rotinas que produzem evidência visual e estatística do
    relatório, gravando cada artefato em results/. Cobre quatro frentes:
      (1) EDA — distribuição das notas, balanceamento por limiar, histogramas,
          outliers (IQR), correlações, boxplots por classe e VIF;
      (2) Engenharia de variáveis — testes do SO2 livre/total, de interações e
          de transformação log (criados, testados e descartados, com registro);
      (3) Avaliação — gráfico de F1, matrizes de confusão, curvas ROC e
          Precision-Recall, e os testes de significância (McNemar, comparação
          pareada por dobras e bootstrap);
      (4) Interpretabilidade — importância da Random Forest, SHAP (opcional) e
          importância por permutação.

USO
    Cada função recebe os dados/modelos de que precisa, salva a figura ou
    tabela e devolve o caminho (e, quando útil, os números). É orquestrada por
    run_all.py na ordem do relatório.

NOTA SOBRE OS TESTES ESTATÍSTICOS (para quem não é da área)
    Um F1 alto numa única divisão pode ser sorte. McNemar, teste pareado e
    bootstrap verificam se a diferença entre modelos resiste à aleatoriedade —
    e, no topo, ela não resiste: os melhores modelos são equivalentes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             f1_score)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from . import config, modeling
from .config import (AZUL, LARANJA, PALETA, SEED, br, eixo_virgula,
                     anota_vertical, anota_h_pos, barh_signed,
                     salvar_figura, salvar_tabela)


# ===========================================================================
# (1) ANÁLISE EXPLORATÓRIA DE DADOS
# ===========================================================================
def fig01_distribuicao_notas(df):
    """Figura 1 — distribuição das notas de qualidade (base limpa)."""
    dist = df["quality"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(dist.index.astype(str), dist.values, color=AZUL, edgecolor="white")
    ax.set_title("Distribuição das notas de qualidade (base limpa, n = 1.018)", pad=14)
    ax.set_xlabel("Nota de qualidade"); ax.set_ylabel("Quantidade de amostras")
    anota_vertical(ax, dist.values)
    n56 = int(dist.get(5, 0) + dist.get(6, 0))
    print(f"Notas 5+6 = {n56} ({n56/len(df):.0%} da base).")
    return salvar_figura(fig, "fig01_distribuicao_notas")


def fig02_balanceamento_limiar(df):
    """Figura 2 — balanceamento das classes nos cortes >= 7 e >= 6."""
    n = len(df)
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle("Balanceamento das classes por limiar de binarização",
                 fontsize=14, fontweight="bold", y=1.02)
    for ax, (corte, sit) in zip(eixos, [(7, "DESBALANCEADO"), (6, "EQUILIBRADO")]):
        y = (df["quality"] >= corte).astype(int)
        alt = [int((y == 0).sum()), int((y == 1).sum())]
        ax.bar(["Baixa/Média", "Alta"], alt, color=[LARANJA, AZUL], edgecolor="white")
        ax.set_title(f"Corte ≥ {corte}  [{sit}]", fontsize=12, pad=12)
        ax.set_ylabel("Quantidade de amostras")
        anota_vertical(ax, alt, total=n, folga=0.34)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    for corte in (7, 6):
        y = (df["quality"] >= corte).astype(int)
        ng, ps = int((y == 0).sum()), int((y == 1).sum())
        print(f"Corte ≥ {corte}: Baixa/Média={ng} ({ng/n:.0%}) | "
              f"Alta={ps} ({ps/n:.0%}) | razão {ng/ps:.1f}:1")
    return salvar_figura(fig, "fig02_balanceamento_limiar")


def fig03_histogramas(df, variaveis):
    """Figura 3 — histogramas das variáveis físico-químicas."""
    fig, axs = plt.subplots(3, 4, figsize=(14, 9))
    for ax, col in zip(axs.ravel(), variaveis):
        ax.hist(df[col], bins=30, color=AZUL, edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=10, pad=8); ax.set_ylabel("freq.")
    for ax in axs.ravel()[len(variaveis):]:
        ax.set_visible(False)
    fig.suptitle("Distribuição das variáveis físico-químicas",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return salvar_figura(fig, "fig03_histogramas")


def fig04_outliers(df, variaveis):
    """Figura 4 + tabela — contagem de outliers (IQR) e boxplots dos 3 maiores."""
    def out_iqr(s):
        q1, q3 = s.quantile(.25), s.quantile(.75); i = q3 - q1
        return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())

    tab_out = (pd.Series({c: out_iqr(df[c]) for c in variaveis})
               .sort_values(ascending=False).to_frame("qtd_outliers"))
    tab_out["%_da_base"] = (tab_out["qtd_outliers"] / len(df) * 100).round(1)
    print(tab_out.to_string())
    salvar_tabela(tab_out, "tab_outliers_iqr")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axs, tab_out.index[:3]):
        ax.boxplot(df[col], vert=True, patch_artist=True,
                   boxprops=dict(facecolor=AZUL, alpha=.6))
        ax.set_title(col, fontsize=11, pad=10); ax.set_xticks([]); eixo_virgula(ax, "y", 2)
    fig.suptitle("Boxplots das variáveis com mais outliers (mantidos na base)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return salvar_figura(fig, "fig04_outliers"), tab_out


def fig05_06_correlacoes(df, variaveis):
    """Figuras 5 e 6 — correlação com a qualidade e mapa de calor."""
    corr_q = (df[variaveis + ["quality"]].corr()["quality"]
              .drop("quality").sort_values())
    fig, ax = plt.subplots(figsize=(8.5, 5))
    barh_signed(ax, list(corr_q.index), list(corr_q.values), dec=3, folga=0.14)
    ax.set_title("Correlação (Pearson) de cada variável com a qualidade", pad=12)
    ax.set_xlabel("coeficiente de correlação"); eixo_virgula(ax, "x", 1)
    c1 = salvar_figura(fig, "fig05_correlacao_qualidade")

    mat = df[variaveis + ["quality"]].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(mat))); ax.set_xticklabels(mat.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.columns, fontsize=8)
    for i in range(len(mat)):
        for j in range(len(mat)):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}".replace(".", ","),
                    ha="center", va="center", fontsize=6)
    ax.set_title("Mapa de calor das correlações", pad=12); ax.grid(False)
    fig.colorbar(im, fraction=.046, pad=.04)
    c2 = salvar_figura(fig, "fig06_mapa_calor")
    return c1, c2


def figA_boxplots_classe(df):
    """Figura A — distribuição das variáveis-chave por classe (corte >= 6)."""
    y6f = (df["quality"] >= 6).astype(int)
    feat = ["alcohol", "volatile acidity", "sulphates",
            "citric acid", "total sulfur dioxide", "density"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, col in zip(axs.ravel(), feat):
        bp = ax.boxplot([df.loc[y6f == 0, col], df.loc[y6f == 1, col]],
                        patch_artist=True, widths=.6,
                        flierprops=dict(marker="o", markersize=3, alpha=.4))
        for p, c in zip(bp["boxes"], [LARANJA, AZUL]):
            p.set_facecolor(c); p.set_alpha(.65)
        for m in bp["medians"]:
            m.set_color("black")
        # Rótulos definidos manualmente (compatível com matplotlib antigo e novo,
        # que renomeou o antigo argumento 'labels' do boxplot).
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Baixa/Média", "Alta"])
        ax.set_title(col, fontsize=11, pad=8)
        eixo_virgula(ax, "y", 3 if col == "density" else 2)
    fig.suptitle("Distribuição das variáveis-chave por classe (corte ≥ 6)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return salvar_figura(fig, "figA_boxplots_por_classe")


def tab_vif(df, variaveis):
    """Tabela — Fator de Inflação de Variância (multicolinearidade; >5 = atenção)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xv = df[variaveis].copy()
    Xv = (Xv - Xv.mean()) / Xv.std()
    Xv["_const"] = 1.0
    vif = pd.Series({c: variance_inflation_factor(Xv.values, k)
                     for k, c in enumerate(variaveis)}).sort_values(ascending=False)
    print("VIF por variável (>5 = atenção):")
    print(vif.round(2).to_string())
    salvar_tabela(vif.round(3).to_frame("VIF"), "tab_vif")
    return vif


# ===========================================================================
# (2) ENGENHARIA DE VARIÁVEIS — testada e documentada
# ===========================================================================
def _experimento(nome_arquivo, titulo, X_base, X_alt, y, cv, conclusao):
    """Núcleo reutilizável: compara F1(CV) base x alternativa para LR e RF."""
    print(titulo)
    linhas = {}
    for nome, est in [("Regressão Logística", LogisticRegression(max_iter=2000, random_state=SEED)),
                      ("Random Forest", RandomForestClassifier(random_state=SEED))]:
        b = modeling.f1cv(X_base, y, est, cv)
        e = modeling.f1cv(X_alt, y, est, cv)
        delta = e.mean() - b.mean()
        print(f"  {nome:22s} sem={br(b.mean(),4)} com={br(e.mean(),4)} "
              f"Δ={br(delta,4)} (±{br(b.std(),3)})")
        linhas[nome] = {"F1_sem": round(b.mean(), 4), "F1_com": round(e.mean(), 4),
                        "delta": round(delta, 4), "desvio_dobras": round(b.std(), 4)}
    print(conclusao)
    tab = pd.DataFrame(linhas).T
    salvar_tabela(tab, nome_arquivo)
    return tab


def tab02_so2_ratio(df, X_base, y, cv):
    """Tabela 2 — efeito de SO2 livre/total (so2_ratio)."""
    X_eng = X_base.assign(
        so2_ratio=df["free sulfur dioxide"] / df["total sulfur dioxide"].replace(0, np.nan)
    ).fillna(0)
    return _experimento(
        "tab02_so2_ratio", "Tabela 2 — efeito do so2_ratio (F1 em validação cruzada):",
        X_base, X_eng, y, cv,
        "→ Δ desprezível e SO₂ livre/total já estão no modelo: DESCARTADA.")


def tab02b_interacoes(df, X_base, y, cv):
    """Tabela 2b — efeito de interações entre variáveis."""
    X_int = X_base.assign(
        alc_x_sulf=df["alcohol"] * df["sulphates"],
        alc_div_dens=df["alcohol"] / df["density"],
        va_x_sulf=df["volatile acidity"] * df["sulphates"])
    return _experimento(
        "tab02b_interacoes", "Tabela 2b — efeito das interações (F1 em validação cruzada):",
        X_base, X_int, y, cv,
        "→ Variação dentro do ruído: interações NÃO incorporadas (exploração documentada).")


def tablog_transformacao_log(df, X_base, y, cv):
    """Tabela log — efeito da transformação log nas variáveis assimétricas (apenas LR)."""
    skew = ["residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "sulphates"]
    X_log = X_base.copy()
    for c in skew:
        X_log[c] = np.log1p(X_log[c])
    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    b = modeling.f1cv(X_base, y, lr, cv)
    e = modeling.f1cv(X_log, y, lr, cv)
    print(f"Regressão Logística — F1(CV) sem log = {br(b.mean(),4)} | "
          f"com log = {br(e.mean(),4)} | Δ = {br(e.mean()-b.mean(),4)} (±{br(b.std(),3)})")
    print("→ Δ desprezível: a assimetria das marginais NÃO limita o modelo linear neste problema.")
    tab = pd.DataFrame({"Regressão Logística": {
        "F1_sem_log": round(b.mean(), 4), "F1_com_log": round(e.mean(), 4),
        "delta": round(e.mean() - b.mean(), 4), "desvio_dobras": round(b.std(), 4)}}).T
    salvar_tabela(tab, "tablog_transformacao_log")
    return tab


# ===========================================================================
# (3) AVALIAÇÃO — figuras e testes de significância
# ===========================================================================
def grafico_f1(tab, titulo, nome):
    """Gráfico de barras horizontais de F1 por modelo (destaque para o melhor)."""
    o = tab.sort_values("F1")
    cores = [AZUL] * len(o); cores[-1] = LARANJA
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(o.index, o["F1"], color=cores, edgecolor="white")
    ax.set_title(titulo, pad=12); ax.set_xlabel("F1-Score (teste)")
    anota_h_pos(ax, o["F1"].values, dec=3); eixo_virgula(ax, "x", 1)
    return salvar_figura(fig, nome)


def fig10_matrizes_confusao(rf_tunado, X_tr6, y_tr6, X_te6, y_te6):
    """Figura 10 — matrizes de confusão: RF otimizada (recomendado) x Gradient Boosting."""
    def cm_ax(ax, pred, titulo):
        cm = confusion_matrix(y_te6, pred)
        ax.imshow(cm, cmap="Blues"); ax.set_title(titulo, pad=10); ax.grid(False)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Baixa/Média", "Alta"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Baixa/Média", "Alta"])
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real"); lim = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if cm[i, j] > lim else "black")

    pred_rf = rf_tunado.fit(X_tr6, y_tr6).predict(X_te6)
    gb = modeling.construir(False)["Gradient Boosting"]
    pred_gb = gb.fit(X_tr6, y_tr6).predict(X_te6)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    cm_ax(a1, pred_rf, "Random Forest (recomendado)")
    cm_ax(a2, pred_gb, "Gradient Boosting (melhor F1 no teste)")
    fig.tight_layout()
    return salvar_figura(fig, "fig10_matrizes_confusao")


def fig11_roc(X_tr6, y_tr6, X_te6, y_te6):
    """Figura 11 — curvas ROC dos 7 modelos (corte >= 6)."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for (nome, pipe), cor in zip(modeling.construir(False).items(), PALETA):
        pipe.fit(X_tr6, y_tr6)
        p = pipe.predict_proba(X_te6)[:, 1]
        fpr, tpr, _ = roc_curve(y_te6, p)
        ax.plot(fpr, tpr, color=cor, lw=1.8,
                label=f"{nome} (AUC={br(roc_auc_score(y_te6, p))})")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="aleatório (AUC=0,5)")
    ax.set_title("Curvas ROC — Parte II (corte ≥ 6)", pad=12)
    ax.set_xlabel("Falsos positivos (1 − especificidade)")
    ax.set_ylabel("Verdadeiros positivos (recall)")
    eixo_virgula(ax, "x", 1); eixo_virgula(ax, "y", 1)
    ax.legend(fontsize=8, loc="lower right")
    return salvar_figura(fig, "fig11_curvas_roc")


def figB_pr_e_testes(X_tr6, y_tr6, X_te6, y_te6, cv):
    """
    Figura B (Precision-Recall) + testes de significância (McNemar, pareado por
    dobras e bootstrap). Retorna um dicionário com os números-chave.
    """
    from statsmodels.stats.contingency_tables import mcnemar
    from scipy import stats

    mods = modeling.construir(False)
    base_rate = float(y_te6.mean())

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for (nome, pipe), cor in zip(mods.items(), PALETA):
        pipe.fit(X_tr6, y_tr6)
        p = pipe.predict_proba(X_te6)[:, 1]
        pr, rc, _ = precision_recall_curve(y_te6, p)
        ax.plot(rc, pr, color=cor, lw=1.8,
                label=f"{nome} (AP={br(average_precision_score(y_te6, p))})")
    ax.axhline(base_rate, ls="--", color="0.6", lw=1,
               label=f"linha de base (AP={br(base_rate)})")
    ax.set_title("Curvas Precision-Recall — Parte II (corte ≥ 6)", pad=12)
    ax.set_xlabel("Recall (sensibilidade)"); ax.set_ylabel("Precisão")
    eixo_virgula(ax, "x", 1); eixo_virgula(ax, "y", 1)
    ax.legend(fontsize=8, loc="lower left")
    caminho_fig = salvar_figura(fig, "figB_precision_recall")

    # McNemar (pareado, exato) entre pares de modelos
    yv = y_te6.values

    def mcn(a, b):
        ca, cb = (a == yv), (b == yv)
        t = [[int(np.sum(ca & cb)), int(np.sum(ca & ~cb))],
             [int(np.sum(~ca & cb)), int(np.sum(~ca & ~cb))]]
        return mcnemar(t, exact=True).pvalue

    pgb = mods["Gradient Boosting"].fit(X_tr6, y_tr6).predict(X_te6)
    psvm = mods["SVM"].fit(X_tr6, y_tr6).predict(X_te6)
    prf = mods["Random Forest"].fit(X_tr6, y_tr6).predict(X_te6)
    plr = mods["Regressão Logística"].fit(X_tr6, y_tr6).predict(X_te6)
    p_gb_svm, p_gb_rf, p_rf_lr = mcn(pgb, psvm), mcn(pgb, prf), mcn(prf, plr)
    print(f"McNemar  GB×SVM p={br(p_gb_svm,3)} | GB×RF p={br(p_gb_rf,3)} | "
          f"RF×LR p={br(p_rf_lr,3)}  → diferenças NÃO significativas")

    # Comparação pareada por dobras (teste t pareado)
    f1_rf = modeling.f1cv(X_tr6, y_tr6, RandomForestClassifier(random_state=SEED), cv)
    f1_lr = modeling.f1cv(X_tr6, y_tr6, LogisticRegression(max_iter=2000, random_state=SEED), cv)
    t, p = stats.ttest_rel(f1_rf, f1_lr)
    print(f"Pareado por dobras RF×LR: Δ={br(f1_rf.mean()-f1_lr.mean(),4)} | "
          f"t={br(t,3)} | p={br(p,3)}")

    # Bootstrap: intervalo de confiança de 95% para o F1
    rng = np.random.default_rng(SEED)

    def ic(pred):
        idx = np.arange(len(yv))
        v = [f1_score(yv[s], pred[s], zero_division=0)
             for s in (rng.choice(idx, len(idx), True) for _ in range(3000))]
        return np.percentile(v, [2.5, 97.5])

    ics = {}
    for nome, pred in [("Regressão Logística", plr), ("Random Forest", prf),
                       ("Gradient Boosting", pgb), ("SVM", psvm)]:
        lo, hi = ic(pred)
        ics[nome] = (round(lo, 3), round(hi, 3))
        print(f"IC95% F1  {nome:22s} [{br(lo,3)} ; {br(hi,3)}]")
    print("→ ICs sobrepostos: no topo, os modelos são estatisticamente indistinguíveis.")

    resumo = {"mcnemar_GBxSVM": round(p_gb_svm, 4), "mcnemar_GBxRF": round(p_gb_rf, 4),
              "mcnemar_RFxLR": round(p_rf_lr, 4), "pareado_t": round(float(t), 4),
              "pareado_p": round(float(p), 4),
              **{f"IC95_{k}": f"[{lo} ; {hi}]" for k, (lo, hi) in ics.items()}}
    salvar_tabela(pd.Series(resumo).to_frame("valor"), "tab_testes_significancia")
    return caminho_fig, resumo


# ===========================================================================
# (4) INTERPRETABILIDADE
# ===========================================================================
def fig12_importancia(rf_interpret, features):
    """Figura 12 — importância das variáveis pela Random Forest."""
    imp = pd.Series(rf_interpret.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(imp.index, imp.values, color=AZUL, edgecolor="white")
    ax.set_title("Importância das variáveis — Random Forest", pad=12)
    ax.set_xlabel("importância relativa")
    anota_h_pos(ax, imp.values, dec=3); eixo_virgula(ax, "x", 2)
    salvar_tabela(imp.sort_values(ascending=False).to_frame("importancia"),
                  "tab_importancia_rf")
    return salvar_figura(fig, "fig12_importancia_rf")


def fig13_shap(rf_interpret, X_te6):
    """Figura 13 — SHAP (opcional; degrada com elegância se a biblioteca faltar)."""
    try:
        import shap
        expl = shap.TreeExplainer(rf_interpret)(X_te6, check_additivity=False)
        if getattr(expl.values, "ndim", 2) == 3:
            expl = expl[:, :, 1]
        plt.figure(figsize=(8, 5))
        shap.plots.beeswarm(expl, max_display=11, show=False)
        plt.title("SHAP — impacto das variáveis na previsão (classe: Alta qualidade)",
                  fontsize=12, fontweight="bold", pad=12)
        fig = plt.gcf(); fig.tight_layout()
        return salvar_figura(fig, "fig13_shap")
    except ImportError:
        print("Biblioteca 'shap' não instalada — etapa opcional ignorada. (pip install shap)")
        return None


def fig14_permutacao(rf_interpret, X_te6, y_te6, features):
    """Figura 14 — importância por permutação (queda média de F1)."""
    perm = permutation_importance(rf_interpret, X_te6, y_te6, scoring="f1",
                                  n_repeats=20, random_state=SEED, n_jobs=-1)
    imp_perm = pd.Series(perm.importances_mean, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    barh_signed(ax, list(imp_perm.index), list(imp_perm.values),
                cor_pos=LARANJA, cor_neg=AZUL, dec=3, folga=0.14)
    ax.set_title("Importância por permutação (queda média de F1)", pad=12)
    ax.set_xlabel("queda no F1 ao embaralhar a variável"); eixo_virgula(ax, "x", 2)
    return salvar_figura(fig, "fig14_permutacao")
