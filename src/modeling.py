"""
modeling.py — Camada de modelagem (Seções 3, 4 e parte da 6 do relatório).

Grupo: Efraim Oliveira, Érica Tarsis, Ricardo Moraes, Rodrigo Bernardino, Thiago Galvão
POSTECH DTAT — 2026

PROPÓSITO
    Define os estimadores e o protocolo de avaliação do estudo. Tudo é montado
    como Pipeline com StandardScaler ANTES do classificador — a padronização
    fica dentro da validação cruzada, evitando vazamento de dados (data leakage).
    Inclui: validação cruzada estratificada, a fábrica dos 7 algoritmos, a
    rotina de métricas (acurácia, precisão, recall, F1, ROC-AUC), o F1 por
    validação cruzada (usado nos testes de engenharia de variáveis) e a
    otimização de hiperparâmetros por GridSearchCV.

USO
    from src import modeling as ml
    cv = ml.criar_cv()
    tab = ml.avaliar(ml.construir(), X_tr, y_tr, X_te, y_te)
    grade_lr, grade_rf = ml.otimizar(X_tr, y_tr, cv)

POR QUE PIPELINE (para quem não é da área)
    O "scaler" coloca todas as variáveis na mesma escala. Se ele aprendesse a
    escala usando também os dados de teste, haveria contaminação. O Pipeline
    garante que cada dobra de validação só "veja" seu próprio treino.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from . import config

SEED = config.SEED


def criar_cv() -> StratifiedKFold:
    """Validação cruzada estratificada de 5 dobras, embaralhada com SEED fixa."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def f1cv(X, y, est, cv) -> np.ndarray:
    """F1 por validação cruzada de um estimador encapsulado em Pipeline."""
    pipe = Pipeline([("sc", StandardScaler()), ("clf", est)])
    return cross_val_score(pipe, X, y, scoring="f1", cv=cv)


def construir(balanced: bool = False) -> dict[str, Pipeline]:
    """
    Fábrica dos 7 algoritmos, cada um como Pipeline (padronização + modelo).
    balanced=True ativa class_weight='balanced' onde aplicável (Parte I, corte >= 7).
    """
    cw = "balanced" if balanced else None
    est = {
        "Regressão Logística": LogisticRegression(max_iter=2000, random_state=SEED, class_weight=cw),
        "Random Forest": RandomForestClassifier(random_state=SEED, class_weight=cw),
        "SVM": SVC(probability=True, random_state=SEED, class_weight=cw),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=SEED, class_weight=cw),
        "Naive Bayes": GaussianNB(),
    }
    return {k: Pipeline([("sc", StandardScaler()), ("clf", v)]) for k, v in est.items()}


def avaliar(modelos: dict[str, Pipeline], Xtr, ytr, Xte, yte) -> pd.DataFrame:
    """Treina cada modelo e devolve a tabela de métricas ordenada por F1."""
    linhas = {}
    for nome, pipe in modelos.items():
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        try:
            auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
        except Exception:
            auc = np.nan
        linhas[nome] = {
            "Acurácia": accuracy_score(yte, pred),
            "Precisão": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred),
            "F1": f1_score(yte, pred),
            "ROC-AUC": auc,
        }
    return pd.DataFrame(linhas).T.sort_values("F1", ascending=False).round(4)


def estabilidade_cv(modelos: dict[str, Pipeline], Xtr, ytr, cv) -> pd.DataFrame:
    """F1 médio e desvio por validação cruzada de cada modelo (Tabela 7)."""
    linhas = {}
    for nome, pipe in modelos.items():
        sc = cross_val_score(pipe, Xtr, ytr, scoring="f1", cv=cv)
        linhas[nome] = {"F1_medio": round(sc.mean(), 4), "desvio": round(sc.std(), 4)}
    return pd.DataFrame(linhas).T.sort_values("F1_medio", ascending=False)


def otimizar(X_tr, y_tr, cv):
    """
    GridSearchCV (5 dobras, métrica F1) para Regressão Logística e Random Forest
    sobre o treino do corte >= 6. Retorna (grade_lr, grade_rf) já ajustados.
    """
    grade_lr = GridSearchCV(
        Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=3000, random_state=SEED, solver="liblinear"))]),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"]},
        scoring="f1", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    grade_rf = GridSearchCV(
        Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(random_state=SEED))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8, 12],
         "clf__min_samples_leaf": [1, 2, 4]},
        scoring="f1", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    return grade_lr, grade_rf


def rf_para_interpretacao(grade_rf, X_tr, y_tr):
    """Random Forest dedicada à interpretabilidade, com os melhores hiperparâmetros."""
    melhores = {k.replace("clf__", ""): v for k, v in grade_rf.best_params_.items()}
    rf = RandomForestClassifier(random_state=SEED, **melhores).fit(X_tr, y_tr)
    return rf, melhores
