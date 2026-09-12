# 🍷 Classificação da Qualidade de Vinhos Tintos — WineQT

**Tech Challenge — Fase 2 · POSTECH DTAT — 2026**

Projeto de *machine learning* que prevê, a partir de características
físico-químicas, se um vinho tinto é de **alta qualidade** ou de **qualidade
baixa/média** — um problema de **classificação binária**.

> **Em uma frase (para quem não é da área):** ensinamos o computador a "provar" o
> vinho pelos números do laboratório (teor de álcool, acidez, açúcar etc.) e a
> dizer se ele tende a ser bem avaliado — e, mais importante, a revelar *quais*
> características mais pesam nessa nota.

**Integrantes (ordem alfabética):** Efraim Oliveira · Érica Tarsis · Ricardo
Moraes · Rodrigo Bernardino · Thiago Galvão

---

## 📌 Objetivo

Construir e comparar modelos de classificação que separem vinhos de **alta**
qualidade dos demais, e **interpretar** o resultado para identificar os fatores
físico-químicos determinantes. O entregável não é só um modelo: é uma decisão
fundamentada sobre **qual algoritmo levar à produção** e **quais variáveis
monitorar** para sustentar a qualidade.

---

## 🗂️ Sobre os dados (WineQT)

O `WineQT.csv` traz medições físico-químicas de vinhos tintos e a respectiva nota
de qualidade. São **11 variáveis preditoras** + o alvo `quality`:

| Variável | Descrição (resumo) | Variável | Descrição (resumo) |
|---|---|---|---|
| `fixed acidity` | acidez fixa | `total sulfur dioxide` | SO₂ total |
| `volatile acidity` | acidez volátil (excesso → sabor avinagrado) | `density` | densidade |
| `citric acid` | ácido cítrico (frescor) | `pH` | acidez/alcalinidade |
| `residual sugar` | açúcar residual | `sulphates` | sulfatos (antimicrobiano) |
| `chlorides` | cloretos (sal) | `alcohol` | teor alcoólico |
| `free sulfur dioxide` | SO₂ livre (conservante ativo) | | |

> Alvo `quality`: nota inteira (faixa ~3 a 8). A coluna técnica `Id` é descartada.

### Auditoria e limpeza (a primeira armadilha)

A coluna `Id` **mascarava duplicatas**: cada linha parecia única por ter um `Id`
diferente. Removendo o `Id` antes da checagem, encontramos **125 linhas
idênticas**.

```
Base bruta:  1.143 amostras
              └─ 125 duplicatas reais (reveladas ao ignorar 'Id')
Base limpa:  1.018 amostras   ← toda estatística do projeto usa esta base
```

> **Regra de governança:** todas as métricas usam a base limpa (**1.018**), nunca
> a bruta (1.143). Relatar números sobre dados duplicados inflaria artificialmente
> a confiança nos resultados.

---

## 🔬 Metodologia (a jornada)

```
WineQT.csv (1.143)
      │  remoção de duplicatas (ignora a coluna 'Id')
      ▼
Base limpa (1.018)
      │  alvo binário: quality ≥ 6 → "Alta"   (equilíbrio 54% / 46%)
      ▼
Split estratificado 80/20   (treino 814 · teste 204 · SEED = 43)
      │
      ▼
7 modelos, cada um em Pipeline (StandardScaler + classificador)  → sem vazamento
      │  validação cruzada (5 dobras) + GridSearchCV
      ▼
Avaliação (F1, ROC-AUC, Precision-Recall)
      │  + testes de significância (McNemar · pareado por dobras · bootstrap)
      ▼
Interpretabilidade (importância da Random Forest · SHAP · permutação)
      │  convergência: álcool, sulphates, volatile acidity
      ▼
Recomendação: Random Forest  (escolha operacional, não métrica)
```

### Por que o corte em ≥ 6 e não em ≥ 7?

Binarizar em **≥ 7** ("apenas excelentes") parece intuitivo, mas gera
**desbalanceamento severo** (~1 vinho "Alto" para cada 6 "Baixo/Médio"). Nesse
cenário, um modelo que simplesmente "chuta tudo como Baixo/Médio" acerta ~86% — a
**ilusão da acurácia**. O corte em **≥ 6** equilibra as classes (**54% / 46%**),
tornando as métricas honestas e o aprendizado significativo. Esse contraste é o
**argumento metodológico central** do trabalho.

### Por que `Pipeline` com `StandardScaler`?

A padronização das variáveis fica **dentro** do *pipeline*, ou seja, é reaprendida
em cada dobra usando só os dados de treino daquela dobra. Isso evita **vazamento
de dados** (*data leakage*) — o erro silencioso de deixar o modelo "espiar" o
conjunto de teste, que produz métricas otimistas e irreais.

---

## 📊 Principais resultados

Desempenho no conjunto de teste (Parte II, corte ≥ 6). Os 7 algoritmos foram
avaliados; o ranking completo está no notebook.

| Modelo | F1 (teste) | Destaque |
|---|---|---|
| **Gradient Boosting** | **0,789** | Melhor F1 e melhor ROC-AUC (**0,826**) no teste |
| SVM | 0,784 | Segundo melhor F1 |
| **Random Forest** | 0,768 | **Mais estável na validação cruzada (0,7623 ± 0,015) → recomendado** |
| Regressão Logística | 0,758 | Melhor *Average Precision* (**0,833**); baseline interpretável |

> ### ✅ Modelo recomendado: **Random Forest**
> No teste, o Gradient Boosting lidera o F1 por margem estreita (0,789 vs 0,768).
> Porém, os testes de significância (McNemar, comparação pareada por dobras e
> *bootstrap*) mostram que as diferenças no topo **não são estatisticamente
> significativas** — os intervalos de confiança se sobrepõem e os valores de p
> ficam acima de 0,5. **Quando os modelos empatam na estatística, a escolha passa
> a ser operacional:** a Random Forest entrega a maior **estabilidade** entre
> dobras e boa interpretabilidade, reduzindo o risco em produção.

Hiperparâmetros vencedores (GridSearchCV, 5 dobras, métrica F1):

- **Regressão Logística:** `C = 0,1`, `penalty = l2` → F1(CV) = 0,7564
- **Random Forest:** `n_estimators = 400`, `max_depth = None`,
  `min_samples_leaf = 2` → F1(CV) = 0,7696

---

## 🧠 Principais achados

**Três variáveis dominam a previsão**, de forma **convergente** entre três métodos
independentes (importância da Random Forest, valores SHAP e importância por
permutação):

1. **Álcool** (`alcohol`) — maior teor associa-se a maior qualidade;
2. **Sulfatos** (`sulphates`);
3. **Acidez volátil** (`volatile acidity`) — em excesso, deprecia a qualidade.

> **Conexão com a decisão de negócio:** o esforço de melhoria de qualidade tende a
> ter melhor retorno ao monitorar e controlar essas três alavancas, em vez de
> distribuir atenção igualmente por todas as 11 variáveis.

**Multicolinearidade (VIF):** `fixed acidity` (~8,0) e `density` (~6,6) ficam
acima do limiar de atenção (5). Como o objetivo é **classificar** (e não
interpretar coeficientes lineares) e os modelos de árvore são robustos a
colinearidade, as variáveis foram **mantidas** — decisão registrada.

**Engenharia de variáveis (testada e descartada):** razão de SO₂ livre/total,
interações (ex.: álcool × sulfatos) e transformação logarítmica das variáveis
assimétricas foram avaliadas e **não melhoraram** o F1 além do ruído. O registro
desses experimentos negativos faz parte da transparência metodológica.

---

## 📁 Estrutura do repositório

```
wine-quality-classification/
├── data/                          # WineQT.csv (entrada)
├── notebooks/
│   └── 01_analise_e_modelagem.ipynb   # análise e modelagem (notebook principal)
├── src/                           # código modular reproduzível (ver src/README.md)
│   ├── config.py                  # SEED, caminhos, paleta, estilo
│   ├── data_prep.py               # carga, deduplicação, alvo, split
│   ├── modeling.py                # 7 modelos, CV, GridSearchCV, métricas
│   ├── analysis.py                # EDA, eng. de variáveis, ROC/PR, testes, SHAP
│   ├── run_all.py                 # orquestrador ponta a ponta
│   └── __init__.py
├── results/                       # figuras e tabelas geradas por src/run_all.py
├── requirements.txt               # dependências (versões congeladas)
└── README.md                      # este arquivo
```

---

## ⚙️ Como reproduzir

### Pré-requisitos
- **Python 3.12** e `venv`

### Instalação
```bash
git clone <INSERIR_LINK_DO_REPOSITORIO>
cd wine-quality-classification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Execução
```bash
# Opção 1 — notebook (análise completa, célula a célula)
jupyter notebook notebooks/01_analise_e_modelagem.ipynb

# Opção 2 — pipeline por linha de comando (reproduz tudo e grava em results/)
python src/run_all.py
```

Saída esperada de `run_all.py`: **16 figuras** em `results/figuras/` (15 caso a
biblioteca SHAP não esteja instalada) e **11 tabelas** em `results/tabelas/`.

> 📄 Guia detalhado de ambiente (macOS/Windows/VS Code, versões testadas e
> solução de problemas) em **`SETUP_AMBIENTE_POSTECH_DTAT.md`**.

---

## 🛡️ Governança e reprodutibilidade

- **Semente fixa** (`SEED = 43`, primo, por decisão de grupo) em todos os passos
  estocásticos — garante que qualquer pessoa reproduza os mesmos números.
- **Sem vazamento de dados:** `StandardScaler` sempre dentro do `Pipeline`,
  após o *split*.
- **Base única de análise:** todas as estatísticas sobre as 1.018 amostras
  deduplicadas.
- **Decisões documentadas:** cada escolha (limiar, outliers mantidos, variáveis
  descartadas) tem justificativa registrada.
- **Rede de segurança:** a pasta `src/` reproduz o estudo por linha de comando,
  caso o notebook falhe na avaliação.
- **Princípios norteadores:** Lei de Murphy (antecipar o que pode falhar), Lei de
  Kidlin (perguntar diante de ambiguidade) e Lei de Gilbert (clareza antes da
  produção).

---

## 📦 Entregáveis e links

| Entregável | Onde |
|---|---|
| Notebook (análise + modelagem) | `notebooks/01_analise_e_modelagem.ipynb` |
| Relatório técnico (PDF) | `relatorio_tecnico_fase_2.pdf` |
| Código modular reproduzível | `src/` — ver `src/README.md` |
| Vídeo de apresentação (≤ 5 min) | `https://youtu.be/24j8LucBWxU?si=hbToJGP8F1KqrZIN` |
| Repositório público | `https://github.com/lgmricardo/Techchallenge2-G12/` |

---

## 👥 Integrantes (ordem alfabética)

- Efraim Oliveira
- Érica Tarsis
- Ricardo Moraes
- Rodrigo Bernardino
- Thiago Galvão

---

## 📚 Fonte dos dados e referências

O conjunto **WineQT** deriva do *Wine Quality* (vinho tinto português "Vinho
Verde"), originalmente publicado por **Cortez et al. (2009)** no *UCI Machine
Learning Repository* e redistribuído na plataforma Kaggle. As referências
completas, em formato **ABNT**, constam no relatório técnico (Seção 11).
