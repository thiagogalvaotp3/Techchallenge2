# `src/` — Scripts auxiliares (pré-processamento e modelagem)

**Tech Challenge — Fase 2 · Classificação de qualidade de vinhos tintos (WineQT)**
Grupo (ordem alfabética): **Efraim Oliveira, Érica Tarsis, Ricardo Moraes, Rodrigo Bernardino, Thiago Galvão**
POSTECH DTAT — 2026

---

## Para que serve esta pasta

O notebook `notebooks/01_analise_e_modelagem.ipynb` é autocontido. Esta pasta
extrai a mesma lógica em **módulos reutilizáveis**, permitindo executar o estudo
inteiro — ou partes dele — sem abrir o Jupyter. Serve também como **rede de
segurança**: se o notebook falhar na banca, os resultados são reproduzidos por
linha de comando.

Posição no repositório (estrutura oficial do desafio):

```
wine-quality-classification/
├── data/              # WineQT.csv (entrada)
├── notebooks/         # 01_analise_e_modelagem.ipynb (análise e modelagem)
├── src/               # <- você está aqui (scripts auxiliares)
├── results/           # figuras e tabelas geradas por src/run_all.py (saída)
├── requirements.txt   # bibliotecas
└── README.md          # descrição do projeto
```

Os scripts **leem** de `../data/WineQT.csv` e **gravam** em `../results/`
(`figuras/` e `tabelas/`). Os caminhos são resolvidos automaticamente a partir
da posição dos arquivos, então funcionam de qualquer diretório.

### Fluxo de dependência entre os módulos

```
config.py    ── SEED = 43, caminhos, paleta, estilo, formatação pt-BR
    │  (importado por todos os demais)
    ▼
data_prep.py ── WineQT.csv → base limpa (1.018) → alvo binário → split 80/20
    │
    ├─► modeling.py  ── 7 algoritmos em Pipeline · CV · GridSearchCV · métricas
    │
    └─► analysis.py  ── EDA · engenharia de variáveis · ROC/PR · testes · SHAP
                  │
                  ▼
         run_all.py  ── orquestra as Seções 2→7 e grava tudo em results/
```

## Módulos (cada um com responsabilidade única)

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Semente (`SEED = 43`), caminhos do repositório, paleta, estilo e auxiliares de gráfico (formatação pt-BR, anotações, barras com sinal). É a única fonte de constantes. |
| `data_prep.py` | Carga do CSV, auditoria, **deduplicação** (1.143 → 1.018, ignorando a coluna `Id`), criação do alvo binário por limiar e divisão estratificada treino/teste. |
| `modeling.py` | Fábrica dos **7 algoritmos** (cada um em `Pipeline` com `StandardScaler` para evitar vazamento), métricas de avaliação, validação cruzada e otimização por `GridSearchCV`. |
| `analysis.py` | EDA (distribuições, outliers, correlações, VIF), experimentos de engenharia de variáveis (SO₂, interações, log — testados e descartados), figuras de avaliação (ROC, Precision-Recall, matrizes de confusão), testes de significância (McNemar, pareado, bootstrap) e interpretabilidade (importância, SHAP, permutação). |
| `run_all.py` | **Orquestrador**: executa tudo na ordem do relatório (Seções 2–7) e grava cada artefato em `results/`. |
| `__init__.py` | Marca a pasta como pacote Python (`import src`). |

## Como executar

A partir da **raiz do repositório** (`wine-quality-classification/`):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_all.py              # ou: python -m src.run_all
```

Saída esperada: **16 figuras** em `results/figuras/` (**15** caso a biblioteca
SHAP não esteja instalada — a Figura 13 é opcional) e **11 tabelas** em
`results/tabelas/` (CSV com separador `;` e decimal `,`, prontos para Excel pt-BR).

### Uso como biblioteca

```python
from src import data_prep as dp, modeling as ml, analysis as an

df = dp.preparar_base()                         # carrega + deduplica (1.018)
variaveis = dp.listar_variaveis(df)
X_tr, X_te, y_tr, y_te = dp.dividir_treino_teste(df, variaveis, corte=6)
tabela = ml.avaliar(ml.construir(), X_tr, y_tr, X_te, y_te)
print(tabela)
```

## Notas técnicas

- **Reprodutibilidade.** `SEED = 43` em todos os passos estocásticos. As
  métricas-âncora batem com o relatório: Random Forest F1(CV) = 0,7623 ± 0,015;
  Regressão Logística F1(CV) = 0,7547; Gradient Boosting F1(teste) = 0,7895.
- **Dependências de estatística.** O VIF e o teste de McNemar usam
  **`statsmodels`**; o teste pareado por dobras usa **`scipy`**. Ambos precisam
  constar no `requirements.txt`. O `scipy` já vem como dependência do
  scikit-learn, mas o `statsmodels` **não** — confirme que ele está congelado no
  `requirements.txt` antes de distribuir o projeto.
- **SHAP é opcional.** A Figura 13 (SHAP) só é gerada se a biblioteca estiver
  instalada; caso contrário, a etapa é ignorada com aviso (sem interromper a
  execução).
- **Compatibilidade Matplotlib.** Os rótulos de boxplot são definidos
  manualmente para funcionar tanto em versões antigas quanto novas do Matplotlib
  (que renomeou o argumento `labels` para `tick_labels`). Valores de p de testes
  muito sensíveis (McNemar) podem variar minimamente conforme a versão do
  `scikit-learn`/`statsmodels`, mas as **conclusões** (diferenças no topo não
  significativas) se mantêm.
- **Governança.** Todas as estatísticas usam a base limpa (1.018), nunca a bruta
  (1.143). A binarização ≥ 6 equilibra as classes (~54% / 46%); ≥ 7 gera
  desbalanceamento severo — esse contraste é o argumento metodológico central.
