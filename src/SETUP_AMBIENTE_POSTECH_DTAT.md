# 📘 Manual Completo: Configuração de Ambiente Python para POSTECH DTAT Tech Challenge

**Última atualização:** 13 de junho de 2026  
**Versão:** 1.0  
**Público:** Efraim Oliveira, Érica Tarsis, Ricardo Moraes, Rodrigo Bernardino, Thiago Galvão

-----

## 📋 Índice

1. [Premissas e Problemas Resolvidos](#premissas)
1. [Configuração macOS (M4 e Intel)](#macos)
1. [Configuração Windows](#windows)
1. [Configuração VS Code](#vscode)
1. [Governança e Reproducibilidade](#governanca)

-----

## 🎯 Premissas e Problemas Resolvidos {#premissas}

### **Lei de Murphy em Ação**

Preparamo-nos para o pior. Documentamos **todos os problemas encontrados e soluções**:

|Problema                                   |Causa Raiz                            |Solução                                           |
|-------------------------------------------|--------------------------------------|--------------------------------------------------|
|`pandas==3.0.2 not found`                  |Versão não existe; Python < 3.10      |Usar pandas 2.2.3 com Python 3.12.13              |
|`numpy==2.4.4 not found`                   |NumPy 2.4.4 exige Python ≥3.12        |Instalar NumPy 2.0.2 ou posterior compatível      |
|Python 3.9 ativado apesar de 3.12 instalado|`/usr/bin/python3` aponta para sistema|Usar alias ou venv explicitamente                 |
|`error: externally-managed-environment`    |Proteção do Homebrew                  |**Usar ambiente virtual (`venv`)**                |
|Jupyter não reconhece venv no VS Code      |Kernel não selecionado corretamente   |Selecionar kernel manualmente: `./venv/bin/python`|

### **Lei de Kidlin: Ambiguidade = Perguntar**

Quando não temos informações completas, **perguntamos antes de agir**. Este manual incorpora:

- ✅ Verificações em cada passo
- ✅ Confirmações visuais esperadas
- ✅ Checklist para validação

### **Lei de Gilbert: Simplicidade Antes de Executar**

Cada seção explica:

1. **O que fazer** (passo técnico)
1. **Por que fazer** (contexto para leigos)
1. **Como verificar** (validação)
1. **Exemplos visuais** (outputs esperados)

-----

## 🍎 Configuração macOS (M4 e Intel) {#macos}

### **Versões Testadas e Validadas**

|Componente      |Versão        |Compatibilidade|
|----------------|--------------|---------------|
|**macOS**       |13.x - 15.x   |✅ Testado      |
|**Python**      |3.12.13       |✅ Recomendado  |
|**Homebrew**    |4.x+          |✅ Obrigatório  |
|**pandas**      |2.2.3         |✅ Testado      |
|**numpy**       |2.0.2 ou 2.4.4|✅ Compatível   |
|**scikit-learn**|1.5.0+        |✅ Testado      |

-----

### **Passo 1: Instalar Homebrew**

**O que é:** Gerenciador de pacotes para macOS (como “App Store” para desenvolvedores)

**Execute:**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**O que esperar:**

- Pedirá sua senha do Mac → digite e pressione Enter
- Mensagens verdes com ✅ ao final
- Levará ~5-10 minutos

**Verifique:**

```bash
brew --version
```

**Output esperado:**

```
Homebrew 4.x.x (ou superior)
```

-----

### **Passo 2: Configurar PATH do Homebrew**

**O que é:** PATH é a “lista de caminhos” onde o Mac procura programas

**Execute:**

```bash
echo >> /Users/$USER/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/$USER/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

**Recarregue:**

```bash
source ~/.zprofile
```

**Verifique:**

```bash
brew --version
```

**Output esperado:** `Homebrew 4.x.x`

-----

### **Passo 3: Instalar Python 3.12**

**O que é:** Python é a linguagem de programação que usamos

**Execute:**

```bash
brew install python@3.12
```

**O que esperar:**

- Levará ~5 minutos
- Compila para Mac M4 (ARM64) ou Intel automaticamente

**Verifique:**

```bash
/opt/homebrew/bin/python3.12 --version
```

**Output esperado:**

```
Python 3.12.13
```

-----

### **Passo 4: Criar Alias (CRÍTICO!)**

**O que é:** Alias é um “atalho” para comandos longos

**Edite o arquivo de configuração:**

```bash
nano ~/.zshrc
```

**Adicione estas linhas no FINAL do arquivo:**

```bash
# Python 3.12 via Homebrew
export PATH="/opt/homebrew/opt/python@3.12/libexec/bin:$PATH"
alias python='/opt/homebrew/bin/python3.12'
alias python3='/opt/homebrew/bin/python3.12'
alias pip='/opt/homebrew/bin/pip3.12'
```

**Como salvar em nano:**

1. Pressione `Ctrl + O`
1. Pressione `Enter`
1. Pressione `Ctrl + X`

**Recarregue:**

```bash
source ~/.zshrc
```

**Verifique (FECHE E ABRA TERMINAL NOVO):**

```bash
python --version
python3 --version
pip --version
```

**Output esperado:**

```
Python 3.12.13
Python 3.12.13
pip 24.x.x
```

-----

### **Passo 5: Criar Ambiente Virtual**

**O que é:** Ambiente virtual é uma “cópia isolada” do Python para seu projeto

**Navegue para o projeto:**

```bash
cd /Users/seu_usuario/Documents/TechChallenge2/Notebook
```

**Crie o venv:**

```bash
python -m venv venv
```

**Ative:**

```bash
source venv/bin/activate
```

**Verifique (você deve ver `(venv)` no prompt):**

```bash
which python
python --version
```

**Output esperado:**

```
/Users/seu_usuario/Documents/TechChallenge2/Notebook/venv/bin/python
Python 3.12.13
```

-----

### **Passo 6: Instalar Dependências**

**Com o venv ativado (`(venv)` no prompt):**

```bash
pip install --upgrade pip

# Instale as dependências
pip install pandas==2.2.3 numpy==2.0.2 scikit-learn matplotlib seaborn shap openpyxl jupyter
```

**Verifique:**

```bash
pip list | grep -E "pandas|numpy|scikit-learn"
```

**Output esperado:**

```
numpy                     2.0.2
pandas                    2.2.3
scikit-learn              1.5.0 (ou superior)
```

-----

### **Passo 7: Salvar requirements.txt**

**Isso garante que todos da equipe usem as MESMAS versões:**

```bash
pip freeze > requirements.txt
```

**Verifique o arquivo:**

```bash
cat requirements.txt
```

**Deve conter:**

```
pandas==2.2.3
numpy==2.0.2
scikit-learn>=1.5.0
matplotlib
seaborn
shap
openpyxl
jupyter
...
```

-----

## 🪟 Configuração Windows {#windows}

### **Versões Testadas e Validadas**

|Componente |Versão        |Compatibilidade|
|-----------|--------------|---------------|
|**Windows**|10 / 11       |✅ Testado      |
|**Python** |3.12.x        |✅ Recomendado  |
|**pip**    |24.x+         |✅ Obrigatório  |
|**pandas** |2.2.3         |✅ Testado      |
|**numpy**  |2.0.2 ou 2.4.4|✅ Compatível   |

-----

### **Passo 1: Instalar Python 3.12**

**Baixe:**

- Acesse <https://www.python.org/downloads/>
- Clique em **“Download Python 3.12.x”**

**Instale:**

1. Execute o `.exe`
1. **IMPORTANTE:** Marque a caixa **“Add Python to PATH”**
1. Clique em **“Install Now”**
1. Aguarde conclusão

**Verifique:**
Abra **PowerShell** ou **cmd** e execute:

```bash
python --version
```

**Output esperado:**

```
Python 3.12.x
```

-----

### **Passo 2: Criar Ambiente Virtual**

**Navegue para o projeto:**

```bash
cd C:\Users\seu_usuario\Documents\TechChallenge2\Notebook
```

**Crie o venv:**

```bash
python -m venv venv
```

**Ative (Windows):**

```bash
venv\Scripts\activate
```

**Verifique (você deve ver `(venv)` no prompt):**

```bash
python --version
```

**Output esperado:**

```
(venv) C:\Users\seu_usuario\...\Notebook> python --version
Python 3.12.x
```

-----

### **Passo 3: Instalar Dependências**

**Com o venv ativado:**

```bash
pip install --upgrade pip

pip install pandas==2.2.3 numpy==2.0.2 scikit-learn matplotlib seaborn shap openpyxl jupyter
```

**Verifique:**

```bash
pip list | findstr pandas
```

**Output esperado:**

```
pandas                    2.2.3
numpy                     2.0.2
```

-----

### **Passo 4: Salvar requirements.txt**

```bash
pip freeze > requirements.txt
```

-----

## 🔧 Configuração VS Code {#vscode}

### **Versão Testada:** VS Code 1.90.x+

-----

### **Passo 1: Instalar Extensões Obrigatórias**

1. Abra **VS Code**
1. Clique em **Extensions** (ícone de quadrado no lado esquerdo)
1. Procure e instale (em ordem):

|Extensão                      |Autor      |Por quê                |
|------------------------------|-----------|-----------------------|
|**Python**                    |Microsoft  |Suporte básico a Python|
|**Jupyter**                   |Microsoft  |Suporte a notebooks    |
|**Pylance**                   |Microsoft  |Autocompletar e dicas  |
|**Python Docstring Generator**|Nils Werner|Gerar docstrings       |

-----

### **Passo 2: Abrir Projeto Corretamente**

**CRÍTICO:** Abra a pasta correta!

```bash
# macOS
cd /Users/seu_usuario/Documents/TechChallenge2/Notebook
code .

# Windows
cd C:\Users\seu_usuario\Documents\TechChallenge2\Notebook
code .
```

**O `code .` abre a pasta atual no VS Code**

-----

### **Passo 3: Selecionar Kernel do venv**

**Dentro do VS Code:**

1. Abra qualquer arquivo `.ipynb` (Jupyter Notebook)
1. Clique no botão azul **“Select Kernel”** (canto superior direito)
1. Clique em **“Python Environments…”**
1. **Procure por opção com `venv`**

**Se não aparecer:**

1. Clique em **“Enter interpreter path…”**
1. Cole o caminho correto:

**macOS:**

```
/Users/seu_usuario/Documents/TechChallenge2/Notebook/venv/bin/python
```

**Windows:**

```
C:\Users\seu_usuario\Documents\TechChallenge2\Notebook\venv\Scripts\python.exe
```

1. Pressione **Enter**

**Verifique:**
O botão azul agora deve mostrar algo como:

```
./venv/bin/python ou venv
```

-----

### **Passo 4: Ativar venv no Terminal Integrado**

**Abra terminal no VS Code:**

- **Terminal → New Terminal** ou **Ctrl + `**

**macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

**Verifique (deve aparecer `(venv)` no prompt):**

```bash
python --version
```

-----

### **Passo 5: Testar Instalação**

**Crie uma célula de teste no notebook:**

```python
import sys
import pandas
import numpy
import sklearn

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Pandas: {pandas.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
```

**Execute:** `Shift + Enter` ou clique em ▶️

**Output esperado:**

```
Python: 3.12.13 ...
Executable: /Users/seu_usuario/.../venv/bin/python
Pandas: 2.2.3
NumPy: 2.0.2
Scikit-learn: 1.5.0+
```

✅ Se aparecer isso, tudo está **100% correto!**

-----

## 📊 Governança e Reproducibilidade {#governanca}

### **Princípio Fundamental**

> **Qualquer membro da equipe deve conseguir executar o projeto em qualquer máquina em menos de 5 minutos.**

-----

### **Checklist de Reproducibilidade**

#### **Antes de compartilhar o projeto:**

```bash
# 1. Documentar versão do Python
python --version > docs/python_version.txt

# 2. Congelar dependências
pip freeze > requirements.txt

# 3. Testar instalação em venv novo
rm -rf venv_test
python -m venv venv_test
source venv_test/bin/activate
pip install -r requirements.txt

# 4. Executar notebook end-to-end
jupyter nbconvert --execute --inplace seu_notebook.ipynb
```

#### **Arquivo `README.md` no projeto:**

```markdown
# TechChallenge Fase 2 - Wine Quality

## Setup Rápido

### macOS
\`\`\`bash
cd path/to/TechChallenge2/Notebook
source venv/bin/activate
pip install -r requirements.txt
code .
\`\`\`

### Windows
\`\`\`bash
cd path\to\TechChallenge2\Notebook
venv\Scripts\activate
pip install -r requirements.txt
code .
\`\`\`

## Verificar Instalação

\`\`\`python
python --version  # Deve ser 3.12.13
pip list | grep pandas  # Deve ser 2.2.3
\`\`\`

## Integrantes (ordem alfabética)
- Efraim Oliveira
- Érica Tarsis
- Ricardo Moraes
- Rodrigo Bernardino
- Thiago Galvão
```

-----

**Documento preparado por:** Ricardo Moraes  
**Data:** 13 de junho de 2026  
**Status:** ✅ Aprovado para uso em produção
