# 🤖 Guia Definitivo: Introdução a Machine Learning

Bem-vindo ao material de consolidação sobre os **fundamentos, processos e algoritmos clássicos de Machine Learning**.

Este guia apresenta desde os conceitos básicos até a avaliação de modelos preditivos utilizados em problemas reais de classificação e regressão.

---

# 🧠 Módulo 1 – A Essência do Machine Learning

Machine Learning representa a transição da programação baseada em regras explícitas para o aprendizado obtido a partir de dados históricos.

Segundo **Tom Mitchell**:

> *"Um programa aprende com a experiência E em relação a uma tarefa T e uma medida de desempenho P se seu desempenho em T, medido por P, melhora com a experiência E."*

## Tipos de Aprendizado

### 📌 Aprendizado Supervisionado

O modelo recebe:

- **Features (X)** → Entradas
- **Target (y)** → Respostas corretas

Objetivo:

Aprender uma função matemática capaz de prever novos valores.

Exemplos:

- Classificação
- Regressão

---

### 📌 Aprendizado Não Supervisionado

Não existem respostas corretas.

O algoritmo procura automaticamente padrões presentes nos dados.

Exemplos:

- Agrupamentos (Clustering)
- Redução de dimensionalidade

---

### 📌 Aprendizado por Reforço

Um agente aprende interagindo com um ambiente.

Cada ação gera:

- recompensa
- penalização

O objetivo é maximizar a recompensa acumulada.

---

## Underfitting

Também chamado de **Subajuste**.

O modelo é simples demais e não consegue aprender os padrões dos dados.

Características:

- baixo desempenho no treino;
- baixo desempenho no teste.

---

## Overfitting

Também chamado de **Sobreajuste**.

O modelo memoriza os dados de treino em vez de aprender padrões gerais.

Características:

- excelente desempenho no treino;
- desempenho ruim em dados novos.

---

## Trade-off Viés × Variância

O objetivo é encontrar o equilíbrio entre modelos simples e modelos complexos.

\[
Erro = Viés^2 + Variância + Erro_{Irredutível}
\]

---

# 📊 Módulo 2 – Datasets Clássicos

| Dataset | Objetivo | Características |
|----------|----------|-----------------|
| Iris Dataset | Classificação | 150 amostras balanceadas, 4 features numéricas e 3 espécies de flores |
| California Housing | Regressão | 20.640 instâncias para previsão de preços de imóveis |
| Titanic | Classificação Binária | Predição de sobrevivência com diversos dados faltantes |

---

# 🛠️ Módulo 3 – Feature Engineering

Feature Engineering consiste em transformar dados brutos em representações matemáticas mais úteis para os algoritmos.

## Min-Max Scaler

Normaliza os valores para um intervalo fixo.

Normalmente:

```
0 → 1
```

Ideal quando o algoritmo depende da escala.

---

## Standard Scaler

Padroniza os dados utilizando o **Z-Score**.

Após a transformação:

- Média = 0
- Desvio padrão = 1

---

## Robust Scaler

Projetado para bases com muitos **outliers**.

Utiliza:

- Mediana
- Intervalo Interquartil (IQR)

Em vez da média e do desvio padrão.

---

## Pipelines

Agrupam todas as etapas do pré-processamento.

Exemplo:

```
Dados
   ↓
Imputação
   ↓
Escalonamento
   ↓
Modelo
```

Vantagens:

- evita Data Leakage;
- facilita reprodução;
- organiza o fluxo de treinamento.

---

## ColumnTransformer

Permite aplicar transformações diferentes para cada tipo de coluna.

Exemplo:

- Colunas numéricas → StandardScaler
- Colunas categóricas → OneHotEncoder

Depois tudo é unido automaticamente.

---

## SMOTE

(Synthetic Minority Over-sampling Technique)

Utilizado quando existe desbalanceamento entre classes.

Em vez de copiar registros da classe minoritária, cria novas amostras sintéticas por interpolação.

---

# 🛡️ Módulo 4 – Validação de Modelos

## Conjunto de Treino

Utilizado para ajustar os parâmetros do modelo.

---

## Conjunto de Validação

Usado para:

- ajuste de hiperparâmetros;
- seleção do melhor modelo.

Não deve ser utilizado para treinamento.

---

## Conjunto de Teste (Hold-Out)

Permanece isolado durante todo o desenvolvimento.

É utilizado apenas uma vez para medir o desempenho final.

---

## K-Fold Cross Validation

O conjunto de dados é dividido em **K partes**.

Em cada execução:

- K−1 partes treinam;
- 1 parte testa.

O processo é repetido até que todas as partes sejam utilizadas como teste.

---

## Stratified K-Fold

Versão do K-Fold utilizada em problemas de classificação.

Mantém a mesma proporção de classes em todas as divisões.

Muito importante para datasets desbalanceados.

---

# ⚙️ Módulo 5 – Algoritmos Preditivos

| Algoritmo | Como Funciona | Ponto Importante |
|------------|---------------|------------------|
| Regressão Linear | Combinação linear das features | Pode sofrer com multicolinearidade |
| Lasso (L1) | Penaliza coeficientes grandes | Pode eliminar variáveis irrelevantes |
| KNN | Baseado na distância entre amostras | Classificação por votação dos vizinhos |
| Regressão Logística | Utiliza função Sigmoide | Retorna probabilidades entre 0 e 1 |
| SVM | Procura o hiperplano com maior margem | Kernel permite separações não lineares |
| Random Forest | Conjunto de árvores de decisão | Utiliza Bootstrap e votação |
| XGBoost / LightGBM | Boosting sequencial | Cada árvore aprende com os erros da anterior |

---

## Regressão Linear

Objetivo:

Encontrar a reta que minimiza o erro entre valores previstos e reais.

---

## Penalização Lasso (L1)

Adiciona uma penalização na função custo.

Benefícios:

- reduz overfitting;
- realiza seleção automática de variáveis.

---

## KNN (K-Nearest Neighbors)

Também conhecido como **Lazy Learning**.

Funcionamento:

1. calcula as distâncias;
2. encontra os K vizinhos mais próximos;
3. realiza votação (classificação) ou média (regressão).

---

## Regressão Logística

Apesar do nome, é um algoritmo de classificação.

Utiliza a função sigmoide:

```
0 ≤ Probabilidade ≤ 1
```

---

## SVM (Support Vector Machine)

Busca o hiperplano que maximiza a distância entre classes.

Quando os dados não são linearmente separáveis, utiliza o **Kernel Trick**.

---

## Random Forest

Método Ensemble baseado em centenas de árvores.

Cada árvore:

- recebe amostras diferentes (Bootstrap);
- utiliza subconjuntos aleatórios de atributos.

A resposta final ocorre por votação.

---

## XGBoost / LightGBM

Métodos baseados em **Boosting**.

Cada árvore nova aprende com os erros cometidos pelas anteriores.

São modelos extremamente utilizados em competições de Machine Learning.

---

# 🎯 Módulo 6 – Métricas de Avaliação

## Precisão (Precision)

Das previsões positivas realizadas pelo modelo, quantas estavam corretas?

\[
Precision=\frac{VP}{VP+FP}
\]

---

## Recall

Dos positivos reais, quantos foram encontrados?

\[
Recall=\frac{VP}{VP+FN}
\]

---

## F1-Score

Média harmônica entre Precisão e Recall.

Ideal para bases desbalanceadas.

\[
F1=\frac{2 \times Precision \times Recall}{Precision+Recall}
\]

---

## Curva ROC (AUC)

Avalia a capacidade do modelo separar corretamente as classes.

Interpretação:

- **1.0** → perfeito
- **0.5** → equivalente ao acaso

---

## MSE (Mean Squared Error)

Erro Quadrático Médio.

Eleva os erros ao quadrado, penalizando erros grandes.

---

## R² Score

Também chamado de **Coeficiente de Determinação**.

Indica quanto da variabilidade dos dados foi explicada pelo modelo.

Valores próximos de **1** representam modelos melhores.

---

# 📚 Resumo Geral

✔ Machine Learning aprende padrões a partir de dados.

✔ Feature Engineering melhora a qualidade das informações fornecidas ao modelo.

✔ A validação evita overfitting e garante boa generalização.

✔ Cada algoritmo possui vantagens e limitações dependendo do problema.

✔ A escolha da métrica correta é tão importante quanto a escolha do algoritmo.

---

# 🚀 Principais Conceitos

- Machine Learning
- Aprendizado Supervisionado
- Aprendizado Não Supervisionado
- Aprendizado por Reforço
- Underfitting
- Overfitting
- Trade-off Viés × Variância
- Feature Engineering
- Min-Max Scaler
- Standard Scaler
- Robust Scaler
- Pipeline
- ColumnTransformer
- SMOTE
- Hold-Out
- K-Fold
- Stratified K-Fold
- Regressão Linear
- Lasso
- KNN
- Regressão Logística
- SVM
- Random Forest
- XGBoost
- LightGBM
- Precisão
- Recall
- F1-Score
- ROC AUC
- MSE
- R² Score