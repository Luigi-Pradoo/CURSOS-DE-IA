# Exercício - Aula 2: Análise Exploratória de Dados (EDA)

## 1. O que é EDA e sua Importância
A Análise Exploratória de Dados (EDA) é uma etapa fundamental no pipeline de Ciência de Dados e Machine Learning. Ela consiste em explorar, limpar e visualizar o conjunto de dados para entender sua estrutura, identificar padrões ocultos, detectar anomalias (como outliers e valores nulos) e extrair insights iniciais. Realizar a EDA é o que garante que os dados estejam qualificados antes de serem submetidos ao treinamento de qualquer modelo preditivo.

## 2. Descrição do Dataset Utilizado
Para este exercício, foi utilizado um conjunto de dados focado em um problema de **Classificação**, extraído da plataforma Kaggle (referência: `aayushgid/csv-classification`).

O dataset possui uma estrutura **tabular (CSV)** composta por:
* **Variáveis Independentes (Features):** Atributos numéricos e/ou categóricos que descrevem as características de cada registro da base de dados.
* **Variável Dependente (Target):** A coluna alvo que define a classe/categoria de cada registro (utilizada para o aprendizado supervisionado de classificação).

## 3. Objetivos Práticos Deste Exercício
* **Carga e Inspeção:** Importar o arquivo CSV utilizando a biblioteca Pandas para verificar o volume de dados (linhas e colunas).
* **Limpeza de Dados:** Identificar e tratar possíveis valores ausentes (nulos) ou inconsistentes.
* **Análise Estatística:** Avaliar médias, medianas e distribuições das variáveis numéricas.
* **Visualização:** Utilizar bibliotecas gráficas (como Matplotlib/Seaborn) para analisar o comportamento das variáveis em relação à classe alvo (Target).