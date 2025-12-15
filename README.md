## 🚒 Dashboard Preditivo de Ocorrências do Corpo de Bombeiros PE (NINO)

Este é um projeto de *Data Science* e Engenharia de Dados que combina a análise exploratória de dados operacionais hipotéticos do Corpo de Bombeiros de Pernambuco (CBPE) com a aplicação de um modelo de *Machine Learning* para prever a severidade das ocorrências.

O projeto utiliza o *framework* **Dash** para criar um dashboard interativo e minimalista, permitindo que gestores do NINO (Núcleo de Inteligência do CBMPE) visualizem tendências e compreendam os fatores que mais impactam a classificação de risco das emergências.

### 🌟 Visão Geral e Objetivo

O objetivo principal é fornecer uma ferramenta analítica que ajude na alocação de recursos e na tomada de decisão estratégica, respondendo a perguntas como:

  * Qual a distribuição espacial e temporal das ocorrências?
  * Qual o impacto do **tempo de resposta** na severidade de uma ocorrência?
  * Quais variáveis (cidade, mês, tipo de ocorrência) são mais **determinantes** para classificar uma emergência como de alta severidade?

### 💻 Tecnologias Utilizadas

| Categoria | Tecnologia | Uso |
| :--- | :--- | :--- |
| **Linguagem** | Python | Linguagem principal do projeto. |
| **Análise de Dados** | Pandas, NumPy | Manipulação e geração de dados hipotéticos. |
| **Machine Learning** | Scikit-learn, XGBoost | Treinamento do modelo preditivo de severidade. |
| **Visualização/Web** | Dash, Plotly Express | Construção do dashboard interativo e dos gráficos. |

### 📊 Estrutura do Dashboard

O dashboard é dividido em quatro áreas principais de análise:

#### 1\. Frequência e Distribuição (Linha 1)

  * **Gráfico de Rosca:** Distribuição percentual dos tipos de ocorrência.
  * **Histograma:** Distribuição do Tempo Médio de Resposta.
  * **Boxplot:** Comparação do Tempo de Resposta agrupado por **Severidade** (Baixa, Média, Alta).

#### 2\. Análise Espaço-Temporal (Linha 2)

  * **Mapa Interativo (Mapbox):** Visualização da distribuição geográfica das ocorrências.
  * **Gráfico de Linha:** Tendência temporal da contagem de ocorrências por mês.

#### 3\. Modelagem Preditiva (XGBoost)

  * **Gráfico de Barras (Feature Importance):** Exibe os 10 fatores mais importantes que o modelo XGBoost utilizou para classificar a severidade das ocorrências.
  * **Métrica de Acurácia:** Apresenta a acurácia do modelo na previsão de severidade sobre o conjunto de testes.

### ⚙️ Como Executar o Projeto Localmente

Siga os passos abaixo para colocar o dashboard no ar:

#### 1\. Pré-requisitos

Certifique-se de ter o Python instalado (versão 3.8+ recomendada).

#### 2\. Instalação de Dependências

Crie e ative um ambiente virtual (opcional, mas recomendado) e instale todas as bibliotecas necessárias:

```bash
pip install pandas numpy scikit-learn xgboost plotly dash
```

#### 3\. Execução

Salve o código Python completo (incluindo a geração de dados e a lógica do Dash) em um arquivo chamado `app.py` e execute-o no terminal:

```bash
python app.py
```

#### 4\. Acesso

O servidor Dash será iniciado. Abra seu navegador e acesse:

```
http://127.0.0.1:8050/
```

### 🎨 Paleta de Cores (Minimalista)

O dashboard foi estilizado com uma paleta de cores minimalista para melhorar a clareza e o foco nos dados, utilizando as seguintes referências HEX:

| Variável | Cor HEX | Descrição |
| :--- | :--- | :--- |
| **Destaque / Primária** | `#e26128` | Títulos e elementos-chave dos gráficos. |
| **Fundo** | `#c3c3cb` | Cor de fundo principal do layout. |
| **Texto / Eixos** | `#4b414e` | Elementos de texto, eixos e linhas de grade. |
| **Neutro / Secundária** | `#a39787` | Fundo de painéis e detalhes de suporte. |