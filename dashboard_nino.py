import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pickle

# --- 1. GERAÇÃO DE DADOS HIPOTÉTICOS (NINO/BOMBEIROS PE) ---

np.random.seed(42)

cidades_pe = ['Recife', 'Jaboatão dos Guararapes', 'Olinda', 'Caruaru', 'Petrolina']
tipos_ocorrencia = ['Incêndio', 'Resgate de Pessoas', 'Resgate de Animais', 'Busca e Salvamento', 'Acidente de Trânsito']

N_OCORRENCIAS = 1000
start_date = datetime(2024, 1, 1)

data = {
    'id_ocorrencia': range(1, N_OCORRENCIAS + 1),
    'data': [start_date + timedelta(days=np.random.randint(0, 300)) for _ in range(N_OCORRENCIAS)],
    'cidade': np.random.choice(cidades_pe, N_OCORRENCIAS, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
    'tipo_ocorrencia': np.random.choice(tipos_ocorrencia, N_OCORRENCIAS, p=[0.3, 0.25, 0.05, 0.1, 0.3]),
    'tempo_resposta': np.random.normal(loc=15, scale=5, size=N_OCORRENCIAS).clip(min=5),
    # Simulação de coordenadas (aproximadamente em PE para o Mapa)
    'latitude': np.random.uniform(low=-8.1, high=-7.9, size=N_OCORRENCIAS),
    'longitude': np.random.uniform(low=-35.0, high=-34.8, size=N_OCORRENCIAS),
}
df = pd.DataFrame(data)

def calcular_severidade(row):
    # Lógica para criar a variável alvo 'severidade' (usada no ML)
    if row['tipo_ocorrencia'] in ['Incêndio', 'Busca e Salvamento'] and row['tempo_resposta'] > 18:
        return 'Alta'
    elif row['tempo_resposta'] > 20: # LINHA CORRIGIDA: Usa 'tempo_resposta'
        return 'Alta'
    elif row['tempo_resposta'] > 10:
        return 'Média'
    else:
        return 'Baixa'

df['severidade'] = df.apply(calcular_severidade, axis=1)

# --- 2. IMPLEMENTAÇÃO DO MODELO DE MACHINE LEARNING (XGBoost) ---

# Criando features adicionais necessárias para o ML
df['mes'] = df['data'].dt.month
df['dia_semana'] = df['data'].dt.dayofweek
df['hora'] = np.random.randint(0, 24, size=N_OCORRENCIAS)

# Seleção de Features e Codificação Categórica
features = ['mes', 'dia_semana', 'hora', 'cidade', 'tipo_ocorrencia']
target = 'severidade'

X = pd.get_dummies(df[features], columns=['cidade', 'tipo_ocorrencia'], drop_first=False)
y = df[target]
severidade_map = {'Baixa': 0, 'Média': 1, 'Alta': 2}
y_encoded = y.map(severidade_map)

# Separação Treino/Teste e Treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Persistindo a importância das features para o gráfico ML
feature_importances = model.feature_importances_
features_names = X.columns
importance_df = pd.DataFrame({
    'Fator': features_names,
    'Importancia': feature_importances
}).sort_values(by='Importancia', ascending=False).head(10) # Top 10 fatores para o gráfico

# --- 3. CRIAÇÃO DO DASHBOARD INTERATIVO COMPLETO ---

app = dash.Dash(__name__)

# Layout do Dashboard
app.layout = html.Div(children=[
    html.H1(children='🚒 Dashboard Completo de Ocorrências - NINO/Bombeiros PE', 
            style={'textAlign': 'center', 'color': '#D9534F'}),
    html.H3(children='Análise de Dados e Modelo Preditivo de Severidade', 
            style={'textAlign': 'center', 'color': '#333'}),

    # --- Filtros Interativos ---
    html.Div([
        html.Label('Filtro por Cidade:', style={'marginRight': '10px', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='filtro-cidade',
            options=[{'label': i, 'value': i} for i in ['Todas'] + cidades_pe],
            value='Todas',
            clearable=False
        ),
    ], style={'width': '40%', 'padding': '10px', 'display': 'inline-block'}),

    html.Hr(),

    # --- Linha 1: Gráficos de Frequência e Distribuição ---
    html.Div([
        # 1. Gráfico de Rosca (Frequência Relativa dos casos)
        dcc.Graph(id='grafico-rosquinha', style={'width': '33%', 'display': 'inline-block'}),
        
        # 2. Histograma (Distribuição de idades/tempo de resposta)
        dcc.Graph(id='grafico-histograma', style={'width': '33%', 'display': 'inline-block'}),

        # 3. Boxplot (Comparação de casos/tempo de resposta)
        dcc.Graph(id='grafico-boxplot', style={'width': '33%', 'display': 'inline-block'}),
    ], style={'display': 'flex'}),

    html.Hr(),

    # --- Linha 2: Gráficos de Espaço e Tempo ---
    html.Div([
        # 4. Mapa (Distribuição espacial dos casos)
        dcc.Graph(id='grafico-mapa', style={'width': '50%', 'display': 'inline-block'}),

        # 5. Gráfico de Linha (Distribuição temporal dos casos)
        dcc.Graph(id='grafico-linha', style={'width': '50%', 'display': 'inline-block'}),
    ], style={'display': 'flex'}),

    html.Hr(),

    # 6. Gráfico de Fatores Determinantes (ML)
    html.Div([
        html.H3(children='📈 Fatores Determinantes na Severidade (Modelo Preditivo)', 
                style={'textAlign': 'center', 'marginTop': '20px', 'color': '#5BC0DE'}),
        dcc.Graph(id='grafico-fatores-determinantes', style={'width': '80%', 'margin': 'auto'}),
    ]),

    html.Div(id='output-ml-accuracy', style={'textAlign': 'center', 'padding': '10px', 'fontWeight': 'bold', 'marginTop': '10px'})
])

# --- Callback para Atualizar TODOS os Gráficos de Dados com base no Filtro de Cidade ---
@app.callback(
    [Output('grafico-rosquinha', 'figure'),
     Output('grafico-histograma', 'figure'),
     Output('grafico-boxplot', 'figure'),
     Output('grafico-mapa', 'figure'),
     Output('grafico-linha', 'figure'),
     Output('output-ml-accuracy', 'children')],
    [Input('filtro-cidade', 'value')]
)
def update_all_graphs(selected_city):
    filtered_df = df.copy()
    if selected_city != 'Todas':
        filtered_df = filtered_df[filtered_df['cidade'] == selected_city]

    # 1. Gráfico de Rosca (Frequência Relativa)
    rosquinha_fig = px.pie(
        filtered_df, names='tipo_ocorrencia', title='1. Frequência Relativa dos Tipos de Ocorrência', hole=0.3
    )

    # 2. Histograma (Distribuição de Tempo de Resposta)
    hist_fig = px.histogram(
        filtered_df, x='tempo_resposta', nbins=15, title='2. Distribuição do Tempo de Resposta (min)'
    )

    # 3. Boxplot (Comparação de Tempo de Resposta por Severidade)
    box_fig = px.box(
        filtered_df, x='severidade', y='tempo_resposta', 
        title='3. Comparação de Tempo de Resposta por Severidade',
        category_orders={"severidade": ["Baixa", "Média", "Alta"]}
    )

    # 4. Mapa (Distribuição Espacial)
    center_lat, center_lon = -8.0578, -34.8827 # Recife, PE
    map_fig = px.scatter_mapbox(
        filtered_df, lat="latitude", lon="longitude", color="tipo_ocorrencia",
        zoom=10, height=400, title='4. Distribuição Espacial das Ocorrências',
        mapbox_style="open-street-map"
    )
    map_fig.update_layout(
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=9),
        margin={"r":0,"t":40,"l":0,"b":0}
    )

    # 5. Gráfico de Linha (Distribuição Temporal)
    df_temporal = filtered_df.groupby(filtered_df['data'].dt.to_period('M')).size().reset_index(name='Contagem')
    df_temporal['data'] = df_temporal['data'].astype(str)
    linha_fig = px.line(
        df_temporal, x='data', y='Contagem', title='5. Distribuição Temporal das Ocorrências (Por Mês)'
    )
    
    # Texto de Acurácia do Modelo
    accuracy_text = f"Acurácia do Modelo XGBoost na Previsão de Severidade (ML): {accuracy_score(y_test, model.predict(X_test)):.2f}"

    return rosquinha_fig, hist_fig, box_fig, map_fig, linha_fig, accuracy_text

# --- Callback para o Gráfico de Fatores Determinantes (ML) ---
@app.callback(
    Output('grafico-fatores-determinantes', 'figure'),
    [Input('filtro-cidade', 'value')] 
)
def update_ml_graph(selected_city):
    # 6. Gráfico de Barras para Fatores Determinantes
    fatores_fig = px.bar(
        importance_df, 
        x='Importancia', 
        y='Fator', 
        orientation='h',
        title='6. Fatores Mais Determinantes na Severidade (Regressão XGBoost)',
        labels={'Importancia': 'Valor de Importância', 'Fator': 'Variável Preditora'}
    )
    fatores_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fatores_fig


# Executar o Dashboard (Ponto de entrada)
if __name__ == '__main__':
    print("\n[Codespace] Servidor Dash iniciado com sucesso.")
    print("Acesse o link abaixo no seu navegador:")
    # LINHA CORRIGIDA: app.run substitui app.run_server
    app.run(debug=True, use_reloader=False)