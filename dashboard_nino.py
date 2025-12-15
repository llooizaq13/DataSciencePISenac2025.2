import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


COLOR_PRIMARY = '#e26128' # Laranja/Destaque
COLOR_BACKGROUND = '#c3c3cb' # Cinza Claro
COLOR_TEXT = '#4b414e' # Roxo Escuro
COLOR_SECONDARY = '#a39787' # Verde Oliva/Neutro


np.random.seed(42)

cidades_pe = ['Recife', 'Jaboatão dos Guararapes', 'Olinda', 'Caruaru', 'Petrolina']
tipos_ocorrencia = ['Incêndio', 'Resgate de Pessoas', 'Resgate de Animais', 'Busca e Salvamento', 'Acidente de Trânsito']

N_OCORRENCIAS = 10000
start_date = datetime(2024, 1, 1)

data = {
    'id_ocorrencia': range(1, N_OCORRENCIAS + 1),
    'data': [start_date + timedelta(days=np.random.randint(0, 300)) for _ in range(N_OCORRENCIAS)],
    'cidade': np.random.choice(cidades_pe, N_OCORRENCIAS, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
    'tipo_ocorrencia': np.random.choice(tipos_ocorrencia, N_OCORRENCIAS, p=[0.3, 0.25, 0.05, 0.1, 0.3]),
    'tempo_resposta': np.random.normal(loc=15, scale=5, size=N_OCORRENCIAS).clip(min=5),
    'latitude': np.random.uniform(low=-8.1, high=-7.9, size=N_OCORRENCIAS),
    'longitude': np.random.uniform(low=-35.0, high=-34.8, size=N_OCORRENCIAS),
}
df = pd.DataFrame(data)

def calcular_severidade(row):
    if row['tipo_ocorrencia'] in ['Incêndio', 'Busca e Salvamento'] and row['tempo_resposta'] > 18:
        return 'Alta'
    elif row['tempo_resposta'] > 20: 
        return 'Alta'
    elif row['tempo_resposta'] > 10:
        return 'Média'
    else:
        return 'Baixa'

df['severidade'] = df.apply(calcular_severidade, axis=1)



df['mes'] = df['data'].dt.month
df['dia_semana'] = df['data'].dt.dayofweek
df['hora'] = np.random.randint(0, 24, size=N_OCORRENCIAS)

features = ['mes', 'dia_semana', 'hora', 'cidade', 'tipo_ocorrencia']
target = 'severidade'

X = pd.get_dummies(df[features], columns=['cidade', 'tipo_ocorrencia'], drop_first=False)
y = df[target]
severidade_map = {'Baixa': 0, 'Média': 1, 'Alta': 2}
y_encoded = y.map(severidade_map)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

feature_importances = model.feature_importances_
features_names = X.columns
importance_df = pd.DataFrame({
    'Fator': features_names,
    'Importancia': feature_importances
}).sort_values(by='Importancia', ascending=False).head(10)



def apply_minimal_style(fig, title_text):
    fig.update_layout(
        title={
            'text': title_text,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': COLOR_TEXT}
        },
        plot_bgcolor=COLOR_BACKGROUND,
        paper_bgcolor=COLOR_BACKGROUND,
        font={'color': COLOR_TEXT, 'family': "Arial"},
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text=None,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=COLOR_SECONDARY)
    )
    if fig.data:
        for trace in fig.data:
            if 'marker' in trace and isinstance(trace.marker, dict) and 'color' not in trace.marker:
                trace.marker.color = COLOR_PRIMARY
            elif 'line' in trace and isinstance(trace.line, dict) and 'color' not in trace.line:
                trace.line.color = COLOR_PRIMARY

            if 'marker' in trace and hasattr(trace.marker, 'color'):
                 trace.marker.color = COLOR_PRIMARY
    return fig

app = dash.Dash(__name__, title="NINO - Dashboard de Ocorrências Bombeiros PE")

APP_STYLE = {
    'backgroundColor': COLOR_BACKGROUND,
    'fontFamily': 'Arial, sans-serif',
    'color': COLOR_TEXT,
    'padding': '10px 30px'
}

HEADER_STYLE = {
    'textAlign': 'center', 
    'color': COLOR_PRIMARY,
    'marginBottom': '5px',
    'fontWeight': 'bold'
}

SUBHEADER_STYLE = {
    'textAlign': 'center', 
    'color': COLOR_TEXT,
    'marginTop': '0px',
    'marginBottom': '20px'
}

CONTROL_PANEL_STYLE = {
    'width': '100%', 
    'padding': '10px 0', 
    'marginBottom': '20px', 
    'textAlign': 'center',
    'borderBottom': f'1px solid {COLOR_SECONDARY}'
}

GRAPH_CONTAINER_STYLE = {
    'display': 'flex', 
    'flexWrap': 'wrap', 
    'justifyContent': 'space-between',
    'marginBottom': '20px'
}


app.layout = html.Div(style=APP_STYLE, children=[
    html.H1(children='Dashboard de Ocorrências - Bombeiros PE', 
            style=HEADER_STYLE),
    html.H3(children='Análise de Dados e Modelo Preditivo de Severidade', 
            style=SUBHEADER_STYLE),

 
    html.Div([
        html.Label('FILTRO POR CIDADE:', style={'marginRight': '10px', 'fontWeight': 'bold', 'color': COLOR_TEXT}),
        dcc.Dropdown(
            id='filtro-cidade',
            options=[{'label': i, 'value': i} for i in ['Todas'] + cidades_pe],
            value='Todas',
            clearable=False,
            style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle', 'backgroundColor': '#fff', 'color': COLOR_TEXT}
        ),
    ], style=CONTROL_PANEL_STYLE),

 
    html.Div(style=GRAPH_CONTAINER_STYLE, children=[
        dcc.Graph(id='grafico-rosquinha', style={'width': '33%'}),
        dcc.Graph(id='grafico-histograma', style={'width': '33%'}),
        dcc.Graph(id='grafico-boxplot', style={'width': '33%'}),
    ]),

  
    html.Div(style=GRAPH_CONTAINER_STYLE, children=[
        dcc.Graph(id='grafico-mapa', style={'width': '50%'}),
        dcc.Graph(id='grafico-linha', style={'width': '50%'}),
    ]),

    html.Div([
        html.H3(children='Fatores Determinantes na Severidade (Modelo Preditivo)', 
                style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '10px', 'color': COLOR_TEXT}),
        dcc.Graph(id='grafico-fatores-determinantes', style={'width': '90%', 'margin': 'auto'}),
    ]),

    html.Div(id='output-ml-accuracy', style={'textAlign': 'center', 'padding': '15px', 'fontWeight': 'bold', 'marginTop': '20px', 'backgroundColor': COLOR_SECONDARY, 'color': '#fff'})
])

EMPTY_FIG = {
    'layout': {
        'title': {
            'text': 'Sem dados para o filtro selecionado',
            'font': {'color': COLOR_TEXT}
        },
        'paper_bgcolor': COLOR_BACKGROUND,
        'plot_bgcolor': COLOR_BACKGROUND,
        'xaxis': {'visible': False},
        'yaxis': {'visible': False},
        'annotations': [{
            'text': 'Nenhuma ocorrência encontrada.',
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16, 'color': COLOR_TEXT}
        }]
    }
}


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

    if filtered_df.empty:
        accuracy_text = "Dados Insuficientes para Análise ML. Por favor, ajuste o filtro."
        # Retorna a figura de aviso para todos os 5 gráficos e o texto de acurácia
        return EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, accuracy_text


    rosquinha_fig = px.pie(
        filtered_df, names='tipo_ocorrencia', 
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY, '#8B4513', '#696969', '#A0522D'], 
        hole=0.3
    )
    rosquinha_fig = apply_minimal_style(rosquinha_fig, '1. Frequência por Tipo de Ocorrência')
    

    hist_fig = px.histogram(
        filtered_df, x='tempo_resposta', nbins=15
    )
    hist_fig = apply_minimal_style(hist_fig, '2. Distribuição do Tempo de Resposta (min)')

  
    box_fig = px.box(
        filtered_df, x='severidade', y='tempo_resposta', 
        category_orders={"severidade": ["Baixa", "Média", "Alta"]}
    )
    box_fig = apply_minimal_style(box_fig, '3. Tempo de Resposta por Severidade')
    box_fig.update_xaxes(title_text='Severidade')
    box_fig.update_yaxes(title_text='Tempo de Resposta (min)')

    center_lat, center_lon = -8.0578, -34.8827 
    map_fig = px.scatter_mapbox(
        filtered_df, lat="latitude", lon="longitude", color="tipo_ocorrencia",
        zoom=9, height=400,
        color_discrete_sequence=px.colors.qualitative.Bold, 
        mapbox_style="carto-positron" 
    )
    map_fig.update_layout(
        title={'text': '4. Distribuição Espacial das Ocorrências', 'x': 0.5, 'font': {'color': COLOR_TEXT}},
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=9),
        margin={"r":0,"t":40,"l":0,"b":0},
        paper_bgcolor=COLOR_BACKGROUND,
        font={'color': COLOR_TEXT}
    )


    df_temporal = filtered_df.groupby(filtered_df['data'].dt.to_period('M')).size().reset_index(name='Contagem')
    df_temporal['data'] = df_temporal['data'].astype(str)
    linha_fig = px.line(
        df_temporal, x='data', y='Contagem'
    )
    linha_fig = apply_minimal_style(linha_fig, '5. Distribuição Temporal (Por Mês)')
    linha_fig.update_traces(line=dict(color=COLOR_PRIMARY, width=2))
    linha_fig.update_xaxes(title_text='Mês')
    linha_fig.update_yaxes(title_text='Contagem de Ocorrências')
    
    accuracy_text = f"ACURÁCIA DO MODELO XGBOOST NA PREVISÃO DE SEVERIDADE (ML): {accuracy_score(y_test, model.predict(X_test)):.2f}"

    return rosquinha_fig, hist_fig, box_fig, map_fig, linha_fig, accuracy_text

@app.callback(
    Output('grafico-fatores-determinantes', 'figure'),
    [Input('filtro-cidade', 'value')] 
)
def update_ml_graph(selected_city):
    fatores_fig = px.bar(
        importance_df, 
        x='Importancia', 
        y='Fator', 
        orientation='h',
        labels={'Importancia': 'Valor de Importância', 'Fator': 'Variável Preditora'}
    )
    fatores_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    fatores_fig = apply_minimal_style(fatores_fig, '6. Fatores Mais Determinantes na Severidade')
    fatores_fig.update_traces(marker_color=COLOR_PRIMARY)
    fatores_fig.update_xaxes(showgrid=True, gridcolor=COLOR_SECONDARY)
    
    return fatores_fig



if __name__ == '__main__':
    print("\n[Codespace] Servidor Dashboar iniciado com todo o sucesso do mundo!.")
    print("Clica aqui para ver dar certo:")
    app.run(debug=True, use_reloader=False)