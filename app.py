# app.py

import os
from dotenv import load_dotenv
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from repository_analysis import GitHubRepoAnalyzer
from ml_model import CommitRiskModel
from deepforest import CascadeForestClassifier
from xgboost import XGBClassifier
from recommendations import generate_recommendations

# 1. Загрузка настроек
load_dotenv()
repo_owner   = os.getenv('GITHUB_REPO_OWNER')
repo_name    = os.getenv('GITHUB_REPO_NAME')
github_token = os.getenv('GITHUB_TOKEN')

# 2. Сбор коммитов
analyzer = GitHubRepoAnalyzer(repo_owner, repo_name, github_token)
commits  = analyzer.analyze_commits()

# 3. Обучение модели
#base_clf = CascadeForestClassifier(
#    n_bins=255,
#    random_state=0,
#    n_estimators=4,
#    max_layers=10,
#    n_jobs=-1,
#    verbose=0
#)
base_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

model = CommitRiskModel(classifier=base_clf)
model.fit(commits)

# 4. Подготовка DataFrame
df = pd.DataFrame(commits)
features = model.features

df['Risk_Proba'] = model.predict_proba(commits)
df['Risk']       = (df['Risk_Proba'] > 0.5).astype(int)

# 5. Важности признаков
feat_imps = model.feature_importances()
fi_values = [feat_imps[f] for f in features]

# 6. Статистика для рекомендаций
repo_stats = {
    f: {
        'mean': df[f].mean(),
        'std':  df[f].std(),
        'quantile_95': df[f].quantile(0.95),
        'quantile_90': df[f].quantile(0.90)
    } for f in features
}
repo_stats['author_stats'] = {
    a: {
        'median_interval': df[df['author_name'] == a]['minutes_since_previous_commit'].median(),
        'median_lines_added': df[df['author_name'] == a]['lines_added'].median()
    } for a in df['author_name'].unique()
}

df['Recommendations']      = df.apply(
    lambda row: generate_recommendations(row, row['Risk_Proba'], repo_stats, feat_imps),
    axis=1
)
df['Recommendations_Text'] = df['Recommendations'].apply(
    lambda recs: "; ".join(recs) if recs else "Нет рекомендаций"
)

# 7. Аналитика по авторам
author_activity = df['author_name'].value_counts().reset_index()
author_activity.columns = ['author_name', 'commit_count']

author_risk = (
    df.groupby('author_name')['Risk_Proba']
      .mean()
      .reset_index()
      .sort_values(by='Risk_Proba', ascending=False)
)

# 8. Календарь активности
df['commit_date'] = pd.to_datetime(df['author_datetime']).dt.date
start_date, end_date = df['commit_date'].min(), df['commit_date'].max()
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

heatmap_df = pd.DataFrame({'date': all_dates})
heatmap_df['count'] = heatmap_df['date'].map(df['commit_date'].value_counts()).fillna(0)
heatmap_df['dow']  = heatmap_df['date'].dt.weekday
heatmap_df['week'] = ((heatmap_df['date'] - heatmap_df['date'].min()).dt.days // 7)

max_weeks = heatmap_df['week'].max() + 1
heat_matrix = np.zeros((7, max_weeks))
for _, row in heatmap_df.iterrows():
    heat_matrix[row['dow'], int(row['week'])] = row['count']

calendar_fig = go.Figure(data=go.Heatmap(
    z=heat_matrix,
    x=[f'Неделя {i+1}' for i in range(max_weeks)],
    y=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'],
    colorscale='Greens',
    hoverongaps=False,
    colorbar=dict(title='Коммитов в день')
))
calendar_fig.update_layout(
    xaxis=dict(scaleanchor='y', showgrid=False),
    yaxis=dict(showgrid=False)
)

# 9. Инициализация Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(fluid=True, children=[

    dbc.Row(dbc.Col(html.H1("Система анализа коммитов GitHub"),
                    className="text-center my-4")),

    dcc.Tabs([

        # --- Общая информация ---
        dcc.Tab(label='Общая информация', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=px.histogram(
                        df,
                        x='lines_added',
                        nbins=30,
                        title='Добавленные строки',
                        color_discrete_sequence=['#636EFA']
                    )
                ), xs=12, md=6),
                dbc.Col(dcc.Graph(
                    figure=px.histogram(
                        df,
                        x='lines_deleted',
                        nbins=30,
                        title='Удалённые строки',
                        color_discrete_sequence=['#EF553B']
                    )
                ), xs=12, md=6),
            ], className="g-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=px.histogram(
                        df,
                        x='files_changed',
                        nbins=30,
                        title='Изменённые файлы',
                        color_discrete_sequence=['#00CC96']
                    )
                ), xs=12, md=6),
                dbc.Col(dcc.Graph(
                    figure=px.histogram(
                        df,
                        x='complexity_score',
                        nbins=30,
                        title='Сложность изменений',
                        color_discrete_sequence=['#AB63FA']
                    )
                ), xs=12, md=6),
            ], className="g-4"),
        ]),

        # --- Анализ риска ---
        dcc.Tab(label='Анализ риска', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=px.bar(
                        x=features,
                        y=fi_values,
                        title='Важность признаков',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                ), xs=12),
            ], className="g-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=px.pie(
                        df,
                        names='Risk',
                        title='Соотношение рискованных и обычных коммитов',
                        color_discrete_sequence=['#00CC96', '#EF553B']
                    )
                ), xs=12, md=6),
                dbc.Col(dcc.Graph(
                    figure=px.scatter(
                        df,
                        x='lines_added',
                        y='complexity_score',
                        color='Risk',
                        title='Риск vs Сложность изменений',
                        color_discrete_sequence=['#00CC96', '#EF553B']
                    )
                ), xs=12, md=6),
            ], className="g-4"),
        ]),

        # --- Авторы ---
        dcc.Tab(label='Авторы', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=px.bar(
                        author_activity,
                        x='author_name',
                        y='commit_count',
                        title='Активность авторов',
                        color='commit_count',
                        color_continuous_scale='Viridis'
                    )
                ), xs=12, md=6),
                dbc.Col(dcc.Graph(
                    figure=px.bar(
                        author_risk,
                        x='author_name',
                        y='Risk_Proba',
                        title='Средний риск по авторам',
                        color='Risk_Proba',
                        color_continuous_scale='Reds'
                    )
                ), xs=12, md=6),
            ], className="g-4"),
        ]),

        # --- Рекомендации ---
        dcc.Tab(label='Рекомендации', children=[
            dbc.Row(dbc.Col(html.Div([
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong(row['commit'][:7]),
                        f" — {row['message'][:60]}... (Автор: {row['author_name']})",
                        html.Br(),
                        html.Ul([html.Li(rec) for rec in row['Recommendations']])
                    ]) for _, row in df[df['Risk'] == 1].iterrows()
                ])
            ]), xs=12), className="g-4"),
        ]),

        # --- Календарь активности ---
        dcc.Tab(label='Календарь активности', children=[
            dbc.Row(dbc.Col(dcc.Graph(
                figure=calendar_fig
            ), xs=12), className="g-4"),
        ]),
    ])
])

if __name__ == '__main__':
    app.run(debug=True)
