# app.py

import os
from dotenv import load_dotenv
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from repository_analysis import GitHubRepoAnalyzer
from ml_model import CommitRiskModel
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
base_clf = XGBClassifier(eval_metric="logloss")
model    = CommitRiskModel(classifier=base_clf)
model.fit(commits)

# 4. Подготовка DataFrame
df = pd.DataFrame(commits)

# 4.1 Подстраховки
if 'author_name' not in df.columns:
    df['author_name'] = 'Unknown'
if 'has_bug_keyword' not in df.columns:
    df['has_bug_keyword'] = df['message'].str.contains(
        r'\b(fix|bug|error)\b', case=False, regex=True, na=False
    ).astype(int)

features = model.features

# 5. Предсказанные риски
df['Risk_Proba'] = model.predict_proba(commits)
df['Risk']       = (df['Risk_Proba'] > 0.5).astype(int)

# 6. Важности признаков
feat_imps = model.feature_importances()
fi_values = [feat_imps.get(f, 0) for f in features]

# 7. Статистика для рекомендаций
repo_stats = {
    f: {
        'mean': df[f].mean() if f in df else 0,
        'std':  df[f].std() if f in df else 0,
        'quantile_95': df[f].quantile(0.95) if f in df else 0,
        'quantile_90': df[f].quantile(0.90) if f in df else 0
    } for f in features
}
repo_stats['author_stats'] = {
    a: {
        'median_interval': df[df['author_name'] == a]['minutes_since_previous_commit'].median() \
                              if 'minutes_since_previous_commit' in df else 0,
        'median_lines_added': df[df['author_name'] == a]['lines_added'].median() \
                              if 'lines_added' in df else 0
    } for a in df['author_name'].unique()
}

df['Recommendations']      = df.apply(
    lambda row: generate_recommendations(row, row['Risk_Proba'], repo_stats, feat_imps),
    axis=1
)
df['Recommendations_Text'] = df['Recommendations'].apply(lambda recs: "; ".join(recs))

# 8. Аналитика по авторам
author_activity = df['author_name'].value_counts().reset_index()
author_activity.columns = ['author_name', 'commit_count']

author_risk = (
    df.groupby('author_name')['Risk_Proba']
      .mean()
      .reset_index()
      .sort_values(by='Risk_Proba', ascending=False)
)

# 9. Календарь активности
df['commit_date'] = pd.to_datetime(df['author_datetime'], errors='coerce').dt.date
all_dates = pd.date_range(start=df['commit_date'].min(), end=df['commit_date'].max(), freq='D')
heatmap_df = pd.DataFrame({'date': all_dates})
heatmap_df['count'] = heatmap_df['date'].map(df['commit_date'].value_counts()).fillna(0)
heatmap_df['dow']  = heatmap_df['date'].dt.weekday
heatmap_df['week'] = ((heatmap_df['date'] - heatmap_df['date'].min()).dt.days // 7).astype(int)

max_weeks = heatmap_df['week'].max() + 1
heat_matrix = np.zeros((7, max_weeks))
for _, row in heatmap_df.iterrows():
    heat_matrix[int(row['dow']), int(row['week'])] = row['count']

calendar_fig = go.Figure(data=go.Heatmap(
    z=heat_matrix,
    x=[f'Неделя {i+1}' for i in range(max_weeks)],
    y=['Пн','Вт','Ср','Чт','Пт','Сб','Вс'],
    colorscale='Greens',
    hoverongaps=False,
    colorbar=dict(title='Коммитов в день')
))
calendar_fig.update_layout(xaxis=dict(scaleanchor='y', showgrid=False),
                           yaxis=dict(showgrid=False))

# 10. Визуализации

# 10.1 File-Risk Map
file_df = df.explode('file_list') if 'file_list' in df else pd.DataFrame()
if not file_df.empty:
    file_risk_df = file_df.groupby('file_list').agg(
        change_count=('file_list','size'),
        avg_risk=('Risk_Proba','mean')
    ).reset_index()
else:
    file_risk_df = pd.DataFrame(columns=['file_list','change_count','avg_risk'])

file_risk_fig = px.scatter(
    file_risk_df, x='change_count', y='avg_risk',
    hover_name='file_list',
    title='Частота изменений vs средний риск по файлам',
    labels={'change_count':'Число изменений','avg_risk':'Средний риск'}
)

# 10.2 Risk Timeline
timeline_df = df.sort_values('commit_date').groupby('commit_date').agg(
    daily_risk=('Risk_Proba','mean'),
    warnings=('Risk','sum')
).reset_index()
timeline_fig = go.Figure([
    go.Scatter(x=timeline_df['commit_date'], y=timeline_df['daily_risk'],
               mode='lines+markers', name='Средний риск'),
    go.Bar(x=timeline_df['commit_date'], y=timeline_df['warnings'],
           name='Кол-во предупреждений', yaxis='y2', opacity=0.5)
])
timeline_fig.update_layout(
    title='Timeline риска и предупреждений',
    yaxis=dict(title='Средний риск'),
    yaxis2=dict(title='Число предупреждений', overlaying='y', side='right')
)

# 10.3 Code Quality — динамический таб
quality_tabs = []
# для Python
if {'pylint_warnings','pylint_errors','bandit_issues'} <= set(df.columns):
    quality_tabs.append(
        dcc.Tab(label='Python Quality', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(
                    df, x='pylint_warnings', nbins=20,
                    title='Pylint Warnings per Commit'
                )), xs=12, md=6),
                dbc.Col(dcc.Graph(figure=px.scatter(
                    df, x='pylint_errors', y='bandit_issues',
                    title='Pylint Errors vs Bandit Issues'
                )), xs=12, md=6),
            ], className="g-4")
        ])
    )
# для JavaScript
if {'eslint_warnings','eslint_errors'} <= set(df.columns):
    quality_tabs.append(
        dcc.Tab(label='JS Quality', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(
                    df, x='eslint_warnings', nbins=20,
                    title='ESLint Warnings per Commit'
                )), xs=12, md=6),
                dbc.Col(dcc.Graph(figure=px.scatter(
                    df, x='eslint_errors', y='eslint_warnings',
                    title='ESLint Errors vs Warnings'
                )), xs=12, md=6),
            ], className="g-4")
        ])
    )
# для Java / Checkstyle
if 'checkstyle_issues' in df.columns:
    quality_tabs.append(
        dcc.Tab(label='Java Quality', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(
                    df, x='checkstyle_issues', nbins=20,
                    title='Checkstyle Issues per Commit'
                )), xs=12, md=6),
            ], className="g-4")
        ])
    )

# 11. Инициализация Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.H1("Система анализа коммитов GitHub"),
                    className="text-center my-4")),

    dcc.Tabs([

        # Общая информация
        dcc.Tab(label='Общая информация', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(
                    df, x='lines_added', nbins=30,
                    title='Добавленные строки', color_discrete_sequence=['#636EFA']
                )), xs=12, md=6),
                dbc.Col(dcc.Graph(figure=px.histogram(
                    df, x='lines_deleted', nbins=30,
                    title='Удалённые строки', color_discrete_sequence=['#EF553B']
                )), xs=12, md=6),
            ], className="g-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(
                    df, x='files_changed', nbins=30,
                    title='Изменённые файлы', color_discrete_sequence=['#00CC96']
                )), xs=12, md=6),
                dbc.Col(dcc.Graph(figure=px.histogram(
                    df, x='complexity_score', nbins=30,
                    title='Сложность изменений', color_discrete_sequence=['#AB63FA']
                )), xs=12, md=6),
            ], className="g-4"),
        ]),

        # Анализ риска
        dcc.Tab(label='Анализ риска', children=[
            dbc.Row([dbc.Col(dcc.Graph(figure=px.bar(
                x=features, y=fi_values, title='Важность признаков',
                color_discrete_sequence=px.colors.qualitative.Set2
            )), xs=12)], className="g-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.pie(
                    df, names='Risk', title='Рискованные vs обычные',
                    color_discrete_sequence=['#00CC96','#EF553B']
                )), xs=12, md=6),
                dbc.Col(dcc.Graph(figure=px.scatter(
                    df, x='lines_added', y='complexity_score',
                    color='Risk', title='Риск vs Сложность изменений',
                    color_discrete_sequence=['#00CC96','#EF553B']
                )), xs=12, md=6),
            ], className="g-4"),
        ]),

        # Авторы
        dcc.Tab(label='Авторы', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.bar(
                    author_activity, x='author_name', y='commit_count',
                    title='Активность авторов', color='commit_count',
                    color_continuous_scale='Viridis'
                )), xs=12, md=6),
                dbc.Col(dcc.Graph(figure=px.bar(
                    author_risk, x='author_name', y='Risk_Proba',
                    title='Средний риск по авторам', color='Risk_Proba',
                    color_continuous_scale='Reds'
                )), xs=12, md=6),
            ], className="g-4"),
        ]),

        # File-Risk Map
        dcc.Tab(label='File-Risk Map', children=[
            dbc.Row(dbc.Col(dcc.Graph(figure=file_risk_fig), xs=12), className="g-4")
        ]),

        # Risk Timeline
        dcc.Tab(label='Risk Timeline', children=[
            dbc.Row(dbc.Col(dcc.Graph(figure=timeline_fig), xs=12), className="g-4")
        ]),

        # Code Quality — только для тех языков, которые найдутся
        *quality_tabs,

        # Commits Table
        dcc.Tab(label='Commits Table', children=[
            dash_table.DataTable(
                columns=[
                    {"name":"SHA",          "id":"commit"},
                    {"name":"Автор",        "id":"author_name"},
                    {"name":"Дата",         "id":"commit_date"},
                    {"name":"Риск",         "id":"Risk_Proba", "type":"numeric", "format":{"specifier":".2f"}},
                    {"name":"Сообщение",    "id":"message"},
                    {"name":"Рекомендации",  "id":"Recommendations_Text"},
                ],
                data=df[[
                    'commit','author_name','commit_date','Risk_Proba','message','Recommendations_Text'
                ]].to_dict('records'),
                page_size=10,
                style_cell={'textAlign':'left','whiteSpace':'normal','height':'auto'},
                style_header={'fontWeight':'bold'}
            )
        ]),

        # Календарь активности
        dcc.Tab(label='Календарь активности', children=[
            dbc.Row(dbc.Col(dcc.Graph(figure=calendar_fig), xs=12), className="g-4")
        ]),
    ])
])

if __name__ == '__main__':
    app.run(debug=True)
