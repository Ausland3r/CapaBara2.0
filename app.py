import os
from dotenv import load_dotenv
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from repository_analysis import GitHubRepoAnalyzer
from ml_model import train_model, get_feature_importances
from recommendations import generate_recommendations

# Загрузка конфигурации
load_dotenv()

repo_owner = os.getenv('GITHUB_REPO_OWNER')
repo_name = os.getenv('GITHUB_REPO_NAME')
github_token = os.getenv('GITHUB_TOKEN')

analyzer = GitHubRepoAnalyzer(repo_owner, repo_name, github_token)
commits = analyzer.analyze_commits()
model = train_model(commits)

df = pd.DataFrame(commits)

features = [
    'lines_added', 'lines_deleted', 'files_changed',
    'avg_file_history', 'message_length',
    'has_bug_keyword', 'complexity_score'
]

X = df[features].values
y = df.apply(lambda row: 1 if (row['lines_added'] > 50 or row['has_bug_keyword'] == 1) else 0, axis=1).values
importances = get_feature_importances(model, X, y)
feature_importances = dict(zip(features, importances))

repo_stats = {
    feature: {
        'mean': df[feature].mean(),
        'std': df[feature].std(),
        'quantile_95': df[feature].quantile(0.95)
    } for feature in features
}
repo_stats['author_stats'] = {
    author: {
        'median_interval': df[df['author_name'] == author]['minutes_since_previous_commit'].median() / (60 * 24),
        'median_lines_added': df[df['author_name'] == author]['lines_added'].median()
    } for author in df['author_name'].unique()
}

df['Risk_Proba'] = df.apply(lambda row: model.predict_proba([[row[f] for f in features]])[0][1], axis=1)
df['Risk'] = df['Risk_Proba'].apply(lambda x: 1 if x > 0.5 else 0)

df['Recommendations'] = df.apply(
    lambda row: generate_recommendations(row, row['Risk_Proba'], repo_stats, feature_importances), axis=1
)
df['Recommendations_Text'] = df['Recommendations'].apply(lambda recs: "; ".join(recs) if recs else "Нет рекомендаций")

author_activity = df['author_name'].value_counts().reset_index()
author_activity.columns = ['author_name', 'commit_count']

author_risk = df.groupby('author_name')['Risk_Proba'].mean().reset_index().sort_values(by='Risk_Proba', ascending=False)

df['commit_date'] = pd.to_datetime(df['author_datetime']).dt.date
start_date, end_date = df['commit_date'].min(), df['commit_date'].max()

all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
heatmap_df = pd.DataFrame({'date': all_dates})
heatmap_df['count'] = heatmap_df['date'].map(df['commit_date'].value_counts()).fillna(0)
heatmap_df['dow'] = heatmap_df['date'].dt.weekday
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
    colorbar=dict(title='Коммитов в день'),
))
calendar_fig.update_layout(xaxis=dict(scaleanchor='y', showgrid=False), yaxis=dict(showgrid=False))

# Dash-приложение
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Система анализа коммитов GitHub"), className="text-center my-4")),

    dcc.Tabs([
        dcc.Tab(label='Общая информация', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(df, x='lines_added', nbins=30,
                                                      title='Добавленные строки',
                                                      color_discrete_sequence=['#636EFA'])), xs=12, sm=12, md=6, lg=6),
                dbc.Col(dcc.Graph(figure=px.histogram(df, x='lines_deleted', nbins=30,
                                                      title='Удалённые строки',
                                                      color_discrete_sequence=['#EF553B'])), xs=12, sm=12, md=6, lg=6),
            ], className="g-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(df, x='files_changed', nbins=30,
                                                      title='Изменённые файлы',
                                                      color_discrete_sequence=['#00CC96'])), xs=12, sm=12, md=6, lg=6),
                dbc.Col(dcc.Graph(figure=px.histogram(df, x='complexity_score', nbins=30,
                                                      title='Сложность изменений',
                                                      color_discrete_sequence=['#AB63FA'])), xs=12, sm=12, md=6, lg=6),
            ], className="g-4"),
        ]),

        dcc.Tab(label='Анализ риска', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.bar(x=features, y=importances,
                                                title='Важность признаков',
                                                color_discrete_sequence=px.colors.qualitative.Set2)), width=12)
            ], className="g-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.pie(df, names='Risk', title='Соотношение рискованных и обычных коммитов',
                                                color_discrete_sequence=['#00CC96', '#EF553B'])), xs=12, sm=12, md=6, lg=6),
                dbc.Col(dcc.Graph(figure=px.scatter(df, x='lines_added', y='complexity_score', color='Risk',
                                                    title='Риск vs Сложность изменений',
                                                    color_discrete_sequence=['#00CC96', '#EF553B'])), xs=12, sm=12, md=6, lg=6),
            ], className="g-4"),
        ]),

        dcc.Tab(label='Авторы', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.bar(author_activity, x='author_name', y='commit_count',
                                                title='Активность авторов',
                                                color='commit_count', color_continuous_scale='Viridis')), xs=12, sm=12, md=6, lg=6),

                dbc.Col(dcc.Graph(figure=px.bar(author_risk, x='author_name', y='Risk_Proba',
                                                title='Средний риск по авторам',
                                                color='Risk_Proba', color_continuous_scale='Reds')), xs=12, sm=12, md=6, lg=6),
            ], className="g-4"),
        ]),

        dcc.Tab(label='Рекомендации', children=[
            dbc.Row([
                dbc.Col(html.Div([
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Strong(row['commit'][:7]),
                            f" — {row['message'][:60]}... (Автор: {row['author_name']})",
                            html.Br(),
                            html.Ul([html.Li(rec) for rec in row['Recommendations']])
                        ]) for _, row in df[df['Risk'] == 1].iterrows()
                    ])
                ]), width=12)
            ], className="g-4"),
        ]),

        dcc.Tab(label='Календарь активности', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=calendar_fig), width=12)
            ], className="g-4"),
        ]),
    ])
], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)
