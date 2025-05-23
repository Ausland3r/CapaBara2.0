# app.py

import os
from time import time

from dotenv import load_dotenv
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from repository_analysis import GitHubRepoAnalyzer
from ml_model import CommitRiskModel
from xgboost import XGBClassifier
from deepforest import CascadeForestClassifier
from sklearn.ensemble import RandomForestClassifier
from recommendations import generate_recommendations


def load_and_analyze_repos():
    load_dotenv()
    github_token = os.getenv('GITHUB_TOKEN')
    repos = [r for r in os.getenv("GITHUB_REPOS", "").split(",") if r]
    analyses = {}
    all_commits = []

    for full_name in repos:
        owner, name = full_name.split("/")
        analyzer = GitHubRepoAnalyzer(owner, name, github_token)
        commits = analyzer.analyze_commits()
        analyzer.analyze_and_pr(commits)

        all_commits.extend(commits)

        df = pd.DataFrame(commits)
        df['Risk_Proba'] = 0
        df['Risk'] = 0
        analyses[full_name] = {
            'df': df,
        }

    return github_token, repos, analyses, all_commits

def compare_models(all_commits):
    model_variants = [
        ("RandomForest", RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)),
        ("XGBoost", XGBClassifier(eval_metric="logloss", use_label_encoder=False, verbosity=0)),
        ("DeepForest", CascadeForestClassifier(random_state=42))
    ]

    print("\n====================== СРАВНЕНИЕ МОДЕЛЕЙ ======================")
    for name, clf in model_variants:
        print(f"\n>>> {name}")
        model = CommitRiskModel(classifier=clf)
        start = time()
        model.fit(all_commits)
        fit_time = time() - start

        start = time()
        _ = model.predict_proba(all_commits)
        predict_time = time() - start

        metrics = model.evaluate_model(all_commits)
        print(f"  Precision : {metrics['precision']:.3f}")
        print(f"  Recall    : {metrics['recall']:.3f}")
        print(f"  F1-score  : {metrics['f1_score']:.3f}")
        print(f"  ROC-AUC   : {metrics['auc']:.3f}")
        print(f"  Fit time  : {fit_time:.2f} сек")
        print(f"  Predict time: {predict_time:.2f} сек")

def train_and_update_model(all_commits, repos, analyses):
    model = CommitRiskModel(XGBClassifier(eval_metric="logloss"))
    model.fit(all_commits)

    for full_name in repos:
        df = analyses[full_name]['df']
        df['Risk_Proba'] = model.predict_proba(df.to_dict('records'))
        df['Risk'] = (df['Risk_Proba'] > 0.5).astype(int)

        analyses[full_name]['model'] = model
        analyses[full_name]['feat_imps'] = model.feature_importances()
        analyses[full_name]['metrics'] = model.evaluate_model(df.to_dict('records'))

    return model, analyses

github_token, repos, analyses, all_commits = load_and_analyze_repos()
model, analyses = train_and_update_model(all_commits, repos, analyses)

# 5. Инициализация Dash-приложения
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(fluid=True, children=[
    html.H1("Мульти-репозиторный анализ коммитов", className="text-center my-4"),
    dcc.Dropdown(
        id="repo-selector",
        options=[{"label": r, "value": r} for r in repos],
        value=repos[0] if repos else None,
        clearable=False,
        style={"width": "60%", "margin": "0 auto 20px auto"}
    ),
    html.Div(id="tabs-container")
])

@app.callback(
    Output("tabs-container", "children"),
    Input("repo-selector", "value")
)
def update_tabs(selected_repo):
    if not selected_repo or selected_repo not in analyses:
        return html.Div("Репозиторий не выбран или недоступен")

    entry = analyses[selected_repo]
    df = entry['df'].copy()
    feat_imps = entry['feat_imps']
    metrics = entry.get('metrics', {})
    metrics_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Метрика"), html.Th("Значение")])),
        html.Tbody([
            html.Tr([html.Td("Precision"), html.Td(f"{metrics.get('precision', 0):.2f}")]),
            html.Tr([html.Td("Recall"), html.Td(f"{metrics.get('recall', 0):.2f}")]),
            html.Tr([html.Td("F1-score"), html.Td(f"{metrics.get('f1_score', 0):.2f}")]),
            html.Tr([html.Td("ROC-AUC"), html.Td(f"{metrics.get('auc', 0):.2f}")]),
        ])
    ], bordered=True, striped=True, hover=True, style={"width": "40%", "marginTop": "20px"})
    features = entry['model'].features

    # Подстраховки
    if 'author_name' not in df:
        df['author_name'] = 'Unknown'
    if 'has_bug_keyword' not in df:
        df['has_bug_keyword'] = df['message'].str.contains(
            r'\b(fix|bug|error)\b', case=False, regex=True, na=False
        ).astype(int)

    # 6. Общая информация
    tab_summary = dcc.Tab(label='Общая информация', children=[
        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=px.histogram(
                    df, x='lines_added', nbins=30,
                    title='Добавленные строки',
                    color_discrete_sequence=['#1f77b4']  # синяя
                )
            ), md=6),
            dbc.Col(dcc.Graph(
                figure=px.histogram(
                    df, x='lines_deleted', nbins=30,
                    title='Удалённые строки',
                    color_discrete_sequence=['#d62728']  # красная
                )
            ), md=6),
        ], className="g-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=px.histogram(
                    df, x='files_changed', nbins=30,
                    title='Изменённые файлы',
                    color_discrete_sequence=['#2ca02c']  # зелёная
                )
            ), md=6),
            dbc.Col(dcc.Graph(
                figure=px.histogram(
                    df, x='complexity_score', nbins=30,
                    title='Сложность изменений',
                    color_discrete_sequence=['#9467bd']  # фиолетовая
                )
            ), md=6),
        ], className="g-4"),
    ])

    # 7. Анализ риска
    fi_vals = [feat_imps.get(f, 0) for f in features]
    tab_risk = dcc.Tab(label='Анализ риска', children=[
        metrics_table,
        dbc.Row(dbc.Col(dcc.Graph(
            figure=px.bar(
                x=features, y=fi_vals,
                title='Важность признаков',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        )), className="g-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=px.pie(
                    df, names='Risk', title='Рискованные vs обычные',
                    color_discrete_sequence=['#2ca02c', '#d62728']
                )
            ), md=6),
            dbc.Col(dcc.Graph(
                figure=px.scatter(
                    df, x='lines_added', y='complexity_score',
                    color='Risk', title='Риск vs Сложность',
                    color_discrete_sequence=['#2ca02c', '#d62728']
                )
            ), md=6),
        ], className="g-4"),
    ])

    # 8. Авторы
    author_activity = df['author_name'].value_counts().reset_index()
    author_activity.columns = ['author_name', 'commit_count']
    author_risk = df.groupby('author_name')['Risk_Proba'] \
                    .mean().reset_index() \
                    .sort_values('Risk_Proba', ascending=False)

    tab_authors = dcc.Tab(label='Авторы', children=[
        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=px.bar(
                    author_activity, x='author_name', y='commit_count',
                    title='Активность авторов',
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
            ), md=6),
            dbc.Col(dcc.Graph(
                figure=px.bar(
                    author_risk, x='author_name', y='Risk_Proba',
                    title='Средний риск по авторам',
                    color_discrete_sequence=px.colors.sequential.Reds
                )
            ), md=6),
        ], className="g-4"),
    ])

    # 9. File-Risk Map
    file_df = df.explode('file_list') if 'file_list' in df else pd.DataFrame()
    if not file_df.empty:
        fr = file_df.groupby('file_list').agg(
            change_count=('file_list','size'),
            avg_risk=('Risk_Proba','mean')
        ).reset_index()
    else:
        fr = pd.DataFrame(columns=['file_list','change_count','avg_risk'])
    tab_file_risk = dcc.Tab(label='File-Risk Map', children=[
        dbc.Row(dbc.Col(dcc.Graph(
            figure=px.scatter(
                fr, x='change_count', y='avg_risk',
                hover_name='file_list',
                title='Частота изменений vs средний риск',
                color_discrete_sequence=['#17becf']  # бирюзовая
            )
        )), className="g-4")
    ])

    # 10. Risk Timeline
    df['commit_date'] = pd.to_datetime(df['author_datetime'], errors='coerce').dt.date
    tl = df.sort_values('commit_date').groupby('commit_date').agg(
        daily_risk=('Risk_Proba','mean'),
        warnings=('Risk','sum')
    ).reset_index()
    fig_tl = go.Figure([
        go.Scatter(
            x=tl['commit_date'], y=tl['daily_risk'],
            mode='lines+markers', name='Средний риск',
            line=dict(color='#1f77b4')
        ),
        go.Bar(
            x=tl['commit_date'], y=tl['warnings'],
            name='Предупреждения', yaxis='y2', opacity=0.6,
            marker_color='#ff7f0e'
        )
    ])
    fig_tl.update_layout(
        title='Timeline риска и предупреждений',
        yaxis=dict(title='Средний риск'),
        yaxis2=dict(title='Кол-во предупреждений', overlaying='y', side='right')
    )
    tab_timeline = dcc.Tab(label='Risk Timeline', children=[
        dbc.Row(dbc.Col(dcc.Graph(figure=fig_tl)), className="g-4")
    ])

    # 11. Code Quality Tabs
    quality_tabs = []
    if {'pylint_warnings','pylint_errors','bandit_issues'} <= set(df.columns):
        quality_tabs.append(dcc.Tab(label='Python Quality', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=px.histogram(
                        df, x='pylint_warnings',
                        title='Pylint Warnings',
                        color_discrete_sequence=['#9467bd']
                    )
                ), md=6),
                dbc.Col(dcc.Graph(
                    figure=px.scatter(
                        df, x='pylint_errors', y='bandit_issues',
                        title='Errors vs Security Issues',
                        color_discrete_sequence=['#8c564b']
                    )
                ), md=6),
            ], className="g-4")
        ]))
    if {'eslint_warnings','eslint_errors'} <= set(df.columns):
        quality_tabs.append(dcc.Tab(label='JS Quality', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=px.histogram(
                        df, x='eslint_warnings',
                        title='ESLint Warnings',
                        color_discrete_sequence=['#e377c2']
                    )
                ), md=6),
                dbc.Col(dcc.Graph(
                    figure=px.scatter(
                        df, x='eslint_errors', y='eslint_warnings',
                        title='Errors vs Warnings',
                        color_discrete_sequence=['#7f7f7f']
                    )
                ), md=6),
            ], className="g-4")
        ]))
    if 'checkstyle_issues' in df.columns:
        quality_tabs.append(dcc.Tab(label='Java Quality', children=[
            dbc.Row(dbc.Col(dcc.Graph(
                figure=px.histogram(
                    df, x='checkstyle_issues',
                    title='Checkstyle Issues',
                    color_discrete_sequence=['#bcbd22']
                )
            )), className="g-4")
        ]))

    # 12. Commits Table
    df['Recommendations'] = df.apply(
        lambda row: generate_recommendations(row, row['Risk_Proba'], {}, feat_imps),
        axis=1
    )
    df['Recommendations_Text'] = df['Recommendations'].apply(lambda recs: "; ".join(recs))
    tab_table = dcc.Tab(label='Commits Table', children=[
        dash_table.DataTable(
            columns=[
                {"name":"SHA","id":"commit"},
                {"name":"Автор","id":"author_name"},
                {"name":"Дата","id":"commit_date"},
                {"name":"Риск","id":"Risk_Proba","type":"numeric","format":{"specifier":".2f"}},
                {"name":"Сообщение","id":"message"},
                {"name":"Рекомендации","id":"Recommendations_Text"},
            ],
            data=df[['commit','author_name','commit_date','Risk_Proba','message','Recommendations_Text']]
                 .to_dict('records'),
            page_size=10,
            style_cell={'textAlign':'left','whiteSpace':'normal','height':'auto'},
            style_header={'fontWeight':'bold'}
        )
    ])

    # 13. Календарь активности
    all_dates = pd.date_range(df['commit_date'].min(), df['commit_date'].max(), freq='D')
    heat = pd.DataFrame({'date': all_dates})
    heat['count'] = heat['date'].map(df['commit_date'].value_counts()).fillna(0)
    heat['dow'] = heat['date'].dt.weekday
    heat['week'] = ((heat['date'] - heat['date'].min()).dt.days // 7).astype(int)
    max_w = heat['week'].max() + 1
    mat = np.zeros((7, max_w))
    for _, r in heat.iterrows():
        mat[int(r['dow']), int(r['week'])] = r['count']
    cal_fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=[f'Неделя {i+1}' for i in range(max_w)],
        y=['Пн','Вт','Ср','Чт','Пт','Сб','Вс'],
        colorscale='Greens', hoverongaps=False,
        colorbar=dict(title='Коммитов/день')
    ))
    cal_fig.update_layout(xaxis=dict(scaleanchor='y', showgrid=False),
                          yaxis=dict(showgrid=False))
    tab_calendar = dcc.Tab(label='Календарь активности', children=[
        dbc.Row(dbc.Col(dcc.Graph(figure=cal_fig)), className="g-4")
    ])

    # Собираем все вкладки
    tabs = [
        tab_summary,
        tab_risk,
        tab_authors,
        tab_file_risk,
        tab_timeline,
        *quality_tabs,
        tab_table,
        tab_calendar
    ]
    return dcc.Tabs(tabs)

if __name__ == '__main__':
    github_token, repos, analyses, all_commits = load_and_analyze_repos()
    model, analyses = train_and_update_model(all_commits, repos, analyses)

    compare_models(all_commits)

    model, analyses = train_and_update_model(all_commits, repos, analyses)


    app.run(debug=True)
