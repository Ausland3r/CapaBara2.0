# repository_analysis.py
import requests
import re
from datetime import datetime
import pandas as pd


class GitHubRepoAnalyzer:
    def __init__(self, repo_owner, repo_name, token):
        self.api_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}'
        self.headers = {'Authorization': f'token {token}'}

    def get_commits(self):
        commits = []
        page = 1
        while True:
            response = requests.get(
                f'{self.api_url}/commits',
                headers=self.headers,
                params={'page': page, 'per_page': 100}
            )
            data = response.json()
            if not data:
                break
            commits.extend(data)
            page += 1
        return commits

    def get_commit_details(self, sha):
        response = requests.get(f'{self.api_url}/commits/{sha}', headers=self.headers)
        return response.json()

    def analyze_commits(self):
        commits_data = []
        file_change_count = {}
        complexity_patterns = [r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\bswitch\b', r'\bcase\b']

        commits = self.get_commits()
        commits.reverse()

        previous_commit_datetime = None

        for commit in commits:
            sha = commit['sha']
            details = self.get_commit_details(sha)

            message = details['commit']['message']
            files = details.get('files', [])

            author_info = details.get('commit', {}).get('author', {})
            author_name = author_info.get('name', 'Unknown')
            author_email = author_info.get('email', 'Unknown')
            author_date_str = author_info.get('date', 'Unknown')

            author_datetime = datetime.strptime(author_date_str, '%Y-%m-%dT%H:%M:%SZ')

            lines_added = sum(file.get('additions', 0) for file in files)
            lines_deleted = sum(file.get('deletions', 0) for file in files)
            files_changed = len(files)

            history_sum = sum(file_change_count.get(file['filename'], 0) for file in files)
            avg_file_history = history_sum / files_changed if files_changed else 0

            complexity_score = 0
            for file in files:
                if 'patch' in file:
                    complexity_score += sum(
                        bool(re.search('|'.join(complexity_patterns), line))
                        for line in file['patch'].splitlines()
                        if line.startswith('+') and not line.startswith('+++')
                    )

            if previous_commit_datetime is not None:
                minutes_since_previous = (author_datetime - previous_commit_datetime).total_seconds() / 60.0
            else:
                minutes_since_previous = None

            commit_data = {
                'commit': sha,
                'author_name': author_name,
                'author_email': author_email,
                'author_datetime': author_datetime,
                'minutes_since_previous_commit': minutes_since_previous,
                'message': message,
                'message_length': len(message),
                'has_bug_keyword': int(bool(re.search(r'\b(fix|bug|error)\b', message, re.IGNORECASE))),
                'lines_added': lines_added,
                'lines_deleted': lines_deleted,
                'files_changed': files_changed,
                'avg_file_history': avg_file_history,
                'complexity_score': complexity_score
            }

            commits_data.append(commit_data)

            for file in files:
                file_change_count[file['filename']] = file_change_count.get(file['filename'], 0) + 1

            previous_commit_datetime = author_datetime

        return commits_data

    def analyze_time_series(self, commits_data):
        df = pd.DataFrame(commits_data).dropna(subset=['minutes_since_previous_commit'])

        df['rolling_mean'] = df['minutes_since_previous_commit'].rolling(window=10).mean()
        df['rolling_std'] = df['minutes_since_previous_commit'].rolling(window=10).std()

        df['is_anomaly'] = df.apply(
            lambda row: row['minutes_since_previous_commit'] > (row['rolling_mean'] + 2 * row['rolling_std']),
            axis=1
        )

        return df