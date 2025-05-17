# repository_analysis.py

import os
import requests
import re
import json
import subprocess
from datetime import datetime
import pandas as pd
from git import Repo, GitCommandError
from typing import List, Dict, Optional

from xgboost import XGBClassifier

from ml_model import CommitRiskModel
from recommendations import generate_recommendations

# mapping extension → analyzer name
LANGUAGE_ANALYZERS = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'javascript',
    '.java': 'java',
}

class GitHubRepoAnalyzer:
    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        token: str,
        clone_dir: str = "/tmp",
    ):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token
        self.api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {"Authorization": f"token {token}"}

        self.local_path = os.path.join(clone_dir, repo_name)
        if not os.path.isdir(self.local_path):
            clone_url = f"https://github.com/{repo_owner}/{repo_name}.git"
            print(f"[INIT] 🔄 Cloning repository {clone_url} into {self.local_path}…")
            Repo.clone_from(clone_url, self.local_path)
            print(f"[INIT] ✅ Clone complete.")
        else:
            print(f"[INIT] ✅ Repository already cloned at {self.local_path}.")
        self.repo = Repo(self.local_path)
        print(f"[INIT] 📂 Repo object ready at {self.local_path}.")

        self.complexity_re = re.compile(r"\b(if|for|while|switch|case)\b")

    def get_commits(self) -> List[Dict]:
        print("[COMMITS] 📬 Fetching commits via GitHub API…")
        commits, page, per_page = [], 1, 100
        while True:
            print(f"[COMMITS]   ▶️  Requesting page {page}")
            resp = requests.get(
                f"{self.api_url}/commits",
                headers=self.headers,
                params={"page": page, "per_page": per_page},
            )
            data = resp.json()
            if resp.status_code == 401:
                raise RuntimeError("Bad credentials: check your GITHUB_TOKEN")
            if not isinstance(data, list):
                print(f"[COMMITS] ⚠️  Unexpected response: {data}")
                break
            commits.extend(data)
            print(f"[COMMITS]   ℹ️  Retrieved {len(data)} commits in page {page}.")
            if len(data) < per_page:
                print(f"[COMMITS] ⛔ Less than {per_page} commits on page {page}, finishing.")
                break
            page += 1
        print(f"[COMMITS] ✅ Total commits fetched: {len(commits)}")
        return commits

    def get_commit_details(self, sha: str) -> Dict:
        print(f"[DETAILS] 🔍 Fetching details for commit {sha}…")
        resp = requests.get(f"{self.api_url}/commits/{sha}", headers=self.headers)
        return resp.json()

    def detect_language(self, filename: str) -> str:
        _, ext = os.path.splitext(filename.lower())
        return LANGUAGE_ANALYZERS.get(ext, "")

    def analyze_python_file(self, full_path: str) -> Dict[str,int]:
        pyl_w = pyl_e = bandit = 0
        try:
            r = subprocess.run(
                ["pylint", full_path, "--output-format=json"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            msgs = json.loads(r.stdout or "[]")
            for m in msgs:
                if m.get("type") == "error":
                    pyl_e += 1
                else:
                    pyl_w += 1
        except Exception:
            print(f"[ANALYZE][PY] ⚠️  Pylint failed on {full_path}")
        try:
            r = subprocess.run(
                ["bandit", "-f", "json", "-r", full_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            js = json.loads(r.stdout or "{}")
            bandit = len(js.get("results", []))
        except Exception:
            print(f"[ANALYZE][PY] ⚠️  Bandit failed on {full_path}")
        return {"pylint_warnings": pyl_w, "pylint_errors": pyl_e, "bandit_issues": bandit}

    def analyze_javascript_file(self, full_path: str) -> Dict[str,int]:
        w = e = 0
        try:
            r = subprocess.run(
                ["eslint", full_path, "-f", "json"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            arr = json.loads(r.stdout or "[]")
            for file_res in arr:
                for msg in file_res.get("messages", []):
                    if msg.get("severity") == 2:
                        e += 1
                    else:
                        w += 1
        except Exception:
            print(f"[ANALYZE][JS] ⚠️  ESLint failed on {full_path}")
        return {"eslint_warnings": w, "eslint_errors": e}

    def analyze_java_file(self, full_path: str) -> Dict[str,int]:
        count = 0
        try:
            r = subprocess.run(
                ["checkstyle", "-f", "plain", full_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            for ln in r.stdout.splitlines():
                if "ERROR" in ln or "WARNING" in ln:
                    count += 1
        except Exception:
            print(f"[ANALYZE][JAVA] ⚠️  Checkstyle failed on {full_path}")
        return {"checkstyle_issues": count}

    def compute_repo_stats(self, commits: List[Dict]) -> Dict:
        import pandas as pd
        df = pd.DataFrame(commits)
        stats = {}
        for f in ['lines_added', 'lines_deleted', 'files_changed',
                  'avg_file_history', 'message_length', 'complexity_score']:
            if f in df:
                stats[f] = {
                    'mean': df[f].mean(),
                    'std': df[f].std(),
                    'quantile_90': df[f].quantile(0.90),
                    'quantile_95': df[f].quantile(0.95),
                }
        # медианы для каждого автора
        stats['author_stats'] = {a: {'median_lines_added': grp.median()}
                                 for a, grp in df.groupby('author_name')['lines_added']}
        # медиана коммит-интервалов
        if 'minutes_since_previous_commit' in df:
            stats['commit_interval'] = {'median': df['minutes_since_previous_commit'].median()}
        return stats

    def analyze_commits(self) -> List[Dict]:
        print("[ANALYZE] 🚀 Starting commit-by-commit analysis…")
        commits_data, file_count = [], {}

        all_commits = self.get_commits()
        all_commits.reverse()
        prev_dt = None

        for idx, c in enumerate(all_commits, 1):
            sha = c["sha"]
            print(f"[ANALYZE] 🔎 ({idx}/{len(all_commits)}) Processing commit {sha}")
            det = self.get_commit_details(sha)

            try:
                print(f"[GIT] ⏪ Checking out {sha}")
                self.repo.git.checkout(sha)
            except GitCommandError:
                print(f"[GIT] ⚠️  Cannot checkout {sha}, skipping FS analysis")

            msg = det["commit"]["message"]
            author = det["commit"]["author"]
            name = author.get("name", "Unknown")
            dt = datetime.strptime(author["date"], "%Y-%m-%dT%H:%M:%SZ")

            files = det.get("files", [])
            print(f"[ANALYZE]   📄 {len(files)} files changed")

            added = sum(f.get("additions", 0) for f in files)
            deleted = sum(f.get("deletions", 0) for f in files)
            hist = sum(file_count.get(f["filename"], 0) for f in files)
            avg_hist = hist / len(files) if files else 0

            comp = 0
            for f in files:
                for ln in f.get("patch", "").splitlines():
                    if ln.startswith("+") and not ln.startswith("+++") and self.complexity_re.search(ln):
                        comp += 1

            delta = (dt - prev_dt).total_seconds() / 60 if prev_dt else None

            metrics = {k: 0 for k in (
                "pylint_warnings","pylint_errors","bandit_issues",
                "eslint_warnings","eslint_errors","checkstyle_issues"
            )}
            for f in files:
                lang = self.detect_language(f["filename"])
                full = os.path.join(self.local_path, f["filename"])
                if lang:
                    print(f"[ANALYZE]    🔧 Running {lang} analysis on {f['filename']}")
                if lang == "python":
                    out = self.analyze_python_file(full)
                elif lang == "javascript":
                    out = self.analyze_javascript_file(full)
                elif lang == "java":
                    out = self.analyze_java_file(full)
                else:
                    out = {}
                for k,v in out.items():
                    metrics[k] += v

            data = {
                "commit": sha,
                "author_name": name,
                "author_datetime": dt,
                "minutes_since_previous_commit": delta,
                "message": msg,
                "message_length": len(msg),
                "lines_added": added,
                "lines_deleted": deleted,
                "files_changed": len(files),
                "avg_file_history": avg_hist,
                "complexity_score": comp,
                "file_list": [f["filename"] for f in files],
                **metrics
            }
            commits_data.append(data)

            for f in files:
                file_count[f["filename"]] = file_count.get(f["filename"], 0) + 1
            prev_dt = dt

        print(f"[ANALYZE] ✅ Completed analysis of {len(commits_data)} commits.")
        return commits_data

    def create_capa_file(self, commits: List[Dict]) -> str:
        path = os.path.join(self.local_path, "CapaRecommendations.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# CAPA Recommendations\n\n")
            for c in commits:
                if c.get("has_capa"):
                    f.write(f"## Commit {c['commit']} — risk={c['Risk_Proba']:.2f}\n")
                    for rec in c["capa_recommendations"]:
                        f.write(f"- {rec}\n")
                    f.write("\n")
        return path

    def push_and_create_pr(self, branch_name: str, file_path: str) -> None:
        """
        Create a new branch from origin/main, commit the CAPA file and open a PR.
        """
        # 1) Fetch latest from origin
        print(f"[PR] 🔄 Fetching origin")
        self.repo.git.fetch('origin')

        # 2) Create and checkout a new branch from origin/main
        base_branch = 'main'  # or 'master' if your default branch is named master
        print(f"[PR] 🔀 Creating branch {branch_name} from origin/{base_branch}")
        self.repo.git.checkout('-b', branch_name, f'origin/{base_branch}')

        # 3) Stage and commit the new file
        rel_path = os.path.relpath(file_path, self.local_path)
        print(f"[PR] ➕ Adding file {rel_path}")
        self.repo.index.add([rel_path])
        print(f"[PR] 💾 Committing changes")
        self.repo.index.commit("Add CAPA recommendations")

        # 4) Push the new branch to origin
        print(f"[PR] 📤 Pushing branch {branch_name}")
        origin = self.repo.remote(name='origin')
        origin.push(branch_name)

        # 5) Create the Pull Request via GitHub API
        pr_data = {
            "title": "Add automated CAPA recommendations",
            "head": f"{self.repo_owner}:{branch_name}",
            "base": base_branch,
            "body": "This PR adds automatically generated corrective/preventive actions."
        }
        print(f"[PR] 📬 Opening PR via GitHub API")
        response = requests.post(
            f"{self.api_url}/pulls",
            headers=self.headers,
            json=pr_data
        )
        if response.status_code in (200, 201):
            pr_url = response.json().get("html_url")
            print(f"[PR] ✅ Pull request created: {pr_url}")
        else:
            print(f"[PR] ⚠️ Failed to create PR: {response.status_code} {response.text}")

    def analyze_and_pr(self, commits: Optional[List[Dict]] = None) -> None:
        if commits is None:
            commits = self.analyze_commits()

        if not commits:
            print("No commits — пропускаем PR.")
            return

        # 1) обучаем модель и предсказываем риск
        model = CommitRiskModel(classifier=XGBClassifier(eval_metric="logloss"))
        model.fit(commits)
        probs = model.predict_proba(commits)

        # 2) считаем статистики репозитория
        repo_stats = self.compute_repo_stats(commits)

        # 3) генерим рекомендации
        for commit, p in zip(commits, probs):
            commit["Risk_Proba"] = float(p)
            commit["has_capa"] = True
            commit["capa_recommendations"] = generate_recommendations(
                commit, p, repo_stats, model.feature_importances()
            )

        # 4) создаём MD-файл и PR
        md_path = self.create_capa_file(commits)
        branch = f"capa-{datetime.utcnow():%Y%m%d%H%M}"
        self.push_and_create_pr(branch, md_path)
