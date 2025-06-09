import os
import requests
import re
import json
import subprocess
from datetime import datetime

from deepforest import CascadeForestClassifier
from git import Repo, GitCommandError
from typing import List, Dict, Optional

from xgboost import XGBClassifier

from ml_model import CommitRiskModel
from recommendations import RecommendationGenerator

recommendation_generator = RecommendationGenerator()

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
            clone_url = f"https://{self.token}@github.com/{repo_owner}/{repo_name}.git"
            print(f"[INIT] Cloning repository {clone_url} into {self.local_path}…")
            Repo.clone_from(clone_url, self.local_path)
            print(f"[INIT] Clone complete.")
        else:
            print(f"[INIT] Repository already cloned at {self.local_path}.")
        self.repo = Repo(self.local_path)
        print(f"[INIT] Repo object ready at {self.local_path}.")

        self.complexity_re = re.compile(r"\b(if|for|while|switch|case)\b")

    def get_commits(self) -> List[Dict]:
        print("[COMMITS] Fetching commits via GitHub API…")
        commits, page, per_page = [], 1, 100
        while True:
            print(f"[COMMITS] Requesting page {page}")
            resp = requests.get(
                f"{self.api_url}/commits",
                headers=self.headers,
                params={"page": page, "per_page": per_page},
            )
            data = resp.json()
            if resp.status_code == 401:
                raise RuntimeError("Bad credentials: check your GITHUB_TOKEN")
            if not isinstance(data, list):
                print(f"[COMMITS] Unexpected response: {data}")
                break
            commits.extend(data)
            print(f"[COMMITS] Retrieved {len(data)} commits in page {page}.")
            if len(data) < per_page:
                print(f"[COMMITS] Less than {per_page} commits on page {page}, finishing.")
                break
            page += 1
        print(f"[COMMITS] Total commits fetched: {len(commits)}")
        return commits

    def get_commit_details(self, sha: str) -> Dict:
        print(f"[DETAILS] Fetching details for commit {sha}…")
        resp = requests.get(f"{self.api_url}/commits/{sha}", headers=self.headers)
        if resp.status_code != 200:
            print(f"[ERROR] Failed to fetch details for {sha}: {resp.status_code} {resp.text}")
            return {}
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
            print(f"[ANALYZE][PY] Pylint failed on {full_path}")
        try:
            r = subprocess.run(
                ["bandit", "-f", "json", "-r", full_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            js = json.loads(r.stdout or "{}")
            bandit = len(js.get("results", []))
        except Exception:
            print(f"[ANALYZE][PY] Bandit failed on {full_path}")
        cyclomatic_complexity = self.analyze_cyclomatic_complexity(full_path)
        return {
            "pylint_warnings": pyl_w,
            "pylint_errors": pyl_e,
            "bandit_issues": bandit,
            "cyclomatic_complexity": cyclomatic_complexity
        }

    def analyze_cyclomatic_complexity(self, full_path: str) -> int:
        try:
            r = subprocess.run(
                ["radon", "cc", "-s", "-j", full_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            js = json.loads(r.stdout or "{}")
            total_complexity = 0
            for file_results in js.values():
                total_complexity += sum(func['complexity'] for func in file_results)
            return total_complexity
        except Exception:
            print(f"[ANALYZE][PY] Radon failed on {full_path}")
            return 0

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
            print(f"[ANALYZE][JS] ESLint failed on {full_path}")
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
            print(f"[ANALYZE][JAVA] Checkstyle failed on {full_path}")
        return {"checkstyle_issues": count}

    def compute_repo_stats(self, commits: List[Dict]) -> Dict:
        import pandas as pd
        df = pd.DataFrame(commits)
        stats = {}
        features_for_stats = [
            'lines_added', 'lines_deleted', 'files_changed',
            'files_added', 'files_deleted', 'lines_changed',
            'avg_file_history', 'message_length', 'complexity_score',
            'cyclomatic_complexity',
            'pylint_warnings', 'pylint_errors', 'bandit_issues',
            'eslint_warnings', 'eslint_errors', 'checkstyle_issues'
        ]
        for f in features_for_stats:
            if f in df:
                stats[f] = {
                    'mean': df[f].mean(),
                    'std': df[f].std(),
                    'quantile_90': df[f].quantile(0.90),
                    'quantile_95': df[f].quantile(0.95),
                }
        stats['author_stats'] = {a: {'median_lines_added': grp.median()}
                                 for a, grp in df.groupby('author_name')['lines_added']}
        if 'minutes_since_previous_commit' in df:
            stats['commit_interval'] = {'median': df['minutes_since_previous_commit'].median()}
        return stats

    def analyze_commits(self) -> List[Dict]:
        print("[ANALYZE] Starting commit-by-commit analysis…")
        commits_data, file_count = [], {}

        all_commits = self.get_commits()
        all_commits.reverse()
        prev_dt = None

        for idx, c in enumerate(all_commits, 1):
            sha = c["sha"]
            print(f"[ANALYZE] ({idx}/{len(all_commits)}) Processing commit {sha}")
            det = self.get_commit_details(sha)

            try:
                print(f"[GIT] Checking out {sha}")
                self.repo.git.checkout(sha)
            except GitCommandError:
                print(f"[GIT] Cannot checkout {sha}, skipping FS analysis")

            if not det or "commit" not in det:
                print(f"[ERROR] Invalid or missing commit data for {sha}, skipping.")
                continue

            msg = det["commit"]["message"]
            author = det["commit"]["author"]
            name = author.get("name", "Unknown")
            dt = datetime.strptime(author["date"], "%Y-%m-%dT%H:%M:%SZ")

            files = det.get("files", [])
            print(f"[ANALYZE]  {len(files)} files changed")

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
                "eslint_warnings","eslint_errors","checkstyle_issues",
                "cyclomatic_complexity"
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

            files_added = sum(1 for f in files if f.get("status") == "added")
            files_deleted = sum(1 for f in files if f.get("status") == "removed")
            lines_changed = added + deleted
            todo_fixme_count = 0
            for f in files:
                patch = f.get("patch", "")
                todo_fixme_count += len(re.findall(r'\bTODO\b|\bFIXME\b', patch))

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
                "files_added": files_added,
                "files_deleted": files_deleted,
                "lines_changed": lines_changed,
                "todo_fixme_count": todo_fixme_count,
                "avg_file_history": avg_hist,
                "complexity_score": comp,
                **metrics,
                "file_list": [f["filename"] for f in files],
            }
            commits_data.append(data)

            for f in files:
                file_count[f["filename"]] = file_count.get(f["filename"], 0) + 1
            prev_dt = dt

        print(f"[ANALYZE] Completed analysis of {len(commits_data)} commits.")
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

    def push_and_create_pr(self, branch_name: str, commits: List[Dict]) -> None:
        print(f"[PR] Fetching origin")
        self.repo.git.fetch('origin')

        base_branches_to_try = ['main', 'master']
        base_branch = None

        for candidate in base_branches_to_try:
            try:
                self.repo.git.rev_parse(f'origin/{candidate}', verify=True)
                base_branch = candidate
                print(f"[PR] Found base branch: origin/{base_branch}")
                break
            except GitCommandError:
                print(f"[PR] origin/{candidate} not found, trying next...")

        if base_branch is None:
            raise RuntimeError("Не удалось найти базовую ветку origin/main или origin/master")

        rel_path = "CapaRecommendations.md"
        full_path = os.path.join(self.local_path, rel_path)

        if os.path.exists(full_path):
            print(f"[PR] Removing conflicting file before checkout: {rel_path}")
            os.remove(full_path)

        if branch_name in self.repo.branches:
            print(f"[PR] Branch {branch_name} already exists locally, checking out.")
            self.repo.git.checkout(branch_name)
        else:
            print(f"[PR] Creating branch {branch_name} from origin/{base_branch}")
            self.repo.git.checkout('-b', branch_name, f'origin/{base_branch}')

        print(f"[PR] Re-creating {rel_path} after checkout")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("# CAPA Recommendations\n\n")
            for c in commits:
                if c.get("has_capa"):
                    f.write(f"## Commit {c['commit']} — risk={c['Risk_Proba']:.2f}\n")
                    for rec in c["capa_recommendations"]:
                        f.write(f"- {rec}\n")
                    f.write("\n")

        print(f"[PR] Adding file {rel_path}")
        self.repo.index.add([rel_path])
        print(f"[PR] Committing changes")
        self.repo.index.commit("Add CAPA recommendations")

        print(f"[PR] Pushing branch {branch_name}")
        origin = self.repo.remote(name='origin')

        try:
            origin.push(branch_name)
        except GitCommandError as e:
            print(f"[PR] ⚠ Push failed: {e}")
            print("[PR] Пропускаем пуш и создание PR из-за ошибки авторизации/доступа.")
            return

        pr_data = {
            "title": "Add automated CAPA recommendations",
            "head": f"{self.repo_owner}:{branch_name}",
            "base": base_branch,
            "body": "This PR adds automatically generated corrective/preventive actions."
        }
        print(f"[PR] Opening PR via GitHub API")
        response = requests.post(
            f"{self.api_url}/pulls",
            headers=self.headers,
            json=pr_data
        )
        if response.status_code in (200, 201):
            pr_url = response.json().get("html_url")
            print(f"[PR] Pull request created: {pr_url}")
        else:
            print(f"[PR] ⚠ Failed to create PR: {response.status_code} {response.text}")

    def analyze_and_pr(self, commits: Optional[List[Dict]] = None) -> None:
        if commits is None:
            commits = self.analyze_commits()

        if not commits:
            print("No commits — пропускаем PR.")
            return

        model = CommitRiskModel(CascadeForestClassifier(random_state=42))
        model.fit(commits)
        probs = model.predict_proba(commits)

        repo_stats = self.compute_repo_stats(commits)

        for commit, p in zip(commits, probs):
            commit["Risk_Proba"] = float(p)
            commit["has_capa"] = True
            commit["capa_recommendations"] = recommendation_generator.generate_recommendations(
                commit, p, repo_stats, model.feature_importances()
            )

        branch = f"capa-{datetime.utcnow():%Y%m%d%H%M}"
        self.push_and_create_pr(branch, commits)
