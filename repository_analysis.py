# repository_analysis.py

import os
import requests
import re
import json
import subprocess
from datetime import datetime
import pandas as pd
from git import Repo
from typing import List, Dict

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
        self.api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {"Authorization": f"token {token}"}

        self.local_path = os.path.join(clone_dir, repo_name)
        if not os.path.isdir(self.local_path):
            clone_url = f"https://github.com/{repo_owner}/{repo_name}.git"
            print(f"Cloning {clone_url}…")
            Repo.clone_from(clone_url, self.local_path)
        self.repo = Repo(self.local_path)

        self.complexity_re = re.compile(r"\b(if|for|while|switch|case)\b")

    def get_commits(self) -> List[Dict]:
        commits = []
        page = 1
        while True:
            resp = requests.get(
                f"{self.api_url}/commits",
                headers=self.headers,
                params={"page": page, "per_page": 100},
            )
            data = resp.json()
            if not data:
                break
            commits.extend(data)
            page += 1
        return commits

    def get_commit_details(self, sha: str) -> Dict:
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
        except:
            pass

        try:
            r = subprocess.run(
                ["bandit", "-f", "json", "-r", full_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            js = json.loads(r.stdout or "{}")
            bandit = len(js.get("results", []))
        except:
            pass

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
        except:
            pass
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
        except:
            pass
        return {"checkstyle_issues": count}

    def analyze_commits(self) -> List[Dict]:
        commits_data = []
        file_count = {}

        all_commits = self.get_commits()
        all_commits.reverse()

        prev_dt = None
        for c in all_commits:
            sha = c["sha"]
            det = self.get_commit_details(sha)
            self.repo.git.checkout(sha)

            msg = det["commit"]["message"]
            author = det["commit"]["author"]
            name = author.get("name", "Unknown")
            dt = datetime.strptime(author["date"], "%Y-%m-%dT%H:%M:%SZ")

            files = det.get("files", [])
            added = sum(f.get("additions", 0) for f in files)
            deleted = sum(f.get("deletions", 0) for f in files)
            changed = len(files)
            hist = sum(file_count.get(f["filename"], 0) for f in files)
            avg_hist = hist / changed if changed else 0

            comp = 0
            for f in files:
                for ln in f.get("patch", "").splitlines():
                    if ln.startswith("+") and not ln.startswith("+++") and self.complexity_re.search(ln):
                        comp += 1

            delta = (dt - prev_dt).total_seconds() / 60 if prev_dt else None

            metrics = {
                "pylint_warnings": 0, "pylint_errors": 0, "bandit_issues": 0,
                "eslint_warnings": 0, "eslint_errors": 0,
                "checkstyle_issues": 0
            }
            for f in files:
                lang = self.detect_language(f["filename"])
                full = os.path.join(self.local_path, f["filename"])
                if lang == "python":
                    out = self.analyze_python_file(full)
                elif lang == "javascript":
                    out = self.analyze_javascript_file(full)
                elif lang == "java":
                    out = self.analyze_java_file(full)
                else:
                    out = {}
                for k, v in out.items():
                    metrics[k] = metrics.get(k, 0) + v

            data = {
                "commit": sha,
                "author_name": name,
                "author_datetime": dt,
                "minutes_since_previous_commit": delta,
                "message": msg,
                "message_length": len(msg),
                "lines_added": added,
                "lines_deleted": deleted,
                "files_changed": changed,
                "avg_file_history": avg_hist,
                "complexity_score": comp,
                "file_list": [f["filename"] for f in files],
                **metrics
            }
            commits_data.append(data)

            for f in files:
                file_count[f["filename"]] = file_count.get(f["filename"], 0) + 1

            prev_dt = dt

        return commits_data

    def analyze_time_series(self, commits_data: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(commits_data).dropna(subset=["minutes_since_previous_commit"])
        df["rolling_mean"] = df["minutes_since_previous_commit"].rolling(10).mean()
        df["rolling_std"]  = df["minutes_since_previous_commit"].rolling(10).std()
        df["is_anomaly"]   = df["minutes_since_previous_commit"] > df["rolling_mean"] + 2 * df["rolling_std"]
        return df
