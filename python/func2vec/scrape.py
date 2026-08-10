"""Scrape GitHub for Python files that import ML libraries.

Two-mode design so it works in either a session with GitHub MCP access
or a standalone shell with a GITHUB_TOKEN:

  Mode A (session): the driver (Claude) calls mcp__github__search_code many
    times, appends each hit to data/hits.jsonl, then runs this script with
    `--from-hits` to fetch full contents via raw.githubusercontent.com.

  Mode B (standalone): with a GITHUB_TOKEN that has code-search permission,
    call `--queries default --pages 10` and the script both searches and fetches.
    (Not usable inside a session-proxied environment that blocks /search/code.)

Resumable: skips SHAs already in data/manifest.jsonl. Re-run to grow the corpus.

Usage:
    # Mode A — after the driver has appended hits:
    python -m func2vec.scrape --from-hits

    # Mode B — end-to-end:
    python -m func2vec.scrape --queries default --pages 5
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "raw"
MANIFEST = DATA / "manifest.jsonl"
HITS = DATA / "hits.jsonl"

GH = "https://api.github.com"
RAW_GH = "https://raw.githubusercontent.com"

DEFAULT_QUERIES = [
    "from sklearn.cluster import language:python",
    "from sklearn.linear_model import language:python",
    "from sklearn.ensemble import language:python",
    "from sklearn.svm import language:python",
    "from sklearn.decomposition import language:python",
    "from sklearn.model_selection import language:python",
    "from sklearn.preprocessing import language:python",
    "from sklearn.metrics import language:python",
    "import torch.nn language:python",
    "from torch.utils.data import language:python",
    "from tensorflow.keras import language:python",
    "from keras.layers import language:python",
    "import xgboost language:python",
    "import lightgbm language:python",
    "from sklearn.pipeline import language:python",
]


@dataclass
class ManifestRow:
    sha: str
    repo: str
    path: str
    size: int
    query: str
    local: str


def load_seen_shas() -> set[str]:
    if not MANIFEST.exists():
        return set()
    seen = set()
    with MANIFEST.open() as f:
        for line in f:
            try:
                seen.add(json.loads(line)["sha"])
            except Exception:
                continue
    return seen


def append_manifest(row: ManifestRow) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as f:
        f.write(json.dumps(asdict(row)) + "\n")


def gh_headers() -> dict:
    tok = os.environ.get("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _sleep_for_rate_limit(resp: requests.Response) -> None:
    if resp.status_code in (403, 429) and "rate limit" in resp.text.lower():
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
        wait = max(60, reset - int(time.time()) + 5)
        print(f"[rate-limit] sleeping {wait}s", file=sys.stderr)
        time.sleep(wait)


def search_code(query: str, page: int, per_page: int = 100) -> list[dict]:
    url = f"{GH}/search/code"
    params = {"q": query, "page": page, "per_page": per_page}
    for attempt in range(4):
        r = requests.get(url, headers=gh_headers(), params=params, timeout=30)
        if r.status_code == 200:
            return r.json().get("items", [])
        if r.status_code in (403, 429):
            _sleep_for_rate_limit(r)
            continue
        if r.status_code == 422:
            return []
        print(f"[search] {r.status_code}: {r.text[:200]}", file=sys.stderr)
        time.sleep(2 ** attempt)
    return []


def fetch_raw(repo: str, path: str, sha: str) -> str | None:
    """Fetch a file via raw.githubusercontent.com (public files, no auth).

    Code search returns a *blob* sha, which raw.githubusercontent.com does not
    accept as a ref. We fall back to HEAD (the default branch), which loses
    exact reproducibility but works for the vast majority of repos. If that's a
    404 (renamed branch, moved file, etc.) we try 'main' and 'master' explicitly.
    """
    refs = ["HEAD", "main", "master"]
    for ref in refs:
        url = f"{RAW_GH}/{repo}/{ref}/{path}"
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=15)
            except requests.RequestException:
                time.sleep(1 + attempt)
                continue
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                break  # try next ref
            time.sleep(1 + attempt)
    return None


def fetch_content_api(item: dict) -> str | None:
    """Fallback that goes through the authenticated contents API."""
    repo = item["repository"]["full_name"] if isinstance(item.get("repository"), dict) else item["repo"]
    path = item["path"]
    ref = item.get("sha")
    url = f"{GH}/repos/{repo}/contents/{path}"
    params = {"ref": ref} if ref else {}
    for attempt in range(3):
        r = requests.get(url, headers=gh_headers(), params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and data.get("encoding") == "base64":
                try:
                    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                except Exception:
                    return None
            return None
        if r.status_code in (403, 429):
            _sleep_for_rate_limit(r)
            continue
        if r.status_code == 404:
            return None
        time.sleep(2 ** attempt)
    return None


def safe_local_path(repo: str, path: str, sha: str) -> Path:
    slug = repo.replace("/", "__") + "__" + path.replace("/", "__")
    return RAW / f"{sha[:12]}__{slug}"


def save_one(repo: str, path: str, sha: str, content: str, query: str) -> None:
    lp = safe_local_path(repo, path, sha)
    lp.write_text(content, encoding="utf-8")
    append_manifest(
        ManifestRow(sha=sha, repo=repo, path=path, size=len(content), query=query, local=lp.name)
    )


def scrape_from_hits(max_bytes: int, workers: int = 20) -> int:
    """Mode A: consume data/hits.jsonl, fetch each via raw with a thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    if not HITS.exists():
        print(f"[scrape] no hits file at {HITS}", file=sys.stderr)
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    seen = load_seen_shas()

    todo: list[dict] = []
    total = 0
    with HITS.open() as f:
        for line in f:
            total += 1
            try:
                hit = json.loads(line)
            except Exception:
                continue
            sha = hit.get("sha")
            if not (sha and hit.get("repo") and hit.get("path")):
                continue
            if sha in seen:
                continue
            seen.add(sha)
            todo.append(hit)

    saved = 0
    lock = Lock()

    def _do(hit: dict) -> bool:
        content = fetch_raw(hit["repo"], hit["path"], hit["sha"])
        if content is None or len(content) > max_bytes or len(content) < 40:
            return False
        with lock:
            save_one(hit["repo"], hit["path"], hit["sha"], content, hit.get("query", ""))
        return True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_do, h) for h in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                if fut.result():
                    saved += 1
            except Exception:
                pass
            if i % 200 == 0:
                print(f"[scrape] progress={i}/{len(todo)} saved={saved}", file=sys.stderr)

    print(f"[scrape] done from-hits. saved={saved} scanned={total} todo={len(todo)}")
    return saved


def scrape_via_search(queries: Iterable[str], pages: int, per_page: int, max_bytes: int) -> int:
    """Mode B: search + fetch in one go. Requires unblocked /search/code."""
    RAW.mkdir(parents=True, exist_ok=True)
    seen = load_seen_shas()
    saved = 0
    for q in queries:
        for page in range(1, pages + 1):
            items = search_code(q, page=page, per_page=per_page)
            if not items:
                break
            for item in items:
                sha = item.get("sha")
                if not sha or sha in seen:
                    continue
                seen.add(sha)
                content = fetch_raw(item["repository"]["full_name"], item["path"], sha)
                if content is None:
                    content = fetch_content_api(item)
                if content is None or len(content) > max_bytes or len(content) < 40:
                    continue
                save_one(item["repository"]["full_name"], item["path"], sha, content, q)
                saved += 1
                if saved % 25 == 0:
                    print(f"[scrape] saved={saved} q={q!r} page={page}", file=sys.stderr)
            time.sleep(2)
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-hits", action="store_true", help="consume data/hits.jsonl instead of running search")
    p.add_argument("--queries", default="default")
    p.add_argument("--pages", type=int, default=5)
    p.add_argument("--per-page", type=int, default=100)
    p.add_argument("--max-bytes", type=int, default=200_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_hits:
        scrape_from_hits(max_bytes=args.max_bytes)
    else:
        queries = DEFAULT_QUERIES if args.queries == "default" else [ln.strip() for ln in Path(args.queries).read_text().splitlines() if ln.strip()]
        n = scrape_via_search(queries, pages=args.pages, per_page=args.per_page, max_bytes=args.max_bytes)
        print(f"[scrape] done via-search. saved={n}")
    if MANIFEST.exists():
        total = sum(1 for _ in MANIFEST.open())
        print(f"[scrape] manifest rows: {total}")


if __name__ == "__main__":
    main()
