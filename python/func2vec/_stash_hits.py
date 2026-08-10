"""Read a JSON blob of MCP search_code results from stdin, append normalized
{sha, repo, path, query} rows to data/hits.jsonl. Query is passed as argv[1]."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
HITS = ROOT / "data" / "hits.jsonl"

query = sys.argv[1] if len(sys.argv) > 1 else ""
blob = json.loads(sys.stdin.read())
items = blob.get("items", [])

HITS.parent.mkdir(parents=True, exist_ok=True)
with HITS.open("a") as f:
    for it in items:
        repo_field = it.get("repository")
        if isinstance(repo_field, dict):
            repo = repo_field.get("full_name") or repo_field.get("fullName")
        else:
            repo = repo_field
        row = {
            "sha": it.get("sha"),
            "repo": repo,
            "path": it.get("path"),
            "query": query,
        }
        if row["sha"] and row["repo"] and row["path"]:
            f.write(json.dumps(row) + "\n")
print(f"[stash] appended {len(items)} items ({query!r}) -> {HITS}", file=sys.stderr)
