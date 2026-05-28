#!/usr/bin/env python3
"""Build ``docs/page/assets/community.json`` with contributor + stargazer
records (login, avatar_url). Avatars are referenced by their GitHub CDN URL —
no binary files committed. Requires ``gh auth login`` or ``GH_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PATH = REPO_ROOT / "docs" / "page" / "assets" / "community.json"

DEFAULT_EXCLUDE_LOGINS: frozenset[str] = frozenset({"jiankangdeng"})


@dataclass(frozen=True)
class Contributor:
    login: str
    contributions: int
    avatar_url: str


@dataclass(frozen=True)
class Stargazer:
    login: str
    starred_at: str
    avatar_url: str


def _gh_api_paginated(path: str, extra_headers: list[str] | None = None) -> list[dict]:
    out: list[dict] = []
    per_page = 100
    page = 1
    sep = "&" if "?" in path else "?"
    while True:
        url = f"{path}{sep}per_page={per_page}&page={page}"
        cmd = ["gh", "api"]
        for h in extra_headers or []:
            cmd.extend(["-H", h])
        cmd.append(url)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"gh api failed ({url}):\n{proc.stderr}")
        batch = json.loads(proc.stdout)
        if not isinstance(batch, list):
            raise RuntimeError(f"unexpected non-list response from {url}: {batch!r}")
        if not batch:
            break
        out.extend(batch)
        print(f"  {path}: page {page} +{len(batch)} (total {len(out)})", file=sys.stderr)
        if len(batch) < per_page:
            break
        page += 1
    return out


def fetch_contributors(owner: str, repo: str, exclude: frozenset[str]) -> list[Contributor]:
    raw = _gh_api_paginated(f"repos/{owner}/{repo}/contributors")
    out: list[Contributor] = []
    for entry in raw:
        if entry.get("type") != "User":
            continue
        login = entry.get("login")
        if not login or login in exclude:
            continue
        out.append(
            Contributor(
                login=login,
                contributions=int(entry.get("contributions", 0)),
                avatar_url=entry["avatar_url"],
            )
        )
    out.sort(key=lambda c: (-c.contributions, c.login.lower()))
    return out


def fetch_stargazers(owner: str, repo: str) -> list[Stargazer]:
    # The star+json Accept header wraps each entry as {"starred_at", "user": {...}};
    # without it the endpoint returns flat user objects and we lose starred_at.
    raw = _gh_api_paginated(
        f"repos/{owner}/{repo}/stargazers",
        extra_headers=["Accept: application/vnd.github.star+json"],
    )
    out: list[Stargazer] = []
    for entry in raw:
        user = entry.get("user") or entry
        login = user.get("login")
        if not login:
            continue
        out.append(
            Stargazer(
                login=login,
                starred_at=entry.get("starred_at", ""),
                avatar_url=user["avatar_url"],
            )
        )
    out.sort(key=lambda s: s.starred_at or "")
    return out


def build_community_json(
    owner: str,
    repo: str,
    out_path: Path,
    exclude_logins: frozenset[str] = DEFAULT_EXCLUDE_LOGINS,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"fetching contributors for {owner}/{repo}...", file=sys.stderr)
    contributors = fetch_contributors(owner, repo, exclude_logins)
    print(f"  -> {len(contributors)} contributors", file=sys.stderr)

    print(f"fetching stargazers for {owner}/{repo}...", file=sys.stderr)
    stargazers = fetch_stargazers(owner, repo)
    print(f"  -> {len(stargazers)} stargazers", file=sys.stderr)

    payload = {
        "owner": owner,
        "repo": repo,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contributors": [
            {
                "login": c.login,
                "contributions": c.contributions,
                "avatar_url": c.avatar_url,
            }
            for c in contributors
        ],
        "stargazers": [
            {
                "login": s.login,
                "starred_at": s.starred_at,
                "avatar_url": s.avatar_url,
            }
            for s in stargazers
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--owner", default="EvolvingLMMs-Lab")
    parser.add_argument("--repo", default="LLaVA-OneVision-2")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"output JSON path (default: {DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDE_LOGINS),
        help="logins to exclude from contributors (bots, etc.)",
    )
    args = parser.parse_args()

    build_community_json(
        owner=args.owner,
        repo=args.repo,
        out_path=args.out,
        exclude_logins=frozenset(args.exclude),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
