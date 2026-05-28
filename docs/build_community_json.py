#!/usr/bin/env python3
"""Build a static community.json + avatar bundle for docs/page/projects/index.html.

Fetches:
  - Contributors (sorted by commit count, desc) via /repos/.../contributors
  - Stargazers (sorted by star time, asc — oldest first) via /repos/.../stargazers

For each user, downloads an 88px JPEG avatar into ``docs/page/assets/avatars/``
and emits a single ``community.json`` consumed by the projects page at runtime.

The JSON shape::

    {
      "owner": "...",
      "repo": "...",
      "generated_at": "2026-05-28T...",
      "avatar_dir": "avatars",
      "contributors": [
        {"login": "...", "contributions": 42, "avatar": "avatars/foo.jpg"},
        ...
      ],
      "stargazers": [
        {"login": "...", "starred_at": "2025-01-02T...", "avatar": "avatars/foo.jpg"},
        ...
      ]
    }

Auth: uses ``gh api`` (must be ``gh auth login``-ed). Avoids the 60/h anonymous
rate limit and works inside CI when ``GH_TOKEN`` is set.

Run from the repo root::

    python docs/build_community_json.py \\
        --owner EvolvingLMMs-Lab \\
        --repo LLaVA-OneVision-2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "page" / "assets"

AVATAR_SIZE_PX = 88  # matches CSS .community-avatar size for crisp rendering on retina
DOWNLOAD_WORKERS = 16
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30

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


def _gh_api(path: str) -> list[dict]:
    """Page through a list-returning GitHub REST endpoint via ``gh api``."""
    out: list[dict] = []
    per_page = 100
    page = 1
    sep = "&" if "?" in path else "?"
    while True:
        url = f"{path}{sep}per_page={per_page}&page={page}"
        cmd = ["gh", "api", url]
        # If the path needs a star-time accept header, the caller injects it via env.
        if path.endswith("/stargazers") or "/stargazers?" in path or "/stargazers&" in path:
            cmd = [
                "gh",
                "api",
                "-H",
                "Accept: application/vnd.github.star+json",
                url,
            ]
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
    raw = _gh_api(f"repos/{owner}/{repo}/contributors")
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
    # /contributors is already sorted by commit count desc, but be defensive.
    out.sort(key=lambda c: (-c.contributions, c.login.lower()))
    return out


def fetch_stargazers(owner: str, repo: str) -> list[Stargazer]:
    raw = _gh_api(f"repos/{owner}/{repo}/stargazers")
    out: list[Stargazer] = []
    for entry in raw:
        # star+json wraps the user in {"starred_at": ..., "user": {...}}
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
    # Oldest-first (the way the API already returns them, but be explicit).
    out.sort(key=lambda s: s.starred_at or "")
    return out


def _download_one(login: str, avatar_url: str, dest_dir: Path, size_px: int) -> tuple[str, str | None, str | None]:
    dest = dest_dir / f"{login}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return login, dest.name, None
    sep = "&" if "?" in avatar_url else "?"
    url = f"{avatar_url}{sep}s={size_px}"
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            r = requests.get(
                url,
                timeout=DOWNLOAD_TIMEOUT,
                headers={"User-Agent": "build_community_json.py"},
            )
            r.raise_for_status()
            dest.write_bytes(r.content)
            return login, dest.name, None
        except Exception as exc:  # noqa: BLE001 — retry-and-report at the boundary
            if attempt == DOWNLOAD_RETRIES - 1:
                return login, None, f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (attempt + 1))
    return login, None, "unknown"


def download_all_avatars(logins: list[tuple[str, str]], dest_dir: Path, size_px: int) -> dict[str, str]:
    """Download avatars for ``[(login, url), ...]`` in parallel.

    Returns ``{login: relative_filename}`` for everything that downloaded successfully.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    # Dedupe — many contributors are also stargazers; download each login once.
    seen: dict[str, str] = {}
    for login, url in logins:
        seen.setdefault(login, url)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futs = {pool.submit(_download_one, login, url, dest_dir, size_px): login for login, url in seen.items()}
        for i, fut in enumerate(as_completed(futs), start=1):
            login, fname, err = fut.result()
            if fname:
                out[login] = fname
            else:
                failures.append((login, err or "unknown"))
            if i % 100 == 0 or i == len(futs):
                print(
                    f"  avatars: {i}/{len(futs)} (ok={len(out)}, fail={len(failures)})",
                    file=sys.stderr,
                )
    print(
        f"avatars downloaded in {time.time() - t0:.1f}s (ok={len(out)}, fail={len(failures)})",
        file=sys.stderr,
    )
    if failures:
        print("  failures (first 10):", file=sys.stderr)
        for login, err in failures[:10]:
            print(f"    {login}: {err}", file=sys.stderr)
    return out


def build_community_json(
    owner: str,
    repo: str,
    out_dir: Path,
    avatar_subdir: str = "avatars",
    exclude_logins: frozenset[str] = DEFAULT_EXCLUDE_LOGINS,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    avatar_dir = out_dir / avatar_subdir

    print(f"fetching contributors for {owner}/{repo}...", file=sys.stderr)
    contributors = fetch_contributors(owner, repo, exclude_logins)
    print(f"  -> {len(contributors)} contributors", file=sys.stderr)

    print(f"fetching stargazers for {owner}/{repo}...", file=sys.stderr)
    stargazers = fetch_stargazers(owner, repo)
    print(f"  -> {len(stargazers)} stargazers", file=sys.stderr)

    print(f"downloading avatars to {avatar_dir}/ ...", file=sys.stderr)
    logins = [(c.login, c.avatar_url) for c in contributors] + [(s.login, s.avatar_url) for s in stargazers]
    avatars = download_all_avatars(logins, avatar_dir, AVATAR_SIZE_PX)

    payload = {
        "owner": owner,
        "repo": repo,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "avatar_dir": avatar_subdir,
        "contributors": [
            {
                "login": c.login,
                "contributions": c.contributions,
                "avatar": f"{avatar_subdir}/{avatars[c.login]}" if c.login in avatars else None,
            }
            for c in contributors
        ],
        "stargazers": [
            {
                "login": s.login,
                "starred_at": s.starred_at,
                "avatar": f"{avatar_subdir}/{avatars[s.login]}" if s.login in avatars else None,
            }
            for s in stargazers
        ],
    }

    out_path = out_dir / "community.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", default="EvolvingLMMs-Lab")
    parser.add_argument("--repo", default="LLaVA-OneVision-2")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory to write community.json + avatars/ into (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--avatar-subdir", default="avatars")
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
        out_dir=args.out_dir,
        avatar_subdir=args.avatar_subdir,
        exclude_logins=frozenset(args.exclude),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
