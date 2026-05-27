#!/usr/bin/env python3
"""Build a self-contained SVG of repository contributors ranked by commit count.

Fetches the contributor list from the GitHub REST API
(``/repos/{owner}/{repo}/contributors``), filters out bots and explicitly
excluded users, downloads each avatar, and embeds them as base64
``<image>`` elements inside a single SVG grid. Also rewrites the README
contributors block in place between two HTML markers.

Usage::

    python docs/build_contributors_svg.py \\
        --owner EvolvingLMMs-Lab \\
        --repo LLaVA-OneVision-2 \\
        --out asset/contributors.svg \\
        --readme README.md

If ``GITHUB_TOKEN`` (or ``GH_TOKEN``) is set, it is sent as a Bearer token.
This is required inside GitHub Actions to avoid the 60/h anon rate limit.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape


DEFAULT_EXCLUDE_LOGINS: frozenset[str] = frozenset({"jiankangdeng"})

AVATAR_PX = 80
CELL_PAD_X = 16
CELL_PAD_Y = 8
LABEL_GAP = 8
LABEL_FONT_SIZE = 13
LABEL_FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
COLUMNS_PER_ROW = 8

README_MARKER_START = "<!-- readme: collaborators,contributors,jiankangdeng/- -start -->"
README_MARKER_END = "<!-- readme: collaborators,contributors,jiankangdeng/- -end -->"


@dataclass(frozen=True)
class Contributor:
    login: str
    contributions: int
    avatar_url: str

    @property
    def profile_url(self) -> str:
        return f"https://github.com/{self.login}"


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "build_contributors_svg.py",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_get(url: str, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}\n{body[:500]}") from exc


def fetch_contributors(owner: str, repo: str) -> list[Contributor]:
    """Fetch human contributors for ``owner/repo``, paginated, bots dropped."""
    headers = _auth_headers()
    per_page = 100
    page = 1
    out: list[Contributor] = []
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page={per_page}&page={page}"
        batch = json.loads(_http_get(url, headers=headers))
        if not batch:
            break
        for entry in batch:
            if entry.get("type") != "User":
                continue
            out.append(
                Contributor(
                    login=entry["login"],
                    contributions=int(entry["contributions"]),
                    avatar_url=entry["avatar_url"],
                )
            )
        if len(batch) < per_page:
            break
        page += 1
    return out


def fetch_avatar_png(avatar_url: str, size: int) -> bytes:
    sep = "&" if "?" in avatar_url else "?"
    sized = f"{avatar_url}{sep}s={size}"
    return _http_get(sized, headers={"User-Agent": "build_contributors_svg.py"})


def _truncate(text: str, max_chars: int = 16) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "\u2026"


def build_svg(contributors: list[Contributor], avatar_pngs: dict[str, bytes]) -> str:
    n = len(contributors)
    if n == 0:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0" viewBox="0 0 0 0"></svg>\n'

    cell_w = AVATAR_PX + 2 * CELL_PAD_X
    cell_h = AVATAR_PX + LABEL_GAP + LABEL_FONT_SIZE + 2 * CELL_PAD_Y
    cols = min(COLUMNS_PER_ROW, n)
    rows = (n + COLUMNS_PER_ROW - 1) // COLUMNS_PER_ROW
    width = cell_w * cols
    height = cell_h * rows
    radius = AVATAR_PX / 2

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Contributors ranked by commit count">'
        f'<defs><clipPath id="avatarClip">'
        f'<circle cx="{radius}" cy="{radius}" r="{radius}"/>'
        f"</clipPath></defs>"
    ]

    for index, contributor in enumerate(contributors):
        png = avatar_pngs.get(contributor.login)
        if png is None:
            continue
        row, col = divmod(index, COLUMNS_PER_ROW)
        cell_x = col * cell_w
        cell_y = row * cell_h
        avatar_x = cell_x + CELL_PAD_X
        avatar_y = cell_y + CELL_PAD_Y
        b64 = base64.b64encode(png).decode("ascii")
        label_x = cell_x + cell_w / 2
        label_y = avatar_y + AVATAR_PX + LABEL_GAP + LABEL_FONT_SIZE
        commit_word = "commit" if contributor.contributions == 1 else "commits"
        title = f"{contributor.login} ({contributor.contributions} {commit_word})"

        out.append(
            f'<a xlink:href="{xml_escape(contributor.profile_url)}" target="_blank">'
            f"<title>{xml_escape(title)}</title>"
            f'<g transform="translate({avatar_x},{avatar_y})">'
            f'<image href="data:image/png;base64,{b64}" '
            f'xlink:href="data:image/png;base64,{b64}" '
            f'width="{AVATAR_PX}" height="{AVATAR_PX}" '
            f'clip-path="url(#avatarClip)" preserveAspectRatio="xMidYMid slice"/>'
            f'<circle cx="{radius}" cy="{radius}" r="{radius - 0.5}" '
            f'fill="none" stroke="#d0d7de" stroke-width="1"/>'
            f"</g>"
            f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
            f'font-family="{LABEL_FONT_FAMILY}" font-size="{LABEL_FONT_SIZE}" '
            f'fill="#24292f">'
            f"{xml_escape(_truncate(contributor.login))}"
            f"</text>"
            f"</a>"
        )

    out.append("</svg>\n")
    return "".join(out)


def rewrite_readme(readme_path: Path, svg_rel_path: str) -> bool:
    """Replace the README marker block. Returns False if already up to date."""
    src = readme_path.read_text(encoding="utf-8")
    start = src.find(README_MARKER_START)
    end = src.find(README_MARKER_END)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            f"Could not find contributors markers in {readme_path}.\n"
            f"  Expected start: {README_MARKER_START!r}\n"
            f"  Expected end:   {README_MARKER_END!r}"
        )
    end_after = end + len(README_MARKER_END)
    new_block = (
        f"{README_MARKER_START}\n"
        f'<p align="center">\n'
        f'  <a href="https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/graphs/contributors">\n'
        f'    <img src="{svg_rel_path}" alt="Contributors ranked by commit count" />\n'
        f"  </a>\n"
        f"</p>\n"
        f"{README_MARKER_END}"
    )
    new_src = src[:start] + new_block + src[end_after:]
    if new_src == src:
        return False
    readme_path.write_text(new_src, encoding="utf-8")
    return True


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a contributors SVG ranked by commit count and update README.")
    parser.add_argument("--owner", default="EvolvingLMMs-Lab")
    parser.add_argument("--repo", default="LLaVA-OneVision-2")
    parser.add_argument("--out", type=Path, default=Path("asset/contributors.svg"))
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Pass an empty string to skip README update.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help=(f"Additional login to exclude (repeatable). Always-on defaults: {sorted(DEFAULT_EXCLUDE_LOGINS)}"),
    )
    parser.add_argument(
        "--readme-svg-path",
        default=None,
        help="Path to embed into README <img src> (default: same as --out).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    excludes: set[str] = set(DEFAULT_EXCLUDE_LOGINS)
    if args.exclude:
        excludes.update(args.exclude)

    print(
        f"[contributors-svg] fetching {args.owner}/{args.repo} contributors ...",
        file=sys.stderr,
    )
    raw = fetch_contributors(args.owner, args.repo)
    contributors = [c for c in raw if c.login not in excludes]
    contributors.sort(key=lambda c: (-c.contributions, c.login.lower()))

    print(
        f"[contributors-svg] {len(contributors)} contributors after filtering "
        f"({len(raw) - len(contributors)} dropped: {sorted(excludes)})",
        file=sys.stderr,
    )
    if not contributors:
        print("[contributors-svg] no contributors to render; aborting.", file=sys.stderr)
        return 1

    avatar_pngs: dict[str, bytes] = {}
    for c in contributors:
        print(
            f"[contributors-svg]   fetching avatar for {c.login} ({c.contributions} commits)",
            file=sys.stderr,
        )
        avatar_pngs[c.login] = fetch_avatar_png(c.avatar_url, AVATAR_PX * 2)

    svg = build_svg(contributors, avatar_pngs)

    if args.dry_run:
        print(
            f"[contributors-svg] dry-run: would write {len(svg)} bytes to {args.out}",
            file=sys.stderr,
        )
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    print(
        f"[contributors-svg] wrote {len(svg)} bytes to {args.out}",
        file=sys.stderr,
    )

    if args.readme and str(args.readme):
        svg_rel = args.readme_svg_path or str(args.out)
        changed = rewrite_readme(args.readme, svg_rel)
        print(
            f"[contributors-svg] README {'updated' if changed else 'already up to date'} ({args.readme})",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
