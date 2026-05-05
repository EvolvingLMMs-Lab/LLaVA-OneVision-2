#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch


OURS_MODEL = "LLaVA-OneVision-2"
TITLE = "Table 1: Benchmark comparison of LLaVA-OneVision-2.0."

GID_HEADER = "tbl-header"
GID_ROW_PREFIX = "tbl-row-"
GID_BEST_PREFIX = "tbl-best-"
GID_SECOND_PREFIX = "tbl-second-"
GID_OURS_COL = "tbl-ours-col"


@dataclass
class Theme:
    name: str
    page_bg: str
    fg: str
    rule: str
    dim_fg: str
    best_bg: str
    best_edge: str
    second_bg: str
    second_edge: str
    ours_tint: str
    zebra_bg: str
    cat_fg: str


LIGHT = Theme(
    name="light",
    page_bg="#ffffff",
    fg="#1a1a1a",
    rule="#1a1a1a",
    dim_fg="#6a737d",
    best_bg="#fef3c7",
    best_edge="#f0d878",
    second_bg="#eef1f4",
    second_edge="none",
    ours_tint="#f6f8fa",
    zebra_bg="#f3f5f7",
    cat_fg="#4a5258",
)


DARK = Theme(
    name="dark",
    page_bg="#0d1117",
    fg="#f0f6fc",
    rule="#f0f6fc",
    dim_fg="#8b949e",
    best_bg="#3d2f08",
    best_edge="#7a5f1a",
    second_bg="#21262d",
    second_edge="none",
    ours_tint="#161b22",
    zebra_bg="#181d24",
    cat_fg="#b8c0c8",
)


def parse_csv(path: Path):
    with path.open() as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def to_float(x: str):
    try:
        return float(x)
    except ValueError:
        return None


def round_box(ax, x, y, w, h, *, radius=0.04, facecolor="none",
              edgecolor="none", linewidth=0, gid=None):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
    )
    if gid:
        p.set_gid(gid)
    ax.add_patch(p)
    return p


def render(csv_path: Path, out_path: Path, theme: Theme, animate: bool = False) -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times"]
    rcParams["mathtext.fontset"] = "dejavuserif"
    rcParams["svg.fonttype"] = "none"

    header, body = parse_csv(csv_path)
    n_rows = len(body)
    n_cols = len(header)
    model_cols = list(range(2, n_cols))

    row_best = []
    for r in body:
        vals = [(i, to_float(r[i])) for i in model_cols]
        vals = [(i, v) for i, v in vals if v is not None]
        vs = sorted(vals, key=lambda kv: kv[1], reverse=True)
        best = vs[0][0] if vs else None
        second = vs[1][0] if len(vs) > 1 else None
        row_best.append((best, second))

    col_w = [1.55, 3.5] + [2.75] * len(model_cols)
    row_h = 0.65
    header_h = 0.84
    pad_x = 0.65
    pad_top = 1.50
    pad_bot = 0.90

    inner_w = sum(col_w)
    fig_w = inner_w + 2 * pad_x
    fig_h = pad_top + header_h + row_h * n_rows + pad_bot

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(theme.page_bg)
    ax.set_facecolor(theme.page_bg)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis("off")

    ax.text(
        pad_x, 0.36, "Table 1: ",
        ha="left", va="top",
        fontsize=18, fontweight="bold", color=theme.dim_fg,
    )
    ax.text(
        pad_x + 1.1, 0.36, "Benchmark comparison of LLaVA-OneVision-2.0.",
        ha="left", va="top",
        fontsize=18, fontweight="bold", color=theme.fg,
    )

    legend_y = 0.92
    legend_x = pad_x
    sw_w, sw_h = 0.32, 0.30
    round_box(
        ax, legend_x, legend_y - sw_h / 2 - 0.05, sw_w, sw_h,
        radius=0.06,
        facecolor=theme.best_bg,
        edgecolor=theme.best_edge,
        linewidth=0.6 if theme.best_edge != "none" else 0,
    )
    ax.text(
        legend_x + sw_w + 0.12, legend_y, "Best",
        ha="left", va="center",
        fontsize=14, color=theme.fg,
    )
    legend_x2 = legend_x + sw_w + 0.12 + 0.65
    round_box(
        ax, legend_x2, legend_y - sw_h / 2 - 0.05, sw_w, sw_h,
        radius=0.06,
        facecolor=theme.second_bg,
        edgecolor="none",
        linewidth=0,
    )
    ax.text(
        legend_x2 + sw_w + 0.12, legend_y, "Second-best",
        ha="left", va="center",
        fontsize=14, color=theme.fg,
    )

    x_off = [pad_x]
    for w in col_w:
        x_off.append(x_off[-1] + w)

    top_rule_y = pad_top
    mid_rule_y = pad_top + header_h
    bot_rule_y = pad_top + header_h + row_h * n_rows

    ours_ci = next((i for i, h in enumerate(header) if h == OURS_MODEL), None)
    if ours_ci is not None:
        ox0, ox1 = x_off[ours_ci], x_off[ours_ci + 1]
        ours_patch = ax.fill_between(
            [ox0, ox1], top_rule_y, bot_rule_y,
            color=theme.ours_tint, linewidth=0, zorder=0.5,
        )
        ours_patch.set_gid(GID_OURS_COL)

    for ri in range(n_rows):
        if ri % 2 == 1:
            zy0 = mid_rule_y + ri * row_h
            ax.fill_between(
                [pad_x, pad_x + inner_w], zy0, zy0 + row_h,
                color=theme.zebra_bg, linewidth=0, zorder=0.3,
            )

    rule_group = []
    rule_group.append(ax.plot(
        [pad_x, pad_x + inner_w], [top_rule_y, top_rule_y],
        color=theme.rule, linewidth=1.0, solid_capstyle="butt",
    )[0])
    rule_group.append(ax.plot(
        [pad_x, pad_x + inner_w], [mid_rule_y, mid_rule_y],
        color=theme.dim_fg, linewidth=0.5, solid_capstyle="butt",
    )[0])
    rule_group.append(ax.plot(
        [pad_x, pad_x + inner_w], [bot_rule_y, bot_rule_y],
        color=theme.rule, linewidth=1.0, solid_capstyle="butt",
    )[0])

    header_artists = []
    for ci, name in enumerate(header):
        x0, x1 = x_off[ci], x_off[ci + 1]
        if ci <= 1:
            tx, ha = x0 + 0.05, "left"
        else:
            tx, ha = (x0 + x1) / 2, "center"
        label = name
        t = ax.text(
            tx, top_rule_y + header_h / 2, label,
            ha=ha, va="center",
            fontsize=17, fontweight="bold", color=theme.fg,
        )
        header_artists.append(t)

    if animate:
        for a in rule_group + header_artists:
            a.set_gid(GID_HEADER)

    prev_category = None
    cat_rule_ys = []
    row_artists_per_row = [[] for _ in range(n_rows)]
    best_box_ids = []
    second_box_ids = []

    for ri, row in enumerate(body):
        y0 = mid_rule_y + ri * row_h
        category = row[0]
        best_idx, second_idx = row_best[ri]

        if prev_category is not None and category != prev_category:
            cat_rule_ys.append(y0)

        for ci, val in enumerate(row):
            x0, x1 = x_off[ci], x_off[ci + 1]
            is_ours_col = ci >= 2 and header[ci] == OURS_MODEL

            if ci == best_idx:
                gid = f"{GID_BEST_PREFIX}{ri}"
                bw = (x1 - x0) - 0.20
                bh = row_h - 0.12
                round_box(
                    ax, x0 + 0.10, y0 + 0.06, bw, bh,
                    radius=0.10,
                    facecolor=theme.best_bg,
                    edgecolor=theme.best_edge,
                    linewidth=0.6 if theme.best_edge != "none" else 0,
                    gid=gid if animate else None,
                )
                best_box_ids.append(gid)
            elif ci == second_idx:
                gid = f"{GID_SECOND_PREFIX}{ri}"
                bw = (x1 - x0) - 0.20
                bh = row_h - 0.12
                round_box(
                    ax, x0 + 0.10, y0 + 0.06, bw, bh,
                    radius=0.10,
                    facecolor=theme.second_bg,
                    edgecolor=theme.second_edge,
                    linewidth=0,
                    gid=gid if animate else None,
                )
                second_box_ids.append(gid)

            if ci == 0:
                if category != prev_category:
                    a = ax.text(
                        x0 + 0.05, y0 + row_h / 2, category.upper(),
                        ha="left", va="center",
                        fontsize=13, color=theme.cat_fg, fontweight="bold",
                    )
                    row_artists_per_row[ri].append(a)
            elif ci == 1:
                if "_" in val:
                    base, _, suffix = val.partition("_")
                    base_artist = ax.text(
                        x0 + 0.05, y0 + row_h / 2, base,
                        ha="left", va="center",
                        fontsize=17, color=theme.fg,
                    )
                    row_artists_per_row[ri].append(base_artist)
                    fig.canvas.draw()
                    bbox = base_artist.get_window_extent(renderer=fig.canvas.get_renderer())
                    bbox_data = bbox.transformed(ax.transData.inverted())
                    suffix_x = bbox_data.x1 + 0.06
                    suffix_artist = ax.text(
                        suffix_x, y0 + row_h / 2, f"({suffix})",
                        ha="left", va="center",
                        fontsize=13, color=theme.dim_fg,
                    )
                    row_artists_per_row[ri].append(suffix_artist)
                else:
                    a = ax.text(
                        x0 + 0.05, y0 + row_h / 2, val,
                        ha="left", va="center",
                        fontsize=17, color=theme.fg,
                    )
                    row_artists_per_row[ri].append(a)
            else:
                if ci == best_idx:
                    weight = "bold"
                elif is_ours_col:
                    weight = "bold"
                else:
                    weight = "normal"
                a = ax.text(
                    (x0 + x1) / 2, y0 + row_h / 2, val,
                    ha="center", va="center",
                    fontsize=17, fontweight=weight, color=theme.fg,
                )
                row_artists_per_row[ri].append(a)

        prev_category = category

    cat_rule_artists = []
    for y in cat_rule_ys:
        line = ax.plot(
            [pad_x, pad_x + inner_w], [y, y],
            color=theme.rule, linewidth=0.4, linestyle=(0, (2, 2)),
        )[0]
        cat_rule_artists.append(line)

    if animate:
        for ri, artists in enumerate(row_artists_per_row):
            for a in artists:
                a.set_gid(f"{GID_ROW_PREFIX}{ri}")

    plt.savefig(out_path, bbox_inches="tight", facecolor=theme.page_bg, format="svg")
    plt.close(fig)

    sanitize_svg_for_github(out_path)

    if animate:
        inject_animation_css(out_path, n_rows, best_box_ids, second_box_ids,
                             ours_present=ours_ci is not None)

    print(f"Saved: {out_path}")


def sanitize_svg_for_github(svg_path: Path) -> None:
    """Make matplotlib SVG render reliably on GitHub.

    1. Strip ``<!DOCTYPE>`` (GitHub's SVG sanitizer occasionally rejects it).
    2. Strip ``<metadata>...</metadata>`` block (RDF/Dublin Core, ~3 KB of dead weight).
    3. Move ``<defs>`` (containing the clipPath) from end of file to right after the
       opening ``<svg>`` tag, eliminating forward references that strict renderers reject.
    """
    svg = svg_path.read_text()

    svg = re.sub(r"<!DOCTYPE[^>]*?>\s*", "", svg, count=1, flags=re.DOTALL)
    svg = re.sub(r"<metadata>.*?</metadata>\s*", "", svg, count=1, flags=re.DOTALL)

    clippath_defs = [
        m for m in re.finditer(r"<defs>\s*<clipPath\b.*?</defs>\s*", svg, flags=re.DOTALL)
    ]
    if clippath_defs:
        m = clippath_defs[-1]
        defs_block = m.group(0)
        svg = svg[: m.start()] + svg[m.end():]
        svg = re.sub(
            r"(<svg[^>]*>)\s*",
            r"\1\n" + defs_block,
            svg,
            count=1,
        )

    svg_path.write_text(svg)


def inject_animation_css(svg_path: Path, n_rows: int,
                         best_ids: list[str], second_ids: list[str],
                         ours_present: bool) -> None:
    svg = svg_path.read_text()

    css_parts = [
        "@keyframes tbl-fade-down { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }",
        "@keyframes tbl-fade-in { from { opacity: 0; transform: translateX(-6px); } to { opacity: 1; transform: translateX(0); } }",
        "@keyframes tbl-pop { 0% { opacity: 0; transform: scale(0.6); } 60% { opacity: 1; transform: scale(1.08); } 100% { opacity: 1; transform: scale(1); } }",
        "@keyframes tbl-pulse { 0%, 100% { filter: none; } 50% { filter: drop-shadow(0 0 4px #f5c518); } }",
        "@keyframes tbl-col-fade { from { opacity: 0; } to { opacity: 1; } }",
        f"g[id='{GID_HEADER}'] {{ animation: tbl-fade-down 0.45s ease-out 0.1s both; transform-box: fill-box; transform-origin: center; }}",
    ]

    if ours_present:
        css_parts.append(
            f"g[id='{GID_OURS_COL}'] {{ animation: tbl-col-fade 0.6s ease-out 0.25s both; }}"
        )

    base_delay = 0.55
    stagger = 0.045
    for ri in range(n_rows):
        delay = base_delay + ri * stagger
        css_parts.append(
            f"g[id='{GID_ROW_PREFIX}{ri}'] {{ animation: tbl-fade-in 0.32s ease-out {delay:.3f}s both; transform-box: fill-box; transform-origin: left center; }}"
        )

    rows_done = base_delay + n_rows * stagger + 0.1
    box_stagger = 0.03
    for i, gid in enumerate(best_ids):
        d = rows_done + i * box_stagger
        css_parts.append(
            f"g[id='{gid}'] {{ animation: tbl-pop 0.5s cubic-bezier(.34,1.56,.64,1) {d:.3f}s both, tbl-pulse 2.4s ease-in-out {d + 0.5:.3f}s 2; transform-box: fill-box; transform-origin: center; }}"
        )
    for i, gid in enumerate(second_ids):
        d = rows_done + i * box_stagger + 0.1
        css_parts.append(
            f"g[id='{gid}'] {{ animation: tbl-pop 0.45s cubic-bezier(.34,1.56,.64,1) {d:.3f}s both; transform-box: fill-box; transform-origin: center; }}"
        )

    style_block = "<style type=\"text/css\"><![CDATA[\n" + "\n".join(css_parts) + "\n]]></style>"

    svg = re.sub(r"(<svg[^>]*>)", r"\1\n" + style_block, svg, count=1)
    svg_path.write_text(svg)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=here / "results.csv")
    p.add_argument("--out-dir", type=Path, default=here.parent.parent / "asset")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir
    render(args.csv, out / "llava_onevision2_performance_light_anim.svg", LIGHT, animate=True)
    render(args.csv, out / "llava_onevision2_performance_dark_anim.svg", DARK, animate=True)


if __name__ == "__main__":
    main()
