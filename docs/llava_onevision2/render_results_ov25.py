#!/usr/bin/env python
"""Render the LLaVA-OneVision-2.5 benchmark figure for the README.

Same look as render_results.py, but the 2.5 results come as three tables with
different columns (video, streaming video, image), so they are stacked as three
blocks inside one figure. Input is the long-format docs/page/assets/results_ov25.csv
(table, category, benchmark, model, score, best).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

from render_results import (
    DARK,
    GID_BEST_PREFIX,
    GID_HEADER,
    GID_OURS_COL,
    GID_ROW_PREFIX,
    LIGHT,
    Theme,
    inject_animation_css,
    round_box,
    sanitize_svg_for_github,
)


OURS_PREFIX = "OV2.5"
TITLE = "Benchmark comparison of LLaVA-OneVision-2.5-4B."

BENCH_W = 6.4
MODEL_W = 3.6
INNER_W = BENCH_W + 3 * MODEL_W
HALF_GAP = 0.8
HALF_BENCH_W = 4.4
HALF_MODEL_W = ((INNER_W - HALF_GAP) / 2 - HALF_BENCH_W) / 2

ROW_H = 0.65
HEADER_H = 1.05
BLOCK_TITLE_H = 0.70
BLOCK_GAP = 0.55
PAD_X = 0.65
PAD_TOP = 1.75
PAD_BOT = 0.90


def load(path: Path):
    """-> {table: {"models": [...], "groups": [(category, [(benchmark, {model: (score, best)})])]}}"""
    tables: dict = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = tables.setdefault(r["table"], {"models": [], "groups": []})
            if r["model"] not in t["models"]:
                t["models"].append(r["model"])
            if not t["groups"] or t["groups"][-1][0] != r["category"]:
                t["groups"].append((r["category"], []))
            rows = t["groups"][-1][1]
            if not rows or rows[-1][0] != r["benchmark"]:
                rows.append((r["benchmark"], {}))
            rows[-1][1][r["model"]] = (r["score"], r["best"] == "1")
    return tables


def split_header(name: str):
    """'OV2.5-4B (codec, 24 fps)' -> ('OV2.5-4B', '(codec, 24 fps)')."""
    base, sep, rest = name.partition(" (")
    return base, (sep.strip() + rest) if sep else ""


class Canvas:
    def __init__(self, ax, theme: Theme, animate: bool):
        self.ax, self.theme, self.animate = ax, theme, animate
        self.n_rows = 0
        self.best_ids: list[str] = []

    def gid(self, artist, gid):
        if self.animate:
            artist.set_gid(gid)
        return artist

    def text(self, x, y, s, gid=None, **kw):
        a = self.ax.text(x, y, s, va=kw.pop("va", "center"), **kw)
        return self.gid(a, gid) if gid else a

    def rule(self, x0, x1, y, gid=None, **kw):
        a = self.ax.plot([x0, x1], [y, y], solid_capstyle="butt", **kw)[0]
        return self.gid(a, gid) if gid else a

    def header(self, x0, y, first, models, bench_w, model_w):
        """Header band of one sub-table; returns the x extents of the 'ours' columns."""
        th = self.theme
        width = bench_w + model_w * len(models)
        self.rule(x0, x0 + width, y, GID_HEADER, color=th.rule, linewidth=1.0)
        self.rule(x0, x0 + width, y + HEADER_H, GID_HEADER, color=th.dim_fg, linewidth=0.5)
        self.text(x0 + 0.05, y + HEADER_H / 2, first, GID_HEADER, ha="left",
                  fontsize=17, fontweight="bold", color=th.fg)
        ours = []
        for i, m in enumerate(models):
            cx = x0 + bench_w + model_w * (i + 0.5)
            base, note = split_header(m)
            if note:
                self.text(cx, y + HEADER_H * 0.34, base, GID_HEADER, ha="center",
                          fontsize=15, fontweight="bold", color=th.fg)
                self.text(cx, y + HEADER_H * 0.70, note, GID_HEADER, ha="center",
                          fontsize=13, color=th.dim_fg)
            else:
                self.text(cx, y + HEADER_H / 2, base, GID_HEADER, ha="center",
                          fontsize=15, fontweight="bold", color=th.fg)
            if m.startswith(OURS_PREFIX):
                ours.append((cx - model_w / 2, cx + model_w / 2))
        return ours

    def tint(self, spans, y0, y1):
        for a, b in spans:
            p = self.ax.fill_between([a, b], y0, y1, color=self.theme.ours_tint, linewidth=0, zorder=0.5)
            p.set_gid(GID_OURS_COL)

    def row(self, x0, y, zebra, bench, models, scores, bench_w, model_w, row_gid):
        th = self.theme
        if zebra:
            self.ax.fill_between([x0, x0 + bench_w + model_w * len(models)], y, y + ROW_H,
                                 color=th.zebra_bg, linewidth=0, zorder=0.3)
        self.text(x0 + 0.05, y + ROW_H / 2, bench, row_gid, ha="left", fontsize=20, color=th.fg)
        for i, m in enumerate(models):
            score, best = scores[m]
            cx0 = x0 + bench_w + model_w * i
            if best:
                gid = f"{GID_BEST_PREFIX}{len(self.best_ids)}"
                round_box(self.ax, cx0 + 0.10, y + 0.06, model_w - 0.20, ROW_H - 0.12, radius=0.10,
                          facecolor=th.best_bg, edgecolor=th.best_edge,
                          linewidth=0.6 if th.best_edge != "none" else 0,
                          gid=gid if self.animate else None)
                self.best_ids.append(gid)
            weight = "bold" if best or m.startswith(OURS_PREFIX) else "normal"
            self.text(cx0 + model_w / 2, y + ROW_H / 2, score, row_gid, ha="center",
                      fontsize=20, fontweight=weight, color=th.fg if score != "--" else th.dim_fg)

    def next_row_gid(self):
        gid = f"{GID_ROW_PREFIX}{self.n_rows}"
        self.n_rows += 1
        return gid


def block_height(table, name):
    if name == "Image":
        n = max(len(rows) for _, rows in table["groups"])
    else:
        n = sum(len(rows) + 1 for _, rows in table["groups"])
    return BLOCK_TITLE_H + HEADER_H + ROW_H * n


def render(csv_path: Path, out_path: Path, theme: Theme, animate: bool = True) -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times"]
    rcParams["mathtext.fontset"] = "dejavuserif"
    rcParams["svg.fonttype"] = "none"

    tables = load(csv_path)
    fig_w = INNER_W + 2 * PAD_X
    fig_h = PAD_TOP + sum(block_height(t, n) for n, t in tables.items()) + BLOCK_GAP * (len(tables) - 1) + PAD_BOT

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(theme.page_bg)
    ax.set_facecolor(theme.page_bg)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis("off")
    c = Canvas(ax, theme, animate)

    label = ax.text(PAD_X, 0.36, "Table 1: ", ha="left", va="top",
                    fontsize=22, fontweight="bold", color=theme.dim_fg)
    fig.canvas.draw()
    bbox = label.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    ax.text(bbox.x1 + 0.16, 0.36, TITLE, ha="left", va="top",
            fontsize=22, fontweight="bold", color=theme.fg)
    sw_w, sw_h, legend_y = 0.32, 0.30, 1.18
    round_box(ax, PAD_X, legend_y - sw_h / 2 - 0.05, sw_w, sw_h, radius=0.06,
              facecolor=theme.best_bg, edgecolor=theme.best_edge,
              linewidth=0.6 if theme.best_edge != "none" else 0)
    ax.text(PAD_X + sw_w + 0.12, legend_y, "Best", ha="left", va="center", fontsize=17, color=theme.fg)

    y = PAD_TOP
    for name, table in tables.items():
        models = table["models"]
        c.text(PAD_X + 0.05, y + BLOCK_TITLE_H / 2, name.upper(), GID_HEADER, ha="left",
               fontsize=16, fontweight="bold", color=theme.cat_fg)
        y += BLOCK_TITLE_H
        top = y

        if name == "Image":
            # two sub-tables side by side, one per category
            half_w = HALF_BENCH_W + HALF_MODEL_W * len(models)
            n = max(len(rows) for _, rows in table["groups"])
            row_gids = [c.next_row_gid() for _ in range(n)]
            for gi, (cat, rows) in enumerate(table["groups"]):
                x0 = PAD_X + gi * (half_w + HALF_GAP)
                ours = c.header(x0, top, cat, models, HALF_BENCH_W, HALF_MODEL_W)
                c.tint(ours, top, top + HEADER_H + ROW_H * n)
                for ri, (bench, scores) in enumerate(rows):
                    c.row(x0, top + HEADER_H + ROW_H * ri, ri % 2 == 1, bench, models, scores,
                          HALF_BENCH_W, HALF_MODEL_W, row_gids[ri])
                c.rule(x0, x0 + half_w, top + HEADER_H + ROW_H * n, GID_HEADER, color=theme.rule, linewidth=1.0)
            y = top + HEADER_H + ROW_H * n
        else:
            ours = c.header(PAD_X, top, "Benchmark", models, BENCH_W, MODEL_W)
            y = top + HEADER_H
            n = sum(len(rows) + 1 for _, rows in table["groups"])
            c.tint(ours, top, y + ROW_H * n)
            for gi, (cat, rows) in enumerate(table["groups"]):
                gid = c.next_row_gid()
                if gi:
                    c.rule(PAD_X, PAD_X + INNER_W, y, gid, color=theme.rule, linewidth=0.4, linestyle=(0, (2, 2)))
                c.text(PAD_X + 0.05, y + ROW_H / 2, cat, gid, ha="left",
                       fontsize=16, fontweight="bold", color=theme.cat_fg)
                y += ROW_H
                for ri, (bench, scores) in enumerate(rows):
                    c.row(PAD_X, y, ri % 2 == 1, bench, models, scores, BENCH_W, MODEL_W, c.next_row_gid())
                    y += ROW_H
            c.rule(PAD_X, PAD_X + INNER_W, y, GID_HEADER, color=theme.rule, linewidth=1.0)
        y += BLOCK_GAP

    plt.savefig(out_path, bbox_inches="tight", facecolor=theme.page_bg, format="svg")
    plt.close(fig)
    sanitize_svg_for_github(out_path)
    if animate:
        inject_animation_css(out_path, c.n_rows, c.best_ids, [], ours_present=True, pulse_color=theme.pulse)
    print(f"Saved: {out_path}")


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=repo_root / "docs" / "page" / "assets" / "results_ov25.csv")
    p.add_argument("--out-dir", type=Path, default=repo_root / "asset")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    render(args.csv, args.out_dir / "llava_onevision2_5_performance_light_anim.svg", LIGHT)
    render(args.csv, args.out_dir / "llava_onevision2_5_performance_dark_anim.svg", DARK)


if __name__ == "__main__":
    main()
