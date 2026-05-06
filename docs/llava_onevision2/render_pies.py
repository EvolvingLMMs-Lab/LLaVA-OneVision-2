#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Wedge


GID_DONUT_PREFIX = "donut-"
GID_WEDGE_PREFIX = "wedge-"
GID_LEGEND_PREFIX = "legend-"


@dataclass
class Theme:
    name: str
    page_bg: str
    fg: str
    dim_fg: str
    title_fg: str
    palette: list[str] = field(default_factory=list)


PALETTE_LIGHT = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac",
    "#86b4d8", "#fbb168", "#88c97e", "#ee8a8b", "#a3d3cf",
    "#f6dd75",
]

PALETTE_DARK = [
    "#79a6d2", "#ffa454", "#7cc274", "#ff8788", "#9bd5d0",
    "#ffe089", "#d8a5cd", "#ffb9c0", "#c89c80", "#d4cdc8",
    "#a4c8e6", "#ffc890", "#aed99e", "#ffadae", "#bee2dd",
    "#fff09a",
]


LIGHT = Theme(
    name="light",
    page_bg="#ffffff",
    fg="#1a1a1a",
    dim_fg="#6a737d",
    title_fg="#1a1a1a",
    palette=PALETTE_LIGHT,
)


DARK = Theme(
    name="dark",
    page_bg="#0d1117",
    fg="#f0f6fc",
    dim_fg="#8b949e",
    title_fg="#f0f6fc",
    palette=PALETTE_DARK,
)


@dataclass
class DonutSpec:
    title: str
    subtitle: str
    items: list[tuple[str, float]]


def sanitize_svg_for_github(svg_path: Path) -> None:
    svg = svg_path.read_text()

    svg = re.sub(r"<!DOCTYPE[^>]*?>\s*", "", svg, count=1, flags=re.DOTALL)
    svg = re.sub(r"<metadata>.*?</metadata>\s*", "", svg, count=1, flags=re.DOTALL)

    clippath_defs = list(
        re.finditer(r"<defs>\s*<clipPath\b.*?</defs>\s*", svg, flags=re.DOTALL)
    )
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


def inject_animation_css(svg_path: Path, n_donuts: int, wedges_per_donut: list[int]) -> None:
    svg = svg_path.read_text()

    css_parts = [
        "@keyframes donut-fade { from { opacity: 0; transform: scale(0.85) rotate(-12deg); } to { opacity: 1; transform: scale(1) rotate(0); } }",
        "@keyframes wedge-pop { from { opacity: 0; } to { opacity: 1; } }",
        "@keyframes hi-bright { 0%, 100% { opacity: 0.32; } 18%, 82% { opacity: 1; } }",
        "@keyframes hi-bright-pct { 0%, 100% { opacity: 0.32; } 18%, 82% { opacity: 0.85; } }",
        "@keyframes hi-bright-wedge { 0%, 100% { opacity: 0.42; } 18%, 82% { opacity: 1; } }",
    ]

    for di in range(n_donuts):
        delay = 0.15 + di * 0.18
        css_parts.append(
            f"g[id='{GID_DONUT_PREFIX}{di}'] {{ animation: donut-fade 0.7s cubic-bezier(.34,1.2,.64,1) {delay:.3f}s both; transform-box: fill-box; transform-origin: center; }}"
        )
        nw = wedges_per_donut[di]
        wedge_base = delay + 0.4
        slot = 1.4
        cycle = nw * slot
        loop_start = wedge_base + nw * 0.04 + 0.6
        for wi in range(nw):
            wd = loop_start + wi * slot
            css_parts.append(
                f"g[id='{GID_WEDGE_PREFIX}{di}-{wi}'] {{ animation: wedge-pop 0.35s ease-out {wedge_base + wi * 0.04:.3f}s both, hi-bright-wedge {cycle:.2f}s ease-in-out {wd:.3f}s infinite; }}"
            )
            for suffix, kf in [("sw", "hi-bright"), ("lb", "hi-bright"), ("pc", "hi-bright-pct")]:
                css_parts.append(
                    f"#{GID_LEGEND_PREFIX}{di}-{wi}-{suffix} {{ animation: {kf} {cycle:.2f}s ease-in-out {wd:.3f}s infinite; }}"
                )

    style_block = "<style type=\"text/css\"><![CDATA[\n" + "\n".join(css_parts) + "\n]]></style>"
    svg = re.sub(r"(<svg[^>]*>)", r"\1\n" + style_block, svg, count=1)
    svg_path.write_text(svg)


def fmt_total(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:.0f}"


def draw_donut(ax, spec: DonutSpec, theme: Theme, donut_idx: int, animate: bool) -> int:
    items = sorted(spec.items, key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = np.array([v for _, v in items], dtype=float)
    total = values.sum()
    fractions = values / total
    n_items = len(items)

    if n_items > 10:
        legend_fontsize = 7.5
        line_h = 0.17
    elif n_items > 6:
        legend_fontsize = 9
        line_h = 0.21
    else:
        legend_fontsize = 10
        line_h = 0.27

    legend_block_h = max(2.5, n_items * line_h + 0.3)
    y_top = legend_block_h / 2
    y_bot = -legend_block_h / 2

    ax.set_xlim(-1.20, 3.20)
    ax.set_ylim(y_bot, y_top)
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy = -0.08, 0.0
    r_outer = 1.10
    r_inner = 0.69

    start_angle = 90.0
    for wi, (frac, color) in enumerate(zip(fractions, theme.palette)):
        sweep = frac * 360.0
        end_angle = start_angle - sweep
        wedge = Wedge(
            (cx, cy),
            r_outer,
            end_angle,
            start_angle,
            width=r_outer - r_inner,
            facecolor=color,
            edgecolor=theme.page_bg,
            linewidth=2.0,
            antialiased=True,
        )
        if animate:
            wedge.set_gid(f"{GID_WEDGE_PREFIX}{donut_idx}-{wi}")
        ax.add_patch(wedge)
        start_angle = end_angle

    ax.text(
        cx, cy + 0.13,
        spec.title,
        ha="center", va="center",
        fontsize=11, fontweight="bold", color=theme.title_fg,
    )
    ax.text(
        cx, cy - 0.18,
        spec.subtitle,
        ha="center", va="center",
        fontsize=9, color=theme.dim_fg,
    )

    legend_x = 1.18
    legend_y_top = (n_items - 1) * line_h / 2
    swatch_w = 0.16
    swatch_h = 0.10
    label_x = legend_x + swatch_w + 0.10
    pct_x = 3.16
    label_max_chars = 14
    for i, ((label, _), color) in enumerate(zip(items, theme.palette)):
        y = legend_y_top - i * line_h
        swatch = plt.Rectangle(
            (legend_x, y - swatch_h / 2),
            swatch_w, swatch_h,
            facecolor=color, edgecolor="none",
        )
        if animate:
            swatch.set_gid(f"{GID_LEGEND_PREFIX}{donut_idx}-{i}-sw")
        ax.add_patch(swatch)
        pct = fractions[i] * 100.0
        if len(label) > label_max_chars:
            label_fs = legend_fontsize * label_max_chars / len(label)
        else:
            label_fs = legend_fontsize
        t_label = ax.text(
            label_x, y,
            f"{label}",
            ha="left", va="center",
            fontsize=label_fs, color=theme.fg,
        )
        t_pct = ax.text(
            pct_x, y,
            f"{pct:>4.1f}%",
            ha="right", va="center",
            fontsize=legend_fontsize, color=theme.dim_fg,
        )
        if animate:
            t_label.set_gid(f"{GID_LEGEND_PREFIX}{donut_idx}-{i}-lb")
            t_pct.set_gid(f"{GID_LEGEND_PREFIX}{donut_idx}-{i}-pc")

    return len(items)


def render(specs: list[DonutSpec], out_path: Path, theme: Theme, animate: bool = False) -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times"]
    rcParams["svg.fonttype"] = "none"

    n = len(specs)
    cols = 2
    rows = (n + cols - 1) // cols

    fig_w_per = 6.4
    fig_h_per = 3.4
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_w_per * cols, fig_h_per * rows),
        gridspec_kw={"wspace": 0.02, "hspace": 0.10},
    )
    fig.patch.set_facecolor(theme.page_bg)

    if rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    wedges_per_donut: list[int] = []
    for di, ax in enumerate(axes_flat):
        ax.set_facecolor(theme.page_bg)
        if di < n:
            spec = specs[di]
            n_wedges = draw_donut(ax, spec, theme, di, animate)
            wedges_per_donut.append(n_wedges)
            if animate:
                for child in ax.get_children():
                    pass
                ax.set_gid(f"axes-{di}")
        else:
            ax.set_xlim(-1.6, 2.8)
            ax.set_ylim(-1.55, 1.55)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.text(
                0.6, 0.0, "Coming soon",
                ha="center", va="center",
                fontsize=14, color=theme.dim_fg, style="italic",
            )

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor=theme.page_bg, format="svg")
    plt.close(fig)

    sanitize_svg_for_github(out_path)

    if animate:
        wrap_donuts_in_groups(out_path, n)
        inject_animation_css(out_path, n, wedges_per_donut)

    print(f"Saved: {out_path}")


def wrap_donuts_in_groups(svg_path: Path, n_donuts: int) -> None:
    svg = svg_path.read_text()
    for di in range(n_donuts):
        pattern = re.compile(
            r"(<g id=\"axes-" + str(di) + r"\">)",
        )
        svg = pattern.sub(
            r'<g id="' + GID_DONUT_PREFIX + str(di) + r'">\1',
            svg,
            count=1,
        )
        end_pattern = re.compile(
            r'(<g id="axes-' + str(di) + r'">.*?)(</g>)(\s*<g id="(?:axes-|axes_|patch_|figure_))',
            re.DOTALL,
        )
        svg = end_pattern.sub(
            r"\1\2</g>\3",
            svg,
            count=1,
        )
    last_pattern = re.compile(
        r'(<g id="axes-' + str(n_donuts - 1) + r'">.*?</g>)(\s*</g>\s*</svg>)',
        re.DOTALL,
    )
    svg = last_pattern.sub(r"\1</g>\2", svg, count=1)
    svg_path.write_text(svg)


SPECS = [
    DonutSpec(
        title="Mid-Training-85M",
        subtitle="85M samples",
        items=[
            ("Obelics", 45.6),
            ("COYO-700M", 18.3),
            ("Zero250M", 11.2),
            ("DataComp-1B", 8.8),
            ("LAION-CN", 7.8),
            ("MINT", 5.0),
            ("ImageNet-21K", 1.9),
            ("SA-1B", 1.4),
        ],
    ),
    DonutSpec(
        title="Instruct",
        subtitle="22M samples",
        items=[
            ("Text", 28.0),
            ("General VQA", 24.7),
            ("Domain-specific", 16.2),
            ("OCR", 11.1),
            ("Chart & Table", 7.3),
            ("Caption", 4.8),
            ("Science", 4.1),
            ("Code/Mathematics", 3.3),
            ("Grounding & Counting", 0.6),
        ],
    ),
    DonutSpec(
        title="VideoCaption",
        subtitle="7.96M clips · 104.1B tokens",
        items=[
            ("30s", 27_700),
            ("30–60s", 34_100),
            ("60–180s", 13_000),
            ("10–15 min", 30_300),
        ],
    ),
    DonutSpec(
        title="Spatial",
        subtitle="2.78M samples",
        items=[
            ("OSD Reasoning", 785_559),
            ("CA-1M Visual Choice", 539_976),
            ("CA-1M Templates", 505_579),
            ("Crosspoint", 377_765),
            ("RoboRefer (Sim)", 245_926),
            ("Pointing & VG", 167_938),
            ("RefCOCO Family", 149_534),
            ("ERQA", 11_172),
        ],
    ),
]


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=here.parent.parent / "asset")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir
    render(SPECS, out / "llava_onevision2_data_distribution_light_anim.svg", LIGHT, animate=True)
    render(SPECS, out / "llava_onevision2_data_distribution_dark_anim.svg", DARK, animate=True)


if __name__ == "__main__":
    main()
