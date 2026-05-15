#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    fg: str
    muted: str
    grid: str
    codec: str
    frame: str
    advantage: str


LIGHT = Theme(
    name="light",
    bg="#ffffff",
    fg="#0f172a",
    muted="#64748b",
    grid="#e2e8f0",
    codec="#2563eb",
    frame="#94a3b8",
    advantage="#d4e8d0",
)

DARK = Theme(
    name="dark",
    bg="#0f172a",
    fg="#e5e7eb",
    muted="#cbd5e1",
    grid="#334155",
    codec="#93c5fd",
    frame="#94a3b8",
    advantage="#25482f",
)


def fmt(v: float) -> str:
    return f"{v:.1f}"


def points_attr(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def x_scale(frame: float, frames: list[float], width: float) -> float:
    lo = math.log2(frames[0])
    hi = math.log2(frames[-1])
    return ((math.log2(frame) - lo) / (hi - lo)) * width


def y_scale(value: float, y_min: float, y_max: float, height: float) -> float:
    return height - ((value - y_min) / (y_max - y_min)) * height


def render_panel(ds: dict, idx: int, tx: float, ty: float) -> str:
    x0, y0 = 48.0, 114.0
    width, height = 190.0, 180.0
    frames = [float(v) for v in ds["frames"]]
    frame_vals = [float(v) for v in ds["frame"]]
    codec_vals = [float(v) for v in ds["codec"]]
    y_min = float(ds["yMin"])
    y_max = float(ds.get("readmeYMax", ds["yMax"]))
    y_step = float(ds["yStep"])

    frame_pts = [(x0 + x_scale(f, frames, width), y0 + y_scale(v, y_min, y_max, height)) for f, v in zip(frames, frame_vals)]
    codec_pts = [(x0 + x_scale(f, frames, width), y0 + y_scale(v, y_min, y_max, height)) for f, v in zip(frames, codec_vals)]
    area_pts = frame_pts + list(reversed(codec_pts))
    deltas = [c - f for c, f in zip(codec_vals, frame_vals)]
    peak_delta = max(deltas)
    title = html.escape(ds.get("readmeName") or ds["name"])
    sub = f"peak {peak_delta:+.1f}  ·  {int(frames[0])}f Δ {deltas[0]:+.1f}  ·  {int(frames[-1])}f Δ {deltas[-1]:+.1f}"

    chunks = [f'<g id="cvf-panel-{idx}" transform="translate({tx:.2f},{ty:.2f})" style="animation-delay:{idx * 0.08:.2f}s">']
    chunks.append(f'<text x="48" y="99" class="panel-title">{title}</text>')
    chunks.append(f'<text x="48" y="111" class="subhead">{html.escape(sub)}</text>')

    tick_count = int(round((y_max - y_min) / y_step))
    if tick_count > 4:
        tick_values = [y_min, (y_min + y_max) / 2, y_max]
    else:
        tick_values = [y_min + y_step * i for i in range(tick_count + 1)]
    for tv in tick_values:
        y = y0 + y_scale(tv, y_min, y_max, height)
        label = str(int(tv)) if tv.is_integer() else fmt(tv)
        chunks.append(f'<line x1="48" y1="{y:.1f}" x2="238" y2="{y:.1f}" class="grid" />')
        chunks.append(f'<text x="42" y="{y + 3:.1f}" text-anchor="end" class="tick">{label}</text>')

    readme_ticks = ds.get("readmeXTicks") or [frames[0], frames[-1]]
    for tick in readme_ticks:
        x = x0 + x_scale(float(tick), frames, width)
        chunks.append(f'<text x="{x:.1f}" y="316.0" text-anchor="middle" class="tick">{int(tick)}</text>')
        chunks.append(f'<line class="grid" x1="{x:.1f}" y1="114" x2="{x:.1f}" y2="294" />')

    chunks.append(f'<polygon points="{points_attr(area_pts)}" opacity="0.4" class="advantage" />')
    chunks.append('<line x1="48" y1="114" x2="48" y2="294" class="axis" />')
    chunks.append('<line x1="48" y1="294" x2="238" y2="294" class="axis" />')
    chunks.append(f'<polyline points="{points_attr(frame_pts)}" fill="none" stroke-width="1.3" stroke-dasharray="3 2" class="frame-line" />')
    chunks.append(f'<polyline class="codec-line" points="{points_attr(codec_pts)}" fill="none" stroke-width="1.8" />')

    codec_end_y = codec_pts[-1][1]
    frame_end_y = frame_pts[-1][1]
    chunks.append(f'<text x="244" y="{codec_end_y + 4:.1f}" class="score-codec">{fmt(codec_vals[-1])}</text>')
    chunks.append(f'<text x="244" y="{frame_end_y + 4:.1f}" class="score-frame">{fmt(frame_vals[-1])}</text>')
    chunks.append('</g>')
    return "\n".join(chunks)


def render_summary(datasets: list[dict], tx: float, ty: float) -> str:
    peak_deltas = []
    codec_max = []
    frame_max = []
    for ds in datasets:
        frame_vals = [float(v) for v in ds["frame"]]
        codec_vals = [float(v) for v in ds["codec"]]
        peak_deltas.append(max(c - f for c, f in zip(codec_vals, frame_vals)))
        codec_max.append(codec_vals[-1])
        frame_max.append(frame_vals[-1])
    avg_peak = sum(peak_deltas) / len(peak_deltas)
    avg_codec = sum(codec_max) / len(codec_max)
    avg_frame = sum(frame_max) / len(frame_max)
    return f'''<g id="cvf-panel-7" transform="translate({tx:.2f},{ty:.2f})" style="animation-delay:0.56s">
<text x="48" y="99" class="panel-title">Overall Averages</text>
<text x="48" y="140" class="subhead summary-text">Average Peak Δ</text>
<text x="210" y="140" class="score-codec summary-text">{avg_peak:+.1f}</text>
<text x="48" y="170" class="subhead summary-text">Codec @ Max (Avg)</text>
<text x="210" y="170" class="score-codec summary-text">{avg_codec:.1f}</text>
<text x="48" y="200" class="subhead summary-text">Frame @ Max (Avg)</text>
<text x="210" y="200" class="score-frame">{avg_frame:.1f}</text>
<text x="48" y="250" class="subhead muted-text">Codec sampling consistently</text>
<text x="48" y="270" class="subhead muted-text">extends temporal coverage and</text>
<text x="48" y="290" class="subhead muted-text">peak performance overall.</text>
</g>'''


def render_svg(datasets: list[dict], theme: Theme) -> str:
    style = f'''<style><![CDATA[
@keyframes cvf-fade{{from{{opacity:0}}to{{opacity:1}}}}
@keyframes cvf-draw{{from{{stroke-dashoffset:900}}to{{stroke-dashoffset:0}}}}
g[id^="cvf-panel-"]{{animation:cvf-fade .4s ease-out both}}
.codec-line{{stroke:{theme.codec};stroke-dasharray:900;animation:cvf-draw 1.2s ease-out .35s both}}
.frame-line{{stroke:{theme.frame}}}
.caption{{font:600 15px "DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times", serif;fill:{theme.fg}}}
.panel-title{{font:600 14px "DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times", serif;fill:{theme.fg}}}
.subhead{{font:12px "DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times", serif;fill:{theme.muted}}}
.tick{{font:12px "DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times", serif;fill:{theme.muted}}}
.score-codec{{font:bold 13px "DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times", serif;fill:{theme.codec}}}
.score-frame{{font:bold 13px "DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times", serif;fill:{theme.frame}}}
.axis{{stroke:{theme.fg};stroke-width:0.6}}
.grid{{stroke:{theme.grid};stroke-width:0.8}}
.advantage{{fill:{theme.advantage}}}
.summary-text{{fill:{theme.fg}}}
.muted-text{{fill:{theme.muted}}}
]]></style>'''
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="580" viewBox="0 0 1100 580" role="img" aria-labelledby="title desc">',
        '<title id="title">Codec vs Frame Sampling</title>',
        '<desc id="desc">Table-style small multiples comparing codec-aligned sampling against uniform frame sampling across seven video benchmarks.</desc>',
        '<defs>', style, '</defs>',
        f'<rect width="1100" height="580" fill="{theme.bg}" />',
        f'<line x1="36" y1="323" x2="1036" y2="323" stroke-width="0.6" stroke="{theme.grid}" />',
        f'<line x1="266" y1="75" x2="266" y2="550" stroke-width="0.6" stroke="{theme.grid}" />',
        f'<line x1="536" y1="75" x2="536" y2="550" stroke-width="0.6" stroke="{theme.grid}" />',
        f'<line x1="806" y1="75" x2="806" y2="550" stroke-width="0.6" stroke="{theme.grid}" />',
        '<text class="caption" x="30" y="30">Codec-aligned sampling vs. uniform frame sampling.</text>',
        '<g transform="translate(730, 26)">',
        f'<line x1="0" y1="0" x2="30" y2="0" stroke-width="1.8" stroke="{theme.codec}" />',
        '<text class="tick" x="36" y="4">codec-aligned</text>',
        f'<line x1="125" y1="0" x2="155" y2="0" stroke-width="1.3" stroke-dasharray="3 2" stroke="{theme.frame}" />',
        '<text class="tick" x="161" y="4">uniform frame</text>',
        f'<rect x="250" y="-6" width="16" height="12" opacity="0.6" fill="{theme.advantage}" />',
        '<text class="tick" x="272" y="4">codec advantage</text>',
        '</g>',
    ]
    positions = [(-12, -24), (258, -24), (528, -24), (798, -24), (-12, 256), (258, 256), (528, 256)]
    for idx, (ds, pos) in enumerate(zip(datasets, positions)):
        chunks.append(render_panel(ds, idx, pos[0], pos[1]))
    chunks.append(render_summary(datasets, 798, 256))
    chunks.append('</svg>')
    return "\n".join(chunks) + "\n"


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=repo_root / "docs" / "page" / "assets" / "codec-vs-frame-data.json")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "asset")
    args = parser.parse_args()

    payload = json.loads(args.data.read_text())
    datasets = payload["datasets"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in (LIGHT, DARK):
        out = args.out_dir / f"method_codec_vs_frame_{theme.name}.svg"
        out.write_text(render_svg(datasets, theme))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
