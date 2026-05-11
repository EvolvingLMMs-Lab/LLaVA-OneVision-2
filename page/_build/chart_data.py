"""Compute chart JSON data (resolution + duration) from a CSV file.
Unified schema:
  resolution: list[{w:int, h:int, count:int}] sorted by count desc
  duration: {unit: 's', bins: list[{lo:int, hi:int, count:int}]} OR None
"""
import csv, math
from collections import Counter

DURATION_BIN_PRESETS = {
    'short':  [0, 5, 10, 15, 20, 25, 30, 35, 40],
    'medium': [0, 30, 60, 90, 120, 180, 240, 300, 600],
    'long':   [0, 60, 300, 600, 900, 1800, 3600, 7200],
    'extra':  [0, 300, 600, 1200, 1800, 2400, 3000, 3600, 7200],
}

def pick_duration_bins(durations):
    if not durations:
        return None
    mn, mx = min(durations), max(durations)
    span = mx - mn
    if mx <= 40:
        return [0, 5, 10, 15, 20, 25, 30, 35, 40]
    if mx <= 600:
        return [0, 30, 60, 90, 120, 180, 240, 300, 600]
    if mx <= 3600:
        if mn >= 60:
            step = max(60, int((span / 8) // 60) * 60)
            edges = [int(mn // step) * step]
            while edges[-1] < mx:
                edges.append(edges[-1] + step)
            return edges
        return [0, 60, 300, 600, 900, 1800, 3600]
    step = max(600, int((span / 8) // 600) * 600)
    edges = [int(mn // step) * step]
    while edges[-1] < mx:
        edges.append(edges[-1] + step)
    return edges

def compute_chart_data(csv_path, type_):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {'resolution': [], 'duration': None}

    has_video = 'video_path' in rows[0] and rows[0].get('video_path', '').strip() not in ('', '0')
    has_duration = 'duration_sec' in rows[0]

    if has_video:
        seen = {}
        for r in rows:
            p = r['video_path']
            if p not in seen:
                try:
                    w, h = int(r['width']), int(r['height'])
                except (ValueError, KeyError):
                    continue
                d = float(r['duration_sec']) if has_duration and r.get('duration_sec') else None
                seen[p] = (w, h, d)
        items = list(seen.values())
    else:
        items = []
        for r in rows:
            try:
                w, h = int(r['width']), int(r['height'])
            except (ValueError, KeyError):
                continue
            items.append((w, h, None))

    res_counter = Counter((w, h) for w, h, _ in items if w > 0 and h > 0)
    resolution = [{'w': w, 'h': h, 'count': c} for (w, h), c in sorted(res_counter.items(), key=lambda x: -x[1])]

    duration = None
    if has_duration:
        durs = [d for _, _, d in items if d is not None and d > 0]
        if durs:
            edges = pick_duration_bins(durs)
            counts = [0] * (len(edges) - 1)
            for d in durs:
                for i in range(len(edges) - 1):
                    if edges[i] <= d < edges[i + 1]:
                        counts[i] += 1
                        break
                else:
                    if d >= edges[-1]:
                        counts[-1] += 1
            unit = 's'
            scale = 1
            if edges[-1] > 600:
                unit = 'min'
                scale = 60
            bins = [{'lo': edges[i] // scale, 'hi': edges[i + 1] // scale, 'count': counts[i]}
                    for i in range(len(counts))]
            duration = {'unit': unit, 'bins': bins}

    return {'resolution': resolution, 'duration': duration}
