import os, sys, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifest import MANIFEST, CSV_DIR, find_arrow_files
from chart_data import compute_chart_data
from extractors import extract_examples
from render import render_charts, render_examples

INDEX_PATH = '/ov2/xiangan/sparrow/index.html'
ALREADY_DONE = {'tempcompass'}

PNG_STEM_OVERRIDE = {
    'robospatial': 'robospatial_home',
}


def csv_stem(name):
    return name[:-4] if name.endswith('.csv') else name


def png_stem_for(slug, csv_name):
    return PNG_STEM_OVERRIDE.get(slug, csv_stem(csv_name))


def find_bench_figs_lines(html, stem):
    pat = re.compile(
        r'^<div class="bench-figs"><div class="bench-fig"><img src="assets/figures/'
        + re.escape(stem)
        + r'_(?:video_)?(?:resolution|duration)\.png".*?</div></div>$',
        re.MULTILINE,
    )
    return list(pat.finditer(html))


def build_replacement(slug, type_, chart_data, examples):
    parts = [render_charts(slug, chart_data, type_), render_examples(slug, examples)]
    return '\n'.join(p for p in parts if p)


def process(dry_run=True, only=None):
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        html = f.read()

    report = []
    new_html = html

    for slug, (csv_name, type_, patterns) in MANIFEST.items():
        if only and slug not in only:
            continue
        if slug in ALREADY_DONE:
            report.append((slug, 'SKIP-already-done', 0))
            continue

        stem = png_stem_for(slug, csv_name)
        matches = find_bench_figs_lines(new_html, stem)
        if not matches:
            report.append((slug, 'NO-MATCH (already replaced?)', 0))
            continue

        csv_path = os.path.join(CSV_DIR, csv_name)
        if not os.path.exists(csv_path):
            report.append((slug, 'NO-CSV', len(matches)))
            continue

        try:
            chart_data = compute_chart_data(csv_path, type_)
        except Exception as e:
            report.append((slug, f'CHART-ERR:{e}', len(matches)))
            continue

        arrow_paths = []
        for p in patterns:
            arrow_paths.extend(find_arrow_files(p))
        try:
            examples = extract_examples(slug, arrow_paths)
        except Exception as e:
            examples = []
            report.append((slug, f'EX-ERR:{e}', len(matches)))

        replacement = build_replacement(slug, type_, chart_data, examples)
        if not replacement:
            report.append((slug, 'EMPTY-REPLACEMENT', len(matches)))
            continue

        m = matches[0]
        new_html = new_html[:m.start()] + replacement + new_html[m.end():]
        report.append((slug, f'OK ex={len(examples)}', len(matches)))

    if not dry_run:
        with open(INDEX_PATH, 'w', encoding='utf-8') as f:
            f.write(new_html)

    return report, new_html


if __name__ == '__main__':
    args = sys.argv[1:]
    dry = '--write' not in args
    only = None
    if '--only' in args:
        idx = args.index('--only')
        only = set(args[idx + 1].split(','))
    report, _ = process(dry_run=dry, only=only)
    for slug, status, n in report:
        print(f'  {slug:25s} matches={n}  {status}')
    print()
    print('DRY RUN' if dry else 'WRITTEN')
