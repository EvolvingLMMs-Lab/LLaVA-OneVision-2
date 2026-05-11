"""Render unified HTML block for one benchmark, given chart_data + examples list.
Output structure (DO NOT change without coordinating across all 39 benchmarks):

  <div class="bench-stat-badges">...badges...</div>
  <div class="bench-charts" data-bench-charts="{slug}">
    <div class="bench-chart">
      <div class="bench-chart-title">[Resolution Distribution]</div>
      <div class="bench-chart-svg" data-chart-type="resolution"></div>
    </div>
    <div class="bench-chart">
      <div class="bench-chart-title">[Duration Distribution]</div>
      <div class="bench-chart-svg" data-chart-type="duration"></div>
    </div>
    <script type="application/json" class="bench-chart-data">...</script>
  </div>
  <div class="bench-examples" data-bench-examples="{slug}">
    <div class="bench-examples-title">[Example Questions]</div>
    {example cards...}
  </div>

Each example dict:
  {tag: str (lowercase, short), question: str, options: list[str] | None, answer: str}
"""
import html, json

def esc(s):
    if s is None:
        return ''
    return html.escape(str(s), quote=True)

def render_badges(num_videos, num_questions, type_):
    badges = []
    if type_ == 'video' and num_videos:
        badges.append(f'<span class="bench-stat-badge"><span class="i18n" data-lang="en">{num_videos:,} videos</span><span class="i18n" data-lang="zh">{num_videos:,} 个视频</span></span>')
    elif type_ == 'image' and num_videos:
        badges.append(f'<span class="bench-stat-badge"><span class="i18n" data-lang="en">{num_videos:,} images</span><span class="i18n" data-lang="zh">{num_videos:,} 张图片</span></span>')
    if num_questions:
        badges.append(f'<span class="bench-stat-badge"><span class="i18n" data-lang="en">{num_questions:,} questions</span><span class="i18n" data-lang="zh">{num_questions:,} 个问题</span></span>')
    return '<div class="bench-stat-badges">' + ''.join(badges) + '</div>'

def render_charts(slug, chart_data, type_):
    has_res = bool(chart_data.get('resolution'))
    has_dur = bool(chart_data.get('duration')) and chart_data['duration'] and chart_data['duration'].get('bins')

    if not has_res and not has_dur:
        return ''

    chart_blocks = []
    if has_res:
        chart_blocks.append(
            '<div class="bench-chart">'
            '<div class="bench-chart-title">'
            '<span class="i18n" data-lang="en">Resolution Distribution</span>'
            '<span class="i18n" data-lang="zh">分辨率分布</span>'
            '</div>'
            '<div class="bench-chart-svg" data-chart-type="resolution"></div>'
            '</div>'
        )
    if has_dur:
        unit = chart_data['duration']['unit']
        unit_zh = '秒' if unit == 's' else '分钟'
        chart_blocks.append(
            '<div class="bench-chart">'
            '<div class="bench-chart-title">'
            f'<span class="i18n" data-lang="en">Duration Distribution ({unit})</span>'
            f'<span class="i18n" data-lang="zh">时长分布 ({unit_zh})</span>'
            '</div>'
            '<div class="bench-chart-svg" data-chart-type="duration"></div>'
            '</div>'
        )

    payload = {}
    if has_res:
        payload['resolution'] = chart_data['resolution']
    if has_dur:
        payload['duration'] = chart_data['duration']
    json_str = json.dumps(payload, separators=(',', ':'))

    return (
        f'<div class="bench-charts" data-bench-charts="{esc(slug)}">'
        + ''.join(chart_blocks)
        + f'<script type="application/json" class="bench-chart-data">{json_str}</script>'
        '</div>'
    )

def render_examples(slug, examples):
    if not examples:
        return ''
    cards = []
    for ex in examples:
        tag = esc(ex.get('tag', ''))
        question_html = esc(ex['question']).replace('\n', '<br>')
        opts = ex.get('options') or []
        opts_html = ''
        if opts:
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            opt_lines = []
            for i, opt in enumerate(opts):
                letter = letters[i] if i < len(letters) else str(i + 1)
                opt_lines.append(f'<span class="bench-opt">{letter}.</span> {esc(opt)}')
            opts_html = '<br>' + '<br>'.join(opt_lines)
        answer_html = esc(ex['answer']).replace('\n', ' ')
        tag_html = f'<span class="bench-example-tag">{tag}</span>' if tag else ''
        cards.append(
            '<div class="bench-example">'
            f'{tag_html}'
            f'<div class="bench-example-q">{question_html}{opts_html}</div>'
            '<div class="bench-example-a">'
            '<span class="bench-example-a-label">'
            '<span class="i18n" data-lang="en">Answer</span>'
            '<span class="i18n" data-lang="zh">答案</span>'
            '</span> '
            f'{answer_html}'
            '</div>'
            '</div>'
        )
    return (
        f'<div class="bench-examples" data-bench-examples="{esc(slug)}">'
        '<div class="bench-examples-title">'
        '<span class="i18n" data-lang="en">Example Questions</span>'
        '<span class="i18n" data-lang="zh">问题示例</span>'
        '</div>'
        + ''.join(cards)
        + '</div>'
    )

def render_full_block(slug, num_videos, num_questions, type_, chart_data, examples):
    parts = [
        render_badges(num_videos, num_questions, type_),
        render_charts(slug, chart_data, type_),
        render_examples(slug, examples),
    ]
    return '\n'.join(p for p in parts if p)
