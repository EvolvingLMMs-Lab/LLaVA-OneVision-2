import pyarrow.ipc as ipc
import pyarrow as pa
import glob, ast, random, re, json
from collections import defaultdict

random.seed(42)

OPT_RE = re.compile(r'\n?[\(\[]?([A-Ha-h])[\)\]\.\:]\s+')
INLINE_OPTS_RE = re.compile(r'(?:^|\n)\s*[\(\[]?([A-Ha-h])[\)\]\.\:]\s+([^\n\(\[]+?)(?=(?:\n\s*[\(\[]?[A-Ha-h][\)\]\.\:])|$)', re.MULTILINE)

def _read_arrow(path):
    with pa.memory_map(path, 'r') as src:
        return ipc.open_stream(src).read_all()

def _load_all(arrow_paths, max_rows=4000):
    all_rows = []
    for p in arrow_paths:
        try:
            tbl = _read_arrow(p)
        except Exception:
            continue
        cols = tbl.column_names
        for i in range(tbl.num_rows):
            row = {}
            for c in cols:
                try:
                    row[c] = tbl[c][i].as_py()
                except Exception:
                    row[c] = None
            all_rows.append(row)
            if len(all_rows) >= max_rows:
                return all_rows
    return all_rows

def _split_inline_options(question):
    matches = list(INLINE_OPTS_RE.finditer(question))
    if len(matches) >= 2:
        first_start = matches[0].start()
        q_text = question[:first_start].strip()
        opts = [m.group(2).strip().rstrip('.;,') for m in matches]
        return q_text, opts
    return question, None

def _parse_listish(s):
    if s is None:
        return None
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if s in ('None', 'null', '', 'nan'):
        return None
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return list(v.values())
    except Exception:
        pass
    return None

def _short_enough(q, a, max_q=260, max_a=120):
    return q and a and len(str(q)) <= max_q and len(str(a)) <= max_a

def _diverse_sample(rows, key_fn, n=4, predicate=lambda r: True, sort_key=None):
    by_dim = defaultdict(list)
    for r in rows:
        if not predicate(r):
            continue
        k = key_fn(r) or '_'
        by_dim[k].append(r)
    if sort_key:
        for k in by_dim:
            by_dim[k].sort(key=sort_key)
    else:
        for k in by_dim:
            random.shuffle(by_dim[k])
    dims = sorted(by_dim.keys())
    random.Random(42).shuffle(dims)
    out = []
    seen_q = set()
    while len(out) < n and dims:
        for d in list(dims):
            if not by_dim[d]:
                dims.remove(d)
                continue
            r = by_dim[d].pop(0)
            q = str(r.get('_q', ''))[:80]
            if q in seen_q:
                continue
            seen_q.add(q)
            out.append(r)
            if len(out) >= n:
                break
    return out


def _generic_qoa(rows, q_field='question', a_field='answer', tag_field=None, options_field=None):
    candidates = []
    for r in rows:
        q = r.get(q_field)
        a = r.get(a_field)
        if not q or a in (None, '', 'None', 'nan', 'hidden'):
            continue
        opts = None
        q_text = str(q).strip()
        if options_field and r.get(options_field):
            opts = _parse_listish(r[options_field])
        if opts is None:
            q_text, opts = _split_inline_options(q_text)
        a_text = str(a).strip()
        if isinstance(a, str) and len(a) <= 3 and opts:
            letters = ['A','B','C','D','E','F','G','H']
            if a in letters:
                idx = letters.index(a)
                if idx < len(opts):
                    a_text = f"{a}. {opts[idx]}"
            elif a.isdigit() and int(a) < len(opts):
                idx = int(a)
                a_text = f"{letters[idx]}. {opts[idx]}"
        elif isinstance(a, str) and a.startswith('(') and a.endswith(')') and opts:
            letter = a.strip('()')
            if letter in 'ABCDEFGH':
                idx = 'ABCDEFGH'.index(letter)
                if idx < len(opts):
                    a_text = f"{letter}. {opts[idx]}"

        if not _short_enough(q_text, a_text):
            continue
        cand = {
            '_q': q_text,
            'tag': str(r.get(tag_field, '')).strip() if tag_field else '',
            'question': q_text,
            'options': opts,
            'answer': a_text,
        }
        candidates.append(cand)
    sel = _diverse_sample(candidates, lambda r: r.get('tag', ''), n=4,
                          sort_key=lambda r: len(r['question']) + len(r['answer']))
    return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]


def extract_examples(slug, arrow_paths):
    if not arrow_paths:
        return []
    rows = _load_all(arrow_paths, max_rows=8000)
    if not rows:
        return []

    if slug == 'tempcompass':
        return _generic_qoa(rows, tag_field='dim')
    if slug == 'lvbench':
        return _generic_qoa(rows, tag_field='type')
    if slug == 'mlvu-dev':
        return _generic_qoa(rows, tag_field='task_type', options_field='candidates')
    if slug == 'videoeval-pro':
        return _generic_qoa(rows, tag_field='qa_type', options_field='options')
    if slug == 'videomme-v2-64':
        return _generic_qoa(rows, tag_field='group_type')
    if slug == 'longvideobench':
        for r in rows:
            opts = [r.get(f'option{i}') for i in range(5)]
            opts = [o for o in opts if o and o != 'N/A']
            r['_opts'] = opts
            cc = r.get('correct_choice')
            if cc is not None and str(cc).lstrip('-').isdigit():
                ci = int(cc)
                r['_a'] = opts[ci] if 0 <= ci < len(opts) else ''
            else:
                r['_a'] = ''
        cands = []
        for r in rows:
            if not r.get('_opts') or not r.get('_a'):
                continue
            q = str(r.get('question', '')).strip()
            if not _short_enough(q, r['_a']):
                continue
            letters = ['A','B','C','D','E']
            try:
                idx = r['_opts'].index(r['_a'])
                a_text = f"{letters[idx]}. {r['_a']}"
            except (ValueError, IndexError):
                a_text = r['_a']
            cands.append({'_q': q, 'tag': str(r.get('question_category', '')),
                          'question': q, 'options': r['_opts'], 'answer': a_text})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]
    if slug == 'nextqa':
        cands = []
        letters = ['A','B','C','D','E']
        for r in rows:
            opts = [r.get(f'a{i}') for i in range(5)]
            opts = [str(o) for o in opts if o]
            try:
                ai = int(r.get('answer'))
                a_text = f"{letters[ai]}. {opts[ai]}"
            except Exception:
                continue
            q = str(r.get('question', '')).strip()
            if not _short_enough(q, a_text):
                continue
            cands.append({'_q': q, 'tag': str(r.get('type', '')),
                          'question': q, 'options': opts, 'answer': a_text})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]
    if slug == 'mmvu-val':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            a = str(r.get('answer', '')).strip()
            ch = r.get('choices')
            opts = None
            if ch:
                cd = _parse_listish(ch) if isinstance(ch, str) else (list(ch.values()) if hasattr(ch, 'values') else None)
                if isinstance(cd, list):
                    cd = [c for c in cd if c]
                    if cd:
                        opts = cd
            if a and a in 'ABCDE' and opts:
                idx = 'ABCDE'.index(a)
                if idx < len(opts):
                    a = f"{a}. {opts[idx]}"
            if not _short_enough(q, a):
                continue
            cands.append({'_q': q, 'tag': str(r.get('question_type', '')),
                          'question': q, 'options': opts, 'answer': a})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]
    if slug in ('t-charades', 't-activitynet', 't-qvhighlights'):
        source_filter = {
            't-charades': 'charades',
            't-activitynet': 'activitynet',
            't-qvhighlights': 'qvhighlights',
        }[slug]
        cands = []
        for r in rows:
            src = str(r.get('source', '')).lower()
            if source_filter not in src:
                continue
            q = str(r.get('query', '')).strip()
            span = r.get('span')
            if not q or not span:
                continue
            spans = _parse_listish(span) if isinstance(span, str) else span
            if not spans:
                continue
            try:
                first = spans[0] if isinstance(spans[0], list) else spans
                a = f"[{first[0]:.1f}s, {first[1]:.1f}s]"
            except Exception:
                continue
            qd = "Find the moment: " + q
            if not _short_enough(qd, a):
                continue
            cands.append({'_q': q, 'tag': '',
                          'question': qd, 'options': None, 'answer': a})
        random.Random(31).shuffle(cands)
        cands.sort(key=lambda r: len(r['question']))
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in cands[:4]]

    if slug == 'vsi-bench':
        return _generic_qoa(rows, q_field='question', a_field='ground_truth',
                            tag_field='question_type', options_field='options')

    if slug == 'cv-bench-2d' or slug == 'cv-bench-3d':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            opts = _parse_listish(r.get('choices'))
            ans = str(r.get('answer', '')).strip().strip('()')
            if not opts or ans not in 'ABCDEFGH':
                continue
            idx = 'ABCDEFGH'.index(ans)
            if idx >= len(opts):
                continue
            a_text = f"{ans}. {opts[idx]}"
            if not _short_enough(q, a_text):
                continue
            cands.append({'_q': q, 'tag': str(r.get('task', '')),
                          'question': q, 'options': opts, 'answer': a_text})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]

    if slug == 'embspatial':
        cands = []
        letters = ['A','B','C','D']
        for r in rows:
            q = str(r.get('question', '')).strip()
            opts = _parse_listish(r.get('answer_options'))
            ans = r.get('answer')
            if not opts or ans is None:
                continue
            try:
                idx = int(ans)
                a_text = f"{letters[idx]}. {opts[idx]}"
            except Exception:
                continue
            if not _short_enough(q, a_text):
                continue
            cands.append({'_q': q, 'tag': str(r.get('relation', '')),
                          'question': q, 'options': opts, 'answer': a_text})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]

    if slug == 'mmsi-bench':
        return _generic_qoa(rows, tag_field='question_type')

    if slug == 'sat-mcq' or slug == 'sat':
        cands = []
        for r in rows:
            q = str(r.get('problem', '')).replace('<image>', '').strip()
            a = str(r.get('answer', '')).replace('<answer>', '').replace('</answer>', '').strip()
            if not _short_enough(q, a):
                continue
            cands.append({'_q': q, 'tag': '', 'question': q, 'options': None, 'answer': a})
        random.Random(7).shuffle(cands)
        cands.sort(key=lambda r: len(r['question']))
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in cands[:4]]

    if slug == 'metavqa':
        return _generic_qoa(rows)

    if slug == 'crosspoint':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            if not q:
                continue
            cands.append({'_q': q, 'tag': str(r.get('type', '')).split()[0].lower() if r.get('type') else '',
                          'question': q, 'options': None,
                          'answer': '[Pixel mask of the grounded region]'})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]

    if slug == 'erqa':
        cands = []
        choices_re = re.compile(r'Choices:\s*(.+?)(?:\s+Please answer|$)', re.DOTALL | re.IGNORECASE)
        opt_re = re.compile(r'(?:^|\s)([A-H])\.\s+(.+?)(?=(?:\s+[A-H]\.\s+)|$)', re.DOTALL)
        for r in rows:
            q_full = str(r.get('question', '')).strip()
            a = str(r.get('answer', '')).strip()
            cm = choices_re.search(q_full)
            if not cm:
                continue
            q_text = q_full[:cm.start()].strip().rstrip(':').strip()
            opts_blob = cm.group(1).strip()
            opts = [m.group(2).strip().rstrip('.;,') for m in opt_re.finditer(opts_blob)]
            if len(opts) < 2:
                continue
            if a in 'ABCDEFGH' and 'ABCDEFGH'.index(a) < len(opts):
                idx = 'ABCDEFGH'.index(a)
                a_text = f"{a}. {opts[idx]}"
            else:
                a_text = a
            if not _short_enough(q_text, a_text, max_q=240, max_a=160):
                continue
            cands.append({'_q': q_text, 'tag': str(r.get('question_type', '')),
                          'question': q_text, 'options': opts, 'answer': a_text})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]

    if slug == 'refspatial':
        cands = []
        for r in rows:
            p = str(r.get('prompt', '')).strip()
            obj = str(r.get('object', '')).strip()
            if not p or not obj:
                continue
            cands.append({'_q': p, 'tag': '',
                          'question': p, 'options': None,
                          'answer': f'2D point coordinates locating "{obj}"'})
        cands.sort(key=lambda r: len(r['question']))
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in cands[:4]]

    if slug == 'robospatial':
        return _generic_qoa(rows, tag_field='category')

    if slug == 'mmou':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            opts_raw = r.get('options')
            opts = None
            if opts_raw:
                if isinstance(opts_raw, dict):
                    opts = [opts_raw.get(k) for k in 'ABCDE' if opts_raw.get(k)]
                else:
                    parsed = _parse_listish(opts_raw)
                    if isinstance(parsed, list):
                        opts = parsed
            qt = r.get('question_type', '')
            if isinstance(qt, str):
                qt_parsed = _parse_listish(qt)
                if isinstance(qt_parsed, list) and qt_parsed:
                    qt = qt_parsed[0]
            cands.append({'_q': q, 'tag': str(qt).lower() if qt else '',
                          'question': q, 'options': opts,
                          'answer': '[See video for correct option]'})
        cands = [c for c in cands if _short_enough(c['question'], c['answer'])]
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]

    if slug == 'ocrbench':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            a = r.get('answer')
            a_parsed = _parse_listish(a) if isinstance(a, str) else a
            if isinstance(a_parsed, list) and a_parsed:
                a = str(a_parsed[0])
            elif a in (None, '', 'None'):
                continue
            else:
                a = str(a)
            if not _short_enough(q, a):
                continue
            cands.append({'_q': q, 'tag': str(r.get('question_type', '')).lower(),
                          'question': q, 'options': None, 'answer': a})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]

    if slug == 'pixmo-count':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).replace('<image>', '').strip()
            a = str(r.get('answer', '')).strip()
            if not _short_enough(q, a):
                continue
            cands.append({'_q': q, 'tag': '', 'question': q, 'options': None, 'answer': a})
        random.Random(13).shuffle(cands)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in cands[:4]]

    if slug == 'countbench':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            a = str(r.get('number', '')).strip()
            if not _short_enough(q, a):
                continue
            cands.append({'_q': q, 'tag': '', 'question': q, 'options': None, 'answer': a})
        random.Random(11).shuffle(cands)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in cands[:4]]

    if slug == 'realworldqa':
        cands = []
        for r in rows:
            q_full = str(r.get('question', '')).strip()
            a = str(r.get('answer', '')).strip()
            q_full = re.sub(r'\nPlease answer.*$', '', q_full, flags=re.DOTALL).strip()
            q_text, opts = _split_inline_options(q_full)
            if not opts:
                continue
            if a in 'ABCDEFGH' and 'ABCDEFGH'.index(a) < len(opts):
                idx = 'ABCDEFGH'.index(a)
                a_text = f"{a}. {opts[idx]}"
            else:
                a_text = a
            if not _short_enough(q_text, a_text, max_q=240):
                continue
            cands.append({'_q': q_text, 'tag': '', 'question': q_text, 'options': opts, 'answer': a_text})
        random.Random(17).shuffle(cands)
        cands.sort(key=lambda r: len(r['question']) + sum(len(o) for o in (r['options'] or [])))
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in cands[:4]]

    if slug == 'v-star':
        cands = []
        for r in rows:
            q_full = str(r.get('text', '')).strip()
            a = str(r.get('label', '')).strip()
            q_full = re.sub(r"Answer with the option.*$", '', q_full, flags=re.DOTALL).strip()
            q_text, opts = _split_inline_options(q_full)
            if not opts:
                continue
            if a in 'ABCDEFGH' and 'ABCDEFGH'.index(a) < len(opts):
                a_text = f"{a}. {opts['ABCDEFGH'.index(a)]}"
            else:
                a_text = a
            if not _short_enough(q_text, a_text):
                continue
            cands.append({'_q': q_text, 'tag': str(r.get('category', '')),
                          'question': q_text, 'options': opts, 'answer': a_text})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]

    if slug == 'mmbenchen':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            opts_raw = [r.get(c) for c in 'ABCD']
            opts = [str(o) for o in opts_raw if o and str(o) != 'nan']
            a = str(r.get('answer', '')).strip()
            if a in 'ABCD' and 'ABCD'.index(a) < len(opts):
                a_text = f"{a}. {opts['ABCD'.index(a)]}"
            else:
                a_text = a
            if not _short_enough(q, a_text):
                continue
            cands.append({'_q': q, 'tag': str(r.get('l2-category', '')),
                          'question': q, 'options': opts or None, 'answer': a_text})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]

    if slug == 'mmstar':
        return _generic_qoa(rows, tag_field='category')

    if slug == 'chartqa':
        return _generic_qoa(rows, tag_field='type')

    if slug == 'blink':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            opts = _parse_listish(r.get('choices'))
            a = str(r.get('answer', '')).strip()
            if a == 'hidden' or not opts:
                continue
            if a in 'ABCDEFGH' and 'ABCDEFGH'.index(a) < len(opts):
                a_text = f"{a}. {opts['ABCDEFGH'.index(a)]}"
            else:
                a_text = a
            if not _short_enough(q, a_text):
                continue
            cands.append({'_q': q, 'tag': str(r.get('sub_task', '')).lower(),
                          'question': q, 'options': opts, 'answer': a_text})
        sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        if not sel:
            for r in rows:
                q = str(r.get('question', '')).strip()
                opts = _parse_listish(r.get('choices'))
                if not q or not opts:
                    continue
                cands.append({'_q': q, 'tag': str(r.get('sub_task', '')).lower(),
                              'question': q, 'options': opts,
                              'answer': '[Hidden in benchmark; see sub_task for category]'})
            sel = _diverse_sample(cands, lambda r: r.get('tag', ''), n=4)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in sel]

    if slug == 'ai2d':
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            opts = _parse_listish(r.get('options'))
            a = r.get('answer')
            if not q or not opts or a is None:
                continue
            try:
                idx = int(a)
                a_text = f"{['A','B','C','D'][idx]}. {opts[idx]}"
            except Exception:
                continue
            if not _short_enough(q, a_text):
                continue
            cands.append({'_q': q, 'tag': '', 'question': q, 'options': opts, 'answer': a_text})
        random.Random(23).shuffle(cands)
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in cands[:4]]

    if slug in ('docvqa', 'infovqa'):
        cands = []
        for r in rows:
            q = str(r.get('question', '')).strip()
            if not q:
                continue
            cands.append({'_q': q, 'tag': '', 'question': q, 'options': None,
                          'answer': '[Free-form text extracted from document]'})
        random.Random(29).shuffle(cands)
        cands.sort(key=lambda r: len(r['question']))
        return [{k: v for k, v in r.items() if not k.startswith('_')} for r in cands[:4]]

    if slug == 'crpe':
        return _generic_qoa(rows, tag_field='task_type')

    if slug == 'videomme' or slug == 'videomme-w-subtitle':
        return []

    return _generic_qoa(rows)
