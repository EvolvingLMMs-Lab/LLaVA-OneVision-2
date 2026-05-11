"""Slug -> {csv, type, hf_dataset_path} manifest for all 39 benchmarks.
Single source of truth. Used by build_chart_data.py and dispatched subagents.
"""
import os, glob

CSV_DIR = '/ov2/xiangan/lmms-eval-ov2/benchmark_stats/csv'
HF_ROOT = '/ov2/xiangan/lmms-eval-ov2/.huggingface_cache/datasets'

# slug -> (csv_filename, type, hf_dataset_glob_pattern OR list of subset glob patterns)
# Where multiple subsets exist (e.g. BLINK has 14, MV-Bench has 20), we list all.
MANIFEST = {
    # ===== Video benchmarks (have duration) =====
    'tempcompass':        ('tempcompass_mc.csv',       'video', ['lmms-lab___temp_compass/multi-choice']),
    'nextqa':             ('nextqa_mc.csv',            'video', ['lmms-lab___n_ex_tqa/MC']),
    'videomme':           ('videomme.csv',             'video', ['lmms-lab___video-mme/videomme']),
    'videomme-w-subtitle':('videomme.csv',             'video', ['lmms-lab___video-mme/videomme']),
    'videomme-v2-64':     ('videommev2.csv',           'video', ['MME-Benchmarks___video-mme-v2/default']),
    'mlvu-dev':           ('mlvu_dev.csv',             'video', ['sy1998___mlvu_dev/default']),
    'lvbench':            ('lvbench.csv',              'video', ['lmms-lab___lv_bench/default']),
    'longvideobench':     ('longvideobench_val.csv',   'video', ['longvideobench___long_video_bench/default']),
    'mmvu-val':           ('mmvu_val.csv',             'video', ['lmms-lab___mmvu/default']),
    'videoeval-pro':      ('videoeval_pro.csv',        'video', ['TIGER-Lab___video_eval-pro/default']),
    't-charades':         ('timelens_charades.csv',    'video', ['kcz358___timelens/default']),
    't-activitynet':      ('timelens_activitynet.csv', 'video', ['kcz358___timelens/default']),
    't-qvhighlights':     ('timelens_qvhighlights.csv','video', ['kcz358___timelens/default']),

    # ===== Spatial benchmarks =====
    'vsi-bench':       ('vsibench.csv',         'video', ['nyu-visionx___vsi-bench/full']),
    'cv-bench-2d':     ('cv_bench_2d.csv',      'image', ['nyu-visionx___cv-bench/2D']),
    'cv-bench-3d':     ('cv_bench_3d.csv',      'image', ['nyu-visionx___cv-bench/3D']),
    'mmsi-bench':      ('mmsi_bench.csv',       'image', ['RunsenXu___mmsi-bench/default']),
    'embspatial':      ('embspatial.csv',       'image', ['FlagEval___emb_spatial-bench/default', 'Phineas476___emb_spatial-bench/default']),
    'sat':             ('sat.csv',              'image', ['hunarbatra___sat-spatial-vqa/default']),
    'sat-mcq':         ('sat_mcq.csv',          'image', ['hunarbatra___sat-spatial-vqa/default']),
    'refspatial':      ('refspatial.csv',       'image', ['BAAI___ref_spatial-bench/default']),
    'robospatial':     ('robospatial.csv',      'image', ['chanhee-luke___robo_spatial-home/default']),
    'metavqa':         ('metavqa.csv',          'image', ['Zray26___metavqa-eval/default', 'Weizhen011210___meta_vqa-eval/default']),
    'crosspoint':      ('crosspoint.csv',       'image', ['WangYipu2002___cross_point-bench/default']),
    'erqa':            ('erqa.csv',             'image', ['FlagEval___erqa/default', 'runoob1___erqa/default']),

    # ===== Image benchmarks =====
    'ai2d':         ('ai2d.csv',         'image', ['lmms-lab___ai2d/default', 'Efficient-Large-Model___ai2d-no-mask/default']),
    'blink':        ('blink.csv',        'image', ['BLINK-Benchmark___blink']),  # all subdirs
    'chartqa':      ('chartqa.csv',      'image', ['lmms-lab___chart_qa/default']),
    'countbench':   ('countbenchqa.csv', 'image', ['vikhyatk___count_bench_qa/default']),
    'crpe':         ('crpe_relation.csv','image', ['OpenGVLab___crpe/default', 'Zray26___crpe-relation/default']),
    'docvqa':       ('docvqa_val.csv',   'image', ['lmms-lab___doc_vqa/DocVQA']),
    'infovqa':      ('infovqa_val.csv',  'image', ['lmms-lab___doc_vqa/InfographicVQA']),
    'mmbenchen':    ('mmbench.csv',      'image', ['lmms-lab___mm_bench_en/default', 'lmms-lab___mm_bench/en']),
    'mmou':         ('mmou.csv',         'image', ['nvidia___mmou/default']),
    'mmstar':       ('mmstar.csv',       'image', ['Lin-Chen___mm_star/val']),
    'ocrbench':     ('ocrbench.csv',     'image', ['echo840___ocr_bench/default']),
    'pixmo-count':  ('pixmo_count.csv',  'image', ['kcz358___pixmo-count/default']),
    'realworldqa':  ('realworldqa.csv',  'image', ['lmms-lab___real_world_qa/default', 'xai-org___realworld_qa/default']),
    'v-star':       ('vstar_bench.csv',  'image', ['lmms-lab___vstar-bench/default']),

    # mv-bench: no CSV, skip
}

def find_arrow_files(subset_pattern):
    """Given 'lmms-lab___temp_compass/multi-choice' return all .arrow files under it."""
    base = os.path.join(HF_ROOT, subset_pattern)
    return sorted(glob.glob(os.path.join(base, '**', '*.arrow'), recursive=True))

if __name__ == '__main__':
    import sys
    for slug, (csv, typ, patterns) in MANIFEST.items():
        csv_ok = os.path.exists(os.path.join(CSV_DIR, csv))
        all_arrows = []
        for p in patterns:
            all_arrows.extend(find_arrow_files(p))
        print(f"{slug:25s} csv={'OK' if csv_ok else 'MISSING':7s} type={typ:5s} arrows={len(all_arrows):3d}  patterns={patterns}")
