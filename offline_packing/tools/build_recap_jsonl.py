#!/usr/bin/env python3
"""
build_recap_jsonl.py

Rewrite the 20 pandas70m_gt10min_fix jsonl files by replacing the assistant
caption (messages[-1].content) with caption from
/data/dataset_frame/pandas70m_gt10min/v2_rewrite_caption/.

Alignment key  : video relative path
                 - in fix jsonl       : record["images_source"][0]
                 - in rewrite jsonl   : record["images_source"][0]  (Schema X)
                                        OR record["source_path"]   (Schema Y)

Output dir     : /data/dataset_frame/pandas70m_gt10min_fix_recap/jsonl_done/
                 pandas70m_gt10min_fix_recap_done_partNN.jsonl  (20 files)

Drop policy for fix records whose video has no rewrite caption: DROP + record.
Drop log       : /data/dataset_frame/pandas70m_gt10min_fix_recap/dropped.jsonl

Multiprocessing: 20 parts in parallel, one process per part.
"""
from __future__ import annotations
import argparse, json, os, sys, time, pickle, multiprocessing as mp
from typing import Optional

FIX_DIR = "/data/dataset_frame/pandas70m_gt10min_fix/jsonl_done"
REW_DIR = "/data/dataset_frame/pandas70m_gt10min/v2_rewrite_caption"
OUT_DIR = "/data/dataset_frame/pandas70m_gt10min_fix_recap/jsonl_done"
DROP_LOG = "/data/dataset_frame/pandas70m_gt10min_fix_recap/dropped.jsonl"
INDEX_PKL = "/data/dataset_frame/pandas70m_gt10min_fix_recap/rewrite_caption_index.pkl"


def build_rewrite_index(force: bool = False) -> dict[str, str]:
    """Build {video_rel_path -> new_assistant_caption} index.

    Cached at INDEX_PKL.
    """
    if not force and os.path.exists(INDEX_PKL):
        print(f"[index] loading cache {INDEX_PKL}", flush=True)
        with open(INDEX_PKL, "rb") as f:
            return pickle.load(f)

    os.makedirs(os.path.dirname(INDEX_PKL), exist_ok=True)
    files = sorted(
        f for f in os.listdir(REW_DIR)
        if f.endswith(".rewritten.jsonl") and "failed" not in f
    )
    print(f"[index] scanning {len(files)} rewrite files...", flush=True)
    idx: dict[str, str] = {}
    n_total = 0
    n_skip = 0
    n_dup = 0
    t0 = time.time()
    for fn in files:
        p = os.path.join(REW_DIR, fn)
        with open(p, "rb") as f:
            for raw in f:
                n_total += 1
                try:
                    r = json.loads(raw)
                except Exception:
                    n_skip += 1
                    continue
                vid = None
                if isinstance(r.get("images_source"), list) and r["images_source"]:
                    vid = r["images_source"][0]
                elif r.get("source_path"):
                    vid = r["source_path"]
                if not vid:
                    n_skip += 1
                    continue
                msgs = r.get("messages") or []
                if len(msgs) < 2:
                    n_skip += 1
                    continue
                cap = msgs[-1].get("content", "")
                if not cap:
                    n_skip += 1
                    continue
                if vid in idx:
                    n_dup += 1
                else:
                    idx[vid] = cap
        if n_total % 5 == 0 or fn == files[-1]:
            print(f"[index]   parsed {n_total:,} recs, idx={len(idx):,}, "
                  f"skip={n_skip:,}, dup={n_dup:,}  ({time.time()-t0:.1f}s)",
                  flush=True)
    with open(INDEX_PKL, "wb") as f:
        pickle.dump(idx, f, protocol=4)
    print(f"[index] saved {len(idx):,} entries → {INDEX_PKL} "
          f"(skip={n_skip:,}, dup={n_dup:,}, elapsed {time.time()-t0:.1f}s)",
          flush=True)
    return idx


def process_part(args):
    pidx, idx_pkl, in_path, out_path, drop_path = args
    # load the shared index per worker (process-local memory)
    with open(idx_pkl, "rb") as f:
        rew_idx = pickle.load(f)
    t0 = time.time()
    n_in = n_out = n_drop = n_no_vid = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(drop_path), exist_ok=True)
    with open(in_path, "rb") as fin, \
         open(out_path, "wb") as fout, \
         open(drop_path, "wb") as fdrop:
        for raw in fin:
            n_in += 1
            r = json.loads(raw)
            vid = None
            if isinstance(r.get("images_source"), list) and r["images_source"]:
                vid = r["images_source"][0]
            if not vid:
                n_no_vid += 1
                fdrop.write(
                    json.dumps({"part": pidx, "id": r.get("id"),
                                "reason": "no_images_source"},
                               ensure_ascii=False).encode() + b"\n"
                )
                continue
            new_cap = rew_idx.get(vid)
            if not new_cap:
                n_drop += 1
                fdrop.write(
                    json.dumps({"part": pidx, "id": r.get("id"),
                                "vid": vid,
                                "reason": "no_rewrite_caption"},
                               ensure_ascii=False).encode() + b"\n"
                )
                continue
            # mutate messages[-1].content (preserve role, keep messages[:-1] intact)
            msgs = r.get("messages")
            if not msgs or len(msgs) < 2:
                n_drop += 1
                fdrop.write(
                    json.dumps({"part": pidx, "id": r.get("id"),
                                "vid": vid,
                                "reason": "bad_messages"},
                               ensure_ascii=False).encode() + b"\n"
                )
                continue
            # Update last message (assistant) content
            new_msgs = list(msgs)
            new_last = dict(new_msgs[-1])
            new_last["content"] = new_cap
            new_msgs[-1] = new_last
            r["messages"] = new_msgs
            # Optional: also tag where caption came from
            r.setdefault("_caption_src", "v2_rewrite_caption")
            fout.write(json.dumps(r, ensure_ascii=False).encode() + b"\n")
            n_out += 1
    return dict(part=pidx, n_in=n_in, n_out=n_out, n_drop=n_drop,
                n_no_vid=n_no_vid, elapsed=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--parts", default="0-19", help="e.g. '0-19' or '0,3,5'")
    args = ap.parse_args()

    # 1) build / load index
    idx = build_rewrite_index(force=args.rebuild_index)
    if not idx:
        print("[fatal] empty rewrite caption index", file=sys.stderr)
        sys.exit(2)

    # 2) prepare per-part tasks
    if "-" in args.parts:
        lo, hi = args.parts.split("-")
        part_ids = list(range(int(lo), int(hi) + 1))
    else:
        part_ids = [int(x) for x in args.parts.split(",")]
    print(f"[run] parts to process: {part_ids}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DROP_LOG), exist_ok=True)

    tasks = []
    for pid in part_ids:
        in_p = os.path.join(
            FIX_DIR, f"pandas70m_gt10min_fix_done_part{pid:02d}.jsonl")
        out_p = os.path.join(
            OUT_DIR, f"pandas70m_gt10min_fix_recap_done_part{pid:02d}.jsonl")
        drop_p = os.path.join(
            os.path.dirname(DROP_LOG),
            f"dropped_part{pid:02d}.jsonl")
        if not os.path.exists(in_p):
            print(f"[skip] missing input: {in_p}")
            continue
        tasks.append((pid, INDEX_PKL, in_p, out_p, drop_p))

    # 3) run in parallel
    print(f"[run] starting pool workers={min(args.workers, len(tasks))} "
          f"for {len(tasks)} parts...", flush=True)
    t0 = time.time()
    with mp.Pool(min(args.workers, len(tasks))) as pool:
        results = []
        for r in pool.imap_unordered(process_part, tasks):
            results.append(r)
            print(f"[done] part{r['part']:02d} "
                  f"in={r['n_in']:,} out={r['n_out']:,} "
                  f"drop_no_cap={r['n_drop']:,} drop_no_vid={r['n_no_vid']:,} "
                  f"({r['elapsed']:.1f}s)", flush=True)

    # 4) summary
    results.sort(key=lambda r: r["part"])
    tot = lambda k: sum(r[k] for r in results)
    print(f"\n=== summary (elapsed {time.time()-t0:.1f}s) ===")
    print(f"  parts processed     : {len(results)}")
    print(f"  total in            : {tot('n_in'):,}")
    print(f"  total out           : {tot('n_out'):,}")
    print(f"  dropped: no caption : {tot('n_drop'):,}")
    print(f"  dropped: no vid     : {tot('n_no_vid'):,}")
    drop_total = tot("n_drop") + tot("n_no_vid")
    print(f"  drop ratio          : {100*drop_total/max(1,tot('n_in')):.3f}%")
    print(f"  out dir             : {OUT_DIR}")
    print(f"  drop log dir        : {os.path.dirname(DROP_LOG)}")

    # consolidate dropped logs (optional)
    drop_files = sorted(
        f for f in os.listdir(os.path.dirname(DROP_LOG))
        if f.startswith("dropped_part") and f.endswith(".jsonl")
    )
    if drop_files:
        with open(DROP_LOG, "wb") as out:
            for fn in drop_files:
                p = os.path.join(os.path.dirname(DROP_LOG), fn)
                with open(p, "rb") as f:
                    out.write(f.read())
        print(f"  combined drop log   : {DROP_LOG} "
              f"({sum(os.path.getsize(os.path.join(os.path.dirname(DROP_LOG), fn)) for fn in drop_files)} bytes)")

if __name__ == "__main__":
    main()
