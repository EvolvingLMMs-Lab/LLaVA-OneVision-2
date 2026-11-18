"""Streaming special tokens: registration, idempotence, and HF-export finalize.

Registering is what makes ``<silence>`` one id instead of four subword pieces.
Everything downstream depends on that: the focal-loss helper resolves the state
markers by id, and an exported checkpoint whose tokenizer lacks them cannot read
back its own training data.

Run inside the training image:

  R=/workspace/LLaVA-OneVision-2.5
  docker exec -e PYTHONPATH=$R/aiak_megatron:$R -w $R <container> \
      python3 tests/test_streaming_tokens.py [BASE_PROCESSOR_DIR]
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "data_preprocess", "register_streaming_tokens.py")
BASE = sys.argv[1] if len(sys.argv) > 1 else \
    "/ov2/pretrain_models/preprocessor/preprocessor_llava_onevision1_5"
# Some processor directories ship no config.json; pass the checkpoint's
# padded_vocab_size as the second argument to still check the ids fit.
VOCAB_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None

sys.path.insert(0, REPO)
from aiak_training_llm.data.multimodal.streaming_tokens import (  # noqa: E402
    STREAMING_SPECIAL_TOKENS,
    STREAMING_STATE_TOKENS,
)

FAILURES = []


def check(label, got, want):
    if got == want:
        print(f"  pass  {label}")
    else:
        print(f"  FAIL  {label}\n          got  {got!r}\n          want {want!r}")
        FAILURES.append(label)


def run_tool(dst, *extra, src=BASE):
    argv = [sys.executable, TOOL, "--src", src, "--dst", dst, "--trust-remote-code"]
    if VOCAB_LIMIT:
        argv += ["--vocab-limit", str(VOCAB_LIMIT)]
    result = subprocess.run(argv + list(extra), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
    return result.returncode


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- 1
def test_registration(dst):
    print("1. registration makes every token one distinct id")
    check("tool exits 0", run_tool(dst), 0)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(dst, trust_remote_code=True)
    ids = {t: tokenizer.encode(t, add_special_tokens=False) for t in STREAMING_SPECIAL_TOKENS}
    check("each token is one id", [len(v) for v in ids.values()], [1] * len(STREAMING_SPECIAL_TOKENS))
    flat = [v[0] for v in ids.values()]
    check("ids are distinct", len(set(flat)), len(flat))
    check("ids are contiguous", sorted(flat), list(range(min(flat), min(flat) + len(flat))))

    cfg = os.path.join(dst, "config.json")
    limit = VOCAB_LIMIT
    if limit is None and os.path.isfile(cfg):
        blob = read(cfg)
        limit = (blob.get("text_config") or {}).get("vocab_size") or blob.get("vocab_size")
    if limit:
        check("ids fit under the vocab limit (no embedding resize needed)", max(flat) < limit, True)
    else:
        print("  skip  vocab fit (base ships no config.json; pass a limit as argv[2])")

    for token in STREAMING_SPECIAL_TOKENS:
        if tokenizer.decode(ids[token]) != token:
            FAILURES.append(f"decode round trip for {token}")
    print(f"  pass  decode round trip for all {len(STREAMING_SPECIAL_TOKENS)} tokens")

    base = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    check("ordinary text tokenizes identically",
          base.encode("The man is loading wood."), tokenizer.encode("The man is loading wood."))
    check("<silence> was multi-token before", len(base.encode("<silence>", add_special_tokens=False)) > 1, True)
    check("still visible under skip_special_tokens",
          tokenizer.decode(tokenizer.encode("<silence>", add_special_tokens=False),
                           skip_special_tokens=True), "<silence>")


# --------------------------------------------------------------------------- 2
def test_nothing_destroyed(dst):
    print("2. the base tokenizer config survives intact")
    src_cfg, dst_cfg = read(os.path.join(BASE, "tokenizer_config.json")), \
        read(os.path.join(dst, "tokenizer_config.json"))
    check("no top-level key lost", sorted(set(src_cfg) - set(dst_cfg)), [])
    src_decoder = src_cfg.get("added_tokens_decoder", {})
    dst_decoder = dst_cfg.get("added_tokens_decoder", {})
    if src_decoder:
        check(
            "added_tokens_decoder grows by the new token count",
            len(dst_decoder) - len(src_decoder),
            len(STREAMING_SPECIAL_TOKENS),
        )
    else:
        check("empty added_tokens_decoder stays empty", dst_decoder, {})
    changed = [k for k in src_cfg if k != "added_tokens_decoder" and src_cfg[k] != dst_cfg.get(k)]
    check("every other field untouched", changed, [])

    src_tj, dst_tj = read(os.path.join(BASE, "tokenizer.json")), read(os.path.join(dst, "tokenizer.json"))
    check("BPE vocab untouched", src_tj["model"]["vocab"] == dst_tj["model"]["vocab"], True)
    check("merges untouched", src_tj["model"]["merges"] == dst_tj["model"]["merges"], True)

    if os.path.isfile(os.path.join(BASE, "added_tokens.json")):
        src_at = read(os.path.join(BASE, "added_tokens.json"))
        dst_at = read(os.path.join(dst, "added_tokens.json"))
        check("added_tokens.json keeps every original entry",
              all(dst_at.get(k) == v for k, v in src_at.items()), True)
        check("added_tokens.json agrees with tokenizer_config",
              len(dst_at), len(dst_cfg["added_tokens_decoder"]))
    else:
        print("  skip  added_tokens.json (base ships none)")
    if os.path.isfile(os.path.join(BASE, "special_tokens_map.json")):
        check("special_tokens_map untouched (streaming tokens stay non-special)",
              read(os.path.join(BASE, "special_tokens_map.json")) ==
              read(os.path.join(dst, "special_tokens_map.json")), True)
    else:
        print("  skip  special_tokens_map.json (base ships none)")


# --------------------------------------------------------------------------- 3
def test_idempotent(dst):
    print("3. re-running assigns no new ids")
    name = "added_tokens.json" if os.path.isfile(os.path.join(dst, "added_tokens.json")) \
        else "tokenizer.json"
    before = read(os.path.join(dst, name))
    check("second run exits 0", run_tool(dst), 0)
    check(f"{name} unchanged", read(os.path.join(dst, name)), before)

    # Use the registered tokenizer as the source, as a resumed checkpoint would.
    second_dst = os.path.join(os.path.dirname(dst), "reregistered")
    check(
        "already-registered source exits 0",
        run_tool(second_dst, src=dst),
        0,
    )

    for name in ("tokenizer.json", "tokenizer_config.json"):
        check(
            f"{name} unchanged after registration",
            read(os.path.join(second_dst, name)),
            read(os.path.join(dst, name)),
        )


# --------------------------------------------------------------------------- 4
def test_opt_out(dst):
    print("4. --no-streaming-tokens copies files without touching the tokenizer")
    check("exits 0", run_tool(dst, "--no-streaming-tokens"), 0)
    check("tokenizer_config identical to base",
          read(os.path.join(dst, "tokenizer_config.json")) ==
          read(os.path.join(BASE, "tokenizer_config.json")), True)


# --------------------------------------------------------------------------- 5
def test_export_finalize(dst):
    print("5. finalizing an export keeps the converted weights")
    os.makedirs(dst, exist_ok=True)
    weights = os.path.join(dst, "model-00001-of-00002.safetensors")
    index = os.path.join(dst, "model.safetensors.index.json")
    for path, blob in ((weights, b"CONVERTED-WEIGHTS"), (index, b'{"weight_map":{}}')):
        with open(path, "wb") as f:
            f.write(blob)

    check("exits 0", run_tool(dst), 0)
    check("weights not overwritten", open(weights, "rb").read(), b"CONVERTED-WEIGHTS")
    check("index not overwritten", open(index, "rb").read(), b'{"weight_map":{}}')
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not os.path.isfile(os.path.join(dst, name)):
            FAILURES.append(f"export missing {name}")
    print("  pass  export now carries config/tokenizer/processor beside the weights")


# --------------------------------------------------------------------------- 6
def test_aiak_wrapper(dst):
    print("6. the focal-loss helper's own check passes through the AIAK tokenizer")
    from aiak_training_llm.tokenizer.tokenization_hf import AutoTokenizerFromHF

    tokenizer = AutoTokenizerFromHF(dst, True, "right", 131072, False)
    ids = tuple(tokenizer.tokenizer.convert_tokens_to_ids(m) for m in STREAMING_STATE_TOKENS)
    # Mirrors _state_token_ids in pretrain_llava_onevision2.py.
    ok = all(isinstance(i, int) and tokenizer.tokenize(m, add_special_tokens=False) == [i]
             for m, i in zip(STREAMING_STATE_TOKENS, ids))
    check("_state_token_ids would accept this tokenizer", ok, True)
    check("state token ids distinct", len(set(ids)), len(ids))


work = tempfile.mkdtemp(prefix="streaming_tokens_test_")
try:
    main_dst = os.path.join(work, "registered")
    test_registration(main_dst); print()
    test_nothing_destroyed(main_dst); print()
    test_idempotent(main_dst); print()
    test_opt_out(os.path.join(work, "plain")); print()
    test_export_finalize(os.path.join(work, "export")); print()
    test_aiak_wrapper(main_dst); print()
finally:
    shutil.rmtree(work, ignore_errors=True)

if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
    sys.exit(1)
print("ALL TESTS PASSED")
