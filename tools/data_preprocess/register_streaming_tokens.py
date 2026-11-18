#!/usr/bin/env python3
"""Copy an HF model/processor directory and register the streaming special tokens.

Two uses, same operation:

  * Prepare the tokenizer training reads, so ``<silence>`` and friends tokenize as
    one id each::

      python register_streaming_tokens.py --src <base processor> --dst <new dir>

  * Finish an mcore -> HF export. The convert scripts write only safetensors, so
    the result is not loadable on its own; this drops the config, processor and
    tokenizer in beside the weights::

      python register_streaming_tokens.py --src <base processor> --dst <export dir>

The tokenizer files are edited field by field rather than round-tripped through
``save_pretrained``: that call rewrites the config in whatever shape the local
transformers version prefers, and has been observed to drop
``added_tokens_decoder`` and ``additional_special_tokens`` entirely.

Tokens are registered the way ``<tool_call>`` is in the Qwen base: an added token
with ``special=false``, kept out of ``additional_special_tokens``. They are then
one id each and never split, but ``decode(..., skip_special_tokens=True)`` still
shows them, which streaming inference needs in order to see that the model chose
to stay silent.

Ids must fit under the model's configured ``vocab_size``; if they would not, this
fails rather than write a tokenizer that indexes past the embedding matrix.
"""

import argparse
import json
import os
import re
import shutil
import sys

try:
    from aiak_training_llm.data.multimodal.streaming_tokens import STREAMING_SPECIAL_TOKENS
except ImportError:  # usable without the training package on sys.path
    _mod = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "aiak_training_llm", "data", "multimodal", "streaming_tokens.py"))
    _ns = {}
    with open(_mod, encoding="utf-8") as _f:
        exec(compile(_f.read(), _mod, "exec"), _ns)  # noqa: S102
    STREAMING_SPECIAL_TOKENS = _ns["STREAMING_SPECIAL_TOKENS"]

# Model weights rather than tokenizer/config artifacts: the destination may already
# hold freshly converted weights that must not be overwritten by the base model's.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".index.json")

# Hand-made and editor backups sit next to the real files in some processor directories
# (preprocessor_config.json.bak_4Mpixels, modeling_llava_onevision2.py.bak_1dpos). Copying
# them puts a stale duplicate of a live file into the released model, where at best it
# confuses a reader and at worst someone restores it over the file that is actually used.
BACKUP_NAME_RE = re.compile(r"(~|\.(bak|orig|old|tmp|save|swp|swo|rej)(_[^.]*)?)$", re.IGNORECASE)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def copy_aux_files(src, dst):
    """Copy config/processor/tokenizer files, leaving any weights in dst alone.

    Backups in ``src`` are skipped and reported rather than shipped; see BACKUP_NAME_RE.
    """
    copied, skipped = [], []
    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        if not os.path.isfile(path) or name.startswith(".") or name.endswith(WEIGHT_SUFFIXES):
            continue
        if BACKUP_NAME_RE.search(name):
            skipped.append(name)
            continue
        shutil.copy2(path, os.path.join(dst, name))
        copied.append(name)
    if skipped:
        print(f"[aux] skipped {len(skipped)} backup file(s) in {src}: " + ", ".join(skipped))
    return copied


def configured_vocab_size(directory):
    """Rows the embedding matrix will have, or None when no config is present."""
    path = os.path.join(directory, "config.json")
    if not os.path.isfile(path):
        return None
    config = read_json(path)
    text_config = config.get("text_config") or config.get("llm_config") or {}
    return text_config.get("vocab_size") or config.get("vocab_size")


def assign_ids(tokenizer_json, tokenizer_config, tokens):
    """Reuse the id of any token already registered, append the rest after the last one."""
    known = {entry["content"]: entry["id"] for entry in tokenizer_json.get("added_tokens", [])}
    for raw_id, entry in (tokenizer_config.get("added_tokens_decoder") or {}).items():
        known.setdefault(entry["content"], int(raw_id))

    base_vocab = tokenizer_json.get("model", {}).get("vocab") or {}
    next_id = max([len(base_vocab)] + [i + 1 for i in known.values()] or [0])

    assigned, fresh = {}, []
    for token in tokens:
        if token in known:
            assigned[token] = known[token]
        else:
            assigned[token] = next_id
            fresh.append(token)
            next_id += 1
    return assigned, fresh


def register(dst, tokens):
    """Insert the tokens into every tokenizer file present in dst."""
    tj_path = os.path.join(dst, "tokenizer.json")
    tc_path = os.path.join(dst, "tokenizer_config.json")
    if not os.path.isfile(tj_path) or not os.path.isfile(tc_path):
        raise SystemExit(f"need both tokenizer.json and tokenizer_config.json in {dst}")

    tokenizer_json = read_json(tj_path)
    tokenizer_config = read_json(tc_path)
    assigned, fresh = assign_ids(tokenizer_json, tokenizer_config, tokens)
    touched = []

    if fresh:
        # tokenizer.json: the fast tokenizer's own added-token table, kept id-sorted.
        entries = tokenizer_json.setdefault("added_tokens", [])
        entries.extend({
            "id": assigned[t], "content": t, "single_word": False, "lstrip": False,
            "rstrip": False, "normalized": False, "special": False,
        } for t in fresh)
        entries.sort(key=lambda e: e["id"])
        write_json(tj_path, tokenizer_json)
        touched.append("tokenizer.json")

        # tokenizer_config.json: the slow-tokenizer view of the same table. Only
        # written when the base model already keeps one. Some checkpoints ship an
        # empty added_tokens_decoder and let tokenizer.json be the sole authority;
        # adding a partial table there makes the loader renumber the whole added
        # vocab, which silently moves <|im_end|> and every id a checkpoint depends on.
        decoder = tokenizer_config.get("added_tokens_decoder")
        if decoder:
            for t in fresh:
                decoder[str(assigned[t])] = {
                    "content": t, "lstrip": False, "normalized": False,
                    "rstrip": False, "single_word": False, "special": False,
                }
            write_json(tc_path, tokenizer_config)
            touched.append("tokenizer_config.json")
        else:
            touched.append("tokenizer_config.json (left alone: no added_tokens_decoder)")

        # added_tokens.json: legacy name -> id map. Stale copies mislead slow loaders,
        # so it is updated whenever the base model shipped one.
        at_path = os.path.join(dst, "added_tokens.json")
        if os.path.isfile(at_path):
            added = read_json(at_path)
            added.update({t: assigned[t] for t in fresh})
            write_json(at_path, dict(sorted(added.items(), key=lambda kv: kv[1])))
            touched.append("added_tokens.json")

    return assigned, fresh, touched


def existing_token_ids(directory, trust_remote_code):
    """Map every token the directory already knows to its id, before we touch it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=trust_remote_code)
    return {t: tokenizer.convert_tokens_to_ids(t) for t in tokenizer.get_vocab()}, len(tokenizer)


def verify(dst, tokens, limit, trust_remote_code, baseline=None, expected_len=None):
    """Load the written files back and confirm each token is one distinct in-range id.

    The load-bearing check is ``baseline``: adding tokens must leave every id that
    already existed exactly where it was. A tokenizer that renumbers its added
    vocab still passes every local check - single id, distinct, decodes - while
    making every checkpoint trained against it wrong.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(dst, trust_remote_code=trust_remote_code)
    ids, problems = {}, []

    if baseline:
        moved = {t: (was, tokenizer.convert_tokens_to_ids(t))
                 for t, was in baseline.items()
                 if tokenizer.convert_tokens_to_ids(t) != was}
        if moved:
            sample = dict(list(moved.items())[:6])
            problems.append(
                f"{len(moved)} pre-existing token(s) changed id, e.g. "
                + ", ".join(f"{t!r} {a}->{b}" for t, (a, b) in sample.items())
                + " - the added vocab was renumbered, which invalidates existing checkpoints")
    if expected_len is not None and len(tokenizer) != expected_len:
        problems.append(
            f"len(tokenizer) is {len(tokenizer)}, expected {expected_len}")
    for token in tokens:
        pieces = tokenizer.encode(token, add_special_tokens=False)
        if len(pieces) != 1:
            problems.append(f"{token!r} tokenizes to {len(pieces)} ids {pieces}, expected 1")
            continue
        ids[token] = pieces[0]
        if tokenizer.decode(pieces) != token:
            problems.append(f"{token!r} does not survive a decode round trip")
    if len(set(ids.values())) != len(ids):
        problems.append(f"ids are not distinct: {ids}")
    if limit is not None and ids:
        over = {t: i for t, i in ids.items() if i >= limit}
        if over:
            problems.append(
                f"ids {over} are >= config vocab_size {limit}: the embedding matrix has no row "
                "for them, so the model must be resized before these tokens can be used")
    return tokenizer, ids, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Base HF processor/model directory to copy from")
    parser.add_argument("--dst", required=True, help="Destination directory (created; existing weights kept)")
    parser.add_argument("--no-streaming-tokens", action="store_true",
                        help="Copy the files but leave the tokenizer unchanged")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--vocab-limit", type=int, default=None,
                        help="Embedding rows the model actually has, for sources that ship no "
                             "config.json (use the checkpoint's padded_vocab_size). Without it "
                             "and without a config.json, the fit check cannot run.")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        parser.error(f"--src is not a directory: {args.src}")
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        parser.error("--src and --dst must differ; this tool never edits the base model in place")
    os.makedirs(args.dst, exist_ok=True)

    copied = copy_aux_files(args.src, args.dst)
    print(f"[aux] copied {len(copied)} file(s) from {args.src}")
    print(f"      {', '.join(copied)}")
    if "config.json" not in copied:
        print(f"[aux] WARNING: {args.src} has no config.json, so {args.dst} still cannot be")
        print("      loaded as a model. Copy the architecture config in separately.")

    if args.no_streaming_tokens:
        print("[tokens] skipped (--no-streaming-tokens)")
        return 0

    limit = args.vocab_limit or configured_vocab_size(args.dst)
    if limit is None:
        print("[tokens] WARNING: no config.json and no --vocab-limit, so the check that the new")
        print("         ids fit the embedding matrix cannot run. Pass the checkpoint's")
        print("         padded_vocab_size as --vocab-limit to enable it.")
    baseline, baseline_len = existing_token_ids(args.dst, args.trust_remote_code)
    assigned, fresh, touched = register(args.dst, STREAMING_SPECIAL_TOKENS)
    print(f"[tokens] {len(fresh)} new, {len(assigned) - len(fresh)} already present"
          f"  (config vocab_size={limit})")
    print(f"[tokens] updated: {', '.join(touched) if touched else 'nothing to do'}")

    tokenizer, ids, problems = verify(
        args.dst,
        STREAMING_SPECIAL_TOKENS,
        limit,
        args.trust_remote_code,
        baseline=baseline,
        expected_len=baseline_len + len(fresh),
    )
    for token in STREAMING_SPECIAL_TOKENS:
        print(f"         {token:22s} id={ids.get(token, '?')}"
              f"  {'new' if token in fresh else 'already present'}")
    print(f"[tokens] len(tokenizer) = {len(tokenizer)}")
    if limit is not None and ids:
        print(f"[tokens] {limit - max(ids.values()) - 1} embedding row(s) still unused")

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK: every streaming token is a single, distinct id inside the configured vocab")
    return 0


if __name__ == "__main__":
    sys.exit(main())
