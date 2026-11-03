#!/usr/bin/env python3
"""Finish a streaming HF export: install the dual-mode (offline / streaming) chat template and verify.

Run by convert_*_mcore_to_hf.sh after the processor/tokenizer files have been copied from the
tokenizer the model was TRAINED with (the checkpoint's --hf-tokenizer-path). It

  1. renders chat_template_ov_dual.jinja with the trainer's two default systems - the qwen2-vl
     template's default_system (offline, from data/chat_templete.py) and STREAMING_SYSTEM_PROMPT
     (streaming, from qwen2vl_task_encoder.py), so neither can drift from training - and writes it
     as <dst>/chat_template.jinja (and into tokenizer/processor configs that embed one). The mode is
     chosen per call: apply_chat_template(..., streaming=True) for streaming, offline by default;
  2. checks every added token of the training tokenizer kept its id in <dst>, the streaming
     tokens are single ids, and all of them fit the embedding rows in config.json;
  3. renders a streaming and an offline conversation with the exported template and compares them
     with the strings the trainer's qwen2-vl template + canvas timestamps produce for the same turns.

Any mismatch exits non-zero.

    python finalize_streaming_export.py --dst <hf_dir> --train-tokenizer <dir> [--repo <OV21 repo>]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
STATE_AND_MEMORY = ("<silence>", "<response>", "<captioning>", "<question_memory>",
                    "</question_memory>", "<answer_memory>", "</answer_memory>")


def trainer_system_prompt(repo):
    path = os.path.join(repo, "aiak_training_llm", "data", "multimodal", "qwen2vl_task_encoder.py")
    m = re.search(r'^STREAMING_SYSTEM_PROMPT\s*=\s*"([^"]+)"', open(path, encoding="utf-8").read(), re.M)
    if not m:
        sys.exit(f"[streaming] cannot find STREAMING_SYSTEM_PROMPT in {path}")
    return m.group(1)


def trainer_offline_system(repo, template_name="qwen2-vl"):
    """default_system of the chat template the trainer ran with (--chat-template qwen2-vl)."""
    path = os.path.join(repo, "aiak_training_llm", "data", "chat_templete.py")
    src = open(path, encoding="utf-8").read()
    m = re.search(r'name="%s",(.*?)\n\)' % re.escape(template_name), src, re.S)
    d = m and re.search(r'default_system="([^"]+)"', m.group(1))
    if not d:
        sys.exit(f"[streaming] cannot find default_system of {template_name} in {path}")
    return d.group(1)


def added_tokens(tok_dir):
    tj = json.load(open(os.path.join(tok_dir, "tokenizer.json"), encoding="utf-8"))
    return {a["content"]: a["id"] for a in tj.get("added_tokens", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--train-tokenizer", required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--template", default=os.path.join(
        DEFAULT_REPO, "examples", "llava_onevision2", "convert", "chat_template_ov_dual.jinja"))
    a = ap.parse_args()
    problems = []

    # ---- 1. template ---------------------------------------------------------------------
    system = trainer_system_prompt(a.repo)
    offline = trainer_offline_system(a.repo)
    template = (open(a.template, encoding="utf-8").read()
                .replace("__STREAMING_SYSTEM_PROMPT__", system).replace("__OFFLINE_SYSTEM_PROMPT__", offline))
    if "_SYSTEM_PROMPT__" in template or system not in template or offline not in template:
        sys.exit("[streaming] template placeholders were not filled")
    with open(os.path.join(a.dst, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(template)
    for name in ("tokenizer_config.json", "processor_config.json"):
        p = os.path.join(a.dst, name)
        if os.path.isfile(p):
            cfg = json.load(open(p, encoding="utf-8"))
            if "chat_template" in cfg:  # an embedded template would shadow chat_template.jinja
                cfg["chat_template"] = template
                json.dump(cfg, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                print(f"[streaming] replaced embedded chat_template in {name}")
    print(f"[streaming] chat_template.jinja installed: offline (default) system = {offline!r}; "
          f"streaming=True system = {system!r}")

    # ---- 2. tokenizer --------------------------------------------------------------------
    want, got = added_tokens(a.train_tokenizer), added_tokens(a.dst)
    moved = {t: (i, got.get(t)) for t, i in want.items() if got.get(t) != i}
    if moved:
        problems.append(f"added-token ids differ from the training tokenizer: {moved}")
    vocab = json.load(open(os.path.join(a.dst, "config.json"), encoding="utf-8"))
    vocab = vocab.get("text_config", {}).get("vocab_size") or vocab.get("vocab_size")
    present = {t: got[t] for t in STATE_AND_MEMORY if t in got}
    if "<silence>" not in present or "<response>" not in present:
        problems.append("training tokenizer has no <silence>/<response>: not a streaming model?")
    over = {t: i for t, i in got.items() if vocab and i >= vocab}
    if over:
        problems.append(f"token ids beyond the embedding rows ({vocab}): {over}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.dst, trust_remote_code=True)
    for t, i in present.items():
        if tok.encode(t, add_special_tokens=False) != [i]:
            problems.append(f"{t} does not encode to the single id {i}")
    print("[streaming] tokens: " + "  ".join(f"{t}={i}" for t, i in present.items())
          + f"  (embedding rows {vocab})")

    # ---- 3. render check against the trainer's format ----------------------------------------
    mem = '<question_memory>{"timestep": "00:00:00", "content": "Q"}</question_memory>\n'
    ans = '<answer_memory>{"timestep": "00:00:01", "content": "At 1 seconds: A"}</answer_memory>'
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": mem}, {"type": "image", "timestamp": 0.0},
                                     {"type": "text", "text": "\n"}]},
        {"role": "assistant", "content": "<silence>"},
        {"role": "user", "content": [{"type": "image", "timestamp": 1.0}, {"type": "text", "text": "\n"}]},
        {"role": "assistant", "content": "<response>At 1 seconds: A"},
        {"role": "user", "content": [{"type": "image", "timestamp": 2.5},
                                     {"type": "text", "text": "\n" + ans + "\n"}]},
    ]
    vis = "<|vision_start|><|image_pad|><|vision_end|>"
    # aiak qwen2-vl: system / user "<|im_start|>user\n{c}<|im_end|>\n<|im_start|>assistant\n" /
    # assistant "{c}<|im_end|>" / separator "\n"; canvas puts "<T.T seconds>" before each frame
    expect = (f"<|im_start|>system\n{system}<|im_end|>\n"
              f"<|im_start|>user\n{mem}<0.0 seconds>{vis}\n<|im_end|>\n<|im_start|>assistant\n<silence><|im_end|>\n"
              f"<|im_start|>user\n<1.0 seconds>{vis}\n<|im_end|>\n<|im_start|>assistant\n"
              f"<response>At 1 seconds: A<|im_end|>\n"
              f"<|im_start|>user\n<2.5 seconds>{vis}\n{ans}\n<|im_end|>\n<|im_start|>assistant\n")
    loaded = getattr(tok, "chat_template", None)
    if loaded is not None and loaded != template:
        problems.append("the tokenizer loads a different chat template than chat_template.jinja")
    rendered = tok.apply_chat_template(msgs, chat_template=template, tokenize=False, add_generation_prompt=True,
                                       streaming=True)
    if rendered != expect:
        problems.append(f"streaming render differs from the training format:\n  got  {rendered!r}\n  want {expect!r}")
    else:
        print("[streaming] streaming=True render matches the training format")

    # offline: same turns, qwen2-vl default system; video frames keep their timestamps, a single
    # image has none; an explicit system message wins in both modes
    off_msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Q1\n"}, {"type": "image", "timestamp": 0.0},
                                     {"type": "text", "text": "\n"}, {"type": "image", "timestamp": 1.0},
                                     {"type": "text", "text": "\n"}]},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": " Q2"}]},
    ]
    off_expect = (f"<|im_start|>system\n{offline}<|im_end|>\n"
                  f"<|im_start|>user\nQ1\n<0.0 seconds>{vis}\n<1.0 seconds>{vis}\n<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n"
                  f"<|im_start|>user\n{vis} Q2<|im_end|>\n<|im_start|>assistant\n")
    for kwargs, label in (({}, "offline (default)"), ({"streaming": False}, "streaming=False")):
        got = tok.apply_chat_template(off_msgs, chat_template=template, tokenize=False,
                                      add_generation_prompt=True, **kwargs)
        if got != off_expect:
            problems.append(f"{label} render differs from the training format:\n  got  {got!r}\n  want {off_expect!r}")
    sys_msgs = [{"role": "system", "content": "S"}] + off_msgs[:1]
    for kwargs in ({}, {"streaming": True}):
        got = tok.apply_chat_template(sys_msgs, chat_template=template, tokenize=False, **kwargs)
        if not got.startswith("<|im_start|>system\nS<|im_end|>\n") or got.count("<|im_start|>system") != 1:
            problems.append(f"explicit system message not honoured ({kwargs})")
    if not any(p.startswith(("offline", "streaming=False", "explicit")) for p in problems):
        print("[streaming] offline render matches the training format; explicit system wins in both modes")

    if problems:
        print("[streaming] FAILED:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    print("[streaming] export finalized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
