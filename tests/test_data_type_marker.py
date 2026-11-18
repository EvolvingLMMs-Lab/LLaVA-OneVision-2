"""Sample-level ``data_type`` marker: jsonl -> webdataset -> encoder.

Streaming samples carry ``data_type="streaming"`` so the encoder can give them
their own system prompt. The marker is positional in packed bins (one entry per
sub-sample), which is where a silent off-by-one would be most damaging: every
sub-sample would get a prompt describing the wrong modality. These tests pin the
alignment, the drop-on-mismatch guard, and the untouched behaviour of data that
predates the marker.

Runs as a plain script from the repo root. pytest collection trips over torch
inductor's subprocess pool inside the training image, so invoke it directly::

  R=<repo root>
  docker run --rm -v /vlm:/vlm -w $R -e PYTHONPATH=$R/aiak_megatron:$R \
      llava_megatron:26.07 python tests/test_data_type_marker.py
"""

import os
import sys

os.environ.setdefault("OFFLINE_PACKING_BMR", "1")
os.environ.pop("BMR_BIN_BLOCKLIST", None)
os.environ.pop("VISION_MAXNUMEL", None)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools", "data_preprocess"))
sys.path.insert(0, os.path.join(REPO, "offline_packing"))

from aiak_training_llm.data.multimodal.flavors import (  # noqa: E402
    MultiMixQASample,
    PackedCaptioningSample,
)
from aiak_training_llm.data.multimodal.qwen2vl_task_encoder import (  # noqa: E402
    DATA_TYPE_STREAMING,
    STREAMING_SYSTEM_PROMPT,
    resolve_system_prompt,
)
from aiak_training_llm.data.multimodal.task_encoder import TaskEncoder  # noqa: E402

FAILURES = []


def check(label, got, want):
    if got == want:
        print(f"  pass  {label}")
    else:
        print(f"  FAIL  {label}\n          got  {got!r}\n          want {want!r}")
        FAILURES.append(label)


# --------------------------------------------------------------------------- 1
def test_resolve_system_prompt():
    print("1. resolve_system_prompt")
    check("streaming -> streaming prompt",
          resolve_system_prompt(None, DATA_TYPE_STREAMING), STREAMING_SYSTEM_PROMPT)
    check("no marker -> None (template default_system applies)",
          resolve_system_prompt(None, None), None)
    check("unknown marker -> None",
          resolve_system_prompt(None, "offline"), None)
    check("explicit system overrides the marker",
          resolve_system_prompt("CUSTOM", DATA_TYPE_STREAMING), "CUSTOM")
    check("empty string is not an override",
          resolve_system_prompt("", DATA_TYPE_STREAMING), STREAMING_SYSTEM_PROMPT)


# --------------------------------------------------------------------------- 2
class _Probe(TaskEncoder):
    """Records the per-sub-sample samples the packed branch builds."""

    def __init__(self):
        self.seen = []

    def encode_multi_mix_qa(self, sample):
        self.seen.append(sample)
        return sample

    def encode_multi_vid_qa(self, sample):  # abstract on the base class, unused here
        raise AssertionError("packed bins must not reach the video path")

    def pack_selected_samples(self, samples):
        return samples


def _bin(data_type, n=3):
    return PackedCaptioningSample(
        __key__="bin_0",
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__={},
        images=[[b"jpg"] for _ in range(n)],
        prompts=[["<image>\nq"] for _ in range(n)],
        captions=[["a"] for _ in range(n)],
        patch_positions=None,
        fps=[None] * n,
        timestamp_decimal=[1] * n,
        data_type=data_type,
    )


def _run(sample):
    return list(TaskEncoder.encode_sample.__wrapped__(_Probe(), sample)), None


def _seen(sample):
    probe = _Probe()
    out = list(TaskEncoder.encode_sample.__wrapped__(probe, sample))
    return probe.seen, out


def test_packed_alignment():
    print("2. packed bin: marker stays with its own sub-sample")
    seen, out = _seen(_bin([None, DATA_TYPE_STREAMING, None]))
    check("all 3 sub-samples encoded", len(seen), 3)
    check("bin yielded", len(out), 1)
    check("sub-sample data_type order",
          [s.data_type for s in seen], [None, DATA_TYPE_STREAMING, None])
    check("resolved system per sub-sample",
          [resolve_system_prompt(s.system, s.data_type) for s in seen],
          [None, STREAMING_SYSTEM_PROMPT, None])
    check("every sub-sample is a MultiMixQASample",
          all(isinstance(s, MultiMixQASample) for s in seen), True)

    print("   all-streaming bin")
    seen, _ = _seen(_bin([DATA_TYPE_STREAMING] * 3))
    check("all three marked",
          [s.data_type for s in seen], [DATA_TYPE_STREAMING] * 3)


# --------------------------------------------------------------------------- 3
def test_packed_length_guard():
    print("3. packed bin: length mismatch drops the bin instead of misaligning")
    for bad in ([DATA_TYPE_STREAMING, None], [None] * 4):
        seen, out = _seen(_bin(bad))
        check(f"data_type len={len(bad)} vs 3 sub-samples -> nothing yielded", out, [])
        check(f"data_type len={len(bad)} -> nothing encoded", seen, [])


# --------------------------------------------------------------------------- 4
def test_backward_compatible():
    print("4. bins written before the marker existed are unchanged")
    seen, out = _seen(_bin(None))
    check("bin still yields", len(out), 1)
    check("every sub-sample data_type is None", [s.data_type for s in seen], [None] * 3)
    check("every resolved system is None",
          [resolve_system_prompt(s.system, s.data_type) for s in seen], [None] * 3)


# --------------------------------------------------------------------------- 5
def _exec_loader(code):
    ns = {}
    exec(compile(code, "<generated sample_loader.py>", "exec"), ns)  # noqa: S102
    return ns["sample_loader"]


def test_generated_loaders():
    print("5. generated sample_loader.py carries data_type through")

    from convert_jsonl_to_webdataset_simple import sample_loader_template
    from s5_convert_to_webdataset import generate_sample_loader_code

    loader = _exec_loader(sample_loader_template())
    marked = loader({
        "__key__": "k", "__restore_key__": (),
        "json": {"messages": [{"role": "user", "content": "hi"}], "image_keys": [],
                 "fps": 30.0, "timestamp_decimal": 1, "data_type": "streaming"},
    })
    check("non-packed loader: data_type", marked.get("data_type"), "streaming")
    plain = loader({
        "__key__": "k", "__restore_key__": (),
        "json": {"messages": [], "image_keys": [], "fps": 30.0},
    })
    check("non-packed loader: absent -> None", plain.get("data_type"), None)

    packed_loader = _exec_loader(generate_sample_loader_code())
    marked = packed_loader({
        "__key__": "b", "__restore_key__": (), "img0_0.jpg": b"x",
        "json": {"images": [["a.jpg"]], "prompts": [["p"]], "captions": [["c"]],
                 "data_type": ["streaming"]},
    })
    check("packed loader: data_type list", marked.get("data_type"), ["streaming"])
    plain = packed_loader({
        "__key__": "b", "__restore_key__": (), "img0_0.jpg": b"x",
        "json": {"images": [["a.jpg"]], "prompts": [["p"]], "captions": [["c"]]},
    })
    check("packed loader: absent -> key omitted", "data_type" in plain, False)


# --------------------------------------------------------------------------- 6
def _manifest_entry(img_path, data_type):
    """A minimal BMR manifest entry; the marker key is absent when data_type is None."""
    raw = {
        "messages": [{"role": "user", "content": "<image>\nq"},
                     {"role": "assistant", "content": "a"}],
        "fps": [30],
        "timestamp_decimal": 1,
    }
    if data_type is not None:
        raw["data_type"] = data_type
    return {"raw": raw, "images_abs": [img_path], "patch_pos_abs": [""]}


def _manifest_payload(markers):
    """Run the manifest writer over one bin and return the payload it wrote."""
    import json
    import tempfile
    from s4_bins_to_webdataset import build_direct_sample_from_manifest

    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "a.jpg")
        with open(img, "wb") as f:
            f.write(b"bytes are copied verbatim without an image policy")
        index = {f"s{i}": _manifest_entry(img, m) for i, m in enumerate(markers)}
        sample = build_direct_sample_from_manifest(
            0, [{"name": f"s{i}"} for i in range(len(markers))], index)
    return json.loads(sample["json"])


def _packer_payload(markers):
    """Run SamplePacker.process_bin over the same bin and return the payload it wrote."""
    import json
    import tempfile
    from s4_pack_samples import PackSamplesConfig, SamplePacker

    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
        with open(os.path.join(src, "a.jpg"), "wb") as f:
            f.write(b"bytes are copied verbatim")
        for i, marker in enumerate(markers):
            doc = {
                "images": ["a.jpg"],
                "messages": [{"role": "user", "content": "<image>\nq"},
                             {"role": "assistant", "content": "a"}],
                "fps": [30],
                "timestamp_decimal": 1,
            }
            if marker is not None:
                doc["data_type"] = marker
            with open(os.path.join(src, f"s{i}.json"), "w", encoding="utf-8") as f:
                json.dump(doc, f)
        packer = SamplePacker(PackSamplesConfig(
            source_dir=src, image_base_dir=src, output_dir=out,
            task_type="bmr", clear_output_dir=False))
        packer.process_bin(0, [{"name": f"s{i}"} for i in range(len(markers))])
        with open(os.path.join(out, "ps_00000000.json"), encoding="utf-8") as f:
            return json.load(f)


def test_packed_writer_payload():
    print("6. packed writers emit data_type positionally, and only when marked")
    for label, payload_of in (("manifest writer (s4_bins_to_webdataset)", _manifest_payload),
                              ("sample packer (s4_pack_samples)", _packer_payload)):
        print(f"   {label}")

        # A bin whose entries never had the key must come out exactly as it used to.
        plain = payload_of([None, None, None])
        check("unmarked bin omits the key", "data_type" in plain, False)

        # Positional: the marker has to land on the sub-sample it came from.
        mixed = payload_of([None, DATA_TYPE_STREAMING, None])
        check("marker keeps its position",
              mixed.get("data_type"), [None, DATA_TYPE_STREAMING, None])
        check("all-marked bin", payload_of([DATA_TYPE_STREAMING] * 3).get("data_type"),
              [DATA_TYPE_STREAMING] * 3)

        # Adding the marker must not disturb anything else in the payload.
        check("the rest of the payload is unchanged by the marker",
              {k: v for k, v in mixed.items() if k != "data_type"}, plain)
        # Both writers record one prompt list per sub-sample; the marker must match it.
        check("one entry per sub-sample",
              len(mixed["data_type"]), len(mixed["prompts"]))

    print("   extractor tuple width (s4_pack_samples._extract_bmr_content)")
    import json
    import tempfile
    import types
    from s4_pack_samples import SamplePacker as _SP

    with tempfile.TemporaryDirectory() as d:
        fake = types.SimpleNamespace(image_base_dir=d)
        base = {"images": ["a.jpg"],
                "messages": [{"role": "user", "content": "<image>\nq"},
                             {"role": "assistant", "content": "a"}]}
        for name, marker in (("plain.json", None), ("marked.json", DATA_TYPE_STREAMING)):
            doc = dict(base)
            if marker is not None:
                doc["data_type"] = marker
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                json.dump(doc, f)
        plain = _SP._extract_bmr_content(fake, os.path.join(d, "plain.json"), image_base_dir=d)
        marked = _SP._extract_bmr_content(fake, os.path.join(d, "marked.json"), image_base_dir=d)
        missing = _SP._extract_bmr_content(fake, os.path.join(d, "nope.json"), image_base_dir=d)

    check("plain sample -> no marker", plain[-1], None)
    check("marked sample -> marker", marked[-1], DATA_TYPE_STREAMING)
    # A short error return would unpack into the caller and fail somewhere unrelated.
    check("error return has the same width as the success path",
          len(missing), len(plain))
    check("error return carries no marker", missing[-1], None)


for fn in (test_resolve_system_prompt, test_packed_alignment, test_packed_length_guard,
           test_backward_compatible, test_generated_loaders, test_packed_writer_payload):
    fn()
    print()

if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
    sys.exit(1)
print("ALL TESTS PASSED")
