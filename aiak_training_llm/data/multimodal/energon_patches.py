"""Runtime fixes for megatron-energon, applied before the train dataset is built.

Per-dataset last-sample retention (energon 5.0.x)
-------------------------------------------------
BlendDataset keeps one suspended iterator chain per blended dataset. While a
chain is suspended at ``yield``, the generator frames of
``RepeatDataset.__iter__`` (local ``sample``) and ``MapDataset.__iter__``
(locals ``sample`` = raw tar bytes, ``mapped_sample`` = decoded sample) keep that
dataset's most recent sample alive, including every decoded PIL image, until
the blend picks the same dataset again. With N blended datasets, each dataloader
worker therefore holds roughly N decoded samples, and its RSS grows with the
number of distinct datasets it has touched.

On a 408-dataset blend this took each worker from ~17 GB to ~95 GB (~700 GB per
8-worker node), starved the host of free memory and halved training throughput.
``gc.collect()`` cannot help (these are strong references, not cycles), and no
allocator setting helps (the objects are genuinely alive).

The fix hands each yielded value out through the evaluation stack
(``box = [next(it)]; yield box.pop()``) and drops locals before yielding, so a
suspended frame references nothing. Sample content, order, and restore keys are
unchanged; the save/restore state fields are the same ones energon uses.

The patch re-implements energon internals, so it is only applied to the energon
version it was written against. Set ``ENERGON_LEAKFIX=0`` to disable it.

BlendDataset restore index (energon 5.0.x)
-----------------------------------------
``BlendDataset.__iter__`` drops the datasets this worker has no samples for and
numbers each sample by its position in that *filtered* list, but
``restore_sample`` looks the number up in the *full* ``dataset_weights`` list.
On a blend with many tiny datasets (single-shard codec parts are empty for most
of 192 workers) the two lists differ by dozens of entries, so restoring an
in-flight sample after a resume picks the wrong dataset: ``KeyError`` on its
shard name, or a silently wrong sample when another dataset happens to have a
shard of the same name. The fix maps the number back through the same filter.
Saved states need no conversion. Set ``ENERGON_BLEND_RESTORE_FIX=0`` to disable it.
"""

import inspect
import os
import warnings
from importlib import metadata
from typing import Generator

_SUPPORTED_ENERGON = ("5.0.",)
_applied = False


def _repeat_iter(self):
    worker_idx = self.worker_config.rank_worker_id()
    assert (
        self.repeats is not None or self.dataset.worker_has_samples()
    ), "Cannot repeat empty dataset indefinitely"
    while self.repeats is None or self._repetition[worker_idx] < self.repeats:
        it = iter(self.dataset)
        while True:
            try:
                box = [next(it)]
            except StopIteration:
                break
            yield box.pop()
        del it
        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "RepeatDataset.repeat",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "offset": self._repetition[worker_idx],
                    "repeats": self.repeats,
                }
            )
        self._repetition[worker_idx] += 1
    if self.restart:
        self._repetition[worker_idx] = 0
    else:
        # No more repeats
        self._repetition[worker_idx] = self.repeats


def _make_map_iter(orig_map_iter):
    from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
    from megatron.energon.flavors.base_dataset import add_sample_restore_key
    from megatron.energon.wrappers import map_dataset as md
    from megatron.energon.wrappers.skip import SkipSample

    get_sample_restore_key = md.get_sample_restore_key

    def _map_iter(self):
        worker_id = self.worker_config.rank_worker_id()
        # Rare path: resuming in the middle of a generator map_fn. Keep energon's own
        # implementation; it only runs once, right after a state restore.
        if self._generator_sample_keys[worker_id] is not None:
            yield from orig_map_iter(self)
            return
        for sample in self.dataset:
            box = []
            skey = sample  # handed to the error handler on failure; dropped before yield
            try:
                with self._sample_index.ctx() as sample_idx:
                    mapped_sample = self.map_fn(sample)
                if isinstance(mapped_sample, Generator):
                    assert inspect.isgeneratorfunction(
                        self.map_fn
                    ), f"Generator in {self.map_fn} but not marked as such."
                    self._generator_sample_keys[worker_id] = get_sample_restore_key(sample)
                    self._generator_offsets[worker_id] = 0
                    gen = self._sample_index.iter_ctx(mapped_sample, sample_idx)
                    del mapped_sample, sample, skey
                    idx = 0
                    while True:
                        try:
                            inner_idx, inner = next(gen)
                        except StopIteration:
                            break
                        self._generator_offsets[worker_id] = idx + 1
                        box.append(add_sample_restore_key(inner, inner_idx, idx, src=self))
                        del inner
                        idx += 1
                        yield box.pop()
                    del gen
                    self._generator_sample_keys[worker_id] = None
                    self._generator_offsets[worker_id] = None
                else:
                    box.append(add_sample_restore_key(mapped_sample, sample_idx, src=self))
                    del mapped_sample, sample, skey
                    yield box.pop()
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(locals().get("skey"))
            except Exception as e:
                self.error_handler(e, locals().get("skey"))

    return _map_iter


def _blend_restore_sample(self, index):
    from megatron.energon.flavors.base_dataset import add_sample_restore_key

    name, ds_idx = index[:2]
    assert name == type(self).__name__
    assert isinstance(ds_idx, int)
    # __iter__ numbers samples by their position among the datasets this worker has
    # samples for; map that position back to the full dataset_weights list.
    active = [i for i, (dataset, _) in enumerate(self.dataset_weights) if dataset.worker_has_samples()]
    return add_sample_restore_key(
        self.dataset_weights[active[ds_idx]][0].restore_sample(index[2:]),
        ds_idx,
        src=self,
    )


def apply_energon_patches():
    """Apply the energon runtime fixes once per process (idempotent).

    Must run before the dataloader forks its workers; the workers inherit the
    patched classes.
    """
    global _applied
    if _applied:
        return
    try:
        version = metadata.version("megatron-energon")
    except metadata.PackageNotFoundError:
        version = "unknown"
    if not version.startswith(_SUPPORTED_ENERGON):
        warnings.warn(
            f"energon_patches: megatron-energon {version} is not a patched version "
            f"{_SUPPORTED_ENERGON}; per-dataset sample-retention fix NOT applied. "
            "Check whether the new release still holds samples in suspended "
            "RepeatDataset/MapDataset frames before extending the version list."
        )
        _applied = True
        return

    from megatron.energon.wrappers import blend_dataset as bd
    from megatron.energon.wrappers import map_dataset as md
    from megatron.energon.wrappers import repeat_dataset as rd

    if os.environ.get("ENERGON_LEAKFIX", "1") != "0":
        rd.RepeatDataset.__iter__ = _repeat_iter
        md.MapDataset.__iter__ = _make_map_iter(md.MapDataset.__iter__)
        print(
            f"[energon_patches] megatron-energon {version}: RepeatDataset/MapDataset "
            "__iter__ patched (no per-dataset sample retention)",
            flush=True,
        )
    if os.environ.get("ENERGON_BLEND_RESTORE_FIX", "1") != "0":
        bd.BlendDataset.restore_sample = _blend_restore_sample
        print(
            f"[energon_patches] megatron-energon {version}: BlendDataset.restore_sample "
            "maps the per-worker dataset index back to the full list",
            flush=True,
        )
    _applied = True
