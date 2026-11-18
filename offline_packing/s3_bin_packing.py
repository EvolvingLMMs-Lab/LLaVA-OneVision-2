#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bin Packing for Token-based Sample Grouping.

This script implements bin packing algorithms to group training samples into
fixed-capacity bins for efficient batch construction during SFT training.
It supports First Fit Decreasing (FFD) and Best Fit Decreasing (BFD) algorithms.

The packing process optimizes GPU memory utilization by minimizing padding waste
when combining multiple samples into a single training batch.

Usage:
    python s3_bin_packing.py \\
        --input token_info.txt \\
        --output bins.pkl \\
        --capacity 16000 \\
        --max-samples 10 \\
        --algorithm bfd

Input Format (token_info.txt):
    sample_name_1:1234
    sample_name_2:5678
    ...

    Output Format (bins.pkl):
        List of numpy structured arrays, each representing a bin:
        [
            array([(w, l, name), ...], dtype=[('w', '<u2'), ('l', '<u4'), ('name', '<U200')]),
            ...
        ]

Author: LLaVA-OneVision Team
License: Apache-2.0
"""

import argparse
import os
import pickle
import re
import sys
import time
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    from sortedcontainers import SortedList

    _HAS_SORTED_CONTAINERS = True
except ImportError:
    _HAS_SORTED_CONTAINERS = False


class BinPacker:
    """
    Token-based bin packer for SFT training sample grouping.

    This class implements bin packing algorithms to efficiently group
    training samples with varying token lengths into fixed-capacity bins.

    Attributes:
        capacity: Maximum token capacity per bin.
        max_samples_per_bin: Maximum number of samples allowed per bin.
        token_data: Dictionary mapping sample names to token counts.
    """

    BIN_DTYPE = np.dtype(
        [
            ("w", "<u2"),  # Width (reserved, set to 0)
            ("l", "<u4"),  # Length (token count)
            ("name", "<U200"),  # Sample name
            ("p", "<u4"),  # Patch count (for ViT load balancing); 0 if unknown
        ]
    )

    def __init__(
        self,
        capacity: int = 16000,
        max_samples_per_bin: int = 10,
        max_patches_per_bin: int = 0,    # 0 = disabled
        min_patches_per_bin: int = 0,    # 0 = disabled
    ) -> None:
        """
        Initialize the bin packer.

        Args:
            capacity: Maximum token capacity per bin.
            max_samples_per_bin: Maximum samples allowed per bin.
            max_patches_per_bin: Upper bound on per-bin ViT patches (0 disables).
            min_patches_per_bin: Lower bound on per-bin ViT patches (0 disables).
        """
        self.capacity = capacity
        self.max_samples_per_bin = max_samples_per_bin
        self.max_patches_per_bin = max_patches_per_bin
        self.min_patches_per_bin = min_patches_per_bin
        # token_data: name -> (token_count, patch_count)
        self.token_data: dict[str, tuple[int, int]] = {}

    def parse_token_info(self, file_path: str) -> None:
        """
        Parse token information file.

        Args:
            file_path: Path to the token info file.
                Format: sample_name:token_count (one per line)

        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        print(f"📖 Parsing token info file: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Token info file not found: {file_path}")

        # Accept both legacy (name:token) and new (name:token:patch) formats.
        pattern = re.compile(r"^(.+?):(\d+)(?::(\d+))?$")
        total_lines = 0
        parse_errors = 0
        with_patch = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                total_lines += 1
                match = pattern.match(line)
                if match:
                    sample_name = match.group(1)
                    token_count = int(match.group(2))
                    patch_count = int(match.group(3)) if match.group(3) is not None else 0
                    if match.group(3) is not None:
                        with_patch += 1
                    self.token_data[sample_name] = (token_count, patch_count)
                else:
                    parse_errors += 1
                    if parse_errors <= 5:
                        print(f"   ⚠️ Parse error at line {total_lines}: {line[:50]}...")

                if total_lines % 100000 == 0:
                    print(f"   Processed {total_lines:,} lines...")

        print(f"✅ Parsing complete: {len(self.token_data):,} samples loaded "
              f"({with_patch:,} with patch_count)")
        if parse_errors > 0:
            print(f"   ⚠️ {parse_errors} lines failed to parse")

    def analyze_distribution(self) -> dict[str, Any]:
        """
        Analyze token length distribution.

        Returns:
            Dictionary containing distribution statistics.
        """
        print("\n📊 Token Distribution Analysis")

        if not self.token_data:
            print("   ❌ No data to analyze")
            return {}

        tokens = [t for t, _ in self.token_data.values()]
        patches = [p for _, p in self.token_data.values()]
        total_tokens = sum(tokens)
        avg_tokens = total_tokens / len(tokens)
        max_tokens = max(tokens)
        min_tokens = min(tokens)

        stats = {
            "total_samples": len(self.token_data),
            "total_tokens": total_tokens,
            "avg_tokens": avg_tokens,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "total_patches": sum(patches),
            "avg_patches": sum(patches) / len(patches) if patches else 0,
            "max_patches": max(patches) if patches else 0,
        }

        print(f"   Total samples: {stats['total_samples']:,}")
        print(f"   Total tokens:  {stats['total_tokens']:,}")
        print(f"   Average:       {stats['avg_tokens']:.1f}")
        print(f"   Max:           {stats['max_tokens']:,}")
        print(f"   Min:           {stats['min_tokens']:,}")
        if stats["total_patches"]:
            print(f"   Total patches: {stats['total_patches']:,}")
            print(f"   Avg patches:   {stats['avg_patches']:.1f}")
            print(f"   Max patches:   {stats['max_patches']:,}")

        # Distribution by size ranges
        ranges = [
            ("tiny", 0, 1000),
            ("small", 1000, 5000),
            ("medium", 5000, 10000),
            ("large", 10000, 15000),
            ("xlarge", 15000, 20000),
            ("xxlarge", 20000, float("inf")),
        ]

        print("\n   Size distribution:")
        for name, low, high in ranges:
            count = sum(1 for t in tokens if low < t <= high)
            pct = count / len(tokens) * 100
            label = f"{name} ({low // 1000}k-{high // 1000 if high != float('inf') else '∞'}k)"
            print(f"     {label:20s}: {count:8,} ({pct:5.1f}%)")
            stats[f"range_{name}"] = count

        return stats

    def pack_first_fit_decreasing(self) -> list[np.ndarray]:
        """
        Pack samples using First Fit Decreasing (FFD) algorithm.

        FFD sorts samples by token count in descending order, then places
        each sample into the first bin that has sufficient capacity.

        Returns:
            List of numpy structured arrays, each representing a bin.
        """
        print("\n📦 Packing with FFD algorithm...")
        print(f"   Capacity: {self.capacity:,} tokens")
        print(f"   Max samples per bin: {self.max_samples_per_bin}")

        # Sort by token count descending
        print("   Sorting samples by token count...")
        start_time = time.time()
        sorted_samples = sorted(self.token_data.items(), key=lambda x: x[1][0], reverse=True)
        print(f"   Sorting complete in {time.time() - start_time:.1f}s")

        bins: list[list[tuple[int, int, str, int]]] = []
        # Track current tokens for each bin to avoid repeated sum() calls
        bin_tokens: list[int] = []

        start_time = time.time()
        total_samples = len(sorted_samples)

        with tqdm(total=total_samples, desc="   Packing samples", unit="sample",
                  bar_format="   {l_bar}{bar:30}{r_bar}") as pbar:
            for sample_name, (token_count, patch_count) in sorted_samples:
                placed = False

                # Try to fit into existing bins
                for i, bin_samples in enumerate(bins):
                    current_tokens = bin_tokens[i]
                    current_count = len(bin_samples)

                    if current_tokens + token_count <= self.capacity and current_count < self.max_samples_per_bin:
                        bin_samples.append((0, token_count, sample_name, patch_count))
                        bin_tokens[i] += token_count
                        placed = True
                        break

                # Create new bin if needed
                if not placed:
                    bins.append([(0, token_count, sample_name, patch_count)])
                    bin_tokens.append(token_count)

                pbar.update(1)
                # Update postfix with current stats
                if pbar.n % 10000 == 0:
                    pbar.set_postfix(bins=len(bins), elapsed=f"{time.time() - start_time:.0f}s")

        elapsed = time.time() - start_time
        print(f"✅ Packing complete: {len(bins):,} bins created in {elapsed:.1f}s")
        print(f"   Speed: {total_samples / elapsed:.0f} samples/sec")

        return self._convert_to_numpy(bins)

    def pack_best_fit_decreasing(self) -> list[np.ndarray]:
        """
        Pack samples using Best Fit Decreasing (BFD) algorithm.

        BFD sorts samples by token count in descending order, then places
        each sample into the bin with minimum remaining space after placement.

        Returns:
            List of numpy structured arrays, each representing a bin.
        """
        print("\n📦 Packing with BFD algorithm...")
        print(f"   Capacity: {self.capacity:,} tokens")
        print(f"   Max samples per bin: {self.max_samples_per_bin}")

        # Sort by token count descending
        print("   Sorting samples by token count...")
        start_time = time.time()
        sorted_samples = sorted(self.token_data.items(), key=lambda x: x[1][0], reverse=True)
        print(f"   Sorting complete in {time.time() - start_time:.1f}s")

        bins: list[list[tuple[int, int, str, int]]] = []
        # Track current tokens for each bin to avoid repeated sum() calls
        bin_tokens: list[int] = []

        start_time = time.time()
        total_samples = len(sorted_samples)

        with tqdm(total=total_samples, desc="   Packing samples", unit="sample",
                  bar_format="   {l_bar}{bar:30}{r_bar}") as pbar:
            for sample_name, (token_count, patch_count) in sorted_samples:
                best_bin_idx = -1
                min_remaining = float("inf")

                # Find best fitting bin (minimum remaining space)
                for i, bin_samples in enumerate(bins):
                    current_tokens = bin_tokens[i]
                    current_count = len(bin_samples)

                    if current_tokens + token_count <= self.capacity and current_count < self.max_samples_per_bin:
                        remaining = self.capacity - (current_tokens + token_count)
                        if remaining < min_remaining:
                            min_remaining = remaining
                            best_bin_idx = i

                # Place in best bin or create new one
                if best_bin_idx >= 0:
                    bins[best_bin_idx].append((0, token_count, sample_name, patch_count))
                    bin_tokens[best_bin_idx] += token_count
                else:
                    bins.append([(0, token_count, sample_name, patch_count)])
                    bin_tokens.append(token_count)

                pbar.update(1)
                # Update postfix with current stats
                if pbar.n % 10000 == 0:
                    pbar.set_postfix(bins=len(bins), elapsed=f"{time.time() - start_time:.0f}s")

        elapsed = time.time() - start_time
        print(f"✅ Packing complete: {len(bins):,} bins created in {elapsed:.1f}s")
        print(f"   Speed: {total_samples / elapsed:.0f} samples/sec")

        return self._convert_to_numpy(bins)

    def _convert_to_numpy(self, bins: list[list[tuple[int, int, str]]]) -> list[np.ndarray]:
        """
        Convert bin lists to numpy structured arrays.

        Args:
            bins: List of bins, each containing tuples of (w, l, name).

        Returns:
            List of numpy structured arrays.
        """
        numpy_bins = []
        for bin_samples in bins:
            if bin_samples:
                bin_array = np.array(bin_samples, dtype=self.BIN_DTYPE)
                numpy_bins.append(bin_array)
        return numpy_bins

    def _pack_optimized(self, algorithm: str) -> list[np.ndarray]:
        """
        Optimized bin packing using sorted containers for O(n log n) performance.

        Instead of scanning all bins linearly [O(n × bins)], uses a SortedList
        to maintain bins sorted by remaining capacity, enabling O(log bins)
        lookup per sample.  Total complexity: O(n log n).

        Both FFD and BFD use tightest-fit selection (smallest remaining capacity
        that still fits the sample), which provides BFD-quality results.

        Requires: pip install sortedcontainers

        Args:
            algorithm: 'ffd' or 'bfd' (both use tightest-fit with sorted lookup).

        Returns:
            List of numpy structured arrays, each representing a bin.
        """
        from sortedcontainers import SortedList

        algo_name = algorithm.upper()
        print(f"\n📦 Packing with {algo_name} algorithm (⚡ optimized O(n log n) mode)...")
        print(f"   Capacity: {self.capacity:,} tokens")
        print(f"   Max samples per bin: {self.max_samples_per_bin}")
        if self.max_patches_per_bin:
            print(f"   Max patches per bin: {self.max_patches_per_bin:,}")
        if self.min_patches_per_bin:
            print(f"   Min patches per bin: {self.min_patches_per_bin:,}")

        # Sort primarily by token count, then by patch count, both descending.
        # This places "heavy in both dims" items first so subsequent fills can patch up the slack.
        print("   Sorting samples by token/patch count...")
        t0 = time.time()
        # When patch_cap is active, sort by (patch desc, token desc) so the
        # heaviest-ViT samples anchor bins first; lighter / zero-patch samples
        # then slot into remaining patch/token room. When no patch_cap, fall
        # back to token-primary ordering for classic BFD.
        if self.max_patches_per_bin:
            sort_key = lambda kv: (kv[1][1], kv[1][0])  # (patch, token)
        else:
            sort_key = lambda kv: (kv[1][0], kv[1][1])  # (token, patch)
        sorted_samples = sorted(
            self.token_data.items(),
            key=sort_key,
            reverse=True,
        )
        print(f"   Sorting complete in {time.time() - t0:.1f}s")

        bins: list[list[tuple[int, int, str, int]]] = []
        bin_tokens: list[int] = []
        bin_patches: list[int] = []
        bin_counts: list[int] = []

        # available: SortedList of (remaining_tokens, bin_index).
        # Bins here still have BOTH token room AND patch room (or no patch_cap).
        available = SortedList()
        # patch_full_avail: bins whose patch_cap is exhausted but still have
        # token room. Only zero-patch (text-only) samples may go here.
        patch_full_avail = SortedList()

        patch_cap = self.max_patches_per_bin or 0  # 0 = no cap
        patch_floor = self.min_patches_per_bin or 0

        start_time = time.time()
        total_samples = len(sorted_samples)
        patch_overflow_skips = 0
        zero_patch_into_full = 0  # text-only samples placed into patch-full bins

        with tqdm(
            total=total_samples,
            desc="   Packing samples",
            unit="sample",
            bar_format="   {l_bar}{bar:30}{r_bar}",
        ) as pbar:
            for sample_name, (token_count, patch_count) in sorted_samples:
                placed = False

                # Search candidates: walk from tightest-fit forward until we find
                # one that also satisfies the patch_cap (if enabled).
                pos = available.bisect_left((token_count,))

                # Snapshot scanning indices we touched to put back rejected candidates.
                rejected: list[tuple[int, int]] = []
                while pos < len(available):
                    rem, bidx = available[pos]
                    if patch_cap and bin_patches[bidx] + patch_count > patch_cap:
                        # Patch overflow — move bin to patch_full_avail (still
                        # has token room for zero-patch samples) and skip.
                        rejected.append((rem, bidx, True))  # True = move to full pool
                        available.pop(pos)
                        patch_overflow_skips += 1
                        continue

                    # Accept this bin.
                    available.pop(pos)
                    bins[bidx].append((0, token_count, sample_name, patch_count))
                    bin_tokens[bidx] += token_count
                    bin_patches[bidx] += patch_count
                    bin_counts[bidx] += 1
                    new_rem = rem - token_count

                    if new_rem > 0 and bin_counts[bidx] < self.max_samples_per_bin:
                        if not patch_cap or bin_patches[bidx] < patch_cap:
                            available.add((new_rem, bidx))
                        else:
                            # Patch exhausted — keep bin alive for zero-patch samples.
                            patch_full_avail.add((new_rem, bidx))
                    placed = True
                    break

                # Restore rejected candidates. Patch-overflow ones go to the
                # zero-patch-only pool; pure scan-bypass ones go back to available.
                for entry in rejected:
                    rem, bidx, to_full = entry
                    if to_full:
                        patch_full_avail.add((rem, bidx))
                    else:
                        available.add((rem, bidx))

                # If still not placed AND this sample is zero-patch, try the
                # patch-full pool: any bin with token room accepts it.
                if not placed and patch_count == 0 and patch_full_avail:
                    pos2 = patch_full_avail.bisect_left((token_count,))
                    if pos2 < len(patch_full_avail):
                        rem, bidx = patch_full_avail.pop(pos2)
                        bins[bidx].append((0, token_count, sample_name, patch_count))
                        bin_tokens[bidx] += token_count
                        bin_counts[bidx] += 1
                        new_rem = rem - token_count
                        if new_rem > 0 and bin_counts[bidx] < self.max_samples_per_bin:
                            patch_full_avail.add((new_rem, bidx))
                        zero_patch_into_full += 1
                        placed = True

                if not placed:
                    new_idx = len(bins)
                    bins.append([(0, token_count, sample_name, patch_count)])
                    bin_tokens.append(token_count)
                    bin_patches.append(patch_count)
                    bin_counts.append(1)
                    new_rem = self.capacity - token_count
                    if new_rem > 0 and 1 < self.max_samples_per_bin:
                        if not patch_cap or patch_count < patch_cap:
                            available.add((new_rem, new_idx))
                        else:
                            # Oversized singleton bin: patch already over cap,
                            # but token room remains → reserve for zero-patch fills.
                            patch_full_avail.add((new_rem, new_idx))

                pbar.update(1)
                if pbar.n % 50000 == 0:
                    pbar.set_postfix(
                        bins=len(bins),
                        avail=len(available),
                        skips=patch_overflow_skips,
                        zfill=zero_patch_into_full,
                        elapsed=f"{time.time() - start_time:.0f}s",
                    )

        elapsed = time.time() - start_time
        print(f"✅ Packing complete: {len(bins):,} bins created in {elapsed:.1f}s")
        print(f"   Speed: {total_samples / elapsed:.0f} samples/sec")
        print(f"   Patch-overflow placements rejected: {patch_overflow_skips:,}")
        print(f"   Zero-patch fills into patch-full bins: {zero_patch_into_full:,}")
        print(f"   Remaining available bins: {len(available)} (+{len(patch_full_avail)} patch-full)")

        # --- Post-pass: enforce min_patches_per_bin (best-effort).
        if patch_floor:
            self._enforce_patch_floor(bins, bin_tokens, bin_patches, patch_floor)

        return self._convert_to_numpy(bins)

    def _enforce_patch_floor(
        self,
        bins: list[list[tuple[int, int, str, int]]],
        bin_tokens: list[int],
        bin_patches: list[int],
        patch_floor: int,
    ) -> None:
        """Best-effort: try to lift bins below ``patch_floor`` by swapping in
        an item with patches from a 'rich' bin. Does not split samples.
        If patch_cap is also active we respect it on the receiving side.
        """
        cap = self.max_patches_per_bin or 0
        donor_order = sorted(range(len(bins)), key=lambda i: -bin_patches[i])
        donor_ptr = 0
        moved = 0
        for ridx in range(len(bins)):
            if bin_patches[ridx] >= patch_floor:
                continue
            # Find a donor whose removing one item still keeps it above floor,
            # and where the item fits in the receiver token-wise.
            for dpos in range(donor_ptr, len(donor_order)):
                didx = donor_order[dpos]
                if didx == ridx or bin_patches[didx] <= patch_floor:
                    continue
                # Pick the smallest-patch item from donor that still helps receiver.
                items = bins[didx]
                # candidates with patches > 0
                cand = sorted(
                    [(i, it) for i, it in enumerate(items) if it[3] > 0],
                    key=lambda x: x[1][3],
                )
                for i, it in cand:
                    w, tlen, name, pc = it
                    if bin_tokens[ridx] + tlen > self.capacity:
                        continue
                    if cap and bin_patches[ridx] + pc > cap:
                        continue
                    if bin_patches[didx] - pc < patch_floor:
                        continue
                    # Move
                    bins[ridx].append(it)
                    bin_tokens[ridx] += tlen
                    bin_patches[ridx] += pc
                    del bins[didx][i]
                    bin_tokens[didx] -= tlen
                    bin_patches[didx] -= pc
                    moved += 1
                    break
                if bin_patches[ridx] >= patch_floor:
                    break
        if moved:
            print(f"   Patch-floor pass: moved {moved} item(s) to lift underloaded bins.")

    def save_bins(self, bins: list[np.ndarray], output_path: str) -> None:
        """
        Save packed bins to a pickle file.

        Args:
            bins: List of numpy structured arrays.
            output_path: Path to save the pickle file.
        """
        print(f"\n💾 Saving bins to: {output_path}")

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, "wb") as f:
            pickle.dump(bins, f)

        print(f"✅ Saved {len(bins):,} bins")

        # Verify saved file
        self._verify_saved_file(output_path)

    def _verify_saved_file(self, file_path: str) -> bool:
        """
        Verify the saved pickle file can be loaded correctly.

        Args:
            file_path: Path to the pickle file.

        Returns:
            True if verification passes, False otherwise.
        """
        try:
            with open(file_path, "rb") as f:
                loaded_bins = pickle.load(f)

            print(f"✅ Verification passed: loaded {len(loaded_bins):,} bins")

            if loaded_bins:
                first_bin = loaded_bins[0]
                print(f"   First bin shape: {first_bin.shape}")
                print(f"   Dtype: {first_bin.dtype}")

                if len(first_bin) > 0:
                    sample = first_bin[0]
                    print(f"   Sample: w={sample['w']}, l={sample['l']}, name={sample['name'][:30]}...")

            return True

        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False

    def analyze_packing_results(self, bins: list[np.ndarray]) -> dict[str, Any]:
        """
        Analyze packing results and compute efficiency metrics.

        Args:
            bins: List of packed bins.

        Returns:
            Dictionary containing packing statistics.
        """
        print("\n📈 Packing Results Analysis")

        if not bins:
            print("   ❌ No bins to analyze")
            return {}

        total_bins = len(bins)
        total_samples = sum(len(b) for b in bins)
        total_tokens = sum(s["l"] for b in bins for s in b)

        # Utilization metrics
        total_capacity = total_bins * self.capacity
        utilization = (total_tokens / total_capacity) * 100
        avg_samples = total_samples / total_bins

        # Theoretical minimum bins
        all_tokens = sum(t for t, _ in self.token_data.values())
        min_bins_theoretical = all_tokens / self.capacity
        efficiency = (min_bins_theoretical / total_bins) * 100

        stats = {
            "total_bins": total_bins,
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "utilization": utilization,
            "avg_samples_per_bin": avg_samples,
            "theoretical_min_bins": min_bins_theoretical,
            "packing_efficiency": efficiency,
        }

        print(f"   Bins created:        {total_bins:,}")
        print(f"   Total samples:       {total_samples:,}")
        print(f"   Total tokens:        {total_tokens:,}")
        print(f"   Capacity utilization: {utilization:.1f}%")
        print(f"   Avg samples/bin:     {avg_samples:.1f}")
        print(f"   Theoretical min bins: {min_bins_theoretical:.1f}")
        print(f"   Packing efficiency:  {efficiency:.1f}%")

        # Utilization distribution
        utilizations = []
        for bin_data in bins:
            bin_tokens = sum(s["l"] for s in bin_data)
            util = (bin_tokens / self.capacity) * 100
            utilizations.append(util)

        utilizations.sort()

        print("\n   Bin utilization distribution:")
        ranges = [(0, 50), (50, 70), (70, 85), (85, 95), (95, 101)]
        for low, high in ranges:
            count = sum(1 for u in utilizations if low <= u < high)
            pct = count / len(utilizations) * 100
            label = f"{low}-{min(high, 100)}%"
            print(f"     {label:8s}: {count:8,} ({pct:5.1f}%)")
            stats[f"util_{low}_{min(high, 100)}"] = count

        # Patch load balance (ViT side) — only meaningful if patch counts are present.
        bin_patches = np.array([int(sum(int(s["p"]) for s in b)) for b in bins], dtype=np.int64)
        if bin_patches.sum() > 0:
            mean_p = float(bin_patches.mean())
            std_p = float(bin_patches.std())
            cv = std_p / mean_p if mean_p > 0 else 0.0
            p50 = float(np.percentile(bin_patches, 50))
            p90 = float(np.percentile(bin_patches, 90))
            p99 = float(np.percentile(bin_patches, 99))
            print("\n   ViT patch load balance (per bin):")
            print(f"     mean={mean_p:,.0f}  std={std_p:,.0f}  CV={cv:.3f}")
            print(f"     min={bin_patches.min():,}  max={bin_patches.max():,}")
            print(f"     p50={p50:,.0f}  p90={p90:,.0f}  p99={p99:,.0f}")
            print(f"     p99/p50 ratio = {p99 / max(1, p50):.2f}x")
            stats.update({
                "patch_mean": mean_p, "patch_std": std_p, "patch_cv": cv,
                "patch_p50": p50, "patch_p90": p90, "patch_p99": p99,
                "patch_min": int(bin_patches.min()), "patch_max": int(bin_patches.max()),
            })

        return stats


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bin packing for token-based sample grouping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to token info file (format: sample_name:token_count)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output pickle file",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=16000,
        help="Maximum token capacity per bin",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum samples per bin",
    )
    parser.add_argument(
        "--algorithm",
        default="bfd",
        choices=["ffd", "bfd"],
        help="Packing algorithm: ffd (First Fit Decreasing) or bfd (Best Fit Decreasing)",
    )
    parser.add_argument(
        "--max-patches-per-bin",
        type=int,
        default=0,
        help="Hard upper bound on per-bin ViT patches (0 = disabled).",
    )
    parser.add_argument(
        "--min-patches-per-bin",
        type=int,
        default=0,
        help="Soft lower bound on per-bin ViT patches (0 = disabled).",
    )
    parser.add_argument(
        "--patch-slack",
        type=float,
        default=0.0,
        help="If > 0 and --max-patches-per-bin is 0, derive cap from baseline "
             "as avg_patches_per_bin * (1 + slack).",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("=" * 70)
    print("🔧 Bin Packing Tool for SFT Training")
    print("=" * 70)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Capacity:    {args.capacity:,} tokens")
    print(f"Max samples: {args.max_samples}")
    print(f"Algorithm:   {args.algorithm.upper()}")
    print("=" * 70)

    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    # Create packer
    packer = BinPacker(
        capacity=args.capacity,
        max_samples_per_bin=args.max_samples,
        max_patches_per_bin=args.max_patches_per_bin,
        min_patches_per_bin=args.min_patches_per_bin,
    )

    # Parse token info
    packer.parse_token_info(args.input)

    # Analyze distribution
    dist_stats = packer.analyze_distribution()

    # If user requested auto patch cap via --patch-slack, derive it now.
    if args.patch_slack > 0 and packer.max_patches_per_bin == 0 and dist_stats.get("total_patches"):
        total_p = dist_stats["total_patches"]
        total_t = dist_stats["total_tokens"]
        # baseline bin count ≈ total_tokens / capacity (lower bound)
        baseline_bins = max(1, total_t / args.capacity)
        avg_patches = total_p / baseline_bins
        packer.max_patches_per_bin = int(avg_patches * (1.0 + args.patch_slack))
        print(f"\n   🤖 Auto patch cap: avg={avg_patches:.0f}, slack={args.patch_slack:.2f} "
              f"-> max_patches_per_bin={packer.max_patches_per_bin:,}")

    # Pack samples (use optimized O(n log n) algorithm if sortedcontainers is available)
    if _HAS_SORTED_CONTAINERS:
        bins = packer._pack_optimized(args.algorithm)
    else:
        print("\n⚠️  sortedcontainers not installed — using O(n²) fallback algorithm.")
        print("   For ~1000x speedup on large datasets:")
        print("   pip install sortedcontainers\n")
        if args.algorithm == "ffd":
            bins = packer.pack_first_fit_decreasing()
        else:
            bins = packer.pack_best_fit_decreasing()

    # Save results
    packer.save_bins(bins, args.output)

    # Analyze results
    packer.analyze_packing_results(bins)

    print("\n" + "=" * 70)
    print("🎉 Packing complete!")
    print(f"   Output: {args.output}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
