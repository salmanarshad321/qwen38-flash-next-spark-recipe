"""Bounded, exact byte cache for immutable PLE mmap rows; NumPy/CPU only."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable

import numpy as np


class ExactRowCache:
    """Direct-mapped cache with complete row-ID tags, never approximate hits.

    Each cache belongs to exactly one immutable table. A collision evicts the
    previous row; it cannot return the wrong row. Only raw uint8 bytes are
    retained, so FP8, BF16 and F16 bit patterns are never converted. The byte
    bound includes both resident arrays (8-byte tags and the row bytes).
    """

    def __init__(self, row_bytes: int, byte_budget: int) -> None:
        if row_bytes <= 0 or byte_budget < 0:
            raise ValueError("row_bytes must be positive; byte_budget nonnegative")
        self.row_bytes = int(row_bytes)
        self.capacity = int(byte_budget) // (self.row_bytes + 8)
        self._keys = np.full(self.capacity, -1, dtype=np.int64)
        self._rows = np.empty((self.capacity, self.row_bytes), dtype=np.uint8)
        self.allocated_bytes = self._keys.nbytes + self._rows.nbytes
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def gather(
        self, ids: np.ndarray, load_rows: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Return fresh bytes in input order; the loader runs outside the lock.

        Callers normally supply sorted unique IDs. This implementation remains
        correct for duplicates and arbitrary order, and simultaneous calls.
        Filtering hits preserves sorted order in the normal loader path.
        """
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        if np.any(ids < 0):
            raise IndexError("PLE cache row IDs must be nonnegative")
        if self.capacity == 0:
            return self._validate_rows(load_rows(ids), ids.size).copy()

        slots = ids % self.capacity
        output = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
        with self._lock:
            hit = self._keys[slots] == ids
            output[hit] = self._rows[slots[hit]]
            hit_count = int(np.count_nonzero(hit))
            self.hits += hit_count
            self.misses += ids.size - hit_count
        if hit_count == ids.size:
            return output

        miss = ~hit
        miss_ids = ids[miss]
        loaded = self._validate_rows(load_rows(miss_ids), miss_ids.size)
        output[miss] = loaded

        # Several misses can target one slot. Select the last explicitly so
        # repeated-index advanced assignments cannot mismatch tags and bytes.
        miss_slots = slots[miss]
        unique_slots, reverse_first = np.unique(
            miss_slots[::-1], return_index=True
        )
        selected = miss_ids.size - 1 - reverse_first
        with self._lock:
            prior = self._keys[unique_slots]
            self.evictions += int(np.count_nonzero(
                (prior >= 0) & (prior != miss_ids[selected])
            ))
            self._rows[unique_slots] = loaded[selected]
            self._keys[unique_slots] = miss_ids[selected]
        return output

    def _validate_rows(self, rows: np.ndarray, count: int) -> np.ndarray:
        rows = np.asarray(rows)
        if rows.dtype != np.uint8 or rows.shape != (count, self.row_bytes):
            raise ValueError("PLE row loader must return uint8 [count, row_bytes]")
        return rows

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(
                capacity=self.capacity, allocated_bytes=self.allocated_bytes,
                hits=self.hits, misses=self.misses, evictions=self.evictions,
            )


class RowCacheBudget:
    """One process-wide cap, including all table instances and their tags.

    The present Qwen checkpoint has one mmap PLE table. Later tables get only
    unreserved capacity, keeping the process bounded if a model adds tables.
    Cache destruction returns its reservation. Per-call result/scratch arrays
    are temporary and are not part of the resident-cache cap.
    """

    def __init__(self, byte_budget: int) -> None:
        self.byte_budget = max(0, int(byte_budget))
        self.reserved_bytes = 0
        self._lock = threading.Lock()

    def make_cache(self, row_bytes: int) -> ExactRowCache:
        with self._lock:
            cache = ExactRowCache(
                row_bytes, self.byte_budget - self.reserved_bytes
            )
            self.reserved_bytes += cache.allocated_bytes
            weakref.finalize(cache, self._release, cache.allocated_bytes)
            return cache

    def _release(self, size: int) -> None:
        with self._lock:
            self.reserved_bytes -= size
