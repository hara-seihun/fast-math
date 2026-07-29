from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
import struct
import time
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lambda_fast._native import (
    NativeUnavailable,
    digest_u64_rows_native,
)


Backend = Literal["auto", "native", "reference"]
FORMAT_TAG = b"fast-math/u64-rows/1\0"


@dataclass(frozen=True)
class UInt64DigestBatch:
    digests: NDArray[np.uint8]
    row_count: int
    field_count: int
    elapsed_seconds: float
    backend: str

    @property
    def hexdigests(self) -> tuple[str, ...]:
        return tuple(row.tobytes().hex() for row in self.digests)


def _prepare_rows(rows: ArrayLike) -> NDArray[np.uint64]:
    array = np.asarray(rows)
    if array.dtype.kind not in {"i", "u", "O"}:
        array = np.asarray(rows, dtype=object)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("rows must have shape (row_count, field_count)")
    if array.dtype.kind == "O":
        maximum = np.iinfo(np.uint64).max
        for value in array.flat:
            if not isinstance(value, Integral):
                raise ValueError("rows must contain integers")
            if value < 0:
                raise ValueError("rows must be nonnegative")
            if value > maximum:
                raise ValueError("row values must fit in uint64")
        return np.ascontiguousarray(array, dtype=np.uint64)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError("rows must contain integers")
    if array.dtype.kind == "i" and np.any(array < 0):
        raise ValueError("rows must be nonnegative")
    if array.dtype.kind == "u" and array.dtype.itemsize > 8:
        if np.any(array > np.iinfo(np.uint64).max):
            raise ValueError("row values must fit in uint64")
    return np.ascontiguousarray(array, dtype=np.uint64)


def _prepare_namespace(namespace: str | bytes) -> bytes:
    if isinstance(namespace, str):
        return namespace.encode("utf-8")
    if isinstance(namespace, bytes):
        return namespace
    raise ValueError("namespace must be str or bytes")


def _prefix(namespace: bytes, field_count: int) -> bytes:
    return (
        FORMAT_TAG
        + struct.pack("<Q", len(namespace))
        + namespace
        + struct.pack("<Q", field_count)
    )


def serialize_u64_row(
    row: ArrayLike,
    *,
    namespace: str | bytes = b"",
) -> bytes:
    prepared = _prepare_rows(row)
    if prepared.shape[0] != 1:
        raise ValueError("serialize_u64_row accepts exactly one row")
    namespace_bytes = _prepare_namespace(namespace)
    little_endian = prepared.astype("<u8", copy=False)
    return _prefix(namespace_bytes, prepared.shape[1]) + little_endian.tobytes()


def _digest_reference(
    rows: NDArray[np.uint64],
    namespace: bytes,
) -> UInt64DigestBatch:
    started = time.perf_counter()
    prefix = _prefix(namespace, rows.shape[1])
    little_endian = rows.astype("<u8", copy=False)
    digests = np.empty((len(rows), 32), dtype=np.uint8)
    for index, row in enumerate(little_endian):
        digest = hashlib.sha256(prefix)
        digest.update(row)
        digests[index] = np.frombuffer(digest.digest(), dtype=np.uint8)
    return UInt64DigestBatch(
        digests=digests,
        row_count=len(rows),
        field_count=rows.shape[1],
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def digest_u64_rows(
    rows: ArrayLike,
    *,
    namespace: str | bytes = b"",
    threads: int = 1,
    backend: Backend = "auto",
) -> UInt64DigestBatch:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    prepared = _prepare_rows(rows)
    namespace_bytes = _prepare_namespace(namespace)
    if backend in {"auto", "native"}:
        try:
            namespace_array = np.frombuffer(
                namespace_bytes,
                dtype=np.uint8,
            )
            digests, stats = digest_u64_rows_native(
                prepared,
                namespace_array,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return UInt64DigestBatch(
                digests=digests,
                row_count=int(stats.row_count),
                field_count=int(stats.field_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _digest_reference(prepared, namespace_bytes)
