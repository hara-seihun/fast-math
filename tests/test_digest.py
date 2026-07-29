from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from fast_math import digest_u64_rows, serialize_u64_row
from lambda_fast import available_backends


NATIVE_AVAILABLE = "native" in available_backends()
FORMAT_TAG = b"fast-math/u64-rows/1\0"


def expected_payload(row: list[int], namespace: bytes = b"") -> bytes:
    return (
        FORMAT_TAG
        + struct.pack("<Q", len(namespace))
        + namespace
        + struct.pack("<Q", len(row))
        + struct.pack(f"<{len(row)}Q", *row)
    )


def test_serializer_has_documented_canonical_bytes() -> None:
    row = [0, 1, 2**32 + 7, 2**64 - 1]
    namespace = b"graph-profile/order=9/max-order=6"
    assert serialize_u64_row(row, namespace=namespace) == expected_payload(
        row,
        namespace,
    )


def test_reference_digest_matches_hashlib_payload() -> None:
    rows = np.array([[0, 1, 2], [3, 5, 8]], dtype=np.uint64)
    result = digest_u64_rows(
        rows,
        namespace="fibonacci",
        backend="reference",
    )
    assert result.hexdigests == tuple(
        hashlib.sha256(
            expected_payload(row.tolist(), b"fibonacci")
        ).hexdigest()
        for row in rows
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("field_count", [1, 7, 8, 9, 31, 208, 1025])
def test_native_digest_matches_reference(field_count: int) -> None:
    rng = np.random.default_rng(7610 + field_count)
    rows = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=(37, field_count),
        dtype=np.uint64,
    )
    reference = digest_u64_rows(
        rows,
        namespace=b"hostile\x00namespace",
        backend="reference",
    )
    native = digest_u64_rows(
        rows,
        namespace=b"hostile\x00namespace",
        threads=5,
        backend="native",
    )
    np.testing.assert_array_equal(native.digests, reference.digests)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_native_digest_is_thread_deterministic() -> None:
    rng = np.random.default_rng(7611)
    rows = rng.integers(0, 1000, size=(1003, 208), dtype=np.uint64)
    single = digest_u64_rows(rows, threads=1, backend="native")
    parallel = digest_u64_rows(rows, threads=5, backend="native")
    np.testing.assert_array_equal(single.digests, parallel.digests)


def test_namespace_and_shape_separate_digests() -> None:
    flat = digest_u64_rows([1, 2], namespace=b"a", backend="reference")
    changed_namespace = digest_u64_rows(
        [1, 2],
        namespace=b"b",
        backend="reference",
    )
    extended = digest_u64_rows([1, 2, 0], namespace=b"a", backend="reference")
    assert flat.hexdigests != changed_namespace.hexdigests
    assert flat.hexdigests != extended.hexdigests


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "shape"),
        ([[]], "shape"),
        ([[1.0]], "integers"),
        ([[-1]], "nonnegative"),
    ],
)
def test_digest_rejects_invalid_rows(rows, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        digest_u64_rows(rows, backend="reference")


def test_digest_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="namespace"):
        digest_u64_rows([1], namespace=object(), backend="reference")
    with pytest.raises(ValueError, match="threads"):
        digest_u64_rows([1], threads=-1, backend="reference")
    with pytest.raises(ValueError, match="backend"):
        digest_u64_rows([1], backend="mystery")
