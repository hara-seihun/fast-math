"""Deployment smoke test: the native library loads, runs, and agrees with the
executable reference model. Run it through the `fast-math` launcher."""

import numpy as np

import fast_math.graphs as graphs

TRIANGLE = np.array([[0b110, 0b101, 0b011]], dtype=np.uint64)

native = graphs.encode_graph6(TRIANGLE, backend="native")
reference = graphs.encode_graph6(TRIANGLE, backend="reference")

assert native.backend == "native", native.backend
assert native.records == reference.records == ("Bw",), (native.records, reference.records)

print(f"fast-math ok: native/reference parity, library {__import__('os').environ['FAST_MATH_LIBRARY']}")
