import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.vector_service import VECTOR_SIZE, validate_embedding


def assert_rejected(vector, expected_message):
    try:
        validate_embedding(vector)
    except ValueError as exc:
        assert expected_message in str(exc)
        return
    raise AssertionError("Expected vector to be rejected")


def test_dense_embedding_is_normalized():
    vector = [float(i + 1) for i in range(VECTOR_SIZE)]
    normalized = validate_embedding(vector)
    norm = math.sqrt(sum(value * value for value in normalized))
    assert len(normalized) == VECTOR_SIZE
    assert abs(norm - 1.0) < 1e-6


def test_bad_embeddings_are_rejected():
    assert_rejected([0.0] * VECTOR_SIZE, "norm is too small")
    assert_rejected([1.0] + [0.0] * (VECTOR_SIZE - 1), "sparse or dummy")
    assert_rejected([math.nan] * VECTOR_SIZE, "not finite")
    assert_rejected([0.1] * (VECTOR_SIZE - 1), "128 dimensions")


if __name__ == "__main__":
    test_dense_embedding_is_normalized()
    test_bad_embeddings_are_rejected()
    print("face vector guard tests passed")
