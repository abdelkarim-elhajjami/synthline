import pytest

from synthline.core.schemas import samples_schema


def test_samples_schema_rejects_non_positive_count():
    with pytest.raises(ValueError, match="at least 1"):
        samples_schema(0)
