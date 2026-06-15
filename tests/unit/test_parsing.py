import pytest

from synthline.errors import StructuredOutputError
from synthline.utils.parsing import parse_completion


def test_parse_completion_accepts_exact_schema():
    result = parse_completion('{"samples": [" First ", "Second"]}', expected_count=2)

    assert result == ["First", "Second"]


@pytest.mark.parametrize(
    ("completion", "message"),
    [
        ("plain text", "invalid JSON"),
        ('{"items": ["A"]}', 'top-level "samples"'),
        ('{"samples": "A"}', '"samples" to be an array'),
        ('{"samples": ["A"]}', "Expected exactly 2 samples"),
        ('{"samples": ["A", ""]}', "non-empty string"),
        ('{"samples": ["A", 2]}', "non-empty string"),
        ('{"samples": ["A", "B"], "extra": true}', 'top-level "samples"'),
    ],
)
def test_parse_completion_rejects_non_schema_output(completion, message):
    with pytest.raises(StructuredOutputError, match=message):
        parse_completion(completion, expected_count=2)
