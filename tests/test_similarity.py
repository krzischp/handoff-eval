import os
from unittest.mock import patch

import pytest

# Ensure OpenAI API Key is set in test environment to avoid import errors
os.environ["OPENAI_API_KEY"] = "test_key"

# Mock OpenAI client to prevent actual API calls
with patch("handoff_eval.similarity.AsyncOpenAI"):
    from handoff_eval.similarity import min_max_normalize


def test_min_max_normalize():
    # Standard case
    assert min_max_normalize(5, 0, 10) == 0.5
    assert min_max_normalize(0, 0, 10) == 0.0
    assert min_max_normalize(10, 0, 10) == 1.0

    # Negative values
    assert min_max_normalize(-5, -10, 0) == 0.5
    assert min_max_normalize(-10, -10, 0) == 0.0
    assert min_max_normalize(0, -10, 0) == 1.0

    # Edge case: min_val == max_val
    assert min_max_normalize(5, 5, 5) == 0  # Avoids division by zero

    # Values outside the min-max range
    assert min_max_normalize(-5, 0, 10) == -0.5
    assert min_max_normalize(15, 0, 10) == 1.5

    # Floating point values
    assert min_max_normalize(2.5, 0, 10) == 0.25
    assert min_max_normalize(7.5, 0, 10) == 0.75
    assert min_max_normalize(3.5, 2.5, 7.5) == 0.2


if __name__ == "__main__":
    pytest.main()
