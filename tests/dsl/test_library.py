import pytest

import ctalearn.dsl.library as lib
from ctalearn.dsl import BUILTIN_FUNCTION_SCHEMA, BUILTIN_FUNCTIONS
from ctalearn.dsl.schema import Arg, DslType


class TestBuiltinRegistry:
    def test_functions_and_schema_share_keys(self) -> None:
        # The runtime and schema dicts must describe exactly the same operators.
        assert set(BUILTIN_FUNCTIONS) == set(BUILTIN_FUNCTION_SCHEMA)
        assert callable(BUILTIN_FUNCTIONS["ts_mean"])
        assert BUILTIN_FUNCTION_SCHEMA["ts_mean"]["return"] == DslType.DATAFRAME

    def test_build_rejects_required_after_optional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A required arg following an optional one is a configuration error."""
        bad_specs = {
            "bad": (
                lambda *_: None,
                [Arg(DslType.INT, default=0), Arg(DslType.INT)],
                DslType.DATAFRAME,
            )
        }
        monkeypatch.setattr(lib, "_SPECS", bad_specs)

        with pytest.raises(ValueError, match="required argument after optional"):
            lib._build()
