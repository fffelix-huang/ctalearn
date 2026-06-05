import pytest

import ctalearn.dsl.library as lib
from ctalearn.dsl import BUILTIN_FUNCTION_SCHEMA, BUILTIN_FUNCTIONS
from ctalearn.dsl.schema import Arg, DslType


class TestBuiltinRegistry:
    def test_functions_and_schema_share_keys(self) -> None:
        # The runtime and schema dicts must describe exactly the same operators.
        assert set(BUILTIN_FUNCTIONS) == set(BUILTIN_FUNCTION_SCHEMA)
        # Each entry is a non-empty list of overloads.
        assert callable(BUILTIN_FUNCTIONS["ts_mean"][0][0])
        assert BUILTIN_FUNCTION_SCHEMA["ts_mean"][0]["return"] == DslType.DATAFRAME

    def test_sqrt_has_dataframe_and_float_overloads(self) -> None:
        """sqrt is overloaded: DataFrame->DataFrame and float->float."""
        overloads = BUILTIN_FUNCTION_SCHEMA["sqrt"]
        returns = {(tuple(o["args"]), o["return"]) for o in overloads}
        assert (
            (Arg(DslType.DATAFRAME),),
            DslType.DATAFRAME,
        ) in returns
        assert ((Arg(DslType.FLOAT),), DslType.FLOAT) in returns

    def test_build_rejects_required_after_optional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A required arg following an optional one is a configuration error."""
        bad_specs = {
            "bad": [
                (
                    lambda *_: None,
                    [Arg(DslType.INT, default=0), Arg(DslType.INT)],
                    DslType.DATAFRAME,
                )
            ]
        }
        monkeypatch.setattr(lib, "_SPECS", bad_specs)

        with pytest.raises(ValueError, match="required argument after optional"):
            lib._build()
