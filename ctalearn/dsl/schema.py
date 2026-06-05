from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DslType(Enum):
    """Enumeration of supported virtual data types in the DSL."""

    DATAFRAME = "DataFrame"
    FLOAT = "float"
    INT = "int"


_MISSING: Any = object()


@dataclass(frozen=True)
class Arg:
    """Type signature of a single DSL function argument.

    Attributes:
        type: The expected DslType of the argument.
        default: The default value. Left as the `_MISSING` sentinel for required
            arguments; any other value (including `None`) marks the argument optional.
    """

    type: DslType
    default: Any = field(default=_MISSING)

    @property
    def required(self) -> bool:
        """Whether the argument must be supplied by the caller."""
        return self.default is _MISSING


def match_signature(params: list[Arg], actual: list[DslType]) -> bool:
    """Check if `actual` arg types satisfy `params` (with INT->FLOAT widening).

    Shared by the analyzer (static check) and interpreter (runtime dispatch) so
    overload resolution agrees in both passes.

    Args:
        params: The declared parameter signature of one overload.
        actual: The inferred types of the call site's arguments.

    Returns:
        True iff arity is within [required, len(params)] and every supplied
        argument is type-compatible (exact match or INT supplied to FLOAT param).
    """
    required = sum(1 for p in params if p.required)
    if not (required <= len(actual) <= len(params)):
        return False
    for a, p in zip(actual, params):
        if a == p.type:
            continue
        if p.type == DslType.FLOAT and a == DslType.INT:
            continue
        return False
    return True


def resolve_binop_type(left: DslType, right: DslType) -> DslType:
    """Resolve the resulting type of a binary operation.

    Handles operator overloading logic, such as ensuring that operations
    involving a DataFrame broadcast the result to a DataFrame.

    Args:
        left: The DSL type of the left operand.
        right: The DSL type of the right operand.

    Returns:
        The resulting DSL type of the binary operation.
    """
    if left == DslType.DATAFRAME or right == DslType.DATAFRAME:
        return DslType.DATAFRAME

    if left == DslType.FLOAT or right == DslType.FLOAT:
        return DslType.FLOAT

    return DslType.INT
