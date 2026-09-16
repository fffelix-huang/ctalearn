import operator
from collections.abc import Callable
from typing import Any

from lark import Token, Transformer, v_args

from ctalearn.core.dataframe import DataFrame
from ctalearn.dsl.exceptions import DslRuntimeError
from ctalearn.dsl.schema import Arg, DslType, match_signature


def _py_to_dsl(value: Any) -> DslType:
    """Map a runtime Python value to its DslType for overload dispatch.

    bool is rejected: the DSL grammar has no bool literal, so a bool here is a
    host-side bug rather than a representable value. Checked before int because
    `bool` is a subclass of `int` in Python.
    """
    if isinstance(value, DataFrame):
        return DslType.DATAFRAME
    if isinstance(value, bool):
        raise DslRuntimeError(f"bool is not a DSL value type: {value!r}")
    if isinstance(value, int):
        return DslType.INT
    if isinstance(value, float):
        return DslType.FLOAT
    if isinstance(value, str):
        return DslType.STRING
    raise DslRuntimeError(f"Unsupported runtime type: {type(value).__name__}")


@v_args(inline=True)
class ExecutionTransformer(Transformer[Any, Any]):
    """Runtime transformer to execute the validated AST."""

    def __init__(
        self,
        functions: dict[str, list[tuple[Callable[..., Any], list[Arg]]]],
        data_loaders: dict[str, Callable[[], Any]],
    ) -> None:
        """Initialize the execution transformer.

        Args:
            functions: A dictionary mapping function names to a list of overloads,
            each `(callable, params)`. First overload whose params accept the
            actual arg types is dispatched.
            data_loaders: A dictionary of lazy-loading callables to fetch data.
        """
        super().__init__()
        self.functions = functions
        self.data_loaders = data_loaders
        self.local_env: dict[str, Any] = {}

    def number(self, token: Token) -> float | int:
        """Parse a numeric token into a float or integer."""
        return float(token.value) if "." in token.value else int(token.value)

    def string(self, token: Token) -> str:
        """Parse a string literal token, stripping the surrounding quotes."""
        return str(token.value)[1:-1]

    def variable(self, token: Token) -> Any:
        """Resolve a variable by fetching from local cache or triggering data loaders.

        Args:
            token: The parsed variable token.

        Returns:
            The underlying data payload (e.g., a DataFrame or Series).

        Raises:
            DslRuntimeError: If data loading fails or returns None.
        """
        var_name = token.value

        # 1. Fetch from local cache (calculated variables)
        if var_name in self.local_env:
            return self.local_env[var_name]

        # 2. Raise error if data loader is not defined
        if var_name not in self.data_loaders:
            raise DslRuntimeError(f"Unknown variable '{var_name}'")

        # 3. Lazy Loading: Execute the callable to fetch data
        try:
            data = self.data_loaders[var_name]()
        except Exception as e:
            raise DslRuntimeError(f"Failed to fetch '{var_name}': {e}")

        if data is None:
            raise DslRuntimeError(f"'{var_name}' is None after fetching.")

        self.local_env[var_name] = data
        return data

    def add(self, left: Any, right: Any) -> Any:
        """Execute an addition operation."""
        return operator.add(left, right)

    def sub(self, left: Any, right: Any) -> Any:
        """Execute a subtraction operation."""
        return operator.sub(left, right)

    def mul(self, left: Any, right: Any) -> Any:
        """Execute a multiplication operation."""
        return operator.mul(left, right)

    def div(self, left: Any, right: Any) -> Any:
        """Execute a division operation."""
        return operator.truediv(left, right)

    def neg(self, val: Any) -> Any:
        """Execute a negation operation."""
        return operator.neg(val)

    def func_call(self, func_name_token: Token, *args: Any) -> Any:
        """Execute a registered function with the provided arguments.

        Args:
            func_name_token: The token containing the function name.
            *args: The evaluated arguments to pass into the function.

        Returns:
            The result of the underlying function execution.

        Raises:
            DslRuntimeError: If the underlying function crashes during execution.
        """
        func_name = func_name_token.value
        overloads = self.functions.get(func_name)
        if overloads is None:
            raise DslRuntimeError(f"Unknown function '{func_name}'")

        # Single overload: analyzer already validated types; dispatch is trivial.
        # Multi-overload: inspect runtime types to pick the right callable.
        if len(overloads) == 1:
            fn: Callable[..., Any] = overloads[0][0]
        else:
            actual = [_py_to_dsl(a) for a in args]
            for cand, params in overloads:
                if match_signature(params, actual):
                    fn = cand
                    break
            else:
                # Analyzer guarantees a match; reachable only if host bypassed it.
                raise DslRuntimeError(f"No matching overload for '{func_name}'")

        try:
            return fn(*args)
        except Exception as e:
            raise DslRuntimeError(f"Failed to execution function '{func_name}': {e}")

    def statement(self, var_name_token: Token, expr_result: Any) -> None:
        """Store the result of an assignment expression into the local cache."""
        self.local_env[var_name_token.value] = expr_result
        return None

    def return_stmt(self, expr_result: Any) -> Any:
        """Process and return the final execution result."""
        return expr_result

    def start(self, *statements_and_return: Any) -> Any:
        """Process the AST entry point and return the final executed result."""
        return statements_and_return[-1]
