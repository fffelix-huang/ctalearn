from typing import Any

from lark import Token, Transformer, v_args

from ctalearn.dsl.exceptions import DslTypeError
from ctalearn.dsl.schema import Arg, DslType, resolve_binop_type


@v_args(inline=True)
class TypeCheckTransformer(Transformer[Any, DslType]):
    """Static analysis transformer to perform type checking on the AST."""

    def __init__(
        self,
        factor_schema: dict[str, DslType],
        function_schema: dict[str, dict[str, Any]],
    ) -> None:
        """Initialize the type checker with provided schemas.

        Args:
            factor_schema: A dictionary mapping factor names to their expected DslType.
            function_schema: A dictionary mapping function names to
            {"args": list[Arg], "return": DslType}. Optional args (those carrying a
            default) must be trailing.
        """
        super().__init__()
        self.factor_schema = factor_schema
        self.function_schema = function_schema
        self.local_env: dict[str, DslType] = {}

    def number(self, token: Token) -> DslType:
        """Infer the type of a numeric token."""
        return DslType.FLOAT if "." in token.value else DslType.INT

    def variable(self, token: Token) -> DslType:
        """Infer the type of a variable token.

        Checks local assignment cache first, then falls back to the global factor
        schema.

        Args:
            token: The parsed variable token.

        Returns:
            The resolved DslType.

        Raises:
            DslTypeError: If the variable is not found in either the environment or
            schema.
        """
        var_name = token.value

        # 1. Check if it's a defined local variable
        if var_name in self.local_env:
            return self.local_env[var_name]

        # 2. Check predefined factor schema (global data sources)
        if var_name in self.factor_schema:
            return self.factor_schema[var_name]

        raise DslTypeError(f"Unknown variable: '{var_name}'")

    def _binop(self, left: DslType, right: DslType) -> DslType:
        """Internal helper to resolve binary operation types."""
        return resolve_binop_type(left, right)

    def add(self, left: DslType, right: DslType) -> DslType:
        """Infer the return type for an addition operation."""
        return self._binop(left, right)

    def sub(self, left: DslType, right: DslType) -> DslType:
        """Infer the return type for a subtraction operation."""
        return self._binop(left, right)

    def mul(self, left: DslType, right: DslType) -> DslType:
        """Infer the return type for a multiplication operation."""
        return self._binop(left, right)

    def div(self, left: DslType, right: DslType) -> DslType:
        """Infer the return type for a division operation."""
        return self._binop(left, right)

    def neg(self, val_type: DslType) -> DslType:
        """Infer the return type for a negation operation."""
        return val_type

    def func_call(self, func_name_token: Token, *args_types: DslType) -> DslType:
        """Validate argument types and infer the return type of a function call.

        Args:
            func_name_token: The parsed function name token.
            *args_types: The inferred types of the arguments passed to the function.

        Returns:
            The predefined return type of the function.

        Raises:
            DslTypeError: If the function is unknown or argument types mismatch.
        """
        func_name = func_name_token.value
        if func_name not in self.function_schema:
            raise DslTypeError(f"Unknown function: '{func_name}'")

        params: list[Arg] = self.function_schema[func_name]["args"]
        required = sum(1 for p in params if p.required)

        if not (required <= len(args_types) <= len(params)):
            expected = (
                str(required)
                if required == len(params)
                else f"{required}-{len(params)}"
            )
            raise DslTypeError(
                f"Function '{func_name}' expect {expected} arguments,"
                f" got {len(args_types)}"
            )

        # zip stops at the supplied count, so omitted optional args are not checked.
        for i, (actual, param) in enumerate(zip(args_types, params)):
            if param.type == DslType.FLOAT and actual == DslType.INT:
                continue

            if actual != param.type:
                raise DslTypeError(
                    f"Function '{func_name}' {i + 1}-th argument should be"
                    f" {param.type.value}, got {actual.value}"
                )

        return_type: DslType = self.function_schema[func_name]["return"]
        return return_type

    def statement(self, var_name_token: Token, expr_type: DslType) -> None:
        """Register a variable assignment type into the local environment.

        Args:
            var_name_token: The variable name token.
            expr_type: The inferred type of the assigned expression.
        """
        self.local_env[var_name_token.value] = expr_type
        return None

    def return_stmt(self, expr_type: DslType) -> DslType:
        """Process and pass through the final return statement type."""
        return expr_type

    def start(self, *statements_and_return: DslType | None) -> DslType:
        """Evaluate the AST entry point and validate the final return type.

        Args:
            *statements_and_return: The evaluated types of all statements
            and the final return.

        Returns:
            The final output type of the script.

        Raises:
            DslTypeError: If the final returned type is not a DataFrame.
        """
        final_type = statements_and_return[-1]

        if final_type is None or final_type != DslType.DATAFRAME:
            got = "None" if final_type is None else final_type.value
            raise DslTypeError(f"Return type should be DataFrame, got {got}")

        return final_type
