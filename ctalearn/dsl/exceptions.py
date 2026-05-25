class DslError(Exception):
    """Base exception class for all DSL-related errors."""

    pass


class DslSyntaxError(DslError):
    """Raised when the parser encounters invalid syntax in the DSL script."""

    pass


class DslTypeError(DslError):
    """Raised during static analysis for type mismatches or unknown variables."""

    pass


class DslRuntimeError(DslError):
    """Raised during AST execution due to failed data fetching or function crashes."""

    pass
