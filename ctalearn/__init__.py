from importlib.metadata import version

from ctalearn.core.dataframe import DataFrame

__version__ = version("ctalearn")

__all__ = ["DataFrame", "__version__"]
