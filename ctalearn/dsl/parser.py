from pathlib import Path

from lark import Lark

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"

with open(_GRAMMAR_PATH, encoding="utf-8") as f:
    parser = Lark(f.read(), parser="lalr", start="start", propagate_positions=True)
