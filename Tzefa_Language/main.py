"""
main.py – Tzefa language compiler entry point.

compile_and_run() is the primary API consumed by Tzefa_Ocr/main.py.
Running this file directly invokes main() and profiles it.
"""
from __future__ import annotations

import cProfile
import importlib
import pstats
import sys
import os
from pstats import SortKey
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tzefa_Language.ErrorCorrection import TzefaParser
from Tzefa_Language.dialects import THREE_WORD, CAPS_ONLY
from Tzefa_Language import topy


def compile_and_run(
    source_lines: List[str],
    dialect: str = THREE_WORD,
    casing: str = CAPS_ONLY,
) -> None:
    """Compile *source_lines* to Python, write test.py, and execute it."""
    parser = TzefaParser(dialect=dialect, casing=casing)
    parser.init_indent_table(len(source_lines))

    bytecode: List[List[str]] = []
    for line in source_lines:
        tokens = line.split(" ")
        normalised = parser.normalize_source_line(tokens)
        bytecode.append(parser.parse_line(normalised))

    topy.make_py_file(bytecode)

    if "test" in sys.modules:
        importlib.reload(sys.modules["test"])
    else:
        import test  # noqa: F401


def main() -> None:
    """Compile and run the hard-coded sample program."""
    source_lines: List[str] = []
    compile_and_run(source_lines)


if __name__ == "__main__":
    cProfile.run("main()", "output.prof")
    with open("output_stats.txt", "w") as _f:
        _p = pstats.Stats("output.prof", stream=_f)
        _p.sort_stats(SortKey.TIME).print_stats()

