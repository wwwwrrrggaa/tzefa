"""
dialects.py – Dialect normalisation for Tzefa source lines.

The canonical internal bytecode is a **4-word tuple**::

    [VERB, TYPE, ARG1, ARG2]

Two source dialects produce these tuples:

  THREE_WORD  – ``OPCODE ARG1 ARG2``  (classic, expanded to 4-word internally)
  FOUR_WORD   – ``VERB TYPE ARG1 ARG2`` (verbose, already native)

Two casing modes:

  CAPS_ONLY   – every token is UPPERCASE
  MIXED_CASE  – commands Titlecase, user vars lowercase; all uppercased internally
"""
from __future__ import annotations

from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

THREE_WORD: str = "three_word"
FOUR_WORD: str = "four_word"

CAPS_ONLY: str = "caps_only"
MIXED_CASE: str = "mixed_case"


# ---------------------------------------------------------------------------
# 3-word → 4-word expansion table
# ---------------------------------------------------------------------------
# Every classic 3-word opcode maps to a (VERB, TYPE) pair.

THREE_TO_FOUR: Dict[str, Tuple[str, str]] = {
    # Variable declarations
    "MAKEINTEGER":    ("MAKE",    "INTEGER"),
    "MAKESTR":        ("MAKE",    "STRING"),
    "MAKEBOOLEAN":    ("MAKE",    "BOOLEAN"),
    "NEWLIST":        ("NEW",     "LIST"),
    "BASICCONDITION": ("NEW",     "CONDITION"),

    # Assignment / copy
    "ASSSIGNINT":     ("SET",     "INTEGER"),
    "STRINGASSIGN":   ("SET",     "STRING"),
    "COPYLIST":       ("SET",     "LIST"),
    "SETINDEX":       ("SET",     "INDEX"),
    "LEFTSIDE":       ("SET",     "LEFT"),
    "RIGHTSIDE":      ("SET",     "RIGHT"),

    # Condition
    "CHANGECOMPARE":  ("CHANGE",  "COMPARE"),

    # Control flow
    "WHILE":          ("WHILE",   "CONDITION"),
    "WHILETRUE":      ("WHILE",   "BOOLEAN"),
    "COMPARE":        ("IF",      "CONDITION"),
    "IFTRUE":         ("IF",      "BOOLEAN"),
    "ELSECOMPARE":    ("ELIF",    "CONDITION"),
    "ELSEIF":         ("ELIF",    "BOOLEAN"),
    "ITERATE":        ("ITERATE", "LIST"),

    # Print
    "PRINTSTRING":    ("PRINT",   "STRING"),
    "PRINTINTEGER":   ("PRINT",   "INTEGER"),

    # List read
    "GETINTEGER":     ("GET",     "INTEGER"),
    "GETSTRING":      ("GET",     "STRING"),
    "GETBOOL":        ("GET",     "BOOLEAN"),
    "GETLIST":        ("GET",     "LIST"),
    "GETTYPE":        ("GET",     "TYPE"),
    "LENGTH":         ("GET",     "LENGTH"),

    # List write
    "WRITEINTEGER":   ("WRITE",   "INTEGER"),
    "WRITESTRING":    ("WRITE",   "STRING"),
    "WRITEBOOL":      ("WRITE",   "BOOLEAN"),
    "WRITELIST":      ("WRITE",   "LIST"),

    # List resize
    "ADDSIZE":        ("ADD",     "SIZE"),

    # String utilities
    "BLANKSPACES":    ("PAD",     "STRING"),

    # Type introspection
    "TYPETOINT":      ("TYPE",    "TOINT"),

    # Functions
    "INTEGERFUNCTION": ("FUNCTION", "INTEGER"),
    "STRINGFUNCTION":  ("FUNCTION", "STRING"),
    "LISTFUNCTION":    ("FUNCTION", "LIST"),
    "RETURN":          ("RETURN",   "VALUE"),
}

# 3-word ALU opcodes that implicitly write to TEMPORARY (or TEMPSTRING for COMBINE).
# They expand differently: OPCODE A B → [VERB, DEST, A, B]
_THREE_WORD_ALU: Dict[str, Tuple[str, str]] = {
    "ADDVALUES":    ("ADD",         "TEMPORARY"),
    "SUBTRACT":     ("SUBTRACT",    "TEMPORARY"),
    "MULTIPLY":     ("MULTIPLY",    "TEMPORARY"),
    "DIVIDE":       ("DIVIDE",      "TEMPORARY"),
    "SIMPLEDIVIDE": ("SIMPLEDIVIDE","TEMPORARY"),
    "MODULO":       ("MODULO",      "TEMPORARY"),
    "MATHPOW":      ("POWER",       "TEMPORARY"),
    "COMBINE":      ("COMBINE",     "TEMPSTRING"),
}

# Reverse lookup: (VERB, DEST) → old 3-word opcode (only for non-ALU ops)
FOUR_TO_THREE: Dict[Tuple[str, str], str] = {v: k for k, v in THREE_TO_FOUR.items()}

# Set of ALU verbs that use the [VERB, DEST, SRC1, SRC2] layout
ALU_VERBS = frozenset(_THREE_WORD_ALU[k][0] for k in _THREE_WORD_ALU)


def words_per_line(dialect: str) -> int:
    """Return the expected token count for the given dialect."""
    return 4 if dialect == FOUR_WORD else 3


# ---------------------------------------------------------------------------
# Normalisation — always produces a 4-word CAPS tuple
# ---------------------------------------------------------------------------

def normalize_line(tokens: List[str], dialect: str, casing: str) -> List[str]:
    """
    Convert a raw token list into a canonical 4-word UPPERCASE tuple.

    For most instructions the layout is  [VERB, TYPE, ARG1, ARG2].
    For ALU operations the layout is     [VERB, DEST, SRC1, SRC2].

    In the 3-word dialect ALU ops have no explicit dest:
        ADDVALUES A B  →  [ADD,  TEMPORARY,  A,  B]
        MODULO    A B  →  [MODULO, TEMPORARY, A, B]

    In the 4-word dialect the dest is already present:
        ADD RESULT A B  →  [ADD, RESULT, A, B]

    Returns
    -------
    list[str]
        Exactly 4 UPPERCASE tokens.
    """
    upper = [t.upper() for t in tokens]

    if dialect == FOUR_WORD:
        while len(upper) < 4:
            upper.append("")
        return upper[:4]

    # THREE_WORD → expand to 4-word
    while len(upper) < 3:
        upper.append("")
    upper = upper[:3]

    opcode, arg1, arg2 = upper[0], upper[1], upper[2]

    # 3-word ALU: inject implicit dest
    alu = _THREE_WORD_ALU.get(opcode)
    if alu is not None:
        return [alu[0], alu[1], arg1, arg2]

    # Standard verb+type expansion
    pair = THREE_TO_FOUR.get(opcode)
    if pair is not None:
        return [pair[0], pair[1], arg1, arg2]

    # Unknown opcode — treat as user-defined function call: FUNCNAME INPUT OUTPUT
    return ["CALL", opcode, arg1, arg2]

