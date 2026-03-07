"""
topy.py – Tzefa IR → Python code generator.

The bytecode is a 4-element tuple::

    [VERB, TYPE, ARG1, ARG2]

Each handler receives (verb, type_word, arg1, arg2, line_num) and returns a
Python source-code string that is later assembled by make_py_file().
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_TICK: str = "tick_line() ;"
_in_function: bool = False
_current_return_type: str = ""

_user_functions: Dict[str, List[str]] = {}

_indent_changes: List[int] = [0] * 1001


# ---------------------------------------------------------------------------
# Tiny code-gen helpers
# ---------------------------------------------------------------------------

def _args(*values: Any) -> str:
    """Parenthesised, comma-separated argument list."""
    return "( " + ", ".join(str(v) for v in values) + " )" if values else "()"


def _q(value: Any) -> str:
    """Single-quote a value for generated code."""
    return f"'{value}'"


def _gv(var_type: str, name: str) -> str:
    """get_var() call expression."""
    return f"get_var({_q(var_type)}, {_q(name)})"


def _lp(n: int) -> str:
    """set_current_line() prefix."""
    return f"set_current_line({n})"


def _stmt(line_num: int, *parts: str) -> str:
    """Standard statement: set_current_line; body; tick_line."""
    return f"{_lp(line_num)}; " + "; ".join(parts) + f"; {_TICK}"


# ---------------------------------------------------------------------------
# Register user-defined functions (called by ErrorCorrection after parsing)
# ---------------------------------------------------------------------------

def register_user_function(name: str, input_type: str, output_type: str) -> None:
    """Register a user-defined function so the code generator can emit calls."""
    _user_functions[name] = [name, input_type, output_type]


def get_user_functions() -> Dict[str, List[str]]:
    return _user_functions


# ---------------------------------------------------------------------------
# Handlers — each takes (type_word, arg1, arg2, line_num) -> str
# ---------------------------------------------------------------------------

# -- MAKE: declare variables -----------------------------------------------

def _make(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    call = "add_local_var" if _in_function else "add_var"
    call_c = "add_local_cond" if _in_function else "add_cond"
    if type_word == "BOOLEAN":
        val = "True" if arg2 == "TRUE" else ("False" if arg2 == "FALSE" else arg2)
        return _stmt(ln, f"{call}{_args(_q('BOOLEAN'), _q(arg1), val)}")
    if type_word == "STRING":
        return _stmt(ln, f"{call}{_args(_q('STR'), _q(arg1), _q(arg2))}")
    if type_word == "INTEGER":
        return _stmt(ln, f"{call}{_args(_q('INT'), _q(arg1), arg2)}")
    if type_word == "LIST":
        return _stmt(ln, f"{call}{_args(_q('LIST'), _q(arg1), int(arg2))}")
    if type_word == "CONDITION":
        return _stmt(ln, f"{call_c}{_args(_q(arg1), _q(arg2))}")
    return ""


# -- SET: assignment / index / condition sides -----------------------------

def _set(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    if type_word == "INTEGER":
        return _stmt(ln, f"vm_assign_int{_args(_q(arg1), _q(arg2))}")
    if type_word == "STRING":
        return _stmt(ln, f"vm_assign_str{_args(_q(arg1), _q(arg2))}")
    if type_word == "LIST":
        return _stmt(ln, f"vm_assign_list{_args(_q(arg1), _q(arg2))}")
    if type_word == "INDEX":
        return _stmt(ln, f"get_var('LIST',{_q(arg1)}).change_index({int(arg2)})")
    if type_word == "LEFT":
        return _stmt(ln, f"get_cond({_q(arg1)}).set_left({_gv('INT', arg2)})")
    if type_word == "RIGHT":
        return _stmt(ln, f"get_cond({_q(arg1)}).set_right({_gv('INT', arg2)})")
    return ""


# -- CHANGE ----------------------------------------------------------------

def _change(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    # Only COMPARE for now
    return _stmt(ln, f"get_cond({_q(arg1)}).set_compare({_q(arg2)})")


# -- Control flow ----------------------------------------------------------

def _while(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    _indent_changes[ln + 1] = 1
    _indent_changes[int(arg2) + 1] = -1
    if type_word == "CONDITION":
        guard = f"set_current_line({ln}) and get_cond({_q(arg1)}).evaluate() and tick_line()"
    else:  # BOOLEAN
        guard = f"set_current_line({ln}) and get_var('BOOLEAN',{_q(arg1)}).read() and tick_line()"
    return f"while( {guard} ):"


def _if(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    _indent_changes[ln + 1] = 1
    _indent_changes[int(arg2) + 1] = -1
    if type_word == "CONDITION":
        guard = f"set_current_line({ln}) and get_cond({_q(arg1)}).evaluate() and tick_line()"
    else:
        guard = f"set_current_line({ln}) and get_var('BOOLEAN',{_q(arg1)}).read() and tick_line()"
    return f"if( {guard} ):"


def _elif(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    _indent_changes[ln + 1] = 1
    _indent_changes[int(arg2) + 1] = -1
    if type_word == "CONDITION":
        guard = f"set_current_line({ln}) and get_cond({_q(arg1)}).evaluate() and tick_line()"
    else:
        guard = f"set_current_line({ln}) and get_var('BOOLEAN',{_q(arg1)}).read() and tick_line()"
    return f"elif( {guard} ):"


def _iterate(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    _indent_changes[ln + 1] = 1
    _indent_changes[int(arg2) + 1] = -1
    return f"for i in vm_loop_list({_gv('LIST', arg1)}, {ln}):"


# -- PRINT -----------------------------------------------------------------

def _print(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    vm_type = "STR" if type_word == "STRING" else "INT"
    newline = "True" if arg2 == "BREAK" else "False"
    return _stmt(ln, f"vm_print(get_var({_q(vm_type)},{_q(arg1)}),{newline})")


# -- GET: read from list ---------------------------------------------------

_GET_TYPE_MAP = {"INTEGER": "INT", "STRING": "STR", "BOOLEAN": "BOOLEAN", "LIST": "LIST"}


def _get(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    if type_word == "TYPE":
        return _stmt(ln, f"get_var('STR',{_q(arg2)}).write(get_var('LIST',{_q(arg1)}).read_type())")
    if type_word == "LENGTH":
        return _stmt(ln, f"get_var('INT',{_q(arg2)}).write(get_var('LIST',{_q(arg1)}).get_size())")
    vm = _GET_TYPE_MAP[type_word]
    return _stmt(ln, f"get_var({_q(vm)},{_q(arg2)}).copy_var(get_var('LIST',{_q(arg1)}).read())")


# -- WRITE: write to list --------------------------------------------------

_WRITE_TYPE_MAP = {"INTEGER": "INT", "STRING": "STR", "BOOLEAN": "BOOLEAN", "LIST": "LIST"}


def _write(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    vm = _WRITE_TYPE_MAP[type_word]
    return _stmt(ln, f"get_var('LIST',{_q(arg1)}).place_value({_q(arg2)},\"{vm}\")")


# -- ADD (dual purpose: list resize / arithmetic with explicit dest) --------

def _add(dest: str, src1: str, src2: str, ln: int) -> str:
    if dest == "SIZE":
        # ADD SIZE listname int_amount  (list resize — dest is literally "SIZE")
        return _stmt(ln, f"vm_list_grow{_args(_q(src1), _q(src2))}")
    # ADD DEST SRC1 SRC2
    return _stmt(ln, f"vm_add_to{_args(_q(dest), _q(src1), _q(src2))}")


# -- Arithmetic verbs — all take (dest, src1, src2, ln) --------------------

def _subtract(dest: str, src1: str, src2: str, ln: int) -> str:
    return _stmt(ln, f"vm_sub_to{_args(_q(dest), _q(src1), _q(src2))}")


def _multiply(dest: str, src1: str, src2: str, ln: int) -> str:
    return _stmt(ln, f"vm_mul_to{_args(_q(dest), _q(src1), _q(src2))}")


def _divide(dest: str, src1: str, src2: str, ln: int) -> str:
    return _stmt(ln, f"vm_float_div_to{_args(_q(dest), _q(src1), _q(src2))}")


def _simpledivide(dest: str, src1: str, src2: str, ln: int) -> str:
    return _stmt(ln, f"vm_div_to{_args(_q(dest), _q(src1), _q(src2))}")


def _modulo(dest: str, src1: str, src2: str, ln: int) -> str:
    return _stmt(ln, f"vm_mod_to{_args(_q(dest), _q(src1), _q(src2))}")


def _power(dest: str, src1: str, src2: str, ln: int) -> str:
    return _stmt(ln, f"vm_pow_to{_args(_q(dest), _q(src1), _q(src2))}")


def _combine(dest: str, src1: str, src2: str, ln: int) -> str:
    return _stmt(ln, f"vm_concat_to{_args(_q(dest), _q(src1), _q(src2))}")


# -- PAD -------------------------------------------------------------------

def _pad(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    return _stmt(ln, f"vm_pad_str{_args(_q(arg1), arg2)}")


# -- TYPE ------------------------------------------------------------------

def _type(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    return _stmt(ln, f"vm_type_to_int{_args(_q(arg1), _q(arg2))}")


# -- FUNCTION: define ------------------------------------------------------

def _function(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    global _in_function, _current_return_type
    _in_function = True
    type_map = {"INTEGER": "INT", "STRING": "STR", "LIST": "LIST"}
    _current_return_type = type_map.get(type_word, "INT")
    _indent_changes[ln + 1] = 1
    return f"def {arg1}():"


# -- RETURN ----------------------------------------------------------------

def _return(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    global _in_function
    if arg2 == "BREAK":
        _indent_changes[ln + 1] = -1
        _in_function = False
    return f"set_current_line({ln}); return(exit_function_call({_q(_current_return_type)}, {_q(arg1)}))"


# -- CALL: user-defined function -------------------------------------------

def _call(type_word: str, arg1: str, arg2: str, ln: int) -> str:
    # type_word = function name, arg1 = input var, arg2 = output var
    func_name = type_word
    spec = _user_functions.get(func_name)
    if spec:
        return (
            f"enter_function_call"
            f"({_q(spec[1])}, {_q(arg1)}, {func_name}, {_q(spec[2])}, {_q(arg2)}, {ln})"
        )
    # Fallback — shouldn't happen if ErrorCorrection registered all functions
    return f"enter_function_call('INT', {_q(arg1)}, {func_name}, 'INT', {_q(arg2)}, {ln})"


# ---------------------------------------------------------------------------
# Dispatch table — keyed by VERB
# ---------------------------------------------------------------------------

_DISPATCH: Dict[str, Callable[[str, str, str, int], str]] = {
    "MAKE":         _make,
    "SET":          _set,
    "CHANGE":       _change,
    "WHILE":        _while,
    "IF":           _if,
    "ELIF":         _elif,
    "ITERATE":      _iterate,
    "PRINT":        _print,
    "GET":          _get,
    "WRITE":        _write,
    "ADD":          _add,
    "SUBTRACT":     _subtract,
    "MULTIPLY":     _multiply,
    "DIVIDE":       _divide,
    "SIMPLEDIVIDE": _simpledivide,
    "MODULO":       _modulo,
    "POWER":        _power,
    "COMBINE":      _combine,
    "PAD":          _pad,
    "TYPE":         _type,
    "FUNCTION":     _function,
    "RETURN":       _return,
    "CALL":         _call,
}


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def make_instruction(quad: List[str], line_num: int) -> str:
    """Dispatch a 4-word bytecode tuple to its code-gen handler."""
    verb = quad[0]
    handler = _DISPATCH.get(verb)
    if handler:
        return handler(quad[1], quad[2], quad[3], line_num)
    # Unknown verb — treat as user-defined function call
    return _call(verb, quad[1], quad[2], line_num)


def make_py_file(instruction_list: List[List[str]]) -> None:
    """Compile *instruction_list* to Python and write it to test.py."""
    from pathlib import Path

    out_path = Path(__file__).parent / "test.py"
    indent_unit = "    "

    with out_path.open("w", encoding="utf-8") as f:
        f.write("import sys\nimport os\n")
        f.write("sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
        f.write("from Tzefa_Language.createdpython import *\n")
        f.write("print('VM TEST START')\n")

        indent_level = 0
        for i, quad in enumerate(instruction_list, start=1):
            indent_level += _indent_changes[i]
            f.write(indent_unit * indent_level + make_instruction(quad, i) + "\n")

        f.write("print_vars()\nprint('VM TEST END')\n")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    register_user_function("GREATESTDIV", "LIST", "LIST")
    _sample = [
        ["MAKE",     "INTEGER",   "THEINT",        "2769"],
        ["MAKE",     "INTEGER",   "THEINTI",       "1065"],
        ["MAKE",     "INTEGER",   "THROWONE",      "1065"],
        ["MAKE",     "INTEGER",   "THROWTWO",      "1065"],
        ["MAKE",     "LIST",      "LISTOFTWO",     "2"],
        ["SET",      "INDEX",     "LISTOFTWO",     "0"],
        ["WRITE",    "INTEGER",   "LISTOFTWO",     "THEINT"],
        ["SET",      "INDEX",     "LISTOFTWO",     "1"],
        ["WRITE",    "INTEGER",   "LISTOFTWO",     "THEINTI"],
        ["MAKE",     "INTEGER",   "ZERO",          "0"],
        ["ADD",      "TEMPORARY", "THEINT",        "THEINTI"],
        ["PRINT",    "INTEGER",   "TEMPORARY",     "BREAK"],
        ["FUNCTION", "LIST",      "GREATESTDIV",   "LIST"],
        ["SET",      "INDEX",     "LISTOFTWO",     "0"],
        ["GET",      "INTEGER",   "LISTOFTWO",     "THROWONE"],
        ["SET",      "INDEX",     "LISTOFTWO",     "1"],
        ["GET",      "INTEGER",   "LISTOFTWO",     "THROWTWO"],
        ["MAKE",     "CONDITION", "EUCLIDCOMPARE", "EQUALS"],
        ["SET",      "LEFT",      "EUCLIDCOMPARE", "THROWTWO"],
        ["SET",      "RIGHT",     "EUCLIDCOMPARE", "ZERO"],
        ["IF",       "CONDITION", "EUCLIDCOMPARE", "23"],
        ["WRITE",    "INTEGER",   "LISTOFTWO",     "THROWTWO"],
        ["RETURN",   "VALUE",     "LISTOFTWO",     "STAY"],
        ["SET",      "RIGHT",     "EUCLIDCOMPARE", "THROWTWO"],
        ["SET",      "INDEX",     "LISTOFTWO",     "0"],
        ["WRITE",    "INTEGER",   "LISTOFTWO",     "THROWTWO"],
        ["MODULO",   "TEMPORARY", "THROWONE",      "THROWTWO"],  # DEST=TEMPORARY
        ["SET",      "INDEX",     "LISTOFTWO",     "1"],
        ["WRITE",    "INTEGER",   "LISTOFTWO",     "TEMPORARY"],
        ["CALL",     "GREATESTDIV","LISTOFTWO",    "LISTOFTWO"],
        ["RETURN",   "VALUE",     "LISTOFTWO",     "BREAK"],
        ["CALL",     "GREATESTDIV","LISTOFTWO",    "LISTOFTWO"],
    ]
    make_py_file(_sample)

