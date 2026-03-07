"""
ErrorCorrection.py – Tzefa source-text parser and error-correcting compiler front-end.

TzefaParser converts raw text lines (e.g. from OCR) into validated 4-word
bytecode tuples consumed by topy.make_instruction().
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from Tzefa_Language import Number2Name
from Tzefa_Language.dialects import (
    THREE_WORD, CAPS_ONLY,
    normalize_line, words_per_line, ALU_VERBS,
)
from Tzefa_Language import topy
from fast_edit_distance import edit_distance


# ---------------------------------------------------------------------------
# Instruction definitions — now in 4-word form
# ---------------------------------------------------------------------------
# Each entry: [VERB, TYPE, ARG1_KIND, ARG2_KIND]
#
# ARG_KIND values:
#   "NEWINT"    – declares a new integer name
#   "NEWSTR"    – declares a new string name
#   "NEWBOOL"   – declares a new boolean name
#   "NEWLIST"   – declares a new list name
#   "NEWCOND"   – declares a new condition name
#   "NEWFUNC"   – declares a new function name
#   "INT"       – existing integer var
#   "STR"       – existing string var
#   "LIST"      – existing list var
#   "BOOL"      – existing boolean var
#   "COND"      – existing condition
#   "STATE"     – STAY / BREAK
#   "TYPE"      – INTEGER / STRING / LIST / BOOLEAN
#   "TRUTH"     – TRUE / FALSE
#   "COMPARE"   – EQUALS / BIGEQUALS / BIGGER
#   "NUMNAME"   – numeric name (ZERO … ONEHUNDRED)
#   "TEXT"      – free text (no correction)
#   "VALUE"     – context-dependent (return var, resolved at parse time)

_BUILTIN_INSTRUCTIONS: List[List[str]] = [
    # Variable declarations
    ["MAKE",      "INTEGER",   "NEWINT",   "NUMNAME"],
    ["MAKE",      "BOOLEAN",   "NEWBOOL",  "TRUTH"],
    ["MAKE",      "STRING",    "NEWSTR",   "TEXT"],
    ["NEW",       "LIST",      "NEWLIST",  "NUMNAME"],
    ["NEW",       "CONDITION", "NEWCOND",  "COMPARE"],

    # Condition manipulation
    ["SET",       "LEFT",      "COND",     "INT"],
    ["SET",       "RIGHT",     "COND",     "INT"],
    ["CHANGE",    "COMPARE",   "COND",     "COMPARE"],

    # Control flow
    ["WHILE",     "CONDITION", "COND",     "NUMNAME"],
    ["IF",        "CONDITION", "COND",     "NUMNAME"],
    ["ELIF",      "CONDITION", "COND",     "NUMNAME"],
    ["ITERATE",   "LIST",      "LIST",     "NUMNAME"],
    ["WHILE",     "BOOLEAN",   "BOOL",     "NUMNAME"],
    ["IF",        "BOOLEAN",   "BOOL",     "NUMNAME"],
    ["ELIF",      "BOOLEAN",   "BOOL",     "NUMNAME"],

    # Function definition
    ["FUNCTION",  "INTEGER",   "NEWFUNC",  "TYPE"],
    ["FUNCTION",  "STRING",    "NEWFUNC",  "TYPE"],
    ["FUNCTION",  "LIST",      "NEWFUNC",  "TYPE"],

    # Return
    ["RETURN",    "VALUE",     "VALUE",    "STATE"],

    # Print
    ["PRINT",     "STRING",    "STR",      "STATE"],
    ["PRINT",     "INTEGER",   "INT",      "STATE"],

    # Assignment / copy
    ["SET",       "INTEGER",   "INT",      "INT"],
    ["SET",       "STRING",    "STR",      "STR"],
    ["SET",       "LIST",      "LIST",     "LIST"],
    ["SET",       "INDEX",     "LIST",     "NUMNAME"],

    # Type introspection
    ["TYPE",      "TOINT",     "STR",      "INT"],

    # List read
    ["GET",       "STRING",    "LIST",     "STR"],
    ["GET",       "INTEGER",   "LIST",     "INT"],
    ["GET",       "BOOLEAN",   "LIST",     "BOOL"],
    ["GET",       "LIST",      "LIST",     "LIST"],
    ["GET",       "TYPE",      "LIST",     "STR"],
    ["GET",       "LENGTH",    "LIST",     "INT"],

    # List write
    ["WRITE",     "INTEGER",   "LIST",     "INT"],
    ["WRITE",     "STRING",    "LIST",     "STR"],
    ["WRITE",     "BOOLEAN",   "LIST",     "BOOL"],
    ["WRITE",     "LIST",      "LIST",     "LIST"],

    # Arithmetic — layout: [VERB, DEST, SRC1, SRC2]
    # arg1_kind=INT is the dest (existing or new), arg2/3 are sources
    ["ADD",       "INT",       "INT",      "INT"],
    ["MULTIPLY",  "INT",       "INT",      "INT"],
    ["POWER",     "INT",       "INT",      "INT"],
    ["DIVIDE",    "INT",       "INT",      "INT"],
    ["SIMPLEDIVIDE","INT",     "INT",      "INT"],
    ["SUBTRACT",  "INT",       "INT",      "INT"],
    ["MODULO",    "INT",       "INT",      "INT"],

    # String ops — COMBINE layout: [COMBINE, DEST, SRC1, SRC2]
    ["COMBINE",   "STR",       "STR",      "STR"],
    ["PAD",       "STRING",    "STR",      "NUMNAME"],

    # List resize — ADD SIZE layout: [ADD, SIZE, listname, int_amount]
    ["ADD",       "SIZE",      "LIST",     "INT"],
]

# Which kinds declare new names (start with "NEW")
_NEW_KINDS = {"NEWINT", "NEWSTR", "NEWBOOL", "NEWLIST", "NEWCOND", "NEWFUNC"}

# Kind → bucket index in the all_names list
_KIND_TO_BUCKET: Dict[str, int] = {
    "INT": 0, "NEWINT": 0,
    "STR": 1, "NEWSTR": 1,
    "LIST": 2, "NEWLIST": 2,
    "BOOL": 3, "NEWBOOL": 3,
    "COND": 4, "NEWCOND": 4,
    "STATE": 5,
    "TYPE": 6,
    "NEWFUNC": 7,
    "TRUTH": 8,
    "COMPARE": 9,
    "NUMNAME": 10,
    "TEXT": 11,
    "VALUE": -1,  # resolved dynamically
}

# The lookup type for function return types
_FUNC_TYPE_MAP: Dict[str, str] = {
    "INTEGER": "INT", "STRING": "STR", "LIST": "LIST", "BOOLEAN": "BOOL",
}


class TzefaParser:
    """Parse and error-correct Tzefa source lines into 4-word bytecode."""

    def __init__(
        self,
        dialect: str = THREE_WORD,
        casing: str = CAPS_ONLY,
    ) -> None:
        self.dialect = dialect
        self.casing = casing

        # Build instruction table from the static definitions
        self.instructions: List[List[str]] = [row[:] for row in _BUILTIN_INSTRUCTIONS]

        # Opcode keys: (VERB, TYPE) tuples for lookup
        self.opcode_keys: List[Tuple[str, str]] = [(r[0], r[1]) for r in self.instructions]

        # Name buckets for fuzzy-matching (index-aligned with _KIND_TO_BUCKET)
        self.all_names: List[List[str]] = [
            # 0: INT names
            ["TEMPORARY", "LOCALINT", "LOOPINTEGER"],
            # 1: STR names
            ["TEMPSTRING", "GLOBALSTR", "LOOPSTRING",
             "INTEGER", "STRING", "LIST", "BOOLEAN"],
            # 2: LIST names
            ["GLOBALLIST", "LOOPLIST"],
            # 3: BOOL names
            ["LOOPBOOL"],
            # 4: COND names
            ["THETRUTH"],
            # 5: STATE
            ["STAY", "BREAK"],
            # 6: TYPE
            ["INTEGER", "STRING", "LIST", "BOOLEAN"],
            # 7: opcode verbs (populated below)
            [],
            # 8: TRUTH
            ["TRUE", "FALSE"],
            # 9: COMPARE
            ["EQUALS", "BIGEQUALS", "BIGGER"],
            # 10: NUMNAME
            [],
            # 11: TEXT (free, no correction)
            [],
        ]

        # Populate bucket 7 (opcode verbs) from instruction table
        seen_verbs: set = set()
        for row in self.instructions:
            key = (row[0], row[1])
            label = f"{row[0]}_{row[1]}"
            if label not in seen_verbs:
                seen_verbs.add(label)
                self.all_names[7].append(label)

        # Numeric name immediates
        self.word_to_num: Dict[str, str] = {}
        for i in range(101):
            name = Number2Name.get_name(i)
            self.all_names[10].append(name)
            self.word_to_num[name] = str(i)

        # Indent tracking
        self.indent_table: List[int] = []

        # Function definition state
        self.function_type_stack: List[str] = []
        self.inside_function: bool = False
        self.line_counter: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def expected_words_per_line(self) -> int:
        return words_per_line(self.dialect)

    def normalize_source_line(self, raw_tokens: List[str]) -> List[str]:
        """Normalize raw tokens into a canonical 4-word CAPS tuple."""
        return normalize_line(raw_tokens, self.dialect, self.casing)

    def init_indent_table(self, line_count: int) -> None:
        """Allocate the indent-change table for *line_count* lines."""
        self.indent_table = [0] * max(line_count + 2, 1002)

    def get_indent_table(self) -> List[int]:
        return self.indent_table

    def match_opcode(self, verb: str, type_word: str) -> Tuple[int, List[str]]:
        """
        Find the instruction row matching (verb, type_word).

        For ALU verbs (ADD, SUBTRACT, etc.) the type_word slot holds the
        destination variable name, not a keyword — so we match on verb alone.

        Returns (index, instruction_row).
        """
        # ALU verbs: match by verb only, type_word is the dest variable
        if verb in ALU_VERBS:
            # Find the primary entry for this verb (first match)
            for i, row in enumerate(self.instructions):
                if row[0] == verb:
                    return i, row
            # Fuzzy-correct the verb itself
            best_verb, _ = self.find_word([v for v in ALU_VERBS], verb, use_ocr_weights=True)
            for i, row in enumerate(self.instructions):
                if row[0] == best_verb:
                    return i, row

        # ADD SIZE is a special non-ALU use of ADD — check for it
        if verb == "ADD" and type_word == "SIZE":
            for i, row in enumerate(self.instructions):
                if row[0] == "ADD" and row[1] == "SIZE":
                    return i, row

        # Standard exact match on (VERB, TYPE)
        key = (verb, type_word)
        for i, k in enumerate(self.opcode_keys):
            if k == key:
                return i, self.instructions[i]

        # Fuzzy match verb+type against bucket 7
        combined = f"{verb}_{type_word}"
        matched, _ = self.find_word(self.all_names[7], combined, use_ocr_weights=True)
        parts = matched.split("_", 1)
        key2 = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")
        for i, k in enumerate(self.opcode_keys):
            if k == key2:
                return i, self.instructions[i]
        return 0, self.instructions[0]

    def parse_line(self, quad: List[str]) -> List[str]:
        """
        Error-correct and validate a 4-word bytecode tuple.

        Non-ALU:  [VERB, TYPE,  ARG1, ARG2]
        ALU:      [VERB, DEST,  SRC1, SRC2]   ← DEST is an INT variable name

        Returns a clean 4-element list.
        """
        while len(quad) < 4:
            quad.append("")

        verb = quad[0]

        # --- ALU fast path ---
        if verb in ALU_VERBS:
            # ADD SIZE list amount — not an ALU op, fall through to normal path
            if not (verb == "ADD" and quad[1] == "SIZE"):
                if verb == "COMBINE":
                    dest = self._resolve_arg("STR", quad[1])
                    src1 = self._resolve_arg("STR", quad[2])
                    src2 = self._resolve_arg("STR", quad[3])
                else:
                    dest = self._resolve_arg("INT", quad[1])
                    src1 = self._resolve_arg("INT", quad[2])
                    src2 = self._resolve_arg("INT", quad[3])
                self.line_counter += 1
                return [verb, dest, src1, src2]

        verb, type_word = quad[0], quad[1]
        idx, spec = self.match_opcode(verb, type_word)
        verb, type_word = spec[0], spec[1]
        arg1_kind, arg2_kind = spec[2], spec[3]

        result = [verb, type_word, "", ""]

        # -- Handle FUNCTION definitions --
        if verb == "FUNCTION":
            if self.inside_function:
                pass  # error: nested function
            else:
                self.inside_function = True
                result[2] = quad[2]  # function name — new, don't correct
                result[3] = self.find_word(self.all_names[6], quad[3])[0]
                self.function_type_stack.append(result[3])
                # Register the new function
                vm_return = _FUNC_TYPE_MAP.get(type_word, "INT")
                vm_input = _FUNC_TYPE_MAP.get(result[3], "INT")
                topy.register_user_function(result[2], vm_return, vm_input)
                # Add to opcode keys so CALL can resolve it
                self.opcode_keys.append(("CALL", result[2]))
                self.instructions.append(["CALL", result[2], "VALUE", "VALUE"])
                label = f"CALL_{result[2]}"
                if label not in self.all_names[7]:
                    self.all_names[7].append(label)

        # -- Handle RETURN --
        elif verb == "RETURN":
            if not self.function_type_stack:
                pass  # error: return outside function
            else:
                ret_type = self.function_type_stack[-1]
                bucket_idx = _KIND_TO_BUCKET.get(
                    _FUNC_TYPE_MAP.get(ret_type, "INT"), 0
                )
                result[2] = self.find_word(self.all_names[bucket_idx], quad[2])[0]
                result[3] = self.find_word(self.all_names[5], quad[3])[0]  # STATE
                if result[3] == "BREAK":
                    self.inside_function = False
                    self.function_type_stack.pop()
                    self.indent_table[self.line_counter] = -1

        # -- Handle CALL (user-defined function) --
        elif verb == "CALL":
            # type_word is the function name; args are input/output vars
            # We can't easily type-check these generically, pass through
            result[1] = type_word
            result[2] = quad[2]
            result[3] = quad[3]

        # -- All other instructions --
        else:
            result[2] = self._resolve_arg(arg1_kind, quad[2])
            result[3] = self._resolve_arg(arg2_kind, quad[3])

        # Control flow indent tracking
        _CONTROL_FLOW = {"WHILE", "IF", "ELIF", "ITERATE"}
        if verb in _CONTROL_FLOW:
            self.indent_table[self.line_counter] = 1
            try:
                self.indent_table[int(result[3])] = -1
            except (ValueError, IndexError):
                pass

        self.line_counter += 1
        return result

    # ------------------------------------------------------------------
    # Argument resolution
    # ------------------------------------------------------------------

    def _resolve_arg(self, kind: str, raw: str) -> str:
        """Resolve a single argument against its kind's name bucket."""
        if not kind or kind == "VALUE":
            return raw

        bucket_idx = _KIND_TO_BUCKET.get(kind, -1)
        if bucket_idx < 0 or bucket_idx >= len(self.all_names):
            return raw

        # New-name kinds: register the raw token, don't fuzzy-correct it
        if kind in _NEW_KINDS:
            if raw and raw not in self.all_names[bucket_idx]:
                self.all_names[bucket_idx].append(raw)
            return raw

        # NUMNAME: if the raw value is already a digit string, pass through
        if kind == "NUMNAME" and raw.isdigit():
            return raw

        # Existing-name kinds: fuzzy-match
        matched = self.find_word(self.all_names[bucket_idx], raw)[0]

        # NUMNAME: replace word with integer value
        if kind == "NUMNAME":
            matched = self.word_to_num.get(matched, matched)

        return matched

    # ------------------------------------------------------------------
    # Edit distance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ocr_edit_distance(word1: str, word2: str) -> float:
        """Levenshtein distance with reduced cost for common OCR confusions."""
        word1, word2 = word1.upper(), word2.upper()

        _LOW_COST: Dict[Tuple[str, str], float] = {
            ('O', '0'): 0.5, ('0', 'O'): 0.5,
            ('I', '1'): 0.5, ('1', 'I'): 0.5,
            ('I', 'L'): 0.5, ('L', 'I'): 0.5,
            ('S', '5'): 0.5, ('5', 'S'): 0.5,
            ('Z', '2'): 0.5, ('2', 'Z'): 0.5,
            ('C', 'O'): 0.5, ('O', 'C'): 0.5,
            ('C', 'G'): 0.5, ('G', 'C'): 0.5,
            ('B', '8'): 0.5, ('8', 'B'): 0.5,
            ('D', 'O'): 0.5, ('O', 'D'): 0.5,
            ('E', 'F'): 0.5, ('F', 'E'): 0.5,
            ('A', '4'): 0.5, ('4', 'A'): 0.5,
        }

        m, n = len(word1), len(word2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = float(i)
        for j in range(n + 1):
            dp[0][j] = float(j)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    cost = 0.0
                else:
                    cost = _LOW_COST.get((word1[i - 1], word2[j - 1]), 2.0)
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,
                    dp[i][j - 1] + 1.0,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[m][n]

    @staticmethod
    def find_word(
        name_list: List[str],
        word: str,
        use_ocr_weights: bool = False,
    ) -> Tuple[str, int]:
        """Return the closest match to *word* in *name_list* and its index."""
        if not name_list:
            return word, 0

        min_dist = 999.0
        best: List[Any] = [word, 0]
        best_len = 16
        word_len = len(word)

        for idx, item in enumerate(name_list):
            if item == word:
                return item, idx

            if use_ocr_weights:
                dist = TzefaParser.ocr_edit_distance(word, item)
            else:
                dist = float(edit_distance(word, item, 32))

            item_len = len(item)
            if dist < min_dist:
                min_dist = dist
                best = [item, idx]
                best_len = item_len
            elif dist == min_dist:
                if abs(word_len - item_len) < abs(word_len - best_len):
                    best = [item, idx]
                    best_len = item_len

        return tuple(best)

