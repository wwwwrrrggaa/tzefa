"""
ErrorCorrection.py – Tzefa source-text parser and error-correcting compiler front-end.

TzefaParser converts raw text lines (e.g. from OCR) into validated 4-word
bytecode tuples consumed by topy.make_instruction().
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from language import Number2Name
from language.dialects import (
    THREE_WORD, CAPS_ONLY,
    normalize_line, words_per_line, ALU_VERBS,
)
from language import topy
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
    ["MAKE",      "LIST",      "NEWLIST",  "NUMNAME"],
    ["MAKE",      "CONDITION", "NEWCOND",  "COMPARE"],

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

        # Build verb→[valid types] lookup for sequential word matching
        self._verb_to_types: Dict[str, List[str]] = {}
        for row in self.instructions:
            v, t = row[0], row[1]
            if v not in self._verb_to_types:
                self._verb_to_types[v] = []
            if t not in self._verb_to_types[v]:
                self._verb_to_types[v].append(t)

        # Deduplicated verb list (order preserved, for fuzzy matching)
        # Always include CALL even before functions are registered
        self._verb_list: List[str] = ["CALL"]
        for row in self.instructions:
            if row[0] not in self._verb_list:
                self._verb_list.append(row[0])

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
        """Exact lookup of (verb, type_word) → instruction row."""
        key = (verb, type_word)
        for i, k in enumerate(self.opcode_keys):
            if k == key:
                return i, self.instructions[i]
        return 0, self.instructions[0]

    def parse_line(self, quad: List[str]) -> List[str]:
        """
        Sequential error-correction:
          W1 → fuzzy match against verb list
          W2 → fuzzy match against valid types for that verb
               (ALU: dest var auto-registered; CALL: known function names)
          W3,W4 → resolved by the spec (arg1_kind, arg2_kind)
        """
        while len(quad) < 4:
            quad.append("")

        # ── W1: verb ─────────────────────────────────────────────────────────
        verb = self.find_word(self._verb_list, quad[0], use_ocr_weights=True)[0]

        # ── ALU fast path (W2 = dest var, W3/W4 = sources) ──────────────────
        if verb in ALU_VERBS:
            # ADD SIZE is the non-ALU outlier — treat normally
            if verb == "ADD":
                size_types = self._verb_to_types.get("ADD", [])
                w2 = self.find_word(size_types, quad[1], use_ocr_weights=True)[0]
                if w2 == "SIZE":
                    # fall through to standard path
                    type_word = "SIZE"
                    verb = "ADD"
                    _, spec = self.match_opcode(verb, type_word)
                    result = [verb, type_word,
                              self._resolve_arg(spec[2], quad[2]),
                              self._resolve_arg(spec[3], quad[3])]
                    self.line_counter += 1
                    return result
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

        # ── CALL (W2 = function name, W3 = input var, W4 = output var) ───────
        if verb == "CALL":
            known_funcs = [k[1] for k in self.opcode_keys if k[0] == "CALL"]
            func_name = self.find_word(known_funcs, quad[1], use_ocr_weights=True)[0] if known_funcs else quad[1]
            func_spec = next((r for r in self.instructions if r[0] == "CALL" and r[1] == func_name), None)
            arg1 = self._resolve_arg(func_spec[2] if func_spec else "INT", quad[2])
            arg2 = self._resolve_arg("INT", quad[3])
            self.line_counter += 1
            return ["CALL", func_name, arg1, arg2]

        # ── W2: type keyword for this verb ───────────────────────────────────
        valid_types = self._verb_to_types.get(verb, [])
        type_word = self.find_word(valid_types, quad[1], use_ocr_weights=True)[0] if valid_types else quad[1]

        # ── Look up full spec ─────────────────────────────────────────────────
        _, spec = self.match_opcode(verb, type_word)
        arg1_kind, arg2_kind = spec[2], spec[3]

        result = [verb, type_word, "", ""]

        # ── FUNCTION ─────────────────────────────────────────────────────────
        if verb == "FUNCTION":
            if not self.inside_function:
                self.inside_function = True
                func_name = quad[2]   # new name, register as-is
                param_type = self.find_word(self.all_names[6], quad[3], use_ocr_weights=True)[0]
                result[2] = func_name
                result[3] = param_type
                self.function_type_stack.append(type_word)
                topy.register_user_function(
                    func_name,
                    _FUNC_TYPE_MAP.get(type_word, "INT"),
                    _FUNC_TYPE_MAP.get(param_type, "INT"),
                )
                self.opcode_keys.append(("CALL", func_name))
                self.instructions.append(["CALL", func_name, "INT", "INT"])
                if "CALL" not in self._verb_to_types:
                    self._verb_to_types["CALL"] = []
                if func_name not in self._verb_to_types["CALL"]:
                    self._verb_to_types["CALL"].append(func_name)
                label = f"CALL_{func_name}"
                if label not in self.all_names[7]:
                    self.all_names[7].append(label)

        # ── RETURN ────────────────────────────────────────────────────────────
        elif verb == "RETURN":
            if self.function_type_stack:
                ret_kind = _FUNC_TYPE_MAP.get(self.function_type_stack[-1], "INT")
                bucket = _KIND_TO_BUCKET.get(ret_kind, 0)
                result[2] = self.find_word(self.all_names[bucket], quad[2], use_ocr_weights=True)[0]
            else:
                result[2] = quad[2]
            result[3] = self.find_word(self.all_names[5], quad[3], use_ocr_weights=True)[0]
            if result[3] == "BREAK" and self.function_type_stack:
                self.inside_function = False
                self.function_type_stack.pop()
                self.indent_table[self.line_counter] = -1

        # ── Everything else ───────────────────────────────────────────────────
        else:
            result[2] = self._resolve_arg(arg1_kind, quad[2])
            result[3] = self._resolve_arg(arg2_kind, quad[3])

        # Control-flow indent tracking
        if verb in {"WHILE", "IF", "ELIF", "ITERATE"}:
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
        """Resolve a single argument against its kind's name bucket via fuzzy-match."""
        if not kind or kind == "VALUE":
            return raw

        bucket_idx = _KIND_TO_BUCKET.get(kind, -1)
        if bucket_idx < 0 or bucket_idx >= len(self.all_names):
            return raw

        # New-name kinds: register as-is, no correction
        if kind in _NEW_KINDS:
            if raw and raw not in self.all_names[bucket_idx]:
                self.all_names[bucket_idx].append(raw)
            return raw

        # NUMNAME: digit strings pass through, words get fuzzy-matched then converted
        if kind == "NUMNAME" and raw.isdigit():
            return raw

        # Fuzzy-match against the bucket — always, no exceptions
        matched, _ = self.find_word(self.all_names[bucket_idx], raw, use_ocr_weights=True)

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

