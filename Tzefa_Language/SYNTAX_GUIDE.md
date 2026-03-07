# Tzefa Language – Syntax Guide

Tzefa is a simple, typed, stack-based language designed to be written on paper and read by an OCR system.

## Instruction Format

Instructions are **variable-length**, typically **3 or 4 tokens**. The parser is context-aware and accepts:
- **3-word form** (classic): `OPCODE ARG1 ARG2`
- **4-word form** (verbose): `VERB TYPE ARG1 ARG2`
- **Special forms** (functions, ALU): may have different layouts as documented below

## Dialects & Casing

Tzefa supports **two source dialects** that both compile to the same **4-word bytecode**.

### 4-Word Dialect (Verbose / Native Bytecode)

Every line is exactly **four space-separated tokens**:

```
VERB  TYPE  ARG1  ARG2
```

Example: `Make Integer counter Five`

### 3-Word Dialect (Classic)

Every line is exactly **three space-separated tokens**:

```
OPCODE  ARG1  ARG2
```

Example: `MAKEINTEGER COUNTER FIVE`

The 3-word opcode is expanded to the equivalent 4-word form internally (e.g. `MAKEINTEGER` → `MAKE INTEGER`). Unknown opcodes are treated as user-defined function calls (`FUNCNAME INPUT OUTPUT` → `CALL FUNCNAME INPUT OUTPUT`).

### Casing Modes

| Mode | Commands | User variables | 3-word example | 4-word example |
|---|---|---|---|---|
| **CAPS ONLY** | `MAKEINTEGER` | `COUNTER` | `MAKEINTEGER COUNTER FIVE` | `MAKE INTEGER COUNTER FIVE` |
| **Mixed Case** | `Makeinteger` | `counter` | `Makeinteger counter Five` | `Make Integer counter Five` |

Numbers are always written as English names (`ZERO`, `ONE`, … `ONEHUNDRED`).  
Variable names are free-form identifiers (UPPERCASE in caps mode, lowercase in mixed mode).

---

## Types

| Type keyword | Meaning |
|---|---|
| `INTEGER` | Whole number (int) |
| `STRING` | Text |
| `BOOLEAN` | `TRUE` or `FALSE` |
| `LIST` | Typed, fixed-size container |

---

## Variables

### Declare an integer
```
MAKE  INTEGER  NAME  VALUE
```

```
MAKE INTEGER COUNTER ZERO
MAKE INTEGER LIMIT   TEN
```

### Declare a string
```
MAKE  STRING  NAME  initialvalue
```

```
MAKE STRING GREETING hello
```

### Declare a boolean
```
MAKE  BOOLEAN  NAME  TRUE|FALSE
```

```
MAKE BOOLEAN RUNNING TRUE
```

### Declare a list
```
NEW  LIST  NAME  SIZE
```

```
NEW LIST MYLIST FIVE
```

---

## Arithmetic

All results are stored in `TEMPORARY` (in 3-word form). In 4-word form, you specify the **destination explicitly**.

### 3-Word Form (classic)

| Instruction | Result |
|---|---|
| `ADDVALUES A B` | `TEMPORARY = A + B` |
| `SUBTRACT A B` | `TEMPORARY = A - B` |
| `MULTIPLY A B` | `TEMPORARY = A * B` |
| `DIVIDE A B` | `TEMPORARY = A / B` (float) |
| `SIMPLEDIVIDE A B` | `TEMPORARY = A // B` (integer) |
| `MODULO A B` | `TEMPORARY = A % B` |
| `MATHPOW A B` | `TEMPORARY = A ** B` |

```
ADDVALUES COUNTER ONEVAR
ASSSIGNINT COUNTER TEMPORARY
```

### 4-Word Form (verbose)

| Instruction | Result |
|---|---|
| `ADD DEST VALUE1 VALUE2` | `DEST = VALUE1 + VALUE2` |
| `SUBTRACT DEST VALUE1 VALUE2` | `DEST = VALUE1 - VALUE2` |
| `MULTIPLY DEST VALUE1 VALUE2` | `DEST = VALUE1 * VALUE2` |
| `DIVIDE DEST VALUE1 VALUE2` | `DEST = VALUE1 / VALUE2` (float) |
| `SIMPLEDIVIDE DEST VALUE1 VALUE2` | `DEST = VALUE1 // VALUE2` (integer) |
| `MODULO DEST VALUE1 VALUE2` | `DEST = VALUE1 % VALUE2` |
| `POWER DEST VALUE1 VALUE2` | `DEST = VALUE1 ** VALUE2` |

```
ADD RESULT COUNTER ONEVAR
SET INTEGER COUNTER RESULT
```

### Copy variables
```
SET  INTEGER  DEST  SOURCE
SET  STRING   DEST  SOURCE
SET  LIST     DEST  SOURCE
```

### String concatenation

3-word: `COMBINE STR1 STR2` (result in `TEMPSTRING`)  
4-word: `COMBINE DEST STR1 STR2`

### Pad a string with spaces

3-word: `BLANKSPACES STRNAME NUMNAME`  
4-word: `PAD STRING STRNAME NUMNAME`

---

## Lists

### Set the active index
```
SET  INDEX  LISTNAME  INDEX
```

### Read from the current slot
```
GET  INTEGER  LISTNAME  DEST_INT
GET  STRING   LISTNAME  DEST_STR
GET  BOOLEAN  LISTNAME  DEST_BOOL
GET  LIST     LISTNAME  DEST_LIST
```

### Write to the current slot
```
WRITE  INTEGER  LISTNAME  SRC_INT
WRITE  STRING   LISTNAME  SRC_STR
WRITE  BOOLEAN  LISTNAME  SRC_BOOL
WRITE  LIST     LISTNAME  SRC_LIST
```

### Get the type of the current slot
```
GET  TYPE  LISTNAME  STR_DEST
```

### Get the length
```
GET  LENGTH  LISTNAME  INT_DEST
```

### Grow a list
```
ADD  SIZE  LISTNAME  INT_AMOUNT
```

---

## Conditions

### Create a condition
```
NEW  CONDITION  CONDNAME  EQUALS|BIGEQUALS|BIGGER
```

### Set operands
```
SET  LEFT   CONDNAME  INTNAME
SET  RIGHT  CONDNAME  INTNAME
```

### Change the operator
```
CHANGE  COMPARE  CONDNAME  EQUALS|BIGEQUALS|BIGGER
```

| Operator | Meaning |
|---|---|
| `EQUALS` | `left == right` |
| `BIGEQUALS` | `left >= right` |
| `BIGGER` | `left > right` |

---

## Control Flow

The last argument is the **end-line number** (as a numeric name).

### While loop (condition-based)
```
WHILE  CONDITION  CONDNAME  END_LINE
```

### While loop (boolean-based)
```
WHILE  BOOLEAN  BOOLNAME  END_LINE
```

### Iterate over a list
```
ITERATE  LIST  LISTNAME  END_LINE
```

### If / else-if (condition-based)
```
IF    CONDITION  CONDNAME  END_LINE
ELIF  CONDITION  CONDNAME  END_LINE
```

### If / else-if (boolean-based)
```
IF    BOOLEAN  BOOLNAME  END_LINE
ELIF  BOOLEAN  BOOLNAME  END_LINE
```

---

## Printing

```
PRINT  STRING   STRNAME   BREAK|STAY
PRINT  INTEGER  INTNAME   BREAK|STAY
```

`BREAK` adds a newline; `STAY` does not.

---

## Functions

### Define a function
```
FUNCTION  INTEGER  FUNCNAME  INPUT_TYPE
FUNCTION  STRING   FUNCNAME  INPUT_TYPE
FUNCTION  LIST     FUNCNAME  INPUT_TYPE
```
`INPUT_TYPE` is `INTEGER`, `STRING`, `LIST`, or `BOOLEAN`.

### Return from a function
```
RETURN  VALUE  VARNAME  BREAK|STAY
```
`BREAK` ends the function scope; `STAY` is a mid-function return.

### Call a function
```
CALL  FUNCNAME  INPUT_VAR  OUTPUT_VAR
```

In the 3-word dialect, calls are written as `FUNCNAME INPUT OUTPUT` and expanded to `CALL FUNCNAME INPUT OUTPUT`.

---

## Built-in Variables

| Name | Type | Description |
|---|---|---|
| `TEMPORARY` | INT | Scratch register for arithmetic |
| `TEMPSTRING` | STR | Scratch register for string concatenation |
| `LOCALINT` | INT | Function I/O slot (integer) |
| `LOCALSTR` | STR | Function I/O slot (string) |
| `LOCALLIST` | LIST | Function I/O slot (list) |
| `LOOPINTEGER` | INT | Read-only loop counter |
| `LOOPSTRING` | STR | Read-only loop string element |
| `LOOPLIST` | LIST | Read-only loop list element |
| `LOOPBOOL` | BOOLEAN | Read-only loop boolean element |
| `THETRUTH` | COND | Always-true condition |

---

## Numeric Names (ZERO … ONEHUNDRED)

Integer literals are written as English words. They are resolved to plain integers at compile time and are **not** runtime variables.

```
ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN
ELEVEN TWELVE ... NINETEEN
TWENTY TWENTYONE ... TWENTYNINE
...
NINETY ... NINETYNINE  ONEHUNDRED
```

---

## Complete Example – GCD via Euclid's Algorithm (4-word dialect)

```
MAKE     INTEGER    FIRST       SEVENTYFIVE
MAKE     INTEGER    SECOND      FIFTY
MAKE     INTEGER    REMAINDER   ZERO
MAKE     INTEGER    TEMP        ZERO
NEW      LIST       PAIR        TWO
SET      INDEX      PAIR        ZERO
WRITE    INTEGER    PAIR        FIRST
SET      INDEX      PAIR        ONE
WRITE    INTEGER    PAIR        SECOND
PRINT    INTEGER    FIRST       BREAK
FUNCTION LIST       GCD         LIST
  SET      INDEX      PAIR        ZERO
  GET      INTEGER    PAIR        FIRST
  SET      INDEX      PAIR        ONE
  GET      INTEGER    PAIR        SECOND
  NEW      CONDITION  ISCONDITION EQUALS
  SET      LEFT       ISCONDITION SECOND
  SET      RIGHT      ISCONDITION ZERO
  IF       CONDITION  ISCONDITION TWENTYFIVE
    WRITE    INTEGER    PAIR        FIRST
    RETURN   VALUE      PAIR        STAY
  SET      RIGHT      ISCONDITION SECOND
  SET      INDEX      PAIR        ZERO
  WRITE    INTEGER    PAIR        SECOND
  MODULO   REMAINDER  FIRST       SECOND
  SET      INDEX      PAIR        ONE
  WRITE    INTEGER    PAIR        REMAINDER
  CALL     GCD        PAIR        PAIR
  RETURN   VALUE      PAIR        BREAK
CALL     GCD        PAIR        PAIR
```

---

## GCD Program (4-Word Dialect) – Clean Version

This program computes the GCD of 75 and 50 using the Euclidean algorithm with explicit destination registers:

```
MAKE     INTEGER    A            SEVENTYFIVE
MAKE     INTEGER    B            FIFTY
PRINT    INTEGER    A            STAY
PRINT    INTEGER    B            BREAK
FUNCTION INTEGER    COMPUTE_GCD  INTEGER
  NEW      CONDITION  IS_ZERO      EQUALS
  SET      LEFT       IS_ZERO      LOCALINT
  SET      RIGHT      IS_ZERO      ZERO
  IF       CONDITION  IS_ZERO      TEN
    RETURN   VALUE      B            BREAK
  MODULO   RESULT     A            B
  SET      INTEGER    A            B
  SET      INTEGER    B            RESULT
  CALL     COMPUTE_GCD A           RESULT
  RETURN   VALUE      RESULT       BREAK
CALL     COMPUTE_GCD A            RESULT
PRINT    INTEGER    RESULT       BREAK
```

**How this works:**
1. Declare A=75, B=50
2. Print both values
3. Define COMPUTE_GCD function (takes INTEGER via LOCALINT, returns INTEGER)
4. Check if B (loaded into LOCALINT) equals ZERO
   - If yes, return the value
   - If no, compute `RESULT = A % B`, swap, and recurse
5. Call COMPUTE_GCD with A as input, receive result
6. Print the GCD

---

## 3-Word ↔ 4-Word Reference Table

| 3-Word Opcode | 4-Word Form |
|---|---|
| `MAKEINTEGER name val` | `MAKE INTEGER name val` |
| `MAKESTR name val` | `MAKE STRING name val` |
| `MAKEBOOLEAN name val` | `MAKE BOOLEAN name val` |
| `NEWLIST name size` | `NEW LIST name size` |
| `BASICCONDITION name cmp` | `NEW CONDITION name cmp` |
| `ASSSIGNINT dest src` | `SET INTEGER dest src` |
| `STRINGASSIGN dest src` | `SET STRING dest src` |
| `COPYLIST dest src` | `SET LIST dest src` |
| `SETINDEX list idx` | `SET INDEX list idx` |
| `LEFTSIDE cond int` | `SET LEFT cond int` |
| `RIGHTSIDE cond int` | `SET RIGHT cond int` |
| `CHANGECOMPARE cond op` | `CHANGE COMPARE cond op` |
| `WHILE cond end` | `WHILE CONDITION cond end` |
| `WHILETRUE bool end` | `WHILE BOOLEAN bool end` |
| `COMPARE cond end` | `IF CONDITION cond end` |
| `IFTRUE bool end` | `IF BOOLEAN bool end` |
| `ELSECOMPARE cond end` | `ELIF CONDITION cond end` |
| `ELSEIF bool end` | `ELIF BOOLEAN bool end` |
| `ITERATE list end` | `ITERATE LIST list end` |
| `PRINTSTRING str state` | `PRINT STRING str state` |
| `PRINTINTEGER int state` | `PRINT INTEGER int state` |
| `GETINTEGER list dest` | `GET INTEGER list dest` |
| `GETSTRING list dest` | `GET STRING list dest` |
| `GETBOOL list dest` | `GET BOOLEAN list dest` |
| `GETLIST list dest` | `GET LIST list dest` |
| `GETTYPE list str` | `GET TYPE list str` |
| `LENGTH list int` | `GET LENGTH list int` |
| `WRITEINTEGER list src` | `WRITE INTEGER list src` |
| `WRITESTRING list src` | `WRITE STRING list src` |
| `WRITEBOOL list src` | `WRITE BOOLEAN list src` |
| `WRITELIST list src` | `WRITE LIST list src` |
| `ADDSIZE list int` | `ADD SIZE list int` |
| `ADDVALUES a b` | `ADD DEST a b` |
| `SUBTRACT a b` | `SUBTRACT DEST a b` |
| `MULTIPLY a b` | `MULTIPLY DEST a b` |
| `DIVIDE a b` | `DIVIDE DEST a b` |
| `SIMPLEDIVIDE a b` | `SIMPLEDIVIDE DEST a b` |
| `MODULO a b` | `MODULO DEST a b` |
| `MATHPOW a b` | `POWER DEST a b` |
| `COMBINE a b` | `COMBINE DEST a b` |
| `BLANKSPACES str num` | `PAD STRING str num` |
| `TYPETOINT str int` | `TYPE TOINT str int` |
| `INTEGERFUNCTION name type` | `FUNCTION INTEGER name type` |
| `STRINGFUNCTION name type` | `FUNCTION STRING name type` |
| `LISTFUNCTION name type` | `FUNCTION LIST name type` |
| `RETURN name state` | `RETURN VALUE name state` |
| `FUNCNAME input output` | `CALL FUNCNAME input output` |

