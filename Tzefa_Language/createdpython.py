import sys
from typing import Dict, List, Any, Callable, Optional

# --- Comparison operator tables ---
COMPARE_OP_INDEX: Dict[str, int] = {"EQUALS": 0, "BIGEQUALS": 1, "BIGGER": 2}
COMPARE_OPS: List[Callable[[Any, Any], bool]] = [
    (lambda x, y: x == y),
    (lambda x, y: x >= y),
    (lambda x, y: x > y)
]

# --- VM execution counters and limits ---
line_count: int = 0
current_line: int = 0
line_limit: int = 1000
function_limit: int = 25
function_count: int = 0
printed_output: str = ""

# --- Forward declarations for types ---
class Value: pass
class VmList: pass
class Condition: pass

# --- Condition registries ---
cond_registry: Dict[str, Condition] = {}
saved_conds: Dict[str, Condition] = {}

# --- Local variable save slots ---
saved_locals: Dict[str, Any] = {}
saved_globals: Dict[str, Dict[str, Any]] = {"INT": {}, "STR": {}, "LIST": {}, "BOOLEAN": {}}


# ---------------------------------------------------------------------------
# Line tracking
# ---------------------------------------------------------------------------

def set_current_line(line_num: int) -> bool:
    """Record the current executing line number for error reporting."""
    global current_line
    current_line = line_num
    return True


# ---------------------------------------------------------------------------
# Node / Stack
# ---------------------------------------------------------------------------

class Node:
    """Singly-linked list node used internally by Stack."""

    def __init__(self, value: Any):
        self.value = value
        self.next: Optional['Node'] = None

    def set_next(self, next_value: Any) -> None:
        """Create and link a new node with the given value."""
        self.next = Node(value=next_value)

    def set_next_node(self, next_node: 'Node') -> None:
        """Directly link to an existing node."""
        self.next = next_node

    def get_next(self) -> Optional['Node']:
        """Return the next node."""
        return self.next

    def get_value(self) -> Any:
        """Return the value stored in this node."""
        return self.value


class Stack:
    """Simple LIFO stack backed by a Python list."""

    def __init__(self):
        self.top: Optional[Node] = None
        self.list: List[Any] = []

    def is_empty(self) -> bool:
        """Return True if the stack holds no elements."""
        return len(self.list) == 0

    def push(self, value: Any) -> None:
        """Push a value onto the top of the stack."""
        self.list.append(value)

    def pop(self) -> Any:
        """Pop and return the top value of the stack."""
        return self.list.pop()


function_call_stack = Stack()


# ---------------------------------------------------------------------------
# Error Handler
# ---------------------------------------------------------------------------

class ErrorHandler(Exception):
    """Centralised VM error reporter: prints diagnostics and terminates execution."""

    def name_error(self, var_type: str, name: str) -> None:
        """Report that a variable of the given type and name does not exist."""
        print(f"Error Line: {current_line}")
        print(f"Var of name {name} doesn't exist as type {var_type}")
        print(" ")
        print_vars()
        sys.exit(0)

    def index_error(self, list_name: str, bad_index: int, list_size: int) -> None:
        """Report an out-of-bounds list index access."""
        print(f"Error Line: {current_line}")
        print(f"Tried to change index of list {list_name} with size of {list_size} to out-of-bounds index {bad_index}")
        print(" ")
        print_vars()
        sys.exit(0)

    def div_zero_error(self, var_name: str) -> None:
        """Report a division-by-zero attempt."""
        print(f"Error Line: {current_line}")
        print(f"Cant divide by zero and var {var_name} has value of zero")
        print(" ")
        print_vars()
        sys.exit(0)

    def doesnt_exist_error(self, name: str) -> None:
        """Report that no object with the given name exists."""
        print(f"Error Line: {current_line}")
        print(f"No object with name {name} exists")
        print_vars()
        sys.exit(0)

    def write_error(self, name: str, value: Any) -> None:
        """Report an attempt to write to an unwritable variable."""
        print(f"Error Line: {current_line}")
        print(f"Tried to write value of {value} to unwritable variable {name}")
        print(" ")
        print_vars()
        sys.exit(0)

    def type_to_int_error(self, value: str) -> None:
        """Report that a string does not correspond to a known type name."""
        print(f"Error Line: {current_line}")
        print(f"No type such as {value}")
        print(" ")
        print_vars()
        sys.exit(0)

    def read_error(self, name: str) -> None:
        """Report an attempt to read from an unreadable variable."""
        print(f"Error Line: {current_line}")
        print(f"Tried to read from unreadable variable {name}")
        print(" ")
        print_vars()
        sys.exit(0)

    def line_limit_error(self) -> None:
        """Report that the program exceeded the maximum allowed line count."""
        print(f"Error Line: {current_line}")
        print("Program ran for too long")
        print(" ")
        print_vars()
        sys.exit(0)

    def overflow_error(self, call_stack: Stack) -> None:
        """Report a function call stack overflow."""
        print(f"Error Line: {current_line}")
        print("Executing too many function calls")
        print(" ")
        print_vars()
        sys.exit(0)

    def cant_change_index_error(self, name: str, added: int) -> None:
        """Report an attempt to resize an unwritable list."""
        print(f"Error Line: {current_line}")
        print(f"Tried to change indexes of list {name} and add size {added} but list is unwritable")

    def var_exists_error(self, name: str) -> None:
        """Report an attempt to create a variable whose name is already taken."""
        print(f"Error Line: {current_line}")
        print(f"Tried to create object with name {name} but var already exists")
        print(" ")
        print_vars()
        sys.exit(0)

    def type_error(self, name: str, type1: str, type2: str) -> None:
        """Report a type mismatch during a variable assignment."""
        print(f"Error Line: {current_line}")
        print(f"Mismatch of types {type1} and {type2} in variable {name}")
        print(" ")
        print_vars()
        sys.exit(0)


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------

class Value:
    """A single typed, named VM variable with read/write permission flags."""

    def __init__(self, name: str, value: Any, readable: bool, writable: bool, type_name: str):
        self.name = name
        self.value = value
        self.readable = readable
        self.writable = writable
        self.type = type_name

    def write(self, value: Any) -> None:
        """Write a value if writable, otherwise raise a write error."""
        if self.writable:
            self.value = value
        else:
            error_handler.write_error(self.name, value)

    def force_write(self, value: Any) -> None:
        """Write a value unconditionally, bypassing the writable flag."""
        self.value = value

    def read(self) -> Any:
        """Read the value if readable, otherwise raise a read error."""
        if self.readable:
            return self.value
        else:
            error_handler.read_error(self.name)

    def force_read(self) -> Any:
        """Read the value unconditionally, bypassing the readable flag."""
        return self.value

    def set_readable(self, readable: bool) -> None:
        """Set the readable permission flag."""
        self.readable = readable

    def set_writable(self, writable: bool) -> None:
        """Set the writable permission flag."""
        self.writable = writable

    def get_name(self) -> str:
        """Return the variable's name."""
        return self.name

    def is_writable(self) -> bool:
        """Return True if the variable is writable."""
        return self.writable

    def is_readable(self) -> bool:
        """Return True if the variable is readable."""
        return self.readable

    def to_string(self) -> str:
        """Return a string representation of the stored value."""
        return str(self.value)

    def give_type(self) -> str:
        """Return the type string of this variable."""
        return self.type

    def override(self, value: Any) -> None:
        """Directly overwrite the stored value, bypassing all checks."""
        self.value = value

    def make_copy(self) -> 'Value':
        """Return a readable, writable copy of this variable."""
        return Value(self.name, self.value, self.readable, True, self.type)

    def copy_var(self, source: 'Value') -> None:
        """Copy the value from source into this variable, with type and read checks."""
        if self.type != source.type:
            error_handler.type_error(self.name, self.type, source.type)
        else:
            if not source.is_readable():
                error_handler.read_error(source.get_name())
            else:
                self.value = source.value


# ---------------------------------------------------------------------------
# VmList
# ---------------------------------------------------------------------------

class VmList:
    """A fixed-size, typed VM list whose elements are Value objects."""

    def __init__(self, name: str, size: int, readable: bool, writable: bool, type_name: str):
        self.size = size
        self.index = 0
        self.values: List[Value] = [
            Value(name=(str(name) + " " + str(i)), value=0, writable=True, readable=True, type_name="INT")
            for i in range(size)
        ]
        self.types: List[str] = ["INT" for _ in range(size)]
        self.readable = readable
        self.writable = writable
        self.name = name
        self.type = type_name

    def add_size(self, added: int) -> None:
        """Grow the list by `added` elements if writable."""
        if self.writable:
            old_size = self.size
            self.size = self.size + added
            # Recreate values list with previous values preserved or new slots added
            # Note: The logic here mirrors the original, which re-checks 'compare_ops' in a
            # way that copies old indices and creates new ones.
            self.values = [
                self.values[i] if i < old_size
                else Value(name=(str(self.name) + " " + str(i)), value=0, writable=True, readable=True, type_name="INT")
                for i in range(self.size)
            ]
            self.types = self.types + ["INT" for _ in range(added)]
        else:
            error_handler.cant_change_index_error(self.name, added)

    def force_add_size(self, added: int) -> None:
        """Grow the list by `added` elements unconditionally."""
        old_size = self.size
        self.size = self.size + added
        self.values = [
            self.values[i] if i < old_size
            else Value(name=(str(self.name) + " " + str(i)), value=0, writable=True, readable=True, type_name="INT")
            for i in range(self.size)
        ]
        self.types = self.types + ["INT" for _ in range(added)]

    def change_index(self, new_index: int) -> None:
        """Set the active index if readable and in bounds."""
        if self.readable:
            if new_index >= self.size:
                error_handler.index_error(self.name, new_index, self.size)
            else:
                self.index = new_index
        else:
            error_handler.read_error(self.name)

    def force_change_index(self, new_index: int) -> None:
        """Set the active index unconditionally, still checking bounds."""
        if new_index >= self.size:
            error_handler.index_error(self.name, new_index, self.size)
        else:
            self.index = new_index

    def place_value(self, name: str, var_type: str) -> None:
        """Copy a variable into the current index slot if writable."""
        if self.writable:
            source = get_var(var_type, name)
            if source.is_readable():
                self.types[self.index] = var_type
                self.values[self.index] = source.make_copy()
            else:
                error_handler.read_error(name)
        else:
            error_handler.write_error(self.name, name)

    def force_place_value(self, name: str, var_type: str) -> None:
        """Copy a variable into the current index slot unconditionally."""
        source = get_var(var_type, name)
        if source.is_readable():
            self.types[self.index] = var_type
            self.values[self.index] = source.make_copy()
        else:
            error_handler.read_error(name)

    def read_value(self) -> Value:
        """Return the Value at the current index if readable."""
        if self.readable:
            return self.values[self.index]
        else:
            error_handler.read_error(self.name)
            return self.values[0] # Should be unreachable due to sys.exit

    def read(self) -> Value:
        """Return the Value at the current index if readable (alias for read_value)."""
        return self.read_value()

    def force_read_value(self) -> Value:
        """Return the Value at the current index unconditionally."""
        return self.values[self.index]

    def copy_element_to(self, dest_value: Value) -> None:
        """Write the current element's value into dest_value, with type checking."""
        if self.types[self.index] == dest_value.give_type():
            dest_value.write(self.values[self.index].read())
        else:
            error_handler.type_error(
                name=self.name,
                type1=self.types[self.index],
                type2=dest_value.give_type()
            )

    def read_type(self) -> str:
        """Return the type string of the element at the current index if readable."""
        if self.readable:
            return self.types[self.index]
        else:
            error_handler.read_error(self.name)
            return ""

    def force_read_type(self) -> str:
        """Return the type string of the element at the current index unconditionally."""
        return self.types[self.index]

    def to_string(self) -> str:
        """Return a bracketed string of all element values."""
        parts = ""
        for val in self.values:
            parts = parts + str(val.to_string()) + " "
        return "[ " + parts + " ]"

    def to_type_string(self) -> str:
        """Return a string of the first-character type codes for all elements if readable."""
        if self.readable:
            return "".join(t[0] for t in self.types)
        else:
            error_handler.read_error(self.name)
            return ""

    def force_to_type_string(self) -> str:
        """Return a string of the first-character type codes for all elements unconditionally."""
        return "".join(t[0] for t in self.types)

    def set_readable(self, readable: bool) -> None:
        """Set the readable permission flag."""
        self.readable = readable

    def set_writable(self, writable: bool) -> None:
        """Set the writable permission flag."""
        self.writable = writable

    def get_name(self) -> str:
        """Return the list's name."""
        return self.name

    def is_writable(self) -> bool:
        """Return True if the list is writable."""
        return self.writable

    def is_readable(self) -> bool:
        """Return True if the list is readable."""
        return self.readable

    def get_values(self) -> List[Value]:
        """Return the raw list of Value elements."""
        return self.values

    def get_types(self) -> List[str]:
        """Return the list of type strings for each element."""
        return self.types

    def get_size(self) -> int:
        """Return the current size of the list."""
        return self.size

    def make_copy(self) -> 'VmList':
        """Return a full deep copy of this list as a writable instance."""
        copy = VmList(self.name, self.size, self.readable, True, self.type)
        copy.types = self.types.copy()
        copy.values = [val.make_copy() for val in self.values]
        return copy

    def override(self, values: List[Value], types: List[str], size: int) -> None:
        """Directly replace the list's contents, bypassing all checks."""
        self.values = values
        self.types = types
        self.size = size

    def give_type(self) -> str:
        """Return the type string of this list object."""
        return self.type

    def copy_var(self, source_list: 'VmList') -> None:
        """Copy all content from source_list into this list, with type and read checks."""
        if self.type != source_list.type:
            error_handler.type_error(self.name, self.type, source_list.type)
        else:
            if not source_list.is_readable():
                error_handler.read_error(source_list.get_name())
            else:
                self.type = 'LIST'
                self.types = source_list.types.copy()
                self.size = source_list.size
                self.values = [var.make_copy() for var in source_list.values]


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------

class Condition:
    """A named conditional that compares two Value objects using a stored operator."""

    def __init__(self, compare: str):
        self.compare_index = COMPARE_OP_INDEX[compare]
        self.left = Value("0", 0, False, False, "INT")
        self.right = Value("0", 0, False, False, "INT")
        self.type = "COND"

    def set_compare(self, compare: str) -> None:
        """Change the comparison operator."""
        self.compare_index = COMPARE_OP_INDEX[compare]

    def set_left(self, left: Value) -> None:
        """Set the left operand Value."""
        self.left = left

    def set_right(self, right: Value) -> None:
        """Set the right operand Value."""
        self.right = right

    def evaluate(self) -> bool:
        """Evaluate the condition and return the boolean result."""
        return COMPARE_OPS[self.compare_index](self.left.read(), self.right.read())

    def give_type(self) -> str:
        """Return the type string of this object."""
        return self.type


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------

def add_cond(name, compare):
    """Create a new global condition with the given comparison operator."""
    global cond_registry
    if name in cond_registry:
        error_handler.var_exists_error(name)
    else:
        cond_registry[name] = Condition(compare)


def add_local_cond(name, compare):
    """Create a new condition scoped to the current function call."""
    global cond_registry, current_local_conds
    if name in cond_registry:
        error_handler.var_exists_error(name)
    else:
        cond_registry[name] = Condition(compare)
        current_local_conds[name] = cond_registry[name]
    return current_local_conds


def push_local_conds(local_conds):
    """Save the current local conditions onto the stack and remove them from the registry."""
    global local_conds_stack, cond_registry, saved_conds
    for name in local_conds:
        del cond_registry[name]
    local_conds_stack.push(local_conds)


def pop_local_conds():
    """Restore the previous function's local conditions from the stack."""
    global local_conds_stack, cond_registry, saved_conds
    popped = local_conds_stack.pop()
    for name in saved_conds:
        if name in cond_registry:
            del cond_registry[name]
    saved_conds = {}
    for name in popped:
        cond_registry[name] = popped[name]
        saved_conds[name] = popped[name]
    return popped


def get_cond(name):
    """Look up and return a condition by name, raising an error if absent."""
    global cond_registry
    if name in cond_registry:
        return cond_registry[name]
    else:
        error_handler.doesnt_exist_error(name)


# ---------------------------------------------------------------------------
# Debug / print helpers
# ---------------------------------------------------------------------------

def print_vars():
    """Print all current variable values and the accumulated output buffer."""
    global var_registry, printed_output
    print("END OF PROGRAM")
    print()
    for var_type in var_registry:
        print("All the vars used from type " + var_type)
        for var_name in var_registry[var_type]:
            if var_registry[var_type][var_name].is_writable():
                print(var_name + " : " + var_registry[var_type][var_name].to_string())
        print("")
    print("All that was printed during the program")
    print(printed_output)


# ---------------------------------------------------------------------------
# Variable helpers
# ---------------------------------------------------------------------------

def add_var(var_type, name, value):
    """Add a new global variable of the given type and initial value."""
    global var_registry
    if name in var_registry[var_type]:
        error_handler.var_exists_error(name)
    if var_type == "LIST":
        var_registry[var_type][name] = VmList(name, value, True, True, var_type)
    else:
        var_registry[var_type][name] = Value(name, value, True, True, var_type)


def get_var(var_type, name):
    """Return the variable object for the given type and name."""
    global var_registry
    if name in var_registry[var_type]:
        return var_registry[var_type][name]
    else:
        error_handler.doesnt_exist_error(name)


def add_local_var(var_type, name, value):
    """Add a new variable scoped to the current function call."""
    global current_local_vars, var_registry
    if name in var_registry[var_type]:
        error_handler.var_exists_error(name)
    if var_type == "LIST":
        var_registry[var_type][name] = VmList(name, value, True, True, var_type)
        current_local_vars[var_type][name] = var_registry[var_type][name]
    else:
        var_registry[var_type][name] = Value(name, value, True, True, var_type)
        current_local_vars[var_type][name] = var_registry[var_type][name]


def push_local_vars(var_registry_ref: dict, local_vars: dict, stack: Stack):
    """Save current local variables to the stack and clear them from the registry."""
    for var_type in local_vars:
        for name in local_vars[var_type]:
            del var_registry_ref[var_type][name]
    stack.push(local_vars)
    return {"INT": {}, "STR": {}, "LIST": {}, "BOOLEAN": {}}


def pop_local_vars(var_registry_ref: dict):
    """Restore the previous call's local variables from the stack."""
    stack = local_vars_stack
    last_call = stack.pop()
    global saved_locals
    for var_type in saved_locals:
        for name in saved_locals[var_type]:
            if name in var_registry_ref[var_type]:
                del var_registry_ref[var_type][name]
    saved_locals = {"INT": {}, "STR": {}, "LIST": {}, "BOOLEAN": {}}
    for var_type in last_call:
        for name in last_call[var_type]:
            var_registry_ref[var_type][name] = last_call[var_type][name]
            saved_locals[var_type][name] = last_call[var_type][name]
    return last_call


# ---------------------------------------------------------------------------
# VM print
# ---------------------------------------------------------------------------

def vm_print(var, newline):
    """Print a VM variable's value and append it to the output buffer."""
    global printed_output
    text = var.to_string() + newline * '\n' + ' ' * (1 - newline)
    print(text, end='')
    printed_output = printed_output + text


# ---------------------------------------------------------------------------
# Arithmetic / string operations
# ---------------------------------------------------------------------------

def vm_add(var1, var2):
    """Add two INT variables and store the result in TEMPORARY."""
    get_var("INT", "TEMPORARY").force_write(get_var("INT", var1).read() + get_var("INT", var2).read())


def vm_sub(var1, var2):
    """Subtract var2 from var1 and store the result in TEMPORARY."""
    get_var("INT", "TEMPORARY").force_write(get_var("INT", var1).read() - get_var("INT", var2).read())


def vm_mul(var1, var2):
    """Multiply two INT variables and store the result in TEMPORARY."""
    get_var("INT", "TEMPORARY").force_write(get_var("INT", var1).read() * get_var("INT", var2).read())


def vm_div(var1, var2):
    """Integer-divide var1 by var2 and store the result in TEMPORARY."""
    if get_var("INT", var2).read() == 0:
        error_handler.div_zero_error(var2)
    get_var("INT", "TEMPORARY").force_write(get_var("INT", var1).read() // get_var("INT", var2).read())


def vm_float_div(var1, var2):
    """Float-divide var1 by var2 and store the result in TEMPORARY."""
    if get_var("INT", var2).read() == 0:
        error_handler.div_zero_error(var2)
    get_var("INT", "TEMPORARY").force_write(get_var("INT", var1).read() / get_var("INT", var2).read())


def vm_pow(var1, var2):
    """Raise var1 to the power of var2 and store the result in TEMPORARY."""
    get_var("INT", "TEMPORARY").force_write(get_var("INT", var1).read() ** get_var("INT", var2).read())


def vm_mod(var1, var2):
    """Compute var1 modulo var2 and store the result in TEMPORARY."""
    if get_var("INT", var2).read() == 0:
        error_handler.div_zero_error(var2)
    get_var("INT", "TEMPORARY").force_write(int(get_var("INT", var1).read() % get_var("INT", var2).read()))


def vm_concat(var1, var2):
    """Concatenate two STR variables and store the result in TEMPSTRING."""
    get_var("STR", "TEMPSTRING").force_write(get_var("STR", var1).read() + get_var("STR", var2).read())


# ---------------------------------------------------------------------------
# Explicit-destination ALU operations (4-word dialect)
# ---------------------------------------------------------------------------

def vm_add_to(dest: str, var1: str, var2: str) -> None:
    """Add two INT variables and store the result in *dest*."""
    get_var("INT", dest).force_write(get_var("INT", var1).read() + get_var("INT", var2).read())


def vm_sub_to(dest: str, var1: str, var2: str) -> None:
    """Subtract var2 from var1 and store the result in *dest*."""
    get_var("INT", dest).force_write(get_var("INT", var1).read() - get_var("INT", var2).read())


def vm_mul_to(dest: str, var1: str, var2: str) -> None:
    """Multiply two INT variables and store the result in *dest*."""
    get_var("INT", dest).force_write(get_var("INT", var1).read() * get_var("INT", var2).read())


def vm_div_to(dest: str, var1: str, var2: str) -> None:
    """Integer-divide var1 by var2 and store the result in *dest*."""
    if get_var("INT", var2).read() == 0:
        error_handler.div_zero_error(var2)
    get_var("INT", dest).force_write(get_var("INT", var1).read() // get_var("INT", var2).read())


def vm_float_div_to(dest: str, var1: str, var2: str) -> None:
    """Float-divide var1 by var2 and store the result in *dest*."""
    if get_var("INT", var2).read() == 0:
        error_handler.div_zero_error(var2)
    get_var("INT", dest).force_write(get_var("INT", var1).read() / get_var("INT", var2).read())


def vm_pow_to(dest: str, var1: str, var2: str) -> None:
    """Raise var1 to the power of var2 and store the result in *dest*."""
    get_var("INT", dest).force_write(get_var("INT", var1).read() ** get_var("INT", var2).read())


def vm_mod_to(dest: str, var1: str, var2: str) -> None:
    """Compute var1 modulo var2 and store the result in *dest*."""
    if get_var("INT", var2).read() == 0:
        error_handler.div_zero_error(var2)
    get_var("INT", dest).force_write(int(get_var("INT", var1).read() % get_var("INT", var2).read()))


def vm_concat_to(dest: str, var1: str, var2: str) -> None:
    """Concatenate two STR variables and store the result in *dest*."""
    get_var("STR", dest).force_write(get_var("STR", var1).read() + get_var("STR", var2).read())


def vm_list_grow(list_name, size_var):
    """Increase the size of a LIST variable by the value of an INT variable."""
    get_var("LIST", list_name).add_size(get_var("INT", size_var).read())


def vm_assign_list(dest, src):
    """Copy a LIST variable from src into dest."""
    get_var("LIST", dest).copy_var(get_var("LIST", src))


def vm_assign_str(dest, src):
    """Copy a STR variable from src into dest."""
    get_var("STR", dest).copy_var(get_var("STR", src))


def vm_assign_int(dest, src):
    """Copy an INT variable from src into dest."""
    get_var("INT", dest).copy_var(get_var("INT", src))


def vm_pad_str(var_name, num_spaces):
    """Append a fixed number of blank spaces to a STR variable."""
    get_var("STR", var_name).write(get_var("STR", var_name).read() + ' ' * num_spaces)


def vm_type_to_int(type_str_var, dest_int_var):
    """Write the integer index of a type-name string variable into an INT variable."""
    lookup = {"INT": 0, "STR": 1, "BOOLEAN": 2, "LIST": 3}
    type_str = get_var('STR', type_str_var).read()
    if type_str in lookup:
        get_var('INT', dest_int_var).write(lookup[type_str])
    else:
        error_handler.type_to_int_error(type_str)


# ---------------------------------------------------------------------------
# Program-locals save/restore
# ---------------------------------------------------------------------------

def restore_program_locals(global_vars: dict, saved_program_locals: dict):
    """Restore program-level local variables into the global registry after a function call."""
    global saved_globals
    for var_type in saved_globals:
        for name in saved_globals[var_type]:
            del global_vars[var_type][name]
    saved_globals = {"INT": {}, "STR": {}, "LIST": {}}
    for var_type in saved_program_locals:
        for name in saved_program_locals[var_type]:
            global_vars[var_type][name] = saved_program_locals[var_type][name]
            saved_globals[var_type][name] = saved_program_locals[var_type][name]


# ---------------------------------------------------------------------------
# Line tick
# ---------------------------------------------------------------------------

def tick_line():
    """Increment the execution line counter and abort if the line limit is exceeded."""
    global line_count
    line_count += 1
    if line_count == line_limit:
        error_handler.line_limit_error()
    else:
        return True


# ---------------------------------------------------------------------------
# Function call enter / exit
# ---------------------------------------------------------------------------

def enter_function_call(input_type, input_var_name, function, output_type, output_var_name, call_line):
    """Push a new function call frame, execute the function, and copy its return value."""
    set_current_line(call_line)
    global var_registry, function_count, program_locals_stack, current_program_locals
    global program_local_names, current_local_vars, function_limit, current_local_conds

    program_locals_stack.push(current_program_locals)
    var_input = get_var(input_type, input_var_name)

    # Reset program-local slots to unreadable/unwritable defaults
    for slot_type in program_local_names:
        slot_name = program_local_names[slot_type]
        if slot_type == 'STR':
            var_registry[slot_type][slot_name] = Value(slot_name, '', False, False, 'STR')
        elif slot_type == 'INT':
            var_registry[slot_type][slot_name] = Value(slot_name, 0, False, False, 'INT')
        else:
            var_registry[slot_type][slot_name] = VmList(slot_name, 8, False, False, 'LIST')

    current_program_locals = {
        "INT":  {"LOCALINT":  var_registry["INT"]["LOCALINT"]},
        "STR":  {"LOCALSTR":  var_registry["STR"]["LOCALSTR"]},
        "LIST": {"LOCALLIST": var_registry["LIST"]["LOCALLIST"]},
    }

    var_registry[input_type][program_local_names[input_type]].copy_var(var_input)
    var_to_send = var_registry[input_type][program_local_names[input_type]]

    function_count += 1
    if function_count == function_limit:
        error_handler.overflow_error(function_call_stack)
    else:
        current_local_vars = push_local_vars(var_registry, current_local_vars, local_vars_stack)
        var_to_send.set_readable(True)
        var_to_send.set_writable(True)
        push_local_conds(current_local_conds)
        current_local_conds = {}
        output = get_var(output_type, output_var_name)
        result = function()
        output.copy_var(result)
        tick_line()


def exit_function_call(return_type, return_var_name):
    """Pop the current function call frame and return the named output variable."""
    global var_registry, program_locals_stack, function_count, current_local_conds
    global current_local_vars, local_vars_stack

    return_var = get_var(return_type, return_var_name)
    function_count -= 1

    saved_program_locals = program_locals_stack.pop()
    # Restore program-level globals
    restore_program_locals(var_registry, saved_program_locals)
    # Restore call-level locals
    current_local_vars = pop_local_vars(var_registry)
    # Restore conditions
    current_local_conds = pop_local_conds()
    tick_line()
    return return_var


# ---------------------------------------------------------------------------
# VM initialisation
# ---------------------------------------------------------------------------

local_vars_stack = Stack()       # stack of local variable dicts per function call
local_vars_stack.push({"INT": {}, "STR": {}, "LIST": {}})
program_locals_stack = Stack()   # stack of program-local variable dicts

LOOP_INTEGER = Value(name="LOOPINTEGER",  value=0,    readable=True,  writable=False, type_name="INT")
LOOP_STRING  = Value(name="LOOPSTRING",   value="",   readable=True,  writable=False, type_name="STR")
LOOP_BOOL    = Value(name="LOOPBOOL",     value=True, readable=True,  writable=False, type_name="BOOL")
LOOP_LIST    = VmList( name="LOOPLIST",     size=8,     readable=True,  writable=False, type_name="LIST")
TEMPORARY    = Value(name="TEMPORARY",    value=0,    readable=True,  writable=True,  type_name="INT")
LOCAL_INT    = Value(name="LOCALINT",     value=0,    readable=False, writable=False, type_name="INT")
TEMP_STRING  = Value(name="TEMPSTRING",   value="",   readable=True,  writable=False, type_name="STR")
LOCAL_STR    = Value(name="LOCALSTR",     value="",   readable=False, writable=False, type_name="STR")
LOCAL_LIST   = VmList( name="LOCALLIST",    size=8,     readable=False, writable=False, type_name="LIST")

TYPE_INT_VAL     = Value(name="INTEGER", value="INT",     readable=True, writable=False, type_name="STR")
TYPE_STR_VAL     = Value(name="STRING",  value="STR",     readable=True, writable=False, type_name="STR")
TYPE_LIST_VAL    = Value(name="LIST",    value="LIST",    readable=True, writable=False, type_name="STR")
TYPE_BOOLEAN_VAL = Value(name="BOOLEAN", value="BOOLEAN", readable=True, writable=False, type_name="STR")

loop_var_registry = {
    "INT":     {"LOOPINTEGER": LOOP_INTEGER},
    "STR":     {"LOOPSTRING":  LOOP_STRING},
    "LIST":    {"LOOPLIST":    LOOP_LIST},
    "BOOLEAN": {"LOOPBOOL":    LOOP_BOOL},
}
loop_var_by_type = {
    "INT":     LOOP_INTEGER,
    "STR":     LOOP_STRING,
    "LIST":    LOOP_LIST,
    "BOOLEAN": LOOP_BOOL,
}

THE_TRUTH = Condition('EQUALS')
THE_TRUTH.set_left(TEMPORARY)
THE_TRUTH.set_right(TEMPORARY)

var_registry = {
    "INT": {
        "LOOPINTEGER": LOOP_INTEGER,
        "TEMPORARY":   TEMPORARY,
        "LOCALINT":    LOCAL_INT,
    },
    "STR": {
        "LOOPSTRING": LOOP_STRING,
        "TEMPSTRING": TEMP_STRING,
        "LOCALSTR":   LOCAL_STR,
        "INTEGER":    TYPE_INT_VAL,
        "STRING":     TYPE_STR_VAL,
        "LIST":       TYPE_LIST_VAL,
        "BOOLEAN":    TYPE_BOOLEAN_VAL,
    },
    "LIST": {
        "LOOPLIST":  LOOP_LIST,
        "LOCALLIST": LOCAL_LIST,
    },
    "BOOLEAN": {
        "LOOPBOOL": LOOP_BOOL,
    },
}

# Number names (ZERO..ONEHUNDRED) are compile-time constants only.
# They are resolved to plain integer literals by the compiler (toline/word_to_num)
# and must NOT live in var_registry["INT"] — that would prevent users from naming
# their own variables ONE, ZERO, etc.

empty_local_vars      = {"INT": {}, "STR": {}, "LIST": {}, "BOOLEAN": {}}
current_local_vars    = empty_local_vars.copy()
current_program_locals = {
    "INT":  {"LOCALINT":  var_registry["INT"]["LOCALINT"]},
    "STR":  {"LOCALSTR":  var_registry["STR"]["LOCALSTR"]},
    "LIST": {"LOCALLIST": var_registry["LIST"]["LOCALLIST"]},
}
program_local_names = {"INT": "LOCALINT", "STR": "LOCALSTR", "LIST": "LOCALLIST"}

local_conds_stack  = Stack()
current_local_conds = {}

program_locals_stack.push(current_program_locals)

error_handler = ErrorHandler()
cond_registry['THETRUTH'] = THE_TRUTH
