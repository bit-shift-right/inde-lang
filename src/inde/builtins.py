from __future__ import annotations

from dataclasses import dataclass

from inde.type_model import (
    BOOL,
    CHAR,
    FLOAT_TYPES,
    INTEGER_TYPES,
    NUMBER_TYPES,
    SIGNED_INTEGER_TYPES,
    STR,
    U32,
    Effect,
    Type,
    option_of,
)


@dataclass(frozen=True)
class Builtin:
    name: str
    forms: tuple[Effect, ...]
    takes: str

    @property
    def arguments(self) -> int:
        return len(self.forms[0].inputs)


@dataclass(frozen=True)
class StackWord:
    name: str
    arguments: int
    result: tuple[int, ...]

    def copied(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.arguments) if self.result.count(i) > 1)

    def discarded(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.arguments) if i not in self.result)


STACK_WORDS: dict[str, StackWord] = {
    word.name: word
    for word in (
        StackWord("dup", 1, (0, 0)),
        StackWord("drop", 1, ()),
        StackWord("over", 2, (0, 1, 0)),
        StackWord("swap", 2, (1, 0)),
        StackWord("rot", 3, (1, 2, 0)),
    )
}


BUILTINS: dict[str, Builtin] = {}


def define(name: str, forms: list[Effect], takes: str) -> None:
    assert name not in BUILTINS and name not in STACK_WORDS, name
    BUILTINS[name] = Builtin(name, tuple(forms), takes)


def binary(types: tuple[Type, ...]) -> list[Effect]:
    return [Effect((t, t), (t,)) for t in types]


def binary_checked(types: tuple[Type, ...]) -> list[Effect]:
    return [Effect((t, t), (option_of(t),)) for t in types]


def unary(types: tuple[Type, ...]) -> list[Effect]:
    return [Effect((t,), (t,)) for t in types]


def unary_checked(types: tuple[Type, ...]) -> list[Effect]:
    return [Effect((t,), (option_of(t),)) for t in types]


def compare(types: tuple[Type, ...]) -> list[Effect]:
    return [Effect((t, t), (BOOL,)) for t in types]


SAME_NUMBERS = "two numbers of the same type"
SAME_INTEGERS = "two integers of the same type"
SAME_FLOATS = "two floats of the same type"

for name in ("+", "-", "*", "/"):
    define(name, binary(NUMBER_TYPES), SAME_NUMBERS)
    define(f"{name}.wrap", binary(INTEGER_TYPES), SAME_INTEGERS)
    define(f"{name}.sat", binary(INTEGER_TYPES), SAME_INTEGERS)
    define(f"{name}.ieee", binary(FLOAT_TYPES), SAME_FLOATS)
    define(f"{name}.checked", binary_checked(NUMBER_TYPES), SAME_NUMBERS)

define("%", binary(INTEGER_TYPES), SAME_INTEGERS)
define("%.wrap", binary(INTEGER_TYPES), SAME_INTEGERS)
define("%.sat", binary(INTEGER_TYPES), SAME_INTEGERS)
define("%.checked", binary_checked(INTEGER_TYPES), SAME_INTEGERS)

SIGNED_NUMBERS = SIGNED_INTEGER_TYPES + FLOAT_TYPES
for name in ("neg", "abs"):
    define(name, unary(SIGNED_NUMBERS), "a signed integer or a float")
    define(f"{name}.wrap", unary(SIGNED_INTEGER_TYPES), "a signed integer")
    define(f"{name}.sat", unary(SIGNED_INTEGER_TYPES), "a signed integer")
    define(f"{name}.checked", unary_checked(SIGNED_INTEGER_TYPES), "a signed integer")

define("sqrt", unary(FLOAT_TYPES), "a float")
define("sqrt.ieee", unary(FLOAT_TYPES), "a float")
define("sqrt.checked", unary_checked(FLOAT_TYPES), "a float")

for name in ("bit-and", "bit-or", "bit-xor"):
    define(name, binary(INTEGER_TYPES), SAME_INTEGERS)
define("bit-not", unary(INTEGER_TYPES), "an integer")

SHIFT = [Effect((t, U32), (t,)) for t in INTEGER_TYPES]
SHIFT_CHECKED = [Effect((t, U32), (option_of(t),)) for t in INTEGER_TYPES]
SHIFT_TAKES = "an integer and a `u32` amount"
for name in ("shl", "shr"):
    define(name, SHIFT, SHIFT_TAKES)
    define(f"{name}.wrap", SHIFT, SHIFT_TAKES)
    define(f"{name}.checked", SHIFT_CHECKED, SHIFT_TAKES)
for name in ("rotl", "rotr"):
    define(name, SHIFT, SHIFT_TAKES)
for name in ("count-ones", "leading-zeros", "trailing-zeros"):
    define(name, [Effect((t,), (U32,)) for t in INTEGER_TYPES], "an integer")

EQUALITY_TYPES = NUMBER_TYPES + (BOOL, CHAR, STR)
ORDERED_TYPES = NUMBER_TYPES + (BOOL, CHAR)
for name in ("==", "!="):
    define(name, compare(EQUALITY_TYPES), "two values of the same type")
for name in ("<", "<=", ">", ">="):
    define(name, compare(ORDERED_TYPES), "two ordered values of the same type")

for name in ("and", "or", "xor"):
    define(name, [Effect((BOOL, BOOL), (BOOL,))], "two `bool` values")
define("not", [Effect((BOOL,), (BOOL,))], "a `bool`")

DISPLAY_TYPES = NUMBER_TYPES + (BOOL, CHAR, STR)
define(
    "print", [Effect((t,), ()) for t in DISPLAY_TYPES], "a value that can be displayed"
)
