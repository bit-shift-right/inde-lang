from __future__ import annotations

from dataclasses import dataclass

from inde.type_model import BOOL, U32, UNIT, Effect, Type, option_of

CONSTRAINTS: tuple[str, ...] = (
    "Unrestricted",
    "Integer",
    "Signed",
    "Float",
    "Num",
    "Eq",
    "PartialEq",
    "Ord",
    "PartialOrd",
    "Display",
)

IMPLIED_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    "Signed": ("Integer",),
    "Integer": ("Num", "Eq", "Ord"),
    "Float": ("Num",),
    "Num": ("PartialEq", "PartialOrd", "Display", "Unrestricted"),
    "Eq": ("PartialEq",),
    "Ord": ("Eq", "PartialOrd"),
}


@dataclass(frozen=True)
class Builtin:
    name: str
    effect: Effect
    takes: str

    @property
    def arguments(self) -> int:
        return len(self.effect.inputs)


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

T = Type("T")


def define(
    name: str,
    inputs: tuple[Type, ...],
    outputs: tuple[Type, ...],
    takes: str,
    constraint: str | None = None,
) -> None:
    assert name not in BUILTINS and name not in STACK_WORDS, name
    where = () if constraint is None else (("T", constraint),)
    BUILTINS[name] = Builtin(name, Effect(inputs, outputs, where), takes)


SAME_NUMBERS = "two numbers of the same type"
SAME_INTEGERS = "two integers of the same type"
SAME_FLOATS = "two floats of the same type"

for name in ("+", "-", "*", "/"):
    define(name, (T, T), (T,), SAME_NUMBERS, "Num")
    define(f"{name}.wrap", (T, T), (T,), SAME_INTEGERS, "Integer")
    define(f"{name}.sat", (T, T), (T,), SAME_INTEGERS, "Integer")
    define(f"{name}.ieee", (T, T), (T,), SAME_FLOATS, "Float")
    define(f"{name}.checked", (T, T), (option_of(T),), SAME_NUMBERS, "Num")

define("%", (T, T), (T,), SAME_INTEGERS, "Integer")
define("%.wrap", (T, T), (T,), SAME_INTEGERS, "Integer")
define("%.sat", (T, T), (T,), SAME_INTEGERS, "Integer")
define("%.checked", (T, T), (option_of(T),), SAME_INTEGERS, "Integer")

for name in ("neg", "abs"):
    define(name, (T,), (T,), "a signed integer or a float", "Signed|Float")
    define(f"{name}.wrap", (T,), (T,), "a signed integer", "Signed")
    define(f"{name}.sat", (T,), (T,), "a signed integer", "Signed")
    define(f"{name}.checked", (T,), (option_of(T),), "a signed integer", "Signed")

define("sqrt", (T,), (T,), "a float", "Float")
define("sqrt.ieee", (T,), (T,), "a float", "Float")
define("sqrt.checked", (T,), (option_of(T),), "a float", "Float")

for name in ("bit-and", "bit-or", "bit-xor"):
    define(name, (T, T), (T,), SAME_INTEGERS, "Integer")
define("bit-not", (T,), (T,), "an integer", "Integer")

SHIFT_TAKES = "an integer and a `u32` amount"
for name in ("shl", "shr"):
    define(name, (T, U32), (T,), SHIFT_TAKES, "Integer")
    define(f"{name}.wrap", (T, U32), (T,), SHIFT_TAKES, "Integer")
    define(f"{name}.checked", (T, U32), (option_of(T),), SHIFT_TAKES, "Integer")
for name in ("rotl", "rotr"):
    define(name, (T, U32), (T,), SHIFT_TAKES, "Integer")
for name in ("count-ones", "leading-zeros", "trailing-zeros"):
    define(name, (T,), (U32,), "an integer", "Integer")

for name in ("==", "!="):
    define(name, (T, T), (BOOL,), "two values of the same type", "PartialEq")
for name in ("<", "<=", ">", ">="):
    define(name, (T, T), (BOOL,), "two ordered values of the same type", "PartialOrd")

for name in ("and", "or", "xor"):
    define(name, (BOOL, BOOL), (BOOL,), "two `bool` values")
define("not", (BOOL,), (BOOL,), "a `bool`")

define("print", (T,), (), "a value that can be displayed", "Display")
define("unit", (), (UNIT,), "nothing")
