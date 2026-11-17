from __future__ import annotations

from dataclasses import dataclass

from inde.type_model import (
    BOOL,
    CHAR,
    STR,
    STRBUF,
    U32,
    UNIT,
    Effect,
    Type,
    array_of,
    option_of,
    quotation_of,
    result_of,
)

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

define("strbuf-new", (), (STRBUF,), "nothing")
define("strbuf-push", (STRBUF, CHAR), (STRBUF,), "a `strbuf` and a `char`")
define("strbuf-append", (STRBUF, STR), (STRBUF,), "a `strbuf` and a `str`")
define("strbuf-len", (STRBUF,), (STRBUF, U32), "a `strbuf`")
define("strbuf-snapshot", (STRBUF,), (STRBUF, STR), "a `strbuf`")
define("strbuf-freeze", (STRBUF,), (STR,), "a `strbuf`")
define("strbuf-from-str", (STR,), (STRBUF,), "a `str`")


Q, Q1, Q2, R = Type("Q"), Type("Q1"), Type("Q2"), Type("R")


def quotation(inputs: tuple[Type, ...], rows: tuple[str, str], qualifier: Type) -> Type:
    return quotation_of(Effect(inputs, (), rows=rows), qualifier)


BUILTINS["call"] = Builtin(
    "call",
    Effect((quotation((), ("..a", "..b"), Q),), (), rows=("..a", "..b")),
    "a quotation",
)
BUILTINS["curry"] = Builtin(
    "curry",
    Effect(
        (T, quotation((T,), ("..b", "..c"), Q)),
        (quotation((), ("..b", "..c"), R),),
    ),
    "a value and a quotation that takes it",
)
BUILTINS["compose"] = Builtin(
    "compose",
    Effect(
        (quotation((), ("..b", "..c"), Q1), quotation((), ("..c", "..d"), Q2)),
        (quotation((), ("..b", "..d"), R),),
    ),
    "two quotations where the first leaves what the second takes",
)


ARRAY = array_of(T)
BUFFER = Type("arraybuf", (T,))
INDEX = "a `u32` index"

define("array-len", (ARRAY,), (U32,), "an `array`", "Unrestricted")
define("array-get", (ARRAY, U32), (T,), f"an `array` and {INDEX}", "Unrestricted")
define(
    "array-get.checked",
    (ARRAY, U32),
    (option_of(T),),
    f"an `array` and {INDEX}",
    "Unrestricted",
)
SLICE_TAKES = "an `array` and two `u32` indices, start and end"
define("array-slice", (ARRAY, U32, U32), (ARRAY,), SLICE_TAKES, "Unrestricted")
define(
    "array-slice.checked",
    (ARRAY, U32, U32),
    (option_of(ARRAY),),
    SLICE_TAKES,
    "Unrestricted",
)

define("arraybuf-new", (), (BUFFER,), "nothing")
define("arraybuf-push", (BUFFER, T), (BUFFER,), "an `arraybuf` and an element")
define("arraybuf-pop", (BUFFER,), (BUFFER, option_of(T)), "an `arraybuf`")
define("arraybuf-len", (BUFFER,), (BUFFER, U32), "an `arraybuf`")
GET_TAKES = f"an `arraybuf` of unrestricted elements and {INDEX}"
define("arraybuf-get", (BUFFER, U32), (BUFFER, T), GET_TAKES, "Unrestricted")
define(
    "arraybuf-get.checked",
    (BUFFER, U32),
    (BUFFER, option_of(T)),
    GET_TAKES,
    "Unrestricted",
)
REPLACE_TAKES = f"an `arraybuf`, {INDEX} and an element"
define("arraybuf-replace", (BUFFER, U32, T), (BUFFER, T), REPLACE_TAKES)
define(
    "arraybuf-replace.checked",
    (BUFFER, U32, T),
    (BUFFER, result_of(T, T)),
    REPLACE_TAKES,
)
UNRESTRICTED_BUFFER = "an `arraybuf` of unrestricted elements"
define(
    "arraybuf-snapshot",
    (BUFFER,),
    (BUFFER, ARRAY),
    UNRESTRICTED_BUFFER,
    "Unrestricted",
)
define("arraybuf-freeze", (BUFFER,), (ARRAY,), UNRESTRICTED_BUFFER, "Unrestricted")
define("arraybuf-from-array", (ARRAY,), (BUFFER,), "an `array`", "Unrestricted")

BUILTINS["arraybuf-consume"] = Builtin(
    "arraybuf-consume",
    Effect(
        (BUFFER, quotation((T,), ("..a", "..a"), Q)),
        (),
        (("Q", "Unrestricted"),),
        ("..a", "..a"),
    ),
    "an `arraybuf` and a quotation to run on each element",
)
