from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inde.lexer import NumberType


@dataclass(frozen=True)
class Type:
    name: str
    args: tuple[Type, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}[{', '.join(str(arg) for arg in self.args)}]"


@dataclass(frozen=True)
class Effect:
    inputs: tuple[Type, ...] = ()
    outputs: tuple[Type, ...] = ()

    def __str__(self) -> str:
        parts = [str(t) for t in self.inputs] + ["->"] + [str(t) for t in self.outputs]
        return f"( {' '.join(parts)} )"


I8, I16, I32, I64 = Type("i8"), Type("i16"), Type("i32"), Type("i64")
U8, U16, U32, U64 = Type("u8"), Type("u16"), Type("u32"), Type("u64")
F32, F64 = Type("f32"), Type("f64")
BOOL, CHAR, STR, UNIT = Type("bool"), Type("char"), Type("str"), Type("unit")
STRBUF = Type("strbuf")

SIGNED_INTEGER_TYPES: tuple[Type, ...] = (I8, I16, I32, I64)
UNSIGNED_INTEGER_TYPES: tuple[Type, ...] = (U8, U16, U32, U64)
INTEGER_TYPES: tuple[Type, ...] = SIGNED_INTEGER_TYPES + UNSIGNED_INTEGER_TYPES
FLOAT_TYPES: tuple[Type, ...] = (F32, F64)
NUMBER_TYPES: tuple[Type, ...] = INTEGER_TYPES + FLOAT_TYPES


CORE_LINEAR_TYPES: tuple[str, ...] = ("strbuf", "arraybuf")


def array_of(element: Type) -> Type:
    return Type("array", (element,))


def option_of(value: Type) -> Type:
    return Type("Option", (value,))


def result_of(value: Type, error: Type) -> Type:
    return Type("Result", (value, error))


def from_number_type(number_type: NumberType) -> Type:
    return Type(number_type.value)


def format_types(types: Sequence[Type]) -> str:
    if not types:
        return "nothing"
    return f"`{' '.join(str(t) for t in types)}`"
