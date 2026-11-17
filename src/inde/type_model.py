from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inde.lexer import NumberType


@dataclass(frozen=True)
class Type:
    name: str
    args: tuple[Type, ...] = ()

    def __str__(self) -> str:
        if is_unknown(self):
            return "_"
        if not self.args:
            return self.name
        return f"{self.name}[{', '.join(str(arg) for arg in self.args)}]"


@dataclass(frozen=True)
class Effect:
    inputs: tuple[Type, ...] = ()
    outputs: tuple[Type, ...] = ()
    where: tuple[tuple[str, str], ...] = ()

    def __str__(self) -> str:
        parts = [str(t) for t in self.inputs] + ["->"] + [str(t) for t in self.outputs]
        text = f"( {' '.join(parts)} )"
        if not self.where:
            return text
        by_variable: dict[str, list[str]] = {}
        for variable, constraint in self.where:
            by_variable.setdefault(variable, []).append(constraint)
        clauses = [f"{v}: {' & '.join(cs)}" for v, cs in by_variable.items()]
        return f"{text} where {', '.join(clauses)}"

    def variables(self) -> tuple[str, ...]:
        found: list[str] = []
        for part in self.inputs + self.outputs:
            for name in variables_of(part):
                if name not in found:
                    found.append(name)
        return tuple(found)


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


def is_unknown(checked_type: Type) -> bool:
    return checked_type.name.startswith("?")


def is_type_variable(checked_type: Type) -> bool:
    name = checked_type.name
    return name[0].isupper() and (len(name) == 1 or name[1:].isdigit())


def variables_of(checked_type: Type) -> list[str]:
    if is_type_variable(checked_type):
        return [checked_type.name]
    found: list[str] = []
    for arg in checked_type.args:
        for name in variables_of(arg):
            if name not in found:
                found.append(name)
    return found


def substitute(checked_type: Type, replacements: dict[str, Type]) -> Type:
    if checked_type.name in replacements:
        return replacements[checked_type.name]
    if not checked_type.args:
        return checked_type
    return Type(
        checked_type.name,
        tuple(substitute(arg, replacements) for arg in checked_type.args),
    )


def format_types(types: Sequence[Type]) -> str:
    if not types:
        return "nothing"
    return f"`{' '.join(str(t) for t in types)}`"
