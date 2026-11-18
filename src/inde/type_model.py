from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inde.lexer import NumberType


@dataclass(frozen=True)
class Type:
    name: str
    args: tuple[Type, ...] = ()
    effect: Effect | None = None

    def __str__(self) -> str:
        if is_unknown(self):
            return "_"
        if self.effect is not None:
            qualifier = self.args[0]
            if is_type_variable(qualifier):
                return qualifier.name
            prefix = "linear " if qualifier == LINEAR else ""
            return f"{prefix}[ {self.effect.describe()} ]"
        if not self.args:
            return self.name
        return f"{self.name}[{', '.join(str(arg) for arg in self.args)}]"


@dataclass(frozen=True)
class Stack:
    row: str
    types: tuple[Type, ...] = ()


@dataclass(frozen=True)
class Effect:
    inputs: tuple[Type, ...] = ()
    outputs: tuple[Type, ...] = ()
    where: tuple[tuple[str, str], ...] = ()
    rows: tuple[str, str] | None = None

    def describe(self) -> str:
        inputs = [str(t) for t in self.inputs]
        outputs = [str(t) for t in self.outputs]
        if self.rows is not None:
            inputs.insert(0, format_row(self.rows[0]))
            outputs.insert(0, format_row(self.rows[1]))
        return " ".join([*inputs, "->", *outputs])

    def __str__(self) -> str:
        text = f"( {self.describe()} )"
        by_variable: dict[str, list[str]] = {}
        for variable, effect in quotation_constraints(self.inputs + self.outputs):
            by_variable.setdefault(variable, []).append(f"[ {effect.describe()} ]")
        for variable, constraint in self.where:
            by_variable.setdefault(variable, []).append(constraint)
        if not by_variable:
            return text
        clauses = [f"{v}: {' & '.join(cs)}" for v, cs in by_variable.items()]
        return f"{text} where {', '.join(clauses)}"

    def variables(self) -> tuple[str, ...]:
        found: list[str] = []
        for part in self.inputs + self.outputs:
            for name in variables_of(part):
                if name not in found:
                    found.append(name)
        return tuple(found)

    def row_variables(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.rows is not None:
            found.extend(row for row in self.rows if row not in found)
        for part in self.inputs + self.outputs:
            for row in rows_of(part):
                if row not in found:
                    found.append(row)
        return tuple(found)


I8, I16, I32, I64 = Type("i8"), Type("i16"), Type("i32"), Type("i64")
U8, U16, U32, U64 = Type("u8"), Type("u16"), Type("u32"), Type("u64")
F32, F64 = Type("f32"), Type("f64")
BOOL, CHAR, STR, UNIT = Type("bool"), Type("char"), Type("str"), Type("unit")
STRBUF = Type("strbuf")
FILE = Type("File")
IO_ERROR = Type("IoError")
IO_ERROR_KIND = Type("IoErrorKind")
OPAQUE_LIBRARY_TYPES: tuple[Type, ...] = (FILE, IO_ERROR)

SIGNED_INTEGER_TYPES: tuple[Type, ...] = (I8, I16, I32, I64)
UNSIGNED_INTEGER_TYPES: tuple[Type, ...] = (U8, U16, U32, U64)
INTEGER_TYPES: tuple[Type, ...] = SIGNED_INTEGER_TYPES + UNSIGNED_INTEGER_TYPES
FLOAT_TYPES: tuple[Type, ...] = (F32, F64)
NUMBER_TYPES: tuple[Type, ...] = INTEGER_TYPES + FLOAT_TYPES


CORE_LINEAR_TYPES: tuple[str, ...] = ("strbuf", "arraybuf", "File")

QUOTATION = "[ ]"
UNRESTRICTED = Type("unrestricted")
LINEAR = Type("linear")

IMPLICIT_ROW = ".."


def quotation_of(effect: Effect, qualifier: Type = UNRESTRICTED) -> Type:
    return Type(QUOTATION, (qualifier,), effect)


def is_quotation(checked_type: Type) -> bool:
    return checked_type.effect is not None


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


def is_unknown_row(row: str) -> bool:
    return row.startswith("?")


def format_row(row: str) -> str:
    return "..?" if is_unknown_row(row) else row


def variables_of(checked_type: Type) -> list[str]:
    if is_type_variable(checked_type):
        return [checked_type.name]
    found: list[str] = []
    parts = list(checked_type.args)
    if checked_type.effect is not None:
        parts.extend(checked_type.effect.inputs + checked_type.effect.outputs)
    for part in parts:
        for name in variables_of(part):
            if name not in found:
                found.append(name)
    return found


def rows_of(checked_type: Type) -> list[str]:
    found: list[str] = []
    effect = checked_type.effect
    if effect is not None and effect.rows is not None:
        found.extend(row for row in effect.rows if row not in found)
    parts = list(checked_type.args)
    if effect is not None:
        parts.extend(effect.inputs + effect.outputs)
    for part in parts:
        for row in rows_of(part):
            if row not in found:
                found.append(row)
    return found


def quotation_constraints(types: Sequence[Type]) -> list[tuple[str, Effect]]:
    found: list[tuple[str, Effect]] = []
    for checked_type in types:
        if checked_type.effect is not None and is_type_variable(checked_type.args[0]):
            entry = (checked_type.args[0].name, checked_type.effect)
            if entry not in found:
                found.append(entry)
        parts = list(checked_type.args)
        if checked_type.effect is not None:
            parts.extend(checked_type.effect.inputs + checked_type.effect.outputs)
        for entry in quotation_constraints(parts):
            if entry not in found:
                found.append(entry)
    return found


def substitute(
    checked_type: Type,
    replacements: dict[str, Type],
    rows: dict[str, str] | None = None,
) -> Type:
    if checked_type.name in replacements:
        return replacements[checked_type.name]
    if not checked_type.args and checked_type.effect is None:
        return checked_type
    effect = checked_type.effect
    if effect is not None:
        effect = substitute_effect(effect, replacements, rows or {})
    return Type(
        checked_type.name,
        tuple(substitute(arg, replacements, rows) for arg in checked_type.args),
        effect,
    )


def substitute_effect(
    effect: Effect, replacements: dict[str, Type], rows: dict[str, str]
) -> Effect:
    return Effect(
        tuple(substitute(t, replacements, rows) for t in effect.inputs),
        tuple(substitute(t, replacements, rows) for t in effect.outputs),
        tuple(
            (
                replacements[variable].name if variable in replacements else variable,
                constraint,
            )
            for variable, constraint in effect.where
        ),
        None
        if effect.rows is None
        else (
            rows.get(effect.rows[0], effect.rows[0]),
            rows.get(effect.rows[1], effect.rows[1]),
        ),
    )


def format_types(types: Sequence[Type]) -> str:
    if not types:
        return "nothing"
    return f"`{' '.join(str(t) for t in types)}`"
