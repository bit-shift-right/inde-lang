from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

from inde.ast import Body
from inde.type_model import (
    BOOL,
    CHAR,
    STR,
    Effect,
    Type,
    array_of,
    option_of,
    quotation_of,
)


@dataclass(frozen=True)
class Data:
    constructor: str
    fields: tuple[Value, ...] = ()


@dataclass(frozen=True)
class Quoted:
    parts: tuple[Body | Value, ...] = ()


@dataclass(frozen=True)
class Closure:
    parts: tuple[str | Value, ...] = ()


@dataclass(frozen=True)
class IoFailure:
    kind: str
    path: str | None
    message: str

    def text(self) -> str:
        return self.message if self.path is None else f"{self.path}: {self.message}"


@dataclass(eq=False)
class OpenFile:
    handle: BinaryIO
    path: str
    writing: bool


Payload = (
    int
    | float
    | bool
    | str
    | tuple["Value", ...]
    | list["Value"]
    | Data
    | Quoted
    | Closure
    | IoFailure
    | OpenFile
)


@dataclass(frozen=True)
class Value:
    type: Type
    data: Payload


def boolean(value: bool) -> Value:
    return Value(BOOL, value)


def character(value: str) -> Value:
    return Value(CHAR, value)


def string(value: str) -> Value:
    return Value(STR, value)


def array(element_type: Type, elements: tuple[Value, ...]) -> Value:
    return Value(array_of(element_type), elements)


def some(value: Value) -> Value:
    return Value(option_of(value.type), Data("some", (value,)))


def none(value_type: Type) -> Value:
    return Value(option_of(value_type), Data("none"))


QUOTED = quotation_of(Effect())


def quoted(parts: tuple[Body | Value, ...]) -> Value:
    return Value(QUOTED, Quoted(parts))
