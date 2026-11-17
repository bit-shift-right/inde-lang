from __future__ import annotations

from dataclasses import dataclass

from inde.type_model import BOOL, CHAR, STR, Type, array_of, option_of


@dataclass(frozen=True)
class Data:
    constructor: str
    fields: tuple[Value, ...] = ()


Payload = int | float | bool | str | tuple["Value", ...] | Data


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
