from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

from inde.lexer import Location
from inde.type_model import (
    Effect,
    Stack,
    Type,
    is_unknown,
    is_unknown_row,
    substitute_effect,
)

ROW_NUMBERS = itertools.count(1)


@dataclass(frozen=True)
class Origin:
    description: str
    location: Location | None


@dataclass
class Inference:
    bindings: dict[str, Type] = field(default_factory=dict)
    origins: dict[str, Origin] = field(default_factory=dict)
    row_bindings: dict[str, Stack] = field(default_factory=dict)

    def fresh(self, description: str, location: Location | None) -> Type:
        name = f"?{len(self.origins) + 1}"
        self.origins[name] = Origin(description, location)
        return Type(name)

    def fresh_row(self) -> str:
        return f"?..{next(ROW_NUMBERS)}"

    def resolve(self, checked_type: Type) -> Type:
        if is_unknown(checked_type):
            if checked_type.name not in self.bindings:
                return checked_type
            return self.resolve(self.bindings[checked_type.name])
        if not checked_type.args and checked_type.effect is None:
            return checked_type
        effect = checked_type.effect
        return Type(
            checked_type.name,
            tuple(self.resolve(arg) for arg in checked_type.args),
            None if effect is None else self.resolve_effect(effect),
        )

    def resolve_effect(self, effect: Effect) -> Effect:
        if effect.rows is None:
            return Effect(
                tuple(self.resolve_all(effect.inputs)),
                tuple(self.resolve_all(effect.outputs)),
                effect.where,
            )
        inputs = self.flatten(Stack(effect.rows[0], effect.inputs))
        outputs = self.flatten(Stack(effect.rows[1], effect.outputs))
        return Effect(
            inputs.types, outputs.types, effect.where, (inputs.row, outputs.row)
        )

    def resolve_all(self, types: Sequence[Type]) -> list[Type]:
        return [self.resolve(checked_type) for checked_type in types]

    def flatten(self, stack: Stack) -> Stack:
        types = list(stack.types)
        row = stack.row
        while row in self.row_bindings:
            below = self.row_bindings[row]
            types = list(below.types) + types
            row = below.row
        return Stack(row, tuple(self.resolve_all(types)))

    def has_unknowns(self, checked_type: Type) -> bool:
        resolved = self.resolve(checked_type)
        if is_unknown(resolved):
            return True
        parts = list(resolved.args)
        if resolved.effect is not None:
            parts.extend(resolved.effect.inputs + resolved.effect.outputs)
        return any(self.has_unknowns(part) for part in parts)

    def occurs(self, name: str, checked_type: Type) -> bool:
        resolved = self.resolve(checked_type)
        if resolved.name == name:
            return True
        parts = list(resolved.args)
        effect = resolved.effect
        if effect is not None:
            if effect.rows is not None and name in effect.rows:
                return True
            parts.extend(effect.inputs + effect.outputs)
        return any(self.occurs(name, part) for part in parts)

    def unify(self, first: Type, second: Type) -> bool:
        first, second = self.resolve(first), self.resolve(second)
        if first == second:
            return True
        if is_unknown(first):
            return self.bind(first.name, second)
        if is_unknown(second):
            return self.bind(second.name, first)
        if first.name != second.name or len(first.args) != len(second.args):
            return False
        if not all(
            self.unify(a, b) for a, b in zip(first.args, second.args, strict=True)
        ):
            return False
        if first.effect is None or second.effect is None:
            return first.effect is None and second.effect is None
        return self.unify_effects(first.effect, second.effect)

    def unify_effects(self, first: Effect, second: Effect) -> bool:
        if first.rows is None and second.rows is None:
            return self.unify_all(first.inputs, second.inputs) and self.unify_all(
                first.outputs, second.outputs
            )
        first_rows = self.rows_of(first)
        second_rows = self.rows_of(second)
        return self.unify_stacks(
            Stack(first_rows[0], first.inputs), Stack(second_rows[0], second.inputs)
        ) and self.unify_stacks(
            Stack(first_rows[1], first.outputs), Stack(second_rows[1], second.outputs)
        )

    def rows_of(self, effect: Effect) -> tuple[str, str]:
        if effect.rows is not None:
            return effect.rows
        row = self.fresh_row()
        return (row, row)

    def bind(self, name: str, checked_type: Type) -> bool:
        if self.occurs(name, checked_type):
            return False
        self.bindings[name] = checked_type
        return True

    def unify_all(self, first: Sequence[Type], second: Sequence[Type]) -> bool:
        if len(first) != len(second):
            return False
        return all(self.unify(a, b) for a, b in zip(first, second, strict=True))

    def unify_stacks(self, first: Stack, second: Stack) -> bool:
        first, second = self.flatten(first), self.flatten(second)
        upper = list(first.types)
        lower = list(second.types)
        while upper and lower:
            if not self.unify(upper.pop(), lower.pop()):
                return False
        if upper or lower:
            rest_first = self.flatten(Stack(first.row, tuple(upper)))
            rest_second = self.flatten(Stack(second.row, tuple(lower)))
            if (rest_first.row, rest_second.row) != (first.row, second.row):
                return self.unify_stacks(rest_first, rest_second)
            if upper:
                return self.bind_row(second.row, rest_first)
            return self.bind_row(first.row, rest_second)
        first_row = self.flatten(Stack(first.row)).row
        second_row = self.flatten(Stack(second.row)).row
        if first_row == second_row:
            return True
        if is_unknown_row(first_row):
            return self.bind_row(first_row, Stack(second_row))
        return self.bind_row(second_row, Stack(first_row))

    def bind_row(self, row: str, stack: Stack) -> bool:
        row = self.flatten(Stack(row)).row
        if not is_unknown_row(row):
            return False
        stack = self.flatten(stack)
        if stack.row == row and not stack.types:
            return True
        if stack.row == row or any(self.occurs(row, t) for t in stack.types):
            return False
        self.row_bindings[row] = stack
        return True

    def instantiate(
        self, name: str, effect: Effect, location: Location | None
    ) -> Effect:
        replacements = {
            variable: self.fresh(f"the type `{variable}` of this `{name}`", location)
            for variable in effect.variables()
        }
        rows = {row: self.fresh_row() for row in effect.row_variables()}
        return substitute_effect(effect, replacements, rows)

    def generalize(self, effect: Effect) -> Effect:
        resolved = self.resolve_effect(effect)
        rows = resolved.rows
        if rows is not None and rows[0] == rows[1] and is_unknown_row(rows[0]):
            return Effect(resolved.inputs, resolved.outputs, resolved.where)
        return resolved
