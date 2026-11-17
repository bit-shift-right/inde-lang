from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from inde.lexer import Location
from inde.type_model import Effect, Type, is_unknown, substitute


@dataclass(frozen=True)
class Origin:
    description: str
    location: Location | None


@dataclass
class Inference:
    bindings: dict[str, Type] = field(default_factory=dict)
    origins: dict[str, Origin] = field(default_factory=dict)

    def fresh(self, description: str, location: Location | None) -> Type:
        name = f"?{len(self.origins) + 1}"
        self.origins[name] = Origin(description, location)
        return Type(name)

    def resolve(self, checked_type: Type) -> Type:
        if is_unknown(checked_type):
            if checked_type.name not in self.bindings:
                return checked_type
            return self.resolve(self.bindings[checked_type.name])
        if not checked_type.args:
            return checked_type
        return Type(
            checked_type.name, tuple(self.resolve(arg) for arg in checked_type.args)
        )

    def resolve_all(self, types: Sequence[Type]) -> list[Type]:
        return [self.resolve(checked_type) for checked_type in types]

    def has_unknowns(self, checked_type: Type) -> bool:
        resolved = self.resolve(checked_type)
        return is_unknown(resolved) or any(
            self.has_unknowns(arg) for arg in resolved.args
        )

    def occurs(self, name: str, checked_type: Type) -> bool:
        resolved = self.resolve(checked_type)
        return resolved.name == name or any(
            self.occurs(name, arg) for arg in resolved.args
        )

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
        return all(
            self.unify(a, b) for a, b in zip(first.args, second.args, strict=True)
        )

    def bind(self, name: str, checked_type: Type) -> bool:
        if self.occurs(name, checked_type):
            return False
        self.bindings[name] = checked_type
        return True

    def unify_all(self, first: Sequence[Type], second: Sequence[Type]) -> bool:
        if len(first) != len(second):
            return False
        return all(self.unify(a, b) for a, b in zip(first, second, strict=True))

    def instantiate(
        self, name: str, effect: Effect, location: Location | None
    ) -> Effect:
        replacements = {
            variable: self.fresh(f"the type `{variable}` of this `{name}`", location)
            for variable in effect.variables()
        }
        return Effect(
            tuple(substitute(t, replacements) for t in effect.inputs),
            tuple(substitute(t, replacements) for t in effect.outputs),
            tuple(
                (replacements[variable].name, constraint)
                for variable, constraint in effect.where
            ),
        )
