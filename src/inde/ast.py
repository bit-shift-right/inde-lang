from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)

from inde.lexer import Location, NumberType


@dataclass(frozen=True, kw_only=True)
class Node:
    location: Location | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, kw_only=True)
class TypeReference(Node):
    name: str
    args: tuple[TypeExpression, ...] = ()


@dataclass(frozen=True, kw_only=True)
class TypeVariable(Node):
    name: str


@dataclass(frozen=True, kw_only=True)
class QuotationType(Node):
    effect: StackEffect
    linear: bool = False


TypeExpression = TypeReference | TypeVariable | QuotationType


@dataclass(frozen=True, kw_only=True)
class StackType(Node):
    row: str | None = None
    types: tuple[TypeExpression, ...] = ()


@dataclass(frozen=True, kw_only=True)
class StackEffect(Node):
    inputs: StackType
    outputs: StackType


@dataclass(frozen=True, kw_only=True)
class ConstraintReference(Node):
    name: str


Bound = ConstraintReference | QuotationType


@dataclass(frozen=True, kw_only=True)
class Constraint(Node):
    var: str
    bounds: tuple[Bound, ...]


@dataclass(frozen=True, kw_only=True)
class IntegerLiteral(Node):
    value: int
    type: NumberType


@dataclass(frozen=True, kw_only=True)
class FloatLiteral(Node):
    value: float
    type: NumberType


@dataclass(frozen=True, kw_only=True)
class CharacterLiteral(Node):
    value: str


@dataclass(frozen=True, kw_only=True)
class StringLiteral(Node):
    value: str


@dataclass(frozen=True, kw_only=True)
class BooleanLiteral(Node):
    value: bool


@dataclass(frozen=True, kw_only=True)
class ArrayLiteral(Node):
    elements: tuple[Literal, ...]


Literal = (
    IntegerLiteral
    | FloatLiteral
    | CharacterLiteral
    | StringLiteral
    | BooleanLiteral
    | ArrayLiteral
)


@dataclass(frozen=True, kw_only=True)
class WordReference(Node):
    name: str


@dataclass(frozen=True, kw_only=True)
class Propagate(Node):
    # Used early return nodes. No body
    pass


@dataclass(frozen=True, kw_only=True)
class Quotation(Node):
    body: Body


@dataclass(frozen=True, kw_only=True)
class If(Node):
    then_body: Body
    else_body: Body | None = None


@dataclass(frozen=True, kw_only=True)
class While(Node):
    condition: Body
    body: Body


@dataclass(frozen=True, kw_only=True)
class ConstructorPattern(Node):
    name: str


@dataclass(frozen=True, kw_only=True)
class LiteralPattern(Node):
    literal: Literal


Pattern = ConstructorPattern | LiteralPattern


@dataclass(frozen=True, kw_only=True)
class MatchArm(Node):
    pattern: Pattern
    body: Body


@dataclass(frozen=True, kw_only=True)
class Match(Node):
    arms: tuple[MatchArm, ...]
    else_body: Body | None = None


@dataclass(frozen=True, kw_only=True)
class Let(Node):
    names: tuple[str, ...]
    body: Body


Item = (
    IntegerLiteral
    | FloatLiteral
    | CharacterLiteral
    | StringLiteral
    | BooleanLiteral
    | ArrayLiteral
    | WordReference
    | Propagate
    | Quotation
    | If
    | While
    | Match
    | Let
)


Body = tuple[Item, ...]


@dataclass(frozen=True, kw_only=True)
class WordDeclaration(Node):
    name: str
    signature: StackEffect
    where: tuple[Constraint, ...] = ()
    body: Body = ()


@dataclass(frozen=True, kw_only=True)
class Variant(Node):
    name: str
    fields: tuple[TypeExpression, ...] = ()


@dataclass(frozen=True, kw_only=True)
class TypeDeclaration(Node):
    name: str
    params: tuple[str, ...] = ()
    variants: tuple[Variant, ...] = ()
    linear: bool = False


Declaration = WordDeclaration | TypeDeclaration


@dataclass(frozen=True, kw_only=True)
class Program(Node):
    decls: tuple[Declaration, ...] = ()
