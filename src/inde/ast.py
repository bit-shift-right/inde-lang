from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING

from inde.lexer import Location, NumberType

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


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
class ConstantReference(Node):
    name: str


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
    literal: Literal | ConstantReference


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
    | ConstantReference
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


@dataclass(frozen=True, kw_only=True)
class ConstantDeclaration(Node):
    name: str
    value: Literal


Declaration = WordDeclaration | TypeDeclaration | ConstantDeclaration


@dataclass(frozen=True, kw_only=True)
class Program(Node):
    declarations: tuple[Declaration, ...] = ()


def dump(node: DataclassInstance, indent: str = "  ") -> str:
    lines: list[str] = []

    def is_node(value: DataclassInstance) -> bool:
        return is_dataclass(value) and not isinstance(value, type)

    def scalar(value: DataclassInstance) -> str:
        if isinstance(value, Enum):
            return str(value.value)
        return repr(value)

    def visit(value: DataclassInstance, depth: int, label: str) -> None:
        prefix = indent * depth + (f"{label}: " if label else "")
        if is_node(value):
            children: list[tuple[str, DataclassInstance]] = []
            attributes: list[str] = []
            for f in fields(value):
                if f.name == "span":
                    continue
                v = getattr(value, f.name)
                if f.default is not MISSING and v == f.default:
                    continue
                if is_node(v) or (isinstance(v, tuple) and v and is_node(v[0])):
                    children.append((f.name, v))
                else:
                    attributes.append(f"{f.name}={scalar(v)}")
            lines.append(
                prefix
                + type(value).__name__
                + ("" if not attributes else " " + " ".join(attributes))
            )
            for name, child in children:
                if isinstance(child, tuple):
                    lines.append(indent * (depth + 1) + f"{name}:")
                    for item in child:
                        visit(item, depth + 2, "")
                else:
                    visit(child, depth + 1, name)
        else:
            lines.append(prefix + scalar(value))

    visit(node, 0, "")
    return "\n".join(lines)
