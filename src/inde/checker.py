from __future__ import annotations

import difflib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NoReturn

from inde.ast import (
    ArrayLiteral,
    Body,
    BooleanLiteral,
    CharacterLiteral,
    ConstantDeclaration,
    ConstantReference,
    FloatLiteral,
    If,
    IntegerLiteral,
    Item,
    Let,
    Literal,
    Match,
    Program,
    Propagate,
    Quotation,
    QuotationType,
    StringLiteral,
    TypeDeclaration,
    TypeExpression,
    TypeVariable,
    While,
    WordDeclaration,
    WordReference,
)
from inde.builtins import BUILTINS, STACK_WORDS, Builtin, StackWord
from inde.errors import IndeError, IndeErrors
from inde.lexer import Location
from inde.parser import CORE_LIBRARY_ARGUMENTS, CORE_TYPE_ARGUMENTS
from inde.type_model import (
    BOOL,
    CHAR,
    CORE_LINEAR_TYPES,
    STR,
    UNIT,
    Effect,
    Type,
    array_of,
    format_types,
    from_number_type,
)


@dataclass(frozen=True)
class TypeInfo:
    name: str
    parameters: int
    location: Location | None = None


@dataclass(frozen=True)
class ConstantInfo:
    name: str
    type: Type
    value: Literal
    location: Location | None = None


@dataclass(frozen=True)
class WordInfo:
    name: str
    effect: Effect
    location: Location | None = None
    declaration: WordDeclaration | None = None


@dataclass(frozen=True)
class CheckedProgram:
    types: dict[str, TypeInfo]
    constants: dict[str, ConstantInfo]
    words: dict[str, WordInfo]
    linear_types: frozenset[str]


class _Abandon(Exception):
    # Stop checking one declaration without reporting anything
    pass


LIBRARY_CONSTRUCTORS: tuple[str, ...] = ("some", "none", "ok", "err")
DISPLAY_TYPE_NAMES: frozenset[str] = frozenset(
    {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64"}
    | {"bool", "char", "str"}
)


class Checker:
    def __init__(self) -> None:
        self.errors: list[IndeError] = []
        self.types: dict[str, TypeInfo] = {
            name: TypeInfo(name, parameters)
            for name, parameters in CORE_LIBRARY_ARGUMENTS.items()
        }
        self.constants: dict[str, ConstantInfo] = {}
        self.words: dict[str, WordInfo] = {}
        self.linear_types: set[str] = set()
        self.field_types: dict[str, list[Type]] = {}
        self.deferred: dict[str, str] = {
            name: "the core library constructors need generics"
            for name in LIBRARY_CONSTRUCTORS
        }
        self.broken: set[str] = set()

    def fail(self, message: str, location: Location | None) -> NoReturn:
        raise IndeError(message, location)

    def unsupported(self, what: str, location: Location | None) -> NoReturn:
        self.fail(f"{what} are not supported by the checker yet", location)

    @contextmanager
    def recover(self) -> Iterator[None]:
        try:
            yield
        except IndeError as error:
            self.errors.append(error)
        except _Abandon:
            pass

    def check_program(
        self, program: Program, require_main: bool = False
    ) -> CheckedProgram:
        declarations = program.declarations
        type_declarations = [d for d in declarations if isinstance(d, TypeDeclaration)]
        constant_declarations = [
            d for d in declarations if isinstance(d, ConstantDeclaration)
        ]
        word_declarations = [d for d in declarations if isinstance(d, WordDeclaration)]

        for type_declaration in type_declarations:
            with self.recover():
                self.declare_type(type_declaration)
        for constant_declaration in constant_declarations:
            with self.recover():
                self.declare_constant(constant_declaration)
        for type_declaration in type_declarations:
            with self.recover():
                self.declare_constructors(type_declaration)
        for word_declaration in word_declarations:
            with self.recover():
                self.declare_word(word_declaration)

        self.derive_linear_types()

        for word in list(self.words.values()):
            with self.recover():
                self.check_word(word)
        with self.recover():
            self.check_main(require_main)

        return CheckedProgram(
            types=self.types,
            constants=self.constants,
            words=self.words,
            linear_types=frozenset(self.linear_types),
        )

    def declare_type(self, declaration: TypeDeclaration) -> None:
        name = declaration.name
        if name in self.types:
            earlier = self.types[name].location
            where = (
                "by the core library" if earlier is None else f"on line {earlier.line}"
            )
            self.fail(f"type `{name}` is already defined {where}", declaration.location)
        for parameter in declaration.params:
            if declaration.params.count(parameter) > 1:
                self.fail(
                    f"type parameter `{parameter}` appears more than once in `{name}`",
                    declaration.location,
                )
        self.types[name] = TypeInfo(name, len(declaration.params), declaration.location)
        if declaration.linear:
            self.linear_types.add(name)

    def declare_constant(self, declaration: ConstantDeclaration) -> None:
        name = declaration.name
        if name in self.constants or name in self.broken:
            self.fail(f"constant `{name}` is already defined", declaration.location)
        try:
            constant_type = self.literal_type(declaration.value)
        except IndeError:
            self.broken.add(name)
            raise
        self.constants[name] = ConstantInfo(
            name, constant_type, declaration.value, declaration.location
        )

    def claim_name(self, name: str, location: Location | None) -> None:
        if name in BUILTINS or name in STACK_WORDS or name in LIBRARY_CONSTRUCTORS:
            self.fail(f"`{name}` is a built-in word and cannot be redefined", location)
        if name in self.words:
            earlier = self.words[name].location
            where = "" if earlier is None else f" on line {earlier.line}"
            self.fail(f"`{name}` is already defined{where}", location)
        if name in self.deferred or name in self.broken:
            self.fail(f"`{name}` is already defined", location)

    def declare_constructors(self, declaration: TypeDeclaration) -> None:
        if declaration.name not in self.types:
            return
        if self.types[declaration.name].location != declaration.location:
            return
        if declaration.params:
            for variant in declaration.variants:
                self.claim_name(variant.name, variant.location)
                self.deferred[variant.name] = (
                    f"the constructors of `{declaration.name}` need generics"
                )
            return

        result = Type(declaration.name)
        all_fields: list[Type] = []
        for variant in declaration.variants:
            try:
                self.claim_name(variant.name, variant.location)
                fields = tuple(self.resolve_type(field) for field in variant.fields)
            except IndeError:
                for unfinished in declaration.variants:
                    if unfinished.name not in self.words:
                        self.broken.add(unfinished.name)
                raise
            all_fields.extend(fields)
            self.words[variant.name] = WordInfo(
                variant.name, Effect(fields, (result,)), variant.location
            )
        self.field_types[declaration.name] = all_fields

    def declare_word(self, declaration: WordDeclaration) -> None:
        self.claim_name(declaration.name, declaration.location)
        try:
            effect = self.resolve_signature(declaration)
        except IndeError:
            self.broken.add(declaration.name)
            raise
        self.words[declaration.name] = WordInfo(
            declaration.name, effect, declaration.location, declaration
        )

    def resolve_signature(self, declaration: WordDeclaration) -> Effect:
        signature = declaration.signature
        if declaration.where:
            self.unsupported("`where` clauses", declaration.where[0].location)
        if signature.inputs.row != signature.outputs.row:
            self.unsupported(
                "signatures with two different row variables", signature.location
            )
        return Effect(
            tuple(self.resolve_type(t) for t in signature.inputs.types),
            tuple(self.resolve_type(t) for t in signature.outputs.types),
        )

    def resolve_type(self, expression: TypeExpression) -> Type:
        if isinstance(expression, TypeVariable):
            self.unsupported("type variables", expression.location)
        if isinstance(expression, QuotationType):
            self.unsupported("quotation types", expression.location)
        name = expression.name
        if name not in CORE_TYPE_ARGUMENTS and name not in self.types:
            self.fail(f"unknown type `{name}`", expression.location)
        return Type(name, tuple(self.resolve_type(arg) for arg in expression.args))

    def derive_linear_types(self) -> None:
        changed = True
        while changed:
            changed = False
            for name, fields in self.field_types.items():
                if name not in self.linear_types and any(
                    self.is_linear(field) for field in fields
                ):
                    self.linear_types.add(name)
                    changed = True

    def is_linear(self, checked_type: Type) -> bool:
        return (
            checked_type.name in CORE_LINEAR_TYPES
            or checked_type.name in self.linear_types
            or any(self.is_linear(arg) for arg in checked_type.args)
        )

    def check_word(self, word: WordInfo) -> None:
        if word.declaration is None:
            return
        stack = list(word.effect.inputs)
        self.check_body(word.declaration.body, stack)
        if tuple(stack) != word.effect.outputs:
            self.fail(
                f"`{word.name}` declares that it leaves "
                f"{format_types(word.effect.outputs)}, "
                f"but its body leaves {format_types(stack)}",
                word.location,
            )

    def check_body(self, body: Body, stack: list[Type]) -> list[Type]:
        for item in body:
            self.check_item(item, stack)
        return stack

    def check_item(self, item: Item, stack: list[Type]) -> None:
        if isinstance(
            item,
            (
                IntegerLiteral,
                FloatLiteral,
                CharacterLiteral,
                StringLiteral,
                BooleanLiteral,
                ArrayLiteral,
            ),
        ):
            stack.append(self.literal_type(item))
        elif isinstance(item, ConstantReference):
            stack.append(self.constant_type(item))
        elif isinstance(item, WordReference):
            self.check_call(item, stack)
        elif isinstance(item, Quotation):
            self.unsupported("quotations", item.location)
        elif isinstance(item, Propagate):
            self.unsupported("`?` forms", item.location)
        elif isinstance(item, If):
            self.unsupported("`if` forms", item.location)
        elif isinstance(item, While):
            self.unsupported("`while` forms", item.location)
        elif isinstance(item, Match):
            self.unsupported("`match` forms", item.location)
        else:
            assert isinstance(item, Let), f"unhandled item {type(item).__name__}"
            self.unsupported("`let` forms", item.location)

    def literal_type(self, literal: Literal) -> Type:
        if isinstance(literal, (IntegerLiteral, FloatLiteral)):
            return from_number_type(literal.type)
        if isinstance(literal, CharacterLiteral):
            return CHAR
        if isinstance(literal, StringLiteral):
            return STR
        if isinstance(literal, BooleanLiteral):
            return BOOL
        return self.array_literal_type(literal)

    def array_literal_type(self, literal: ArrayLiteral) -> Type:
        if not literal.elements:
            self.fail(
                "the element type of `{ }` comes from its context, "
                "which the checker cannot use yet",
                literal.location,
            )
        element_types = [
            self.constant_type(element)
            if isinstance(element, ConstantReference)
            else self.literal_type(element)
            for element in literal.elements
        ]
        first = element_types[0]
        for element, element_type in zip(literal.elements, element_types, strict=False):
            if element_type != first:
                self.fail(
                    "the elements of an array literal must have one type; "
                    f"found `{first}` and `{element_type}`",
                    element.location,
                )
        return array_of(first)

    def constant_type(self, reference: ConstantReference) -> Type:
        if reference.name in self.constants:
            return self.constants[reference.name].type
        if reference.name in self.broken:
            raise _Abandon
        self.fail(
            f"unknown constant `{reference.name}`"
            f"{self.suggestion(reference.name, self.constants)}",
            reference.location,
        )

    def check_call(self, call: WordReference, stack: list[Type]) -> None:
        name = call.name
        if name in self.words:
            self.apply_effect(name, self.words[name].effect, stack, call.location)
        elif name in BUILTINS:
            self.apply_builtin(BUILTINS[name], stack, call.location)
        elif name in STACK_WORDS:
            self.apply_stack_word(STACK_WORDS[name], stack, call.location)
        elif name in self.broken:
            raise _Abandon
        elif name in self.deferred:
            self.fail(
                f"`{name}` cannot be used yet: {self.deferred[name]}", call.location
            )
        else:
            known = [*self.words, *BUILTINS, *STACK_WORDS]
            self.fail(
                f"unknown word `{name}`{self.suggestion(name, known)}", call.location
            )

    def require_depth(
        self, name: str, needed: int, stack: list[Type], location: Location | None
    ) -> None:
        if len(stack) >= needed:
            return
        values = "1 value" if needed == 1 else f"{needed} values"
        holds = "is empty" if not stack else f"holds only {format_types(stack)}"
        self.fail(f"`{name}` takes {values}, but the stack {holds}", location)

    def apply_effect(
        self, name: str, effect: Effect, stack: list[Type], location: Location | None
    ) -> None:
        count = len(effect.inputs)
        self.require_depth(name, count, stack, location)
        found = stack[len(stack) - count :]
        if tuple(found) != effect.inputs:
            self.fail(
                f"`{name}` takes {format_types(effect.inputs)}, "
                f"found {format_types(found)}",
                location,
            )
        del stack[len(stack) - count :]
        stack.extend(effect.outputs)

    def apply_builtin(
        self, builtin: Builtin, stack: list[Type], location: Location | None
    ) -> None:
        count = builtin.arguments
        self.require_depth(builtin.name, count, stack, location)
        found = tuple(stack[len(stack) - count :])
        for form in builtin.forms:
            if form.inputs == found:
                del stack[len(stack) - count :]
                stack.extend(form.outputs)
                return
        self.fail(
            f"`{builtin.name}` takes {builtin.takes}, found {format_types(found)}",
            location,
        )

    def apply_stack_word(
        self, word: StackWord, stack: list[Type], location: Location | None
    ) -> None:
        self.require_depth(word.name, word.arguments, stack, location)
        taken = stack[len(stack) - word.arguments :]
        for position in word.copied():
            if self.is_linear(taken[position]):
                self.fail(
                    f"`{word.name}` cannot copy a value of the linear type "
                    f"`{taken[position]}`",
                    location,
                )
        for position in word.discarded():
            if self.is_linear(taken[position]):
                self.fail(
                    f"`{word.name}` cannot discard a value of the linear type "
                    f"`{taken[position]}`",
                    location,
                )
        del stack[len(stack) - word.arguments :]
        stack.extend(taken[position] for position in word.result)

    def suggestion(self, name: str, known: Iterable[str]) -> str:
        matches = difflib.get_close_matches(name, list(known), n=1)
        return f"; did you mean `{matches[0]}`?" if matches else ""

    def check_main(self, require_main: bool) -> None:
        if "main" not in self.words:
            if "main" in self.broken or not require_main:
                return
            self.fail("the program has no `main`", None)
        main = self.words["main"]
        outputs = main.effect.outputs
        returns_result = (
            len(outputs) == 1
            and outputs[0].name == "Result"
            and outputs[0].args[0] == UNIT
        )
        if main.effect.inputs or not (outputs == () or returns_result):
            self.fail(
                "`main` must have the signature `( -> )` or "
                f"`( -> Result[unit, E] )`, found `{main.effect}`",
                main.location,
            )
        if returns_result and outputs[0].args[1].name not in DISPLAY_TYPE_NAMES:
            self.fail(
                f"the error type of `main` must be displayable; "
                f"`{outputs[0].args[1]}` is not",
                main.location,
            )


def error_order(error: IndeError) -> tuple[int, int, int]:
    if error.location is None:
        return (1, 0, 0)
    return (0, error.location.line, error.location.column)


def check(program: Program, require_main: bool = False) -> CheckedProgram:
    checker = Checker()
    checked = checker.check_program(program, require_main)
    if checker.errors:
        raise IndeErrors(sorted(checker.errors, key=error_order))
    return checked
