from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import NoReturn

from inde.ast import (
    ArrayLiteral,
    Body,
    BooleanLiteral,
    CharacterLiteral,
    ConstantDeclaration,
    ConstantReference,
    ConstructorPattern,
    FloatLiteral,
    If,
    IntegerLiteral,
    Item,
    Let,
    Literal,
    Match,
    Pattern,
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
from inde.builtins import (
    BUILTINS,
    CONSTRAINTS,
    IMPLIED_CONSTRAINTS,
    STACK_WORDS,
    StackWord,
)
from inde.errors import IndeError, IndeErrors
from inde.inference import Inference
from inde.lexer import Location, quote
from inde.parser import CORE_TYPE_ARGUMENTS, parse
from inde.type_model import (
    BOOL,
    CHAR,
    CORE_LINEAR_TYPES,
    FLOAT_TYPES,
    INTEGER_TYPES,
    NUMBER_TYPES,
    SIGNED_INTEGER_TYPES,
    STR,
    UNIT,
    Effect,
    Type,
    array_of,
    format_types,
    from_number_type,
    is_type_variable,
    is_unknown,
    substitute,
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
    kind: str = "word"


@dataclass(frozen=True)
class Obligation:
    type: Type
    constraint: str
    location: Location | None
    describe: Callable[[Type], str]


@dataclass(frozen=True)
class CheckedProgram:
    types: dict[str, TypeInfo]
    constants: dict[str, ConstantInfo]
    words: dict[str, WordInfo]
    linear_types: frozenset[str]


Scope = dict[str, Type]


class _Abandon(Exception):
    pass


LIBRARY = parse(
    "type Option[T] = none | some( T )\n"
    "type Result[T, E] = ok( T ) | err( E )\n"
    "type Pair[A, B] = pair( A B )\n"
)


class Checker:
    def __init__(self) -> None:
        self.errors: list[IndeError] = []
        self.types: dict[str, TypeInfo] = {}
        self.constants: dict[str, ConstantInfo] = {}
        self.words: dict[str, WordInfo] = {}
        self.linear_types: set[str] = set()
        self.field_types: dict[str, list[Type]] = {}
        self.variants: dict[str, tuple[str, ...]] = {}
        self.local_names: dict[str, str] = {}
        self.current_word: str | None = None
        self.broken: set[str] = set()
        self.inference = Inference()
        self.assumptions: dict[str, frozenset[str]] = {}
        self.obligations: list[Obligation] = []
        self.declare_library()

    def declare_library(self) -> None:
        for declaration in LIBRARY.declarations:
            assert isinstance(declaration, TypeDeclaration)
            self.declare_type(declaration)
            self.declare_constructors(declaration)
        self.types = {
            name: replace(info, location=None) for name, info in self.types.items()
        }
        self.words = {
            name: replace(info, location=None) for name, info in self.words.items()
        }

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
                self.check_array_elements(word)
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
            self.begin()
            constant_type = self.literal_type(declaration.value)
            self.finish()
        except IndeError:
            self.broken.add(name)
            raise
        self.constants[name] = ConstantInfo(
            name,
            self.inference.resolve(constant_type),
            declaration.value,
            declaration.location,
        )

    def claim_name(self, name: str, location: Location | None) -> None:
        if name in BUILTINS or name in STACK_WORDS:
            self.fail(f"`{name}` is a built-in word and cannot be redefined", location)
        if name in self.words:
            earlier = self.words[name].location
            if earlier is None:
                self.fail(
                    f"`{name}` is a built-in word and cannot be redefined", location
                )
            self.fail(f"`{name}` is already defined on line {earlier.line}", location)
        if name in self.broken:
            self.fail(f"`{name}` is already defined", location)
        if name in self.local_names:
            self.fail(
                f"`{name}` is a local in `{self.local_names[name]}`, "
                "so it cannot name a word",
                location,
            )

    def declare_constructors(self, declaration: TypeDeclaration) -> None:
        if declaration.name not in self.types:
            return
        if self.types[declaration.name].location != declaration.location:
            return
        result = Type(declaration.name, tuple(Type(p) for p in declaration.params))
        names = [variant.name for variant in declaration.variants]
        if len(names) == 1:
            names.append(f"{names[0]}-unpack")
        all_fields: list[Type] = []
        try:
            for variant in declaration.variants:
                self.claim_name(variant.name, variant.location)
                for field in variant.fields:
                    self.check_type_names(field, declaration)
                fields = tuple(self.resolve_type(field) for field in variant.fields)
                all_fields.extend(fields)
                self.words[variant.name] = WordInfo(
                    variant.name,
                    Effect(fields, (result,)),
                    variant.location,
                    kind="constructor",
                )
            if len(declaration.variants) == 1:
                only = declaration.variants[0]
                self.claim_name(names[-1], only.location)
                self.words[names[-1]] = WordInfo(
                    names[-1], Effect((result,), fields), only.location, kind="unpack"
                )
        except IndeError:
            for name in names:
                if name not in self.words:
                    self.broken.add(name)
            raise
        self.field_types[declaration.name] = all_fields
        self.variants[declaration.name] = tuple(
            variant.name for variant in declaration.variants
        )

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
        if signature.inputs.row != signature.outputs.row:
            self.unsupported(
                "signatures with two different row variables", signature.location
            )
        effect = Effect(
            tuple(self.resolve_type(t) for t in signature.inputs.types),
            tuple(self.resolve_type(t) for t in signature.outputs.types),
        )
        variables = effect.variables()
        where: list[tuple[str, str]] = []
        for constraint in declaration.where:
            if constraint.var not in variables:
                self.fail(
                    f"`{constraint.var}` has a constraint but does not appear in "
                    f"the signature of `{declaration.name}`",
                    constraint.location,
                )
            for bound in constraint.bounds:
                if isinstance(bound, QuotationType):
                    self.unsupported("quotation constraints", bound.location)
                if bound.name not in CONSTRAINTS:
                    self.fail(
                        f"unknown constraint `{bound.name}`"
                        f"{self.suggestion(bound.name, CONSTRAINTS)}",
                        bound.location,
                    )
                if (constraint.var, bound.name) not in where:
                    where.append((constraint.var, bound.name))
        return replace(effect, where=tuple(where))

    def resolve_type(self, expression: TypeExpression) -> Type:
        if isinstance(expression, TypeVariable):
            return Type(expression.name)
        if isinstance(expression, QuotationType):
            self.unsupported("quotation types", expression.location)
        name = expression.name
        if name not in CORE_TYPE_ARGUMENTS and name not in self.types:
            self.fail(f"unknown type `{name}`", expression.location)
        return Type(name, tuple(self.resolve_type(arg) for arg in expression.args))

    def check_type_names(
        self, expression: TypeExpression, declaration: TypeDeclaration
    ) -> None:
        if isinstance(expression, TypeVariable):
            if expression.name not in declaration.params:
                self.fail(
                    f"`{expression.name}` is not a parameter of `{declaration.name}`",
                    expression.location,
                )
        elif isinstance(expression, QuotationType):
            for side in (expression.effect.inputs, expression.effect.outputs):
                for part in side.types:
                    self.check_type_names(part, declaration)
        else:
            name = expression.name
            if name not in CORE_TYPE_ARGUMENTS and name not in self.types:
                self.fail(f"unknown type `{name}`", expression.location)
            for arg in expression.args:
                self.check_type_names(arg, declaration)

    def derive_linear_types(self) -> None:
        changed = True
        while changed:
            changed = False
            for name, fields in self.field_types.items():
                if name not in self.linear_types and any(
                    self.always_linear(field) for field in fields
                ):
                    self.linear_types.add(name)
                    changed = True

    def always_linear(self, checked_type: Type) -> bool:
        if is_type_variable(checked_type):
            return False
        return (
            checked_type.name in CORE_LINEAR_TYPES
            or checked_type.name in self.linear_types
            or any(self.always_linear(arg) for arg in checked_type.args)
        )

    def check_array_elements(self, word: WordInfo) -> None:
        if word.kind == "unpack":
            return
        self.begin(word.effect.where)
        for checked_type in word.effect.inputs + word.effect.outputs:
            self.require_unrestricted_elements(checked_type, word)

    def require_unrestricted_elements(self, checked_type: Type, word: WordInfo) -> None:
        if checked_type.name == "array" and self.is_linear(checked_type.args[0]):
            element = checked_type.args[0]
            reason = (
                f"`{element}` may be linear; add `where {element}: Unrestricted`"
                if is_type_variable(element)
                else f"`{element}` is linear"
            )
            self.fail(
                f"the elements of an `array` must be unrestricted, but "
                f"{reason} (in `{word.name}`)",
                word.location,
            )
        for arg in checked_type.args:
            self.require_unrestricted_elements(arg, word)

    def is_linear(self, checked_type: Type) -> bool:
        if is_type_variable(checked_type):
            return "Unrestricted" not in self.assumptions.get(checked_type.name, ())
        return (
            checked_type.name in CORE_LINEAR_TYPES
            or checked_type.name in self.linear_types
            or any(self.is_linear(arg) for arg in checked_type.args)
        )

    def satisfies(self, checked_type: Type, constraint: str) -> bool:
        if "|" in constraint:
            return any(self.satisfies(checked_type, c) for c in constraint.split("|"))
        if constraint == "Unrestricted":
            return not self.is_linear(checked_type)
        if is_type_variable(checked_type):
            return constraint in self.assumptions.get(checked_type.name, ())
        if constraint == "Integer":
            return checked_type in INTEGER_TYPES
        if constraint == "Signed":
            return checked_type in SIGNED_INTEGER_TYPES
        if constraint == "Float":
            return checked_type in FLOAT_TYPES
        if constraint == "Num":
            return checked_type in NUMBER_TYPES
        if constraint == "Eq":
            return self.is_eq(checked_type, set())
        if constraint == "PartialEq":
            return checked_type in FLOAT_TYPES or self.is_eq(checked_type, set())
        if constraint == "Ord":
            return checked_type in INTEGER_TYPES or checked_type in (BOOL, CHAR, STR)
        if constraint == "PartialOrd":
            return checked_type in NUMBER_TYPES or checked_type in (BOOL, CHAR, STR)
        assert constraint == "Display", constraint
        return checked_type in NUMBER_TYPES or checked_type in (BOOL, CHAR, STR, UNIT)

    def is_eq(self, checked_type: Type, visiting: set[Type]) -> bool:
        if is_type_variable(checked_type):
            return "Eq" in self.assumptions.get(checked_type.name, ())
        if checked_type in INTEGER_TYPES or checked_type in (BOOL, CHAR, STR, UNIT):
            return True
        if checked_type.name == "array":
            return self.is_eq(checked_type.args[0], visiting)
        if checked_type.name not in self.variants:
            return False
        if checked_type in visiting:
            return True
        visiting.add(checked_type)
        return all(
            self.is_eq(field, visiting)
            for constructor in self.variants[checked_type.name]
            for field in self.variant_fields(constructor, checked_type)
        )

    def variant_fields(self, constructor: str, of_type: Type) -> tuple[Type, ...]:
        effect = self.words[constructor].effect
        parameters = effect.outputs[0].args
        replacements = {
            parameter.name: argument
            for parameter, argument in zip(parameters, of_type.args, strict=True)
        }
        return tuple(substitute(field, replacements) for field in effect.inputs)

    def require(
        self,
        checked_type: Type,
        constraint: str,
        location: Location | None,
        describe: Callable[[Type], str],
    ) -> None:
        resolved = self.inference.resolve(checked_type)
        if self.inference.has_unknowns(resolved):
            self.obligations.append(
                Obligation(checked_type, constraint, location, describe)
            )
        elif not self.satisfies(resolved, constraint):
            self.fail(self.unmet(resolved, constraint, describe), location)

    def unmet(
        self, resolved: Type, constraint: str, describe: Callable[[Type], str]
    ) -> str:
        message = describe(resolved)
        if is_type_variable(resolved) and "|" not in constraint:
            message += f"; add `where {resolved}: {constraint}` to the signature"
        return message

    def begin(self, where: tuple[tuple[str, str], ...] = ()) -> None:
        self.inference = Inference()
        self.obligations = []
        assumed: dict[str, set[str]] = {}
        for variable, constraint in where:
            pending = [constraint]
            while pending:
                current = pending.pop()
                if current not in assumed.setdefault(variable, set()):
                    assumed[variable].add(current)
                    pending.extend(IMPLIED_CONSTRAINTS.get(current, ()))
        self.assumptions = {v: frozenset(cs) for v, cs in assumed.items()}

    def finish(self) -> None:
        obligations, self.obligations = self.obligations, []
        for obligation in obligations:
            resolved = self.inference.resolve(obligation.type)
            if self.inference.has_unknowns(resolved):
                continue
            if not self.satisfies(resolved, obligation.constraint):
                self.fail(
                    self.unmet(resolved, obligation.constraint, obligation.describe),
                    obligation.location,
                )
        for name, origin in self.inference.origins.items():
            if self.inference.has_unknowns(Type(name)):
                self.fail(
                    f"cannot work out {origin.description}; "
                    "nothing here says what it is",
                    origin.location,
                )

    def show(self, types: Iterable[Type]) -> str:
        return format_types(self.inference.resolve_all(list(types)))

    def check_line(self, body: Body, stack: list[Type]) -> list[Type]:
        self.begin()
        self.check_body(body, stack)
        self.finish()
        stack[:] = self.inference.resolve_all(stack)
        return stack

    def check_word(self, word: WordInfo) -> None:
        if word.declaration is None:
            return
        self.begin(word.effect.where)
        stack = list(word.effect.inputs)
        self.current_word = word.name
        try:
            self.check_body(word.declaration.body, stack)
        finally:
            self.current_word = None
        if not self.inference.unify_all(stack, word.effect.outputs):
            self.fail(
                f"`{word.name}` declares that it leaves "
                f"{format_types(word.effect.outputs)}, "
                f"but its body leaves {self.show(stack)}",
                word.location,
            )
        self.finish()

    def check_body(
        self, body: Body, stack: list[Type], scope: Scope | None = None
    ) -> list[Type]:
        scope = scope or {}
        for item in body:
            self.check_item(item, stack, scope)
        return stack

    def check_item(self, item: Item, stack: list[Type], scope: Scope) -> None:
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
            if item.name in scope:
                stack.append(scope[item.name])
            else:
                self.check_call(item, stack, scope)
        elif isinstance(item, Quotation):
            self.unsupported("quotations", item.location)
        elif isinstance(item, Propagate):
            self.unsupported("`?` forms", item.location)
        elif isinstance(item, If):
            self.check_if(item, stack, scope)
        elif isinstance(item, While):
            self.check_while(item, stack, scope)
        elif isinstance(item, Match):
            self.check_match(item, stack, scope)
        else:
            assert isinstance(item, Let), f"unhandled item {type(item).__name__}"
            self.check_let(item, stack, scope)

    def describe_stacks(self, *stacks: list[Type]) -> list[str]:
        resolved = [self.inference.resolve_all(stack) for stack in stacks]
        depth = 0
        shortest = min(len(stack) for stack in resolved)
        while depth < shortest and all(
            stack[depth] == resolved[0][depth] for stack in resolved
        ):
            depth += 1
        return [format_types(stack[depth:]) for stack in resolved]

    def check_if(self, item: If, stack: list[Type], scope: Scope) -> None:
        self.require_depth("if", 1, stack, item.location)
        condition = stack.pop()
        if not self.inference.unify(condition, BOOL):
            self.fail(
                f"`if` takes a `bool`, found {self.show([condition])}", item.location
            )

        then_stack = list(stack)
        self.check_body(item.then_body, then_stack, scope)
        else_stack = list(stack)
        if item.else_body is not None:
            self.check_body(item.else_body, else_stack, scope)

        if not self.inference.unify_all(then_stack, else_stack):
            _, first, second = self.describe_stacks(stack, then_stack, else_stack)
            if item.else_body is None:
                self.fail(
                    "an `if` without `else` must leave the stack as it found it, "
                    f"but its branch turns {second} into {first}",
                    item.location,
                )
            self.fail(
                "the branches of `if` leave different stacks: "
                f"{first} before `else`, {second} after it",
                item.location,
            )
        stack[:] = then_stack

    def check_while(self, item: While, stack: list[Type], scope: Scope) -> None:
        condition_stack = list(stack)
        self.check_body(item.condition, condition_stack, scope)
        if not self.inference.unify_all(condition_stack, [*stack, BOOL]):
            before, after = self.describe_stacks(stack, condition_stack)
            self.fail(
                "the condition of `while` must add one `bool` to the stack, "
                f"but it turns {before} into {after}",
                item.location,
            )

        body_stack = list(stack)
        self.check_body(item.body, body_stack, scope)
        if not self.inference.unify_all(body_stack, stack):
            before, after = self.describe_stacks(stack, body_stack)
            self.fail(
                "the body of `while` must leave the stack as it found it, "
                f"but it turns {before} into {after}",
                item.location,
            )

    def names_a_word(self, name: str) -> bool:
        return (
            name in self.words
            or name in BUILTINS
            or name in STACK_WORDS
            or name in self.broken
        )

    def check_let(self, item: Let, stack: list[Type], scope: Scope) -> None:
        count = len(item.names)
        self.require_depth("let", count, stack, item.location)
        bound = stack[len(stack) - count :]
        del stack[len(stack) - count :]

        inner = dict(scope)
        for name, bound_type in zip(item.names, bound, strict=True):
            if name in inner:
                self.fail(
                    f"`{name}` is already a local here; a local cannot be shadowed",
                    item.location,
                )
            if self.names_a_word(name):
                self.fail(
                    f"`{name}` already names a word, so it cannot name a local",
                    item.location,
                )
            if self.current_word is not None:
                self.local_names.setdefault(name, self.current_word)
            self.require(
                bound_type,
                "Unrestricted",
                item.location,
                lambda _: "linear locals are not supported by the checker yet",
            )
            inner[name] = bound_type
        self.check_body(item.body, stack, inner)

    def describe_pattern(self, pattern: Pattern) -> str:
        if isinstance(pattern, ConstructorPattern):
            return pattern.name
        literal = pattern.literal
        if isinstance(literal, ConstantReference):
            return literal.name
        if isinstance(literal, BooleanLiteral):
            return "true" if literal.value else "false"
        if isinstance(literal, CharacterLiteral):
            return quote(literal.value, "'")
        if isinstance(literal, StringLiteral):
            return quote(literal.value, '"')
        assert isinstance(literal, (IntegerLiteral, FloatLiteral))
        return str(literal.value)

    def check_match(self, item: Match, stack: list[Type], scope: Scope) -> None:
        self.require_depth("match", 1, stack, item.location)
        scrutinee = self.inference.resolve(stack.pop())

        if is_unknown(scrutinee):
            self.fail(
                "the type of the value being matched is not known at this point",
                item.location,
            )
        if scrutinee in INTEGER_TYPES or scrutinee in (BOOL, CHAR, STR):
            self.check_literal_patterns(item, scrutinee)
            pushed: list[tuple[Type, ...]] = [() for _ in item.arms]
        elif scrutinee.name in self.variants:
            pushed = self.check_constructor_patterns(item, scrutinee)
        elif scrutinee in FLOAT_TYPES:
            self.fail(
                f"a `{scrutinee}` cannot be matched, "
                "because float equality is only partial",
                item.location,
            )
        elif scrutinee.name in self.types:
            raise _Abandon
        else:
            self.fail(f"a value of type `{scrutinee}` cannot be matched", item.location)

        ends: list[tuple[str, list[Type], Location | None]] = []
        for arm, fields in zip(item.arms, pushed, strict=True):
            arm_stack = [*stack, *fields]
            self.check_body(arm.body, arm_stack, scope)
            label = f"`case {self.describe_pattern(arm.pattern)}`"
            ends.append((label, arm_stack, arm.location))
        if item.else_body is not None:
            else_stack = list(stack)
            self.check_body(item.else_body, else_stack, scope)
            ends.append(("`else`", else_stack, item.location))

        first_label, first_stack, _ = ends[0]
        for label, arm_stack, location in ends[1:]:
            if not self.inference.unify_all(first_stack, arm_stack):
                _, first, other = self.describe_stacks(stack, first_stack, arm_stack)
                self.fail(
                    "the arms of `match` leave different stacks: "
                    f"{first} after {first_label}, {other} after {label}",
                    location,
                )
        stack[:] = first_stack

    def check_literal_patterns(self, item: Match, scrutinee: Type) -> None:
        seen: set[object] = set()
        for arm in item.arms:
            pattern = arm.pattern
            text = self.describe_pattern(pattern)
            if isinstance(pattern, ConstructorPattern):
                self.fail(
                    f"`{text}` is a constructor, but a `{scrutinee}` is matched "
                    "against literals",
                    arm.location,
                )
            literal = pattern.literal
            if isinstance(literal, ConstantReference):
                pattern_type = self.constant_type(literal)
                literal = self.constants[literal.name].value
            else:
                pattern_type = self.literal_type(literal)
            if pattern_type != scrutinee:
                self.fail(
                    f"`case {text}` is a `{pattern_type}`, but the value being "
                    f"matched is a `{scrutinee}`",
                    arm.location,
                )
            assert not isinstance(literal, ArrayLiteral)
            if literal.value in seen:
                self.fail(
                    f"the value of `case {text}` already has a `case`", arm.location
                )
            seen.add(literal.value)

        if scrutinee == BOOL and len(seen) == 2:
            if item.else_body is not None:
                self.fail(
                    "both `true` and `false` have a `case`, so `else` can never run",
                    item.location,
                )
        elif item.else_body is None:
            self.fail(
                f"a `match` on a `{scrutinee}` needs an `else` arm for the values "
                "without a `case`",
                item.location,
            )

    def check_constructor_patterns(
        self, item: Match, scrutinee: Type
    ) -> list[tuple[Type, ...]]:
        variants = self.variants[scrutinee.name]
        seen: set[str] = set()
        pushed: list[tuple[Type, ...]] = []
        for arm in item.arms:
            pattern = arm.pattern
            text = self.describe_pattern(pattern)
            if not isinstance(pattern, ConstructorPattern):
                self.fail(
                    f"`case {text}` is not a constructor, but a `{scrutinee}` is "
                    "matched against its constructors",
                    arm.location,
                )
            if pattern.name not in variants:
                self.fail(
                    f"`{pattern.name}` is not a constructor of `{scrutinee}`"
                    f"{self.suggestion(pattern.name, variants)}",
                    arm.location,
                )
            if pattern.name in seen:
                self.fail(f"`case {pattern.name}` appears more than once", arm.location)
            seen.add(pattern.name)
            pushed.append(self.variant_fields(pattern.name, scrutinee))

        if item.else_body is not None:
            self.fail(
                "a `match` on constructors has no `else`; "
                "every constructor gets its own `case`",
                item.location,
            )
        missing = [f"`case {name}`" for name in variants if name not in seen]
        if missing:
            self.fail(
                f"this `match` on a `{scrutinee}` is missing {', '.join(missing)}",
                item.location,
            )
        return pushed

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
            return array_of(
                self.inference.fresh("the element type of this `{ }`", literal.location)
            )
        element_types = [
            self.constant_type(element)
            if isinstance(element, ConstantReference)
            else self.literal_type(element)
            for element in literal.elements
        ]
        first = element_types[0]
        for element, element_type in zip(literal.elements, element_types, strict=False):
            if not self.inference.unify(first, element_type):
                self.fail(
                    "the elements of an array literal must have one type; "
                    f"found {self.show([first])} and {self.show([element_type])}",
                    element.location,
                )
        return array_of(self.inference.resolve(first))

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

    def check_call(
        self, call: WordReference, stack: list[Type], scope: Scope | None = None
    ) -> None:
        name = call.name
        if name in self.words:
            self.apply_effect(name, self.words[name].effect, stack, call.location)
        elif name in BUILTINS:
            builtin = BUILTINS[name]
            self.apply_effect(name, builtin.effect, stack, call.location, builtin.takes)
        elif name in STACK_WORDS:
            self.apply_stack_word(STACK_WORDS[name], stack, call.location)
        elif name in self.broken:
            raise _Abandon
        else:
            known = [*(scope or {}), *self.words, *BUILTINS, *STACK_WORDS]
            self.fail(
                f"unknown word `{name}`{self.suggestion(name, known)}", call.location
            )

    def require_depth(
        self, name: str, needed: int, stack: list[Type], location: Location | None
    ) -> None:
        if len(stack) >= needed:
            return
        values = "1 value" if needed == 1 else f"{needed} values"
        holds = "is empty" if not stack else f"holds only {self.show(stack)}"
        self.fail(f"`{name}` takes {values}, but the stack {holds}", location)

    def apply_effect(
        self,
        name: str,
        declared: Effect,
        stack: list[Type],
        location: Location | None,
        takes: str | None = None,
    ) -> None:
        count = len(declared.inputs)
        self.require_depth(name, count, stack, location)
        effect = self.inference.instantiate(name, declared, location)
        found = stack[len(stack) - count :]
        expected = takes or format_types(declared.inputs)
        if not self.inference.unify_all(found, effect.inputs):
            self.fail(f"`{name}` takes {expected}, found {self.show(found)}", location)
        del stack[len(stack) - count :]
        stack.extend(effect.outputs)

        for (unknown, constraint), (variable, _) in zip(
            effect.where, declared.where, strict=True
        ):
            if takes is not None:
                message = f"`{name}` takes {takes}, found"
                describe = self.describe_operands(message, found)
            else:
                describe = self.describe_unmet(name, variable, constraint)
            self.require(Type(unknown), constraint, location, describe)

    def describe_operands(
        self, message: str, found: list[Type]
    ) -> Callable[[Type], str]:
        return lambda _: f"{message} {self.show(found)}"

    def describe_unmet(
        self, name: str, variable: str, constraint: str
    ) -> Callable[[Type], str]:
        def describe(resolved: Type) -> str:
            return (
                f"`{resolved}` does not satisfy `{constraint}`, "
                f"which `{name}` requires of its `{variable}`"
            )

        return describe

    def apply_stack_word(
        self, word: StackWord, stack: list[Type], location: Location | None
    ) -> None:
        self.require_depth(word.name, word.arguments, stack, location)
        taken = stack[len(stack) - word.arguments :]
        for verb, positions in (("copy", word.copied()), ("discard", word.discarded())):
            for position in positions:
                self.require(
                    taken[position],
                    "Unrestricted",
                    location,
                    self.describe_linear(word.name, verb),
                )
        del stack[len(stack) - word.arguments :]
        stack.extend(taken[position] for position in word.result)

    def describe_linear(self, name: str, verb: str) -> Callable[[Type], str]:
        def describe(resolved: Type) -> str:
            if is_type_variable(resolved):
                return (
                    f"`{name}` cannot {verb} a value of type `{resolved}`, "
                    "which may be linear"
                )
            return f"`{name}` cannot {verb} a value of the linear type `{resolved}`"

        return describe

    def suggestion(self, name: str, known: Iterable[str]) -> str:
        matches = difflib.get_close_matches(name, list(known), n=1)
        return f"; did you mean `{matches[0]}`?" if matches else ""

    def check_main(self, require_main: bool) -> None:
        if "main" not in self.words:
            if "main" in self.broken or not require_main:
                return
            self.fail("the program has no `main`", None)
        main = self.words["main"]
        self.begin(main.effect.where)
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
        if returns_result and not self.satisfies(outputs[0].args[1], "Display"):
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
