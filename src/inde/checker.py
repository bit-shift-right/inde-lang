from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import NoReturn

from inde import ast
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
    BIT_CAST_TAKES,
    BIT_CASTS,
    BUILTINS,
    CONSTRAINTS,
    CONVERSIONS,
    IMPLIED_CONSTRAINTS,
    STACK_WORDS,
    StackWord,
    conversion_forms,
)
from inde.errors import IndeError, IndeErrors
from inde.folding import CertainFailure, Folder
from inde.inference import Inference
from inde.lexer import Location, quote
from inde.parser import CORE_TYPE_ARGUMENTS, parse
from inde.type_model import (
    BOOL,
    CHAR,
    CORE_LINEAR_TYPES,
    F32,
    FLOAT_TYPES,
    I32,
    IMPLICIT_ROW,
    INTEGER_TYPES,
    LINEAR,
    NUMBER_TYPES,
    SIGNED_INTEGER_TYPES,
    STR,
    UNIT,
    UNRESTRICTED,
    Effect,
    Stack,
    Type,
    array_of,
    format_types,
    from_number_type,
    is_quotation,
    is_type_variable,
    is_unknown,
    is_unknown_row,
    option_of,
    quotation_constraints,
    quotation_of,
    result_of,
    rows_of,
    substitute,
    variables_of,
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
class Capture:
    name: str
    operands: tuple[Type, ...]
    qualifier: Type
    location: Location | None


@dataclass(frozen=True)
class CheckedProgram:
    types: dict[str, TypeInfo]
    constants: dict[str, ConstantInfo]
    words: dict[str, WordInfo]
    linear_types: frozenset[str]


@dataclass
class Local:
    name: str
    type: Type
    location: Location | None
    loop_depth: int
    uses: int = 0
    second_use: Location | None = None
    in_loop: Location | None = None
    disagreement: tuple[str, str, Location | None] | None = None
    at_exit: Location | None = None


Scope = dict[str, Local]


class TypeStack:
    def __init__(self, row: str, types: Iterable[Type] = ()) -> None:
        self.row = row
        self.types = list(types)

    def copy(self) -> TypeStack:
        return TypeStack(self.row, self.types)

    def whole(self) -> Stack:
        return Stack(self.row, tuple(self.types))

    def become(self, other: TypeStack | Stack) -> None:
        self.row = other.row
        self.types = list(other.types)


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
        self.type_parameters: dict[str, tuple[str, ...]] = {}
        self.unrestricted_parameters: dict[str, frozenset[int]] = {}
        self.local_names: dict[str, str] = {}
        self.current_word: str | None = None
        self.broken: set[str] = set()
        self.inference = Inference()
        self.assumptions: dict[str, frozenset[str]] = {}
        self.obligations: list[Obligation] = []
        self.captures: list[Capture] = []
        self.locals: list[Local] = []
        self.loop_depth = 0
        self.bound_syntax: dict[str, QuotationType] = {}
        self.quotation_bounds: dict[str, Effect] = {}
        self.resolving_bounds: set[str] = set()
        self.hidden_locals: frozenset[str] = frozenset()
        self.quotation_depth = 0
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
        self.derive_unrestricted_parameters()

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
        self.type_parameters[name] = declaration.params
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
        if (
            name in BUILTINS
            or name in STACK_WORDS
            or name in CONVERSIONS
            or name in BIT_CASTS
        ):
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
        rows = None
        if signature.inputs.row is not None and signature.outputs.row is not None:
            rows = (f"..{signature.inputs.row}", f"..{signature.outputs.row}")

        syntax: dict[str, QuotationType] = {}
        for constraint in declaration.where:
            for bound in constraint.bounds:
                if not isinstance(bound, QuotationType):
                    continue
                if bound.linear:
                    self.fail(
                        "a quotation constraint accepts either qualifier, "
                        "so it is written without `linear`",
                        bound.location,
                    )
                if constraint.var in syntax:
                    self.fail(
                        f"`{constraint.var}` has two quotation constraints",
                        bound.location,
                    )
                syntax[constraint.var] = bound
        self.bound_syntax = syntax
        try:
            for variable, bound in syntax.items():
                self.quotation_bound(variable, bound.location)
            effect = Effect(
                tuple(self.resolve_type(t) for t in signature.inputs.types),
                tuple(self.resolve_type(t) for t in signature.outputs.types),
                rows=rows,
            )
        finally:
            self.bound_syntax = {}
            self.quotation_bounds = {}
        if rows is not None and rows[0] == rows[1]:
            inner = [row for t in effect.inputs + effect.outputs for row in rows_of(t)]
            if rows[0] not in inner:
                effect = replace(effect, rows=None)

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
                    continue
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
            name = expression.name
            if name in self.bound_syntax:
                bound = self.quotation_bound(name, expression.location)
                return quotation_of(bound, Type(name))
            return Type(name)
        if isinstance(expression, QuotationType):
            qualifier = LINEAR if expression.linear else UNRESTRICTED
            return quotation_of(self.resolve_effect(expression.effect), qualifier)
        name = expression.name
        if name not in CORE_TYPE_ARGUMENTS and name not in self.types:
            self.fail(f"unknown type `{name}`", expression.location)
        return Type(name, tuple(self.resolve_type(arg) for arg in expression.args))

    def quotation_bound(self, variable: str, location: Location | None) -> Effect:
        if variable not in self.quotation_bounds:
            if variable in self.resolving_bounds:
                self.fail(
                    f"the quotation constraint on `{variable}` refers back "
                    f"to `{variable}`",
                    location,
                )
            self.resolving_bounds.add(variable)
            try:
                effect = self.resolve_effect(self.bound_syntax[variable].effect)
            finally:
                self.resolving_bounds.discard(variable)
            self.quotation_bounds[variable] = effect
        return self.quotation_bounds[variable]

    def resolve_effect(self, effect: ast.StackEffect) -> Effect:
        rows = None
        if effect.inputs.row is not None and effect.outputs.row is not None:
            rows = (f"..{effect.inputs.row}", f"..{effect.outputs.row}")
        return Effect(
            tuple(self.resolve_type(t) for t in effect.inputs.types),
            tuple(self.resolve_type(t) for t in effect.outputs.types),
            rows=rows,
        )

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
            if expression.effect.inputs.row is not None:
                self.fail(
                    "a quotation type in a type declaration cannot name its rows",
                    expression.location,
                )
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
        if checked_type == LINEAR:
            return True
        return (
            checked_type.name in CORE_LINEAR_TYPES
            or checked_type.name in self.linear_types
            or any(self.always_linear(arg) for arg in checked_type.args)
        )

    def derive_unrestricted_parameters(self) -> None:
        required: dict[str, set[int]] = {"array": {0}}
        changed = True
        while changed:
            changed = False
            for name, fields in self.field_types.items():
                parameters = self.type_parameters.get(name, ())
                for field in fields:
                    for variable in self.variables_held_in_arrays(field, required):
                        position = parameters.index(variable)
                        if position not in required.setdefault(name, set()):
                            required[name].add(position)
                            changed = True
        self.unrestricted_parameters = {
            name: frozenset(positions) for name, positions in required.items()
        }

        for word in list(self.words.values()):
            if word.kind == "word":
                continue
            declared = (
                word.effect.outputs[0]
                if word.kind == "constructor"
                else word.effect.inputs[0]
            )
            positions = self.unrestricted_parameters.get(declared.name, frozenset())
            where = tuple(
                (declared.args[position].name, "Unrestricted")
                for position in sorted(positions)
            )
            self.words[word.name] = replace(
                word, effect=replace(word.effect, where=where)
            )

    def variables_held_in_arrays(
        self, checked_type: Type, required: dict[str, set[int]]
    ) -> list[str]:
        found: list[str] = []
        for position, arg in enumerate(checked_type.args):
            if position in required.get(checked_type.name, ()):
                found.extend(variables_of(arg))
            found.extend(self.variables_held_in_arrays(arg, required))
        return found

    def check_array_elements(self, word: WordInfo) -> None:
        if word.kind == "unpack":
            return
        self.begin(word.effect.where)
        for checked_type in word.effect.inputs + word.effect.outputs:
            self.require_unrestricted_elements(checked_type, word)

    def require_unrestricted_elements(self, checked_type: Type, word: WordInfo) -> None:
        positions = self.unrestricted_parameters.get(checked_type.name, frozenset())
        for position in sorted(positions):
            element = checked_type.args[position]
            if not self.is_linear(element):
                continue
            reason = (
                f"`{element}` may be linear; add `where {element}: Unrestricted`"
                if is_type_variable(element)
                else f"`{element}` is linear"
            )
            if checked_type.name == "array":
                rule = "the elements of an `array` must be unrestricted"
            else:
                parameter = self.type_parameters[checked_type.name][position]
                rule = (
                    f"`{checked_type.name}` holds an `array` of its `{parameter}`, "
                    "which must be unrestricted"
                )
            self.fail(f"{rule}, but {reason} (in `{word.name}`)", word.location)
        for arg in checked_type.args:
            self.require_unrestricted_elements(arg, word)

    def is_linear(self, checked_type: Type) -> bool:
        if is_type_variable(checked_type):
            return "Unrestricted" not in self.assumptions.get(checked_type.name, ())
        if checked_type == LINEAR:
            return True
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

    def explain_no_equality(self, checked_type: Type) -> str:
        culprit = self.without_equality(checked_type, set())
        if culprit is None:
            return ""
        if culprit == checked_type:
            return f"; `{checked_type}` has no equality"
        if culprit in FLOAT_TYPES:
            held = f"`{culprit}` values, whose equality is only partial"
        elif is_quotation(culprit):
            held = "quotations, which have no equality"
        elif is_type_variable(culprit):
            held = (
                f"`{culprit}` values, which may not have equality; "
                f"add `where {culprit}: Eq` to the signature"
            )
        else:
            held = f"`{culprit}` values, which have no equality"
        return f"; `{checked_type}` has no equality, because it holds {held}"

    def without_equality(self, checked_type: Type, visiting: set[Type]) -> Type | None:
        if self.is_eq(checked_type, set()):
            return None
        if checked_type.name == "array":
            return self.without_equality(checked_type.args[0], visiting)
        if checked_type.name in self.variants and checked_type not in visiting:
            visiting.add(checked_type)
            for constructor in self.variants[checked_type.name]:
                for field in self.variant_fields(constructor, checked_type):
                    culprit = self.without_equality(field, visiting)
                    if culprit is not None:
                        return culprit
        return checked_type

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

    def qualifier_variable(self, resolved: Type) -> Type | None:
        if is_quotation(resolved) and is_type_variable(resolved.args[0]):
            return resolved.args[0]
        return None

    def begin(self, where: tuple[tuple[str, str], ...] = ()) -> None:
        self.inference = Inference()
        self.obligations = []
        self.captures = []
        self.locals = []
        self.loop_depth = 0
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
        self.settle_captures()
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
        self.check_locals()
        for name, origin in self.inference.origins.items():
            if self.inference.has_unknowns(Type(name)):
                self.fail(
                    f"cannot work out {origin.description}; "
                    "nothing here says what it is",
                    origin.location,
                )

    def show(self, types: Iterable[Type]) -> str:
        return format_types(self.inference.resolve_all(list(types)))

    def show_stack(self, stack: Stack) -> str:
        flat = self.inference.flatten(stack)
        if flat.row == IMPLICIT_ROW or is_unknown_row(flat.row):
            return format_types(flat.types)
        return f"`{' '.join([flat.row, *(str(t) for t in flat.types)])}`"

    def settle(self, stack: TypeStack) -> None:
        stack.become(self.inference.flatten(stack.whole()))

    def check_line(self, body: Body, stack: list[Type]) -> list[Type]:
        self.begin()
        working = TypeStack(IMPLICIT_ROW, stack)
        self.check_body(body, working)
        self.finish()
        self.settle(working)
        stack[:] = working.types
        return stack

    def check_word(self, word: WordInfo) -> None:
        if word.declaration is None:
            return
        self.begin(word.effect.where)
        input_row, output_row = word.effect.rows or (IMPLICIT_ROW, IMPLICIT_ROW)
        stack = TypeStack(input_row, word.effect.inputs)
        self.current_word = word.name
        try:
            self.check_body(word.declaration.body, stack)
        finally:
            self.current_word = None
        declared = Stack(output_row, word.effect.outputs)
        if not self.inference.unify_stacks(stack.whole(), declared):
            self.fail(
                f"`{word.name}` declares that it leaves {self.show_stack(declared)}, "
                f"but its body leaves {self.show_stack(stack.whole())}",
                word.location,
            )
        self.finish()

    def check_body(
        self, body: Body, stack: TypeStack, scope: Scope | None = None
    ) -> TypeStack:
        scope = scope or {}
        folder = Folder(self.words, self.constants)
        for item in body:
            self.check_item(item, stack, scope)
            local = isinstance(item, WordReference) and item.name in scope
            try:
                folder.step(item, local)
            except CertainFailure as failure:
                self.fail(failure.message, failure.location)
        return stack

    def check_item(self, item: Item, stack: TypeStack, scope: Scope) -> None:
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
            stack.types.append(self.literal_type(item))
        elif isinstance(item, ConstantReference):
            stack.types.append(self.constant_type(item))
        elif isinstance(item, WordReference):
            if item.name in scope:
                local = scope[item.name]
                self.use_local(local, item.location)
                stack.types.append(local.type)
            else:
                self.check_call(item, stack, scope)
        elif isinstance(item, Quotation):
            self.check_quotation(item, stack, scope)
        elif isinstance(item, Propagate):
            self.check_propagate(item, stack, scope)
        elif isinstance(item, If):
            self.check_if(item, stack, scope)
        elif isinstance(item, While):
            self.check_while(item, stack, scope)
        elif isinstance(item, Match):
            self.check_match(item, stack, scope)
        else:
            assert isinstance(item, Let), f"unhandled item {type(item).__name__}"
            self.check_let(item, stack, scope)

    def describe_stacks(self, *stacks: TypeStack) -> list[str]:
        resolved = [
            list(self.inference.flatten(stack.whole()).types) for stack in stacks
        ]
        depth = 0
        shortest = min(len(stack) for stack in resolved)
        while depth < shortest and all(
            stack[depth] == resolved[0][depth] for stack in resolved
        ):
            depth += 1
        return [format_types(stack[depth:]) for stack in resolved]

    def check_propagate(self, item: Propagate, stack: TypeStack, scope: Scope) -> None:
        location = item.location
        if self.quotation_depth > 0:
            self.fail("`?` is not allowed inside a quotation", location)
        if self.current_word is None:
            self.fail("`?` can only be used in the body of a word", location)
        name = self.current_word
        outputs = self.words[name].effect.outputs
        result = outputs[0] if len(outputs) == 1 else None
        if result is None or result.name not in ("Result", "Option"):
            self.fail(
                f"`?` needs `{name}` to return a `Result` or an `Option`, "
                f"but it returns {format_types(outputs)}",
                location,
            )

        rows = self.words[name].effect.rows
        if rows is not None and rows[0] != rows[1]:
            self.fail(
                f"`?` cannot exit `{name}` early, because `{name}` turns "
                f"`{rows[0]}` into `{rows[1]}`, and an early exit would leave "
                f"`{rows[0]}`",
                location,
            )
        self.require_depth("?", 1, stack, location)
        row = self.inference.flatten(stack.whole()).row
        if row != (IMPLICIT_ROW if rows is None else rows[0]):
            self.fail(
                f"`?` cannot exit `{name}` early here, because the stack below "
                f"`{name}`'s own values is no longer the one it was called with",
                location,
            )
        operand = stack.types.pop()
        value = self.inference.fresh("the value `?` leaves", location)
        if result.name == "Result":
            expected = result_of(value, result.args[1])
        else:
            expected = option_of(value)
        if not self.inference.unify(operand, expected):
            self.fail(self.describe_propagate(name, result, operand), location)

        for below in stack.types:
            self.require(below, "Unrestricted", location, self.describe_discarded())
        for local in scope.values():
            if local.uses == 0 and local.at_exit is None:
                local.at_exit = location
        stack.types.append(value)

    def describe_propagate(self, name: str, result: Type, operand: Type) -> str:
        found = self.inference.resolve(operand)
        kind = result.name
        if found.name in ("Result", "Option") and found.name != kind:
            return (
                f"`?` here takes {'a `Result`' if kind == 'Result' else 'an `Option`'},"
                f" because `{name}` returns `{result}`, found `{found}`; "
                "`Option` and `Result` cannot be mixed"
            )
        if found.name == "Result":
            return (
                f"the error types differ: `?` found `{found}`, but `{name}` "
                f"returns errors of type `{result.args[1]}`; convert with `match`"
            )
        article = "a `Result`" if kind == "Result" else "an `Option`"
        return f"`?` takes {article}, found {self.show([found])}"

    def describe_discarded(self) -> Callable[[Type], str]:
        def describe(resolved: Type) -> str:
            variable = self.qualifier_variable(resolved)
            if variable is not None or is_type_variable(resolved):
                shown = variable or resolved
                return (
                    f"a value of type `{shown}` on the stack would be discarded if "
                    f"`?` exits early, and `{shown}` may be linear"
                    + (
                        f"; add `where {shown}: Unrestricted` to the signature"
                        if variable is not None
                        else ""
                    )
                )
            return (
                f"the linear `{resolved}` on the stack would be discarded if `?` "
                "exits early; handle the failure with `match` instead"
            )

        return describe

    def check_quotation(self, item: Quotation, stack: TypeStack, scope: Scope) -> None:
        row = self.inference.fresh_row()
        inner = TypeStack(row)
        hidden = self.hidden_locals
        self.hidden_locals = hidden | scope.keys()
        self.quotation_depth += 1
        try:
            self.check_body(item.body, inner, {})
        finally:
            self.hidden_locals = hidden
            self.quotation_depth -= 1
        effect = Effect(outputs=tuple(inner.types), rows=(row, inner.row))
        stack.types.append(quotation_of(self.inference.generalize(effect)))

    def check_if(self, item: If, stack: TypeStack, scope: Scope) -> None:
        self.require_depth("if", 1, stack, item.location)
        condition = stack.types.pop()
        if not self.inference.unify(condition, BOOL):
            self.fail(
                f"`if` takes a `bool`, found {self.show([condition])}", item.location
            )

        before = self.uses_of(scope)
        then_stack = stack.copy()
        self.check_body(item.then_body, then_stack, scope)
        then_uses = self.uses_of(scope)
        self.restore_uses(scope, before)
        else_stack = stack.copy()
        if item.else_body is not None:
            self.check_body(item.else_body, else_stack, scope)
            labels = ("before `else`", "after it")
        else:
            labels = ("in its body", "when the body is skipped")
        self.join_branches(
            "the branches of `if`",
            scope,
            before,
            [(labels[0], then_uses), (labels[1], self.uses_of(scope))],
            item.location,
        )

        if not self.inference.unify_stacks(then_stack.whole(), else_stack.whole()):
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
        stack.become(then_stack)

    def check_while(self, item: While, stack: TypeStack, scope: Scope) -> None:
        self.loop_depth += 1
        try:
            self.check_loop(item, stack, scope)
        finally:
            self.loop_depth -= 1

    def check_loop(self, item: While, stack: TypeStack, scope: Scope) -> None:
        condition_stack = stack.copy()
        self.check_body(item.condition, condition_stack, scope)
        expected = Stack(stack.row, (*stack.types, BOOL))
        if not self.inference.unify_stacks(condition_stack.whole(), expected):
            before, after = self.describe_stacks(stack, condition_stack)
            self.fail(
                "the condition of `while` must add one `bool` to the stack, "
                f"but it turns {before} into {after}",
                item.location,
            )

        body_stack = stack.copy()
        self.check_body(item.body, body_stack, scope)
        if not self.inference.unify_stacks(body_stack.whole(), stack.whole()):
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
            or name in CONVERSIONS
            or name in BIT_CASTS
            or name in self.broken
        )

    def check_let(self, item: Let, stack: TypeStack, scope: Scope) -> None:
        count = len(item.names)
        self.require_depth("let", count, stack, item.location)
        bound = stack.types[len(stack.types) - count :]
        del stack.types[len(stack.types) - count :]

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
            inner[name] = Local(name, bound_type, item.location, self.loop_depth)
        self.check_body(item.body, stack, inner)
        for name in item.names:
            if not self.judge_local(inner[name]):
                self.locals.append(inner[name])

    def use_local(self, local: Local, location: Location | None) -> None:
        local.uses += 1
        if local.uses == 2 and local.second_use is None:
            local.second_use = location
        if self.loop_depth > local.loop_depth and local.in_loop is None:
            local.in_loop = location

    def uses_of(self, scope: Scope) -> dict[str, int]:
        return {name: local.uses for name, local in scope.items()}

    def restore_uses(self, scope: Scope, uses: dict[str, int]) -> None:
        for name, count in uses.items():
            scope[name].uses = count

    def join_branches(
        self,
        construct: str,
        scope: Scope,
        before: dict[str, int],
        ends: list[tuple[str, dict[str, int]]],
        location: Location | None,
    ) -> None:
        for name, local in scope.items():
            counts = [uses[name] - before[name] for _, uses in ends]
            if len(set(counts)) > 1 and local.disagreement is None:
                detail = ", ".join(
                    f"{times(count)} {label}"
                    for (label, _), count in zip(ends, counts, strict=True)
                )
                local.disagreement = (construct, detail, location)
            local.uses = max(uses[name] for _, uses in ends)

    def check_locals(self) -> None:
        for local in self.locals:
            self.judge_local(local)

    def judge_local(self, local: Local) -> bool:
        resolved = self.inference.resolve(local.type)
        if self.inference.has_unknowns(resolved):
            return False
        if not self.is_linear(resolved):
            return True
        variable = (
            None if self.always_linear(resolved) else self.linear_variable(resolved)
        )
        subject = (
            f"the linear local `{local.name}`"
            if variable is None
            else f"the local `{local.name}`"
        )
        location: Location | None
        if local.in_loop is not None:
            message = f"{subject} is used inside a loop, which may run more than once"
            if variable is None:
                message += "; pass the value through the loop on the stack instead"
            location = local.in_loop
        elif local.second_use is not None:
            message = f"{subject} is used {times(local.uses)}"
            location = local.second_use
        elif local.disagreement is not None:
            construct, detail, location = local.disagreement
            message = f"{construct} use {subject} differently: {detail}"
        elif local.at_exit is not None:
            message = (
                f"{subject} would be discarded if this `?` exits early; use it "
                "before the `?`, or handle the failure with `match`"
            )
            location = local.at_exit
        elif local.uses == 0:
            message = f"{subject} is never used"
            location = local.location
        else:
            return True
        if variable is not None:
            message += (
                f"; `{local.name}` has type `{resolved}`, which may be linear; "
                f"add `where {variable}: Unrestricted` to the signature"
            )
        self.fail(message, location)

    def linear_variable(self, checked_type: Type) -> str | None:
        if is_type_variable(checked_type):
            if "Unrestricted" in self.assumptions.get(checked_type.name, ()):
                return None
            return checked_type.name
        for arg in checked_type.args:
            variable = self.linear_variable(arg)
            if variable is not None:
                return variable
        return None

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

    def check_match(self, item: Match, stack: TypeStack, scope: Scope) -> None:
        self.require_depth("match", 1, stack, item.location)
        scrutinee = self.inference.resolve(stack.types.pop())

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

        before = self.uses_of(scope)
        uses: list[tuple[str, dict[str, int]]] = []
        ends: list[tuple[str, TypeStack, Location | None]] = []
        for arm, fields in zip(item.arms, pushed, strict=True):
            self.restore_uses(scope, before)
            arm_stack = TypeStack(stack.row, [*stack.types, *fields])
            self.check_body(arm.body, arm_stack, scope)
            label = f"`case {self.describe_pattern(arm.pattern)}`"
            ends.append((label, arm_stack, arm.location))
            uses.append((f"in {label}", self.uses_of(scope)))
        if item.else_body is not None:
            self.restore_uses(scope, before)
            else_stack = stack.copy()
            self.check_body(item.else_body, else_stack, scope)
            ends.append(("`else`", else_stack, item.location))
            uses.append(("in `else`", self.uses_of(scope)))
        self.join_branches("the arms of `match`", scope, before, uses, item.location)

        first_label, first_stack, _ = ends[0]
        for label, arm_stack, location in ends[1:]:
            if not self.inference.unify_stacks(first_stack.whole(), arm_stack.whole()):
                _, first, other = self.describe_stacks(stack, first_stack, arm_stack)
                self.fail(
                    "the arms of `match` leave different stacks: "
                    f"{first} after {first_label}, {other} after {label}",
                    location,
                )
        stack.become(first_stack)

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
        self, call: WordReference, stack: TypeStack, scope: Scope | None = None
    ) -> None:
        name = call.name
        if name in self.words:
            self.apply_effect(name, self.words[name].effect, stack, call.location)
        elif name.partition(".")[0] in CONVERSIONS:
            self.check_conversion(name, stack, call.location)
        elif name in BIT_CASTS:
            self.check_bit_cast(name, stack, call.location)
        elif name in BUILTINS:
            builtin = BUILTINS[name]
            self.require_depth(name, builtin.arguments, stack, call.location)
            operands = stack.types[len(stack.types) - builtin.arguments :]
            self.apply_effect(name, builtin.effect, stack, call.location, builtin.takes)
            if name in ("curry", "compose"):
                self.settle_captured(name, operands, stack, call.location)
        elif name in STACK_WORDS:
            self.apply_stack_word(STACK_WORDS[name], stack, call.location)
        elif name in self.broken:
            raise _Abandon
        elif name in self.hidden_locals:
            self.fail(
                f"`{name}` is a local, and locals are not visible inside a "
                f"quotation; push `{name}` before the quotation and `curry` it",
                call.location,
            )
        else:
            known = [
                *(scope or {}),
                *self.words,
                *BUILTINS,
                *STACK_WORDS,
                *CONVERSIONS,
                *BIT_CASTS,
            ]
            if name == "to-bool":
                self.fail(
                    "there is no conversion to `bool`; use a comparison, such as "
                    "`0 !=`",
                    call.location,
                )
            base, dot, modifier = name.partition(".")
            if dot and base in known:
                self.fail(self.missing_form(base, modifier, known), call.location)
            self.fail(
                f"unknown word `{name}`{self.suggestion(name, known)}", call.location
            )

    def missing_form(self, base: str, modifier: str, known: list[str]) -> str:
        forms = [name for name in known if name.partition(".")[0] == base]
        message = f"`{base}` has no `.{modifier}` form"
        if len(forms) == 1:
            return message
        suggestion = self.suggestion(f"{base}.{modifier}", forms)
        if suggestion:
            return message + suggestion
        return message + "; its forms are " + ", ".join(f"`{form}`" for form in forms)

    def check_conversion(
        self, name: str, stack: TypeStack, location: Location | None
    ) -> None:
        word, _, modifier = name.partition(".")
        target = CONVERSIONS[word]
        self.require_depth(name, 1, stack, location)
        source = self.inference.resolve(stack.types[-1])
        if self.inference.has_unknowns(source):
            self.fail(
                f"`{name}` converts the value below it, but its type is not known "
                "at this point",
                location,
            )
        if is_type_variable(source):
            forms = self.generic_conversion_forms(source, target)
        else:
            forms = conversion_forms(source, target)
        if not forms:
            message = f"there is no conversion from `{source}` to `{target}`"
            if target == CHAR:
                message += "; only a `u32` converts to a `char`"
            elif source == CHAR:
                message += "; a `char` converts only to `u32`"
            elif source == BOOL:
                message += "; use `select`, as in `1 0 select`"
            elif is_type_variable(source):
                message += (
                    f"; add `where {source}: Integer`, `where {source}: Float` or "
                    f"`where {source}: Num` to the signature"
                )
            self.fail(message, location)
        if modifier not in forms:
            available = ", ".join(
                f"`{word}.{form}`" if form else f"`{word}`" for form in forms
            )
            self.fail(
                f"`{word}` has no `.{modifier}` form when converting from "
                f"`{source}`; its forms are {available}",
                location,
            )
        stack.types[-1] = option_of(target) if modifier == "checked" else target

    def check_bit_cast(
        self, name: str, stack: TypeStack, location: Location | None
    ) -> None:
        self.require_depth(name, 1, stack, location)
        source = self.inference.resolve(stack.types[-1])
        if self.inference.has_unknowns(source):
            self.fail(
                f"`{name}` reinterprets the value below it, but its type is not "
                "known at this point",
                location,
            )
        pairs = BIT_CASTS[name]
        if source not in pairs:
            self.fail(
                f"`{name}` takes {BIT_CAST_TAKES[name]}, found `{source}`", location
            )
        stack.types[-1] = pairs[source]

    def generic_conversion_forms(self, variable: Type, target: Type) -> tuple[str, ...]:
        assumed = self.assumptions.get(variable.name, frozenset())
        if "Integer" in assumed:
            kinds: tuple[Type, ...] = (I32,)
        elif "Float" in assumed:
            kinds = (F32,)
        elif "Num" in assumed:
            kinds = (I32, F32)
        else:
            return ()
        forms = conversion_forms(kinds[0], target)
        for kind in kinds[1:]:
            forms = tuple(
                form for form in forms if form in conversion_forms(kind, target)
            )
        return forms

    def settle_captured(
        self,
        name: str,
        operands: list[Type],
        stack: TypeStack,
        location: Location | None,
    ) -> None:
        result = self.inference.resolve(stack.types[-1])
        assert result.effect is not None
        effect = self.inference.generalize(result.effect)
        stack.types[-1] = quotation_of(effect, result.args[0])
        capture = Capture(name, tuple(operands), result.args[0], location)
        if not self.decide_qualifier(capture):
            self.captures.append(capture)

    def decide_qualifier(self, capture: Capture) -> bool:
        linear = False
        for operand in self.inference.resolve_all(capture.operands):
            if is_quotation(operand):
                operand = operand.args[0]
            if self.inference.has_unknowns(operand):
                return False
            linear = linear or self.is_linear(operand)
        decided = LINEAR if linear else UNRESTRICTED
        if not self.inference.unify(capture.qualifier, decided):
            if linear:
                self.fail(
                    f"the quotation `{capture.name}` builds here holds something "
                    "linear, so it is a linear quotation, but it is used as an "
                    "unrestricted one",
                    capture.location,
                )
            self.fail(
                f"the quotation `{capture.name}` builds here holds nothing linear, "
                "so it is an unrestricted quotation, but it is used as a linear one",
                capture.location,
            )
        return True

    def settle_captures(self) -> None:
        pending, self.captures = self.captures, []
        while pending:
            waiting = [c for c in pending if not self.decide_qualifier(c)]
            if len(waiting) == len(pending):
                break
            pending = waiting
        for capture in pending:
            self.fail(
                f"`{capture.name}` cannot tell whether what it captures is linear, "
                "because its type is never worked out",
                capture.location,
            )

    def require_depth(
        self, name: str, needed: int, stack: TypeStack, location: Location | None
    ) -> None:
        self.settle(stack)
        if len(stack.types) >= needed:
            return
        if is_unknown_row(stack.row):
            below = [
                self.inference.fresh(
                    f"the type of a value that `{name}` takes", location
                )
                for _ in range(needed - len(stack.types))
            ]
            row = self.inference.fresh_row()
            self.inference.bind_row(stack.row, Stack(row, tuple(below)))
            stack.become(Stack(row, (*below, *stack.types)))
            return
        values = "1 value" if needed == 1 else f"{needed} values"
        holds = (
            "is empty" if not stack.types else f"holds only {self.show(stack.types)}"
        )
        self.fail(f"`{name}` takes {values}, but the stack {holds}", location)

    def apply_effect(
        self,
        name: str,
        declared: Effect,
        stack: TypeStack,
        location: Location | None,
        takes: str | None = None,
    ) -> None:
        count = len(declared.inputs)
        self.require_depth(name, count, stack, location)
        effect = self.inference.instantiate(name, declared, location)
        found = stack.types[len(stack.types) - count :]
        expected = takes or format_types(declared.inputs)
        constraints = quotation_constraints(declared.inputs)
        if takes is None and constraints:
            clauses = ", ".join(f"{v}: [ {e.describe()} ]" for v, e in constraints)
            expected += f" where `{clauses}`"
        if not self.inference.unify_all(found, effect.inputs):
            message = f"`{name}` takes {expected}, found {self.show(found)}"
            if name == "compose":
                first, second = self.inference.resolve_all(found)
                if first.effect is not None and second.effect is not None:
                    message = (
                        f"`compose` takes {expected}, but the first leaves "
                        f"{format_types(first.effect.outputs)} and the second "
                        f"takes {format_types(second.effect.inputs)}"
                    )
            self.fail(message, location)
        del stack.types[len(stack.types) - count :]

        if effect.rows is None:
            stack.types.extend(effect.outputs)
        else:
            input_row, output_row = effect.rows
            below = self.inference.flatten(Stack(input_row))
            if not self.inference.unify_stacks(stack.whole(), Stack(input_row)):
                self.fail(self.describe_rest(name, below, stack), location)
            stack.become(self.inference.flatten(Stack(output_row, effect.outputs)))

        for (unknown, constraint), (variable, _) in zip(
            effect.where, declared.where, strict=True
        ):
            if takes is not None:
                message = f"`{name}` takes {takes}, found"
                describe = self.describe_operands(message, found, constraint)
            else:
                describe = self.describe_unmet(name, variable, constraint)
            self.require(Type(unknown), constraint, location, describe)

    def describe_rest(self, name: str, below: Stack, stack: TypeStack) -> str:
        subject = "the quotation given to `call`" if name == "call" else f"`{name}`"
        have = self.inference.flatten(stack.whole())
        if below.row == IMPLICIT_ROW or is_unknown_row(below.row):
            count = len(below.types)
            expected = format_types(below.types)
            found = have.types[len(have.types) - count :] if count else have.types
            found_text = self.show(found)
        else:
            expected = self.show_stack(below)
            found_text = self.show_stack(have)
        return (
            f"{subject} takes {expected} from the rest of the stack, found {found_text}"
        )

    def describe_operands(
        self, message: str, found: list[Type], constraint: str
    ) -> Callable[[Type], str]:
        def describe(resolved: Type) -> str:
            text = f"{message} {self.show(found)}"
            if is_quotation(resolved):
                if constraint == "Display":
                    text += "; quotations cannot be displayed"
                elif constraint != "Unrestricted":
                    text += "; quotations have no equality or order"
            elif is_type_variable(resolved):
                pass
            elif constraint in ("Eq", "PartialEq"):
                text += self.explain_no_equality(resolved)
            elif constraint in ("Ord", "PartialOrd"):
                text += f"; `{resolved}` has no order"
            elif resolved == LINEAR:
                text += "; a linear quotation can be called only once"
            return text

        return describe

    def describe_unmet(
        self, name: str, variable: str, constraint: str
    ) -> Callable[[Type], str]:
        def describe(resolved: Type) -> str:
            subject = "a linear quotation" if resolved == LINEAR else f"`{resolved}`"
            return (
                f"{subject} does not satisfy `{constraint}`, "
                f"which `{name}` requires of its `{variable}`"
            )

        return describe

    def apply_stack_word(
        self, word: StackWord, stack: TypeStack, location: Location | None
    ) -> None:
        self.require_depth(word.name, word.arguments, stack, location)
        taken = stack.types[len(stack.types) - word.arguments :]
        for verb, positions in (("copy", word.copied()), ("discard", word.discarded())):
            for position in positions:
                self.require(
                    taken[position],
                    "Unrestricted",
                    location,
                    self.describe_linear(word.name, verb),
                )
        del stack.types[len(stack.types) - word.arguments :]
        stack.types.extend(taken[position] for position in word.result)

    def describe_linear(self, name: str, verb: str) -> Callable[[Type], str]:
        def describe(resolved: Type) -> str:
            variable = self.qualifier_variable(resolved)
            if variable is not None:
                return (
                    f"`{name}` cannot {verb} a value of type `{variable}`, which may "
                    f"be linear; add `where {variable}: Unrestricted` to the signature"
                )
            if is_type_variable(resolved):
                return (
                    f"`{name}` cannot {verb} a value of type `{resolved}`, "
                    "which may be linear"
                )
            return f"`{name}` cannot {verb} a value of the linear type `{resolved}`"

        return describe

    def suggestion(self, name: str, known: Iterable[str]) -> str:
        matches = difflib.get_close_matches(name, list(known), n=1, cutoff=0.7)
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


def times(count: int) -> str:
    return {0: "never", 1: "once", 2: "twice"}.get(count, f"{count} times")
