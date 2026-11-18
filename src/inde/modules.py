from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from inde.ast import (
    ArrayLiteral,
    Body,
    ConstantDeclaration,
    ConstantReference,
    Constraint,
    ConstructorPattern,
    Declaration,
    If,
    Import,
    Item,
    Let,
    LiteralPattern,
    Match,
    MatchArm,
    Program,
    Quotation,
    QuotationType,
    StackEffect,
    StackType,
    TypeDeclaration,
    TypeExpression,
    TypeReference,
    Variant,
    While,
    WordDeclaration,
    WordReference,
)
from inde.builtins import BIT_CASTS, BUILTINS, CONVERSIONS, STACK_WORDS
from inde.checker import Checker, error_order
from inde.errors import IndeError, IndeErrors
from inde.lexer import Location, Token, TokenType, tokenize
from inde.parser import CORE_LIBRARY_ARGUMENTS, CORE_TYPE_ARGUMENTS, Parser

LIBRARY = Path(__file__).parent / "library"
PRELUDE = "prelude"
SEPARATOR = ":"
EXTENSION = ".inde"

WORDS = "word"
TYPES = "type"
CONSTANTS = "constant"
KINDS = (WORDS, TYPES, CONSTANTS)


def core_names() -> dict[str, frozenset[str]]:
    checker = Checker()
    return {
        WORDS: frozenset(
            {*checker.words, *BUILTINS, *STACK_WORDS, *CONVERSIONS, *BIT_CASTS}
        ),
        TYPES: frozenset({*checker.types, *CORE_TYPE_ARGUMENTS}),
        CONSTANTS: frozenset(),
    }


CORE = core_names()


def is_core(kind: str, name: str) -> bool:
    names = CORE[kind]
    return name in names or (kind == WORDS and name.partition(".")[0] in names)


def leading_imports(tokens: list[Token]) -> list[tuple[str, Location]]:
    found: list[tuple[str, Location]] = []
    index = 0
    while (
        index + 1 < len(tokens)
        and tokens[index].type is TokenType.KEYWORD
        and tokens[index].text == "import"
        and tokens[index + 1].type is TokenType.NAME
    ):
        found.append((tokens[index + 1].text, tokens[index].location))
        index += 2
    return found


@dataclass
class Module:
    name: str
    path: Path
    prefix: str
    program: Program = field(default_factory=Program)
    imports: list[Module] = field(default_factory=list)
    own: dict[str, dict[str, str]] = field(default_factory=dict)
    public: dict[str, set[str]] = field(default_factory=dict)
    arities: dict[str, int] = field(default_factory=dict)
    failed: bool = False

    def declares(self, kind: str, name: str) -> bool:
        return name in self.own.get(kind, {})

    def exports(self, kind: str, name: str) -> bool:
        return name in self.public.get(kind, set())

    def internal(self, kind: str, name: str) -> str:
        return self.own[kind][name]


def declared_names(
    program: Program,
) -> dict[str, list[tuple[str, bool, Location | None]]]:
    found: dict[str, list[tuple[str, bool, Location | None]]] = {k: [] for k in KINDS}
    for declaration in program.declarations:
        public = declaration.public
        if isinstance(declaration, WordDeclaration):
            found[WORDS].append((declaration.name, public, declaration.location))
        elif isinstance(declaration, ConstantDeclaration):
            found[CONSTANTS].append((declaration.name, public, declaration.location))
        else:
            found[TYPES].append((declaration.name, public, declaration.location))
            for variant in declaration.variants:
                found[WORDS].append((variant.name, public, variant.location))
    return found


class Loader:
    def __init__(self, library: Path | None = None) -> None:
        self.library = LIBRARY if library is None else library
        self.modules: dict[Path, Module] = {}
        self.order: list[Module] = []
        self.loading: list[Module] = []
        self.prefixes: set[str] = set()
        self.errors: list[IndeError] = []
        self.main: Module | None = None

    def load(self, path: Path, source: str | None = None) -> Program:
        main = Module("", path, "")
        self.main = main
        self.visit(main, source, None)
        if not self.errors:
            for module in self.order:
                if not module.failed:
                    Resolver(self, module).run()
        if self.errors:
            raise IndeErrors(sorted(self.errors, key=error_order))
        declarations: list[Declaration] = []
        for module in self.order:
            declarations.extend(module.program.declarations)
        return Program(declarations=tuple(declarations), location=main.program.location)

    def start_session(self, directory: Path) -> list[Declaration]:
        main = Module("", directory / "repl", "")
        self.main = main
        prelude = self.prelude()
        if prelude is None:
            return []
        return self.session_attach(PRELUDE, prelude, None)

    def session_import(self, request: Import) -> list[Declaration]:
        main = self.main
        assert main is not None
        if any(imported.name == request.name for imported in main.imports):
            raise IndeErrors(
                [IndeError(f"`{request.name}` is already imported", request.location)]
            )
        target = self.find(request.name, main.path)
        if target is None:
            raise IndeErrors(
                [
                    IndeError(
                        f"no module named `{request.name}` (looked in this directory "
                        "and in the standard library)",
                        request.location,
                    )
                ]
            )
        return self.session_attach(request.name, target, request.location)

    def session_attach(
        self, name: str, path: Path, location: Location | None
    ) -> list[Declaration]:
        main = self.main
        assert main is not None
        saved = (
            dict(self.modules),
            list(self.order),
            set(self.prefixes),
            list(main.imports),
        )
        start = len(self.order)
        self.errors = []
        self.attach(main, name, path, location)
        added = self.order[start:]
        if not self.errors:
            for module in added:
                if not module.failed:
                    Resolver(self, module).run()
        if not self.errors:
            Resolver(self, main).check_session_names()
        if self.errors:
            errors = sorted(self.errors, key=error_order)
            self.errors = []
            self.modules, self.order, self.prefixes, main.imports = saved
            raise IndeErrors(errors)
        return [d for module in added for d in module.program.declarations]

    def session_resolve(self, item: Declaration | Body) -> Declaration | Body:
        main = self.main
        assert main is not None
        self.errors = []
        resolver = Resolver(self, main)
        if isinstance(item, tuple):
            result: Declaration | Body = resolver.body(item, frozenset())
        else:
            resolver.check_declarations(Program(declarations=(item,)))
            result = resolver.declaration(item)
        if self.errors:
            errors = sorted(self.errors, key=error_order)
            self.errors = []
            raise IndeErrors(errors)
        return result

    def session_declared(self, declaration: Declaration) -> None:
        main = self.main
        assert main is not None
        for kind, entries in declared_names(
            Program(declarations=(declaration,))
        ).items():
            for name, _, _ in entries:
                main.own.setdefault(kind, {})[name] = name

    def session_types(self) -> dict[str, int]:
        main = self.main
        known: dict[str, int] = {}
        if main is not None:
            for imported in main.imports:
                known.update(imported.arities)
        return known

    def source_name(self, module: Module) -> str | None:
        return None if module is self.main else str(module.path)

    def find(self, name: str, near: Path) -> Path | None:
        for directory in (near.parent, self.library):
            candidate = directory / f"{name}{EXTENSION}"
            if candidate.is_file():
                return candidate
        return None

    def prelude(self) -> Path | None:
        candidate = self.library / f"{PRELUDE}{EXTENSION}"
        return candidate if candidate.is_file() else None

    def unique_prefix(self, name: str) -> str:
        prefix = name
        count = 1
        while prefix in self.prefixes:
            count += 1
            prefix = f"{name}~{count}"
        self.prefixes.add(prefix)
        return f"{prefix}{SEPARATOR}"

    def visit(
        self, module: Module, source: str | None, location: Location | None
    ) -> None:
        if source is None:
            try:
                source = module.path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                self.errors.append(
                    IndeError(
                        f"cannot read module `{module.name}` ({module.path}): {error}",
                        location,
                    )
                )
                module.failed = True
                return
        self.loading.append(module)
        try:
            tokens = tokenize(source, self.source_name(module))
        except IndeError as error:
            self.errors.append(error)
            module.failed = True
            self.loading.pop()
            return
        prelude = self.prelude()
        if prelude is not None and module.path.resolve() != prelude.resolve():
            self.attach(module, PRELUDE, prelude, None)
        loaded: set[str] = set()
        for name, place in leading_imports(tokens):
            if name in loaded:
                continue
            loaded.add(name)
            target = self.find(name, module.path)
            if target is None:
                self.fail(
                    f"no module named `{name}` (looked beside this file and in the "
                    "standard library)",
                    place,
                )
                module.failed = True
            elif target.resolve() == module.path.resolve():
                self.fail("a module cannot import itself", place)
                module.failed = True
            else:
                self.attach(module, name, target, place)
        known = dict(CORE_TYPE_ARGUMENTS)
        known.update(CORE_LIBRARY_ARGUMENTS)
        for imported in module.imports:
            known.update(imported.arities)
        parser = Parser(tokens, known)
        program = parser.parse_program()
        self.errors.extend(parser.errors)
        if parser.errors:
            module.failed = True
        first: dict[str, Import] = {}
        for request in program.imports:
            if request.name in first:
                earlier = first[request.name].location
                where = (
                    "" if earlier is None else f"; the first is on line {earlier.line}"
                )
                self.fail(
                    f"`{request.name}` is imported twice{where}", request.location
                )
                module.failed = True
            first.setdefault(request.name, request)
        module.program = program
        self.tabulate(module)
        self.loading.pop()
        self.order.append(module)

    def fail(self, message: str, location: Location | None) -> None:
        self.errors.append(IndeError(message, location))

    def attach(
        self, module: Module, name: str, path: Path, location: Location | None
    ) -> None:
        key = path.resolve()
        cycle = [m for m in self.loading if m.path.resolve() == key]
        if cycle:
            start = self.loading.index(cycle[0])
            names = [self.display(m) for m in self.loading[start:]]
            chain = " → ".join(f"`{n}`" for n in [*names, names[0]])
            self.fail(f"import cycle {chain}", location)
            module.failed = True
            return
        imported = self.modules.get(key)
        if imported is None:
            imported = Module(name, path, self.unique_prefix(name))
            self.modules[key] = imported
            self.visit(imported, None, location)
        if imported.failed:
            module.failed = True
        module.imports.append(imported)

    def display(self, module: Module) -> str:
        return module.name or module.path.stem

    def tabulate(self, module: Module) -> None:
        names = declared_names(module.program)
        for kind in KINDS:
            module.own[kind] = {
                name: f"{module.prefix}{name}" for name, _, _ in names[kind]
            }
            module.public[kind] = {name for name, public, _ in names[kind] if public}
        for declaration in module.program.declarations:
            if isinstance(declaration, TypeDeclaration) and declaration.public:
                module.arities[declaration.name] = len(declaration.params)


class Resolver:
    def __init__(self, loader: Loader, module: Module) -> None:
        self.loader = loader
        self.module = module
        self.errors = loader.errors
        self.scope: dict[str, dict[str, list[tuple[str, Module]]]] = {}
        for kind in KINDS:
            table: dict[str, list[tuple[str, Module]]] = {}
            for imported in module.imports:
                for name in sorted(imported.public[kind]):
                    entry = (imported.internal(kind, name), imported)
                    if entry not in table.setdefault(name, []):
                        table[name].append(entry)
            self.scope[kind] = table

    def fail(self, message: str, location: Location | None) -> None:
        self.errors.append(IndeError(message, location))

    def run(self) -> None:
        self.check_declarations(self.module.program)
        declarations = tuple(
            self.declaration(declaration)
            for declaration in self.module.program.declarations
        )
        self.module.program = replace(self.module.program, declarations=declarations)

    def check_session_names(self) -> None:
        for kind in KINDS:
            for name, providers in self.scope[kind].items():
                if self.module.declares(kind, name):
                    label = "type " if kind == TYPES else ""
                    self.fail(
                        f"{label}`{name}` from `{providers[0][1].name}` is already "
                        "defined in this session",
                        None,
                    )

    def check_declarations(self, program: Program) -> None:
        main = self.module is self.loader.main
        for kind, entries in declared_names(program).items():
            for name, _, location in entries:
                providers = self.scope[kind].get(name, [])
                if providers:
                    source = providers[0][1].name
                    label = "type " if kind == TYPES else ""
                    self.fail(
                        f"{label}`{name}` is already provided by the import of "
                        f"`{source}`",
                        location,
                    )
                elif not main and is_core(kind, name):
                    if kind == TYPES:
                        self.fail(
                            f"type `{name}` is already defined by the core library",
                            location,
                        )
                    else:
                        self.fail(
                            f"`{name}` is a built-in word and cannot be redefined",
                            location,
                        )

    def lookup(self, kind: str, name: str, location: Location | None) -> str:
        if self.module.declares(kind, name):
            return self.module.internal(kind, name)
        providers = self.scope[kind].get(name, [])
        if len(providers) == 1:
            return providers[0][0]
        if providers:
            modules = sorted({provider[1].name for provider in providers})
            listed = " and ".join(f"`{m}`" for m in modules)
            self.fail(f"`{name}` is provided by both {listed}", location)
            return name
        if kind == WORDS and "." in name:
            base, dot, rest = name.partition(".")
            if self.module.declares(kind, base) or base in self.scope[kind]:
                return f"{self.lookup(kind, base, location)}{dot}{rest}"
        if is_core(kind, name):
            return name
        self.explain(kind, name, location)
        return name

    def explain(self, kind: str, name: str, location: Location | None) -> None:
        label = {WORDS: "word", TYPES: "type", CONSTANTS: "constant"}[kind]
        for imported in self.module.imports:
            if imported.declares(kind, name):
                self.fail(
                    f"{label} `{name}` is private to module `{imported.name}`",
                    location,
                )
                return
        main = self.loader.main
        for other in self.loader.order:
            if other is self.module or other is main or not other.exports(kind, name):
                continue
            self.fail(
                f"unknown {label} `{name}`; it is in module `{other.name}`, which "
                f"this module does not import",
                location,
            )
            return
        if self.module is not main and main is not None and main.declares(kind, name):
            self.fail(f"unknown {label} `{name}`", location)

    def declaration(self, declaration: Declaration) -> Declaration:
        prefix = self.module.prefix
        if isinstance(declaration, WordDeclaration):
            if declaration.public:
                self.check_exposure(
                    f"the public word `{declaration.name}`",
                    self.effect_types(declaration.signature),
                    self.where_types(declaration.where),
                )
            return replace(
                declaration,
                name=f"{prefix}{declaration.name}",
                signature=self.effect(declaration.signature),
                where=tuple(self.constraint(c) for c in declaration.where),
                body=self.body(declaration.body, frozenset()),
            )
        if isinstance(declaration, TypeDeclaration):
            if declaration.public:
                self.check_exposure(
                    f"the public type `{declaration.name}`",
                    [f for v in declaration.variants for f in v.fields],
                )
            return replace(
                declaration,
                name=f"{prefix}{declaration.name}",
                variants=tuple(self.variant(v) for v in declaration.variants),
            )
        return replace(declaration, name=f"{prefix}{declaration.name}")

    def effect_types(self, effect: StackEffect) -> list[TypeExpression]:
        return [*effect.inputs.types, *effect.outputs.types]

    def where_types(self, where: tuple[Constraint, ...]) -> list[TypeExpression]:
        return [b for c in where for b in c.bounds if isinstance(b, QuotationType)]

    def check_exposure(self, subject: str, *groups: Iterable[TypeExpression]) -> None:
        reported: set[str] = set()
        for group in groups:
            for expression in group:
                for name, location in self.type_names(expression):
                    if (
                        self.module.declares(TYPES, name)
                        and not self.module.exports(TYPES, name)
                        and name not in reported
                    ):
                        reported.add(name)
                        self.fail(
                            f"{subject} mentions the private type `{name}`", location
                        )

    def type_names(
        self, expression: TypeExpression | QuotationType
    ) -> list[tuple[str, Location | None]]:
        if isinstance(expression, TypeReference):
            found = [(expression.name, expression.location)]
            for argument in expression.args:
                found.extend(self.type_names(argument))
            return found
        if isinstance(expression, QuotationType):
            found = []
            for inner in self.effect_types(expression.effect):
                found.extend(self.type_names(inner))
            return found
        return []

    def variant(self, variant: Variant) -> Variant:
        return replace(
            variant,
            name=f"{self.module.prefix}{variant.name}",
            fields=tuple(self.type(f) for f in variant.fields),
        )

    def type(self, expression: TypeExpression) -> TypeExpression:
        if isinstance(expression, TypeReference):
            return replace(
                expression,
                name=self.lookup(TYPES, expression.name, expression.location),
                args=tuple(self.type(a) for a in expression.args),
            )
        if isinstance(expression, QuotationType):
            return replace(expression, effect=self.effect(expression.effect))
        return expression

    def stack(self, stack: StackType) -> StackType:
        return replace(stack, types=tuple(self.type(t) for t in stack.types))

    def effect(self, effect: StackEffect) -> StackEffect:
        return replace(
            effect, inputs=self.stack(effect.inputs), outputs=self.stack(effect.outputs)
        )

    def constraint(self, constraint: Constraint) -> Constraint:
        bounds = tuple(
            replace(b, effect=self.effect(b.effect))
            if isinstance(b, QuotationType)
            else b
            for b in constraint.bounds
        )
        return replace(constraint, bounds=bounds)

    def body(self, body: Body, locals_: frozenset[str]) -> Body:
        return tuple(self.item(item, locals_) for item in body)

    def optional(self, body: Body | None, locals_: frozenset[str]) -> Body | None:
        return None if body is None else self.body(body, locals_)

    def item(self, item: Item, locals_: frozenset[str]) -> Item:
        if isinstance(item, WordReference):
            if item.name in locals_:
                return item
            return replace(item, name=self.lookup(WORDS, item.name, item.location))
        if isinstance(item, ConstantReference):
            return self.constant(item)
        if isinstance(item, ArrayLiteral):
            return self.array(item)
        if isinstance(item, Quotation):
            return replace(item, body=self.body(item.body, locals_))
        if isinstance(item, If):
            return replace(
                item,
                then_body=self.body(item.then_body, locals_),
                else_body=self.optional(item.else_body, locals_),
            )
        if isinstance(item, While):
            return replace(
                item,
                condition=self.body(item.condition, locals_),
                body=self.body(item.body, locals_),
            )
        if isinstance(item, Match):
            return replace(
                item,
                arms=tuple(self.arm(arm, locals_) for arm in item.arms),
                else_body=self.optional(item.else_body, locals_),
            )
        if isinstance(item, Let):
            main = self.module is self.loader.main
            for name in item.names:
                own = self.module.declares(WORDS, name) and not main
                if own or name in self.scope[WORDS]:
                    self.fail(
                        f"`{name}` already names a word, so it cannot name a local",
                        item.location,
                    )
            return replace(item, body=self.body(item.body, locals_ | set(item.names)))
        return item

    def constant(self, reference: ConstantReference) -> ConstantReference:
        name = self.lookup(CONSTANTS, reference.name, reference.location)
        return replace(reference, name=name)

    def array(self, literal: ArrayLiteral) -> ArrayLiteral:
        elements = tuple(
            self.constant(e)
            if isinstance(e, ConstantReference)
            else self.array(e)
            if isinstance(e, ArrayLiteral)
            else e
            for e in literal.elements
        )
        return replace(literal, elements=elements)

    def arm(self, arm: MatchArm, locals_: frozenset[str]) -> MatchArm:
        pattern = arm.pattern
        if isinstance(pattern, ConstructorPattern):
            pattern = replace(
                pattern, name=self.lookup(WORDS, pattern.name, pattern.location)
            )
        elif isinstance(pattern.literal, ConstantReference):
            pattern = LiteralPattern(
                literal=self.constant(pattern.literal), location=pattern.location
            )
        return replace(arm, pattern=pattern, body=self.body(arm.body, locals_))


def load_program(
    path: Path, source: str | None = None, library: Path | None = None
) -> Program:
    return Loader(library).load(path, source)
