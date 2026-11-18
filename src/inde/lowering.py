from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from inde import ir
from inde.ast import (
    ArrayLiteral,
    Body,
    BooleanLiteral,
    CharacterLiteral,
    ConstantReference,
    ConstructorPattern,
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
    StringLiteral,
    While,
    WordReference,
)
from inde.builtins import BIT_CASTS, BUILTINS, CONVERSIONS, STACK_WORDS
from inde.checker import Checker, Scope, TypeStack, WordInfo, error_order
from inde.errors import IndeError, IndeErrors
from inde.inference import Inference
from inde.lexer import Location
from inde.type_model import (
    BOOL,
    CHAR,
    IMPLICIT_ROW,
    NUMBER_TYPES,
    STR,
    U32,
    UNIT,
    Effect,
    Stack,
    Type,
    is_type_variable,
    quotation_of,
    substitute_effect,
)

INSTANCE = "#instance"
MAX_TYPE_DEPTH = 24

ARITHMETIC = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "rem"}
COMPARISONS = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}

PRIMITIVE_WORDS = frozenset(
    [
        "neg",
        "abs",
        "sqrt",
        "floor",
        "ceil",
        "trunc",
        "round",
        "copysign",
        "fma",
        "bit-and",
        "bit-or",
        "bit-xor",
        "bit-not",
        "shl",
        "shr",
        "rotl",
        "rotr",
        "count-ones",
        "leading-zeros",
        "trailing-zeros",
    ]
)
BOOLEAN_WORDS = frozenset({"and", "or", "xor", "not"})

PLAIN_RUNTIME_WORDS = frozenset(
    [
        "str-len",
        "str-concat",
        "str-slice",
        "str-char-at",
        "str-to-bytes",
        "bytes-to-str",
        "str-from-char",
        "strbuf-new",
        "strbuf-push",
        "strbuf-append",
        "strbuf-len",
        "strbuf-snapshot",
        "strbuf-freeze",
        "strbuf-from-str",
    ]
)


def written(name: str) -> str:
    return name.rpartition(":")[2]


def underscored(word: str) -> str:
    return word.replace("-", "_")


@dataclass
class RecordingInference(Inference):
    last: dict[str, Type] | None = None

    def instantiate(
        self, name: str, effect: Effect, location: Location | None
    ) -> Effect:
        replacements = {
            variable: self.fresh(f"the type `{variable}` of this `{name}`", location)
            for variable in effect.variables()
        }
        rows = {row: self.fresh_row() for row in effect.row_variables()}
        self.last = replacements
        return substitute_effect(effect, replacements, rows)


@dataclass(frozen=True)
class Record:
    before: tuple[Type, ...]
    after: tuple[Type, ...]
    variables: dict[str, Type] | None


class Elaborator(Checker):
    def __init__(self) -> None:
        self.recording: dict[int, tuple[Stack, Stack, dict[str, Type] | None]] | None
        self.recording = None
        self.kept = 0
        super().__init__()

    def begin(self, where: tuple[tuple[str, str], ...] = ()) -> None:
        super().begin(where)
        self.inference = RecordingInference()

    def check_body(
        self, body: Body, stack: TypeStack, scope: Scope | None = None
    ) -> TypeStack:
        if self.recording is None:
            return super().check_body(body, stack, scope)
        scope = scope or {}
        for item in body:
            self.check_item(item, stack, scope)
        return stack

    def check_item(self, item: Item, stack: TypeStack, scope: Scope) -> None:
        if self.recording is None:
            super().check_item(item, stack, scope)
            return
        inference = self.inference
        assert isinstance(inference, RecordingInference)
        before = stack.whole()
        inference.last = None
        super().check_item(item, stack, scope)
        self.recording[id(item)] = (before, stack.whole(), inference.last)

    def check_propagate(self, item: Propagate, stack: TypeStack, scope: Scope) -> None:
        if self.recording is None or self.kept == 0:
            super().check_propagate(item, stack, scope)
            return
        kept = stack.types[: self.kept]
        own = TypeStack(IMPLICIT_ROW, stack.types[self.kept :])
        super().check_propagate(item, own, scope)
        stack.types = kept + own.types

    def check_instance(
        self, word: WordInfo, effect: Effect, kept: int
    ) -> dict[int, Record]:
        assert word.declaration is not None
        self.recording = {}
        self.kept = kept
        self.begin()
        own = Effect(effect.inputs[kept:], effect.outputs[kept:])
        module = word.name.rpartition(":")[0]
        instance = f"{module}:{INSTANCE}" if module else INSTANCE
        self.words[instance] = WordInfo(instance, own, word.location)
        self.current_word = instance
        try:
            stack = TypeStack(IMPLICIT_ROW, effect.inputs)
            self.check_body(word.declaration.body, stack)
            declared = Stack(IMPLICIT_ROW, effect.outputs)
            if not self.inference.unify_stacks(stack.whole(), declared):
                raise AssertionError(f"instance of `{word.name}` does not check")
            self.finish()
            return {
                key: Record(
                    self.resolve_stack(before),
                    self.resolve_stack(after),
                    None
                    if variables is None
                    else {
                        name: self.concrete(self.inference.resolve(t), None)
                        for name, t in variables.items()
                    },
                )
                for key, (before, after, variables) in self.recording.items()
            }
        finally:
            del self.words[instance]
            self.current_word = None
            self.recording = None

    def resolve_stack(self, stack: Stack) -> tuple[Type, ...]:
        flat = self.inference.flatten(stack)
        return tuple(self.concrete(t, None) for t in flat.types)

    def concrete(self, checked_type: Type, location: Location | None) -> Type:
        effect = checked_type.effect
        if effect is None:
            if not checked_type.args:
                return checked_type
            return Type(
                checked_type.name,
                tuple(self.concrete(arg, location) for arg in checked_type.args),
            )
        if effect.rows is not None and effect.rows[0] != effect.rows[1]:
            raise IndeErrors(
                [
                    IndeError(
                        "this program cannot be compiled: the stack effect of a "
                        f"quotation, `{checked_type}`, is not fixed",
                        location,
                    )
                ]
            )
        return quotation_of(
            Effect(
                tuple(self.concrete(t, location) for t in effect.inputs),
                tuple(self.concrete(t, location) for t in effect.outputs),
            ),
            checked_type.args[0],
        )


def row_lengths(
    declared: Effect, taken: tuple[Type, ...], left: tuple[Type, ...]
) -> dict[str, int]:
    parent: dict[str, str] = {}
    offset: dict[str, int] = {}
    bounds: list[tuple[str, int]] = []

    def find(row: str) -> tuple[str, int]:
        parent.setdefault(row, row)
        offset.setdefault(row, 0)
        if parent[row] == row:
            return row, 0
        root, above = find(parent[row])
        return root, above + offset[row]

    def relate(row: str, other: str, difference: int) -> None:
        root, row_offset = find(row)
        other_root, other_offset = find(other)
        if root == other_root:
            assert row_offset - other_offset == difference, (row, other)
            return
        parent[root] = other_root
        offset[root] = difference - row_offset + other_offset

    def visit(pattern: Type, actual: Type) -> None:
        pattern_effect, actual_effect = pattern.effect, actual.effect
        if pattern_effect is not None and actual_effect is not None:
            if pattern_effect.rows is not None:
                below_in = len(actual_effect.inputs) - len(pattern_effect.inputs)
                below_out = len(actual_effect.outputs) - len(pattern_effect.outputs)
                row, other = pattern_effect.rows
                relate(row, other, below_in - below_out)
                bounds.extend([(row, below_in), (other, below_out)])
            return
        for pattern_arg, actual_arg in zip(pattern.args, actual.args, strict=False):
            visit(pattern_arg, actual_arg)

    for pattern, actual in zip(declared.inputs, taken, strict=True):
        visit(pattern, actual)
    for pattern, actual in zip(declared.outputs, left, strict=True):
        visit(pattern, actual)
    for row in declared.row_variables():
        bounds.append((row, 0))

    base: dict[str, int] = {}
    for row, bound in bounds:
        root, above = find(row)
        base[root] = max(base.get(root, 0), bound - above)
    lengths: dict[str, int] = {}
    for row in declared.row_variables():
        root, above = find(row)
        lengths[row] = base[root] + above
    return lengths


def quotation_arguments(
    patterns: tuple[Type, ...], actuals: tuple[Type, ...], found: dict[str, Type]
) -> None:
    for pattern, actual in zip(patterns, actuals, strict=True):
        if pattern.effect is not None:
            if is_type_variable(pattern.args[0]):
                found.setdefault(pattern.args[0].name, actual)
        else:
            quotation_arguments(pattern.args, actual.args[: len(pattern.args)], found)


def type_depth(argument: ir.TypeArgument) -> int:
    if isinstance(argument, tuple):
        return max((type_depth(t) for t in argument), default=0)
    parts = list(argument.args)
    if argument.effect is not None:
        parts.extend(argument.effect.inputs + argument.effect.outputs)
    return 1 + max((type_depth(part) for part in parts), default=0)


def is_generic(effect: Effect) -> bool:
    return bool(effect.variables() or effect.row_variables())


@dataclass
class Root:
    function: ir.Function
    quotations: list[ir.Function] = field(default_factory=list)
    numbers: Iterator[int] = field(default_factory=itertools.count)

    def quotation(
        self, inputs: tuple[Type, ...], outputs: tuple[Type, ...]
    ) -> ir.Function:
        function = ir.Function(
            f"{self.function.name}.q{next(self.numbers)}", inputs, outputs
        )
        self.quotations.append(function)
        return function


class Lowerer:
    def __init__(self, program: Program) -> None:
        self.checker = Elaborator()
        self.checker.check_program(program, require_main=True)
        if self.checker.errors:
            raise IndeErrors(sorted(self.checker.errors, key=error_order))
        self.roots: list[Root] = []
        self.instances: dict[tuple[str, tuple[Type, ...], tuple[Type, ...]], Root] = {}
        self.pending: deque[tuple[WordInfo, Root, int]] = deque()
        self.data: dict[tuple[str, str], ir.DataEntry] = {}
        self.runtime: dict[str, ir.RuntimeDeclaration] = {}
        self.equalities: dict[Type, str] = {}

    def compile(self) -> ir.Module:
        for word in list(self.checker.words.values()):
            if (
                word.kind == "word"
                and word.declaration is not None
                and not is_generic(word.effect)
            ):
                effect = word.effect
                self.request(word, effect.inputs, effect.outputs, (), 0, word.location)
        while self.pending:
            word, root, kept = self.pending.popleft()
            effect = Effect(root.function.inputs, root.function.outputs)
            records = self.checker.check_instance(word, effect, kept)
            assert word.declaration is not None
            lowering = FunctionLowering(self, root, root.function, records, kept)
            lowering.lower_function(word.declaration.body)

        main = self.checker.words["main"]
        functions = [
            function
            for root in self.roots
            for function in (root.function, *root.quotations)
        ]
        module = ir.Module(
            data=list(self.data.values()),
            runtime=list(self.runtime.values()),
            functions=functions,
            entry=("$main", "result" if main.effect.outputs else "unit"),
        )
        module.types = self.type_table(module)
        return module

    def request(
        self,
        word: WordInfo,
        inputs: tuple[Type, ...],
        outputs: tuple[Type, ...],
        arguments: tuple[ir.TypeArgument, ...],
        kept: int,
        location: Location | None,
    ) -> ir.Function:
        key = (word.name, inputs, outputs)
        if key in self.instances:
            return self.instances[key].function
        if any(type_depth(argument) > MAX_TYPE_DEPTH for argument in arguments):
            raise IndeErrors(
                [
                    IndeError(
                        f"this program cannot be compiled: `{word.name}` is used "
                        "at ever larger types, so it would need endless copies "
                        "(polymorphic recursion)",
                        location,
                    )
                ]
            )
        name = ir.function_name(word.name, arguments)
        taken = {root.function.name for root in self.roots}
        if name in taken:
            name = next(
                f"{name}#{n}" for n in itertools.count(2) if f"{name}#{n}" not in taken
            )
        root = Root(ir.Function(name, inputs, outputs))
        self.roots.append(root)
        self.instances[key] = root
        self.pending.append((word, root, kept))
        return root.function

    def string(self, value: str) -> str:
        key = ("str", ir.quote_text(value))
        if key not in self.data:
            self.data[key] = ir.DataEntry(f"$s{self.count('$s')}", STR, value)
        return self.data[key].name

    def array(self, value: tuple[object, ...], array_type: Type) -> str:
        entry = ir.DataEntry("", array_type, value)
        key = (ir.ir_type(array_type), ir.format_data(entry))
        if key not in self.data:
            self.data[key] = ir.DataEntry(f"$a{self.count('$a')}", array_type, value)
        return self.data[key].name

    def count(self, prefix: str) -> int:
        return sum(1 for entry in self.data.values() if entry.name.startswith(prefix))

    def declare(
        self, call: ir.Runtime, inputs: tuple[Type, ...], outputs: tuple[Type, ...]
    ) -> ir.Runtime:
        name = call.full_name
        declaration = ir.RuntimeDeclaration(name, inputs, outputs)
        known = self.runtime.setdefault(name, declaration)
        assert known == declaration, (known, declaration)
        return call

    def is_linear(self, checked_type: Type) -> bool:
        self.checker.assumptions = {}
        return self.checker.is_linear(checked_type)

    def equality(self, compared: Type) -> str:
        if compared in self.equalities:
            return self.equalities[compared]
        name = ir.function_name("==", (compared,))
        self.equalities[compared] = name
        function = ir.Function(name, (compared, compared), (BOOL,))
        self.roots.append(Root(function))
        if compared.name == "array":
            self.array_equality(function, compared.args[0])
        else:
            self.variant_equality(function, compared)
        return name

    def compare(self, word: str, compared: Type) -> list[ir.Instruction]:
        op = COMPARISONS[word]
        if compared in NUMBER_TYPES or compared in (BOOL, CHAR):
            return [ir.Primitive(type=compared, op=op)]
        if compared == STR:
            call = ir.Runtime(name=f"str_{op}")
            return [self.declare(call, (STR, STR), (BOOL,))]
        if compared == UNIT:
            drop = ir.StackOp(op="drop", type=UNIT)
            return [drop, drop, ir.Const(type=BOOL, value=word == "==")]
        instructions: list[ir.Instruction] = [ir.Call(function=self.equality(compared))]
        if word == "!=":
            instructions.append(ir.Primitive(type=BOOL, op="not"))
        return instructions

    def variant_equality(self, function: ir.Function, compared: Type) -> None:
        variants = self.variants(compared)
        function.slots.append(("$b", compared))
        outer: list[tuple[str, ir.Body]] = []
        for variant, fields in variants:
            mine = [f"$a.{variant}.{i}" for i in range(len(fields))]
            function.slots.extend(zip(mine, fields, strict=True))
            inner: list[tuple[str, ir.Body]] = []
            for other, other_fields in variants:
                body: list[ir.Instruction] = []
                if other == variant:
                    theirs = [f"$b.{variant}.{i}" for i in range(len(fields))]
                    function.slots.extend(zip(theirs, fields, strict=True))
                    body.extend(ir.LocalOp(op="set", slot=s) for s in reversed(theirs))
                    for index, field_type in enumerate(fields):
                        body.append(ir.LocalOp(op="take", slot=mine[index]))
                        body.append(ir.LocalOp(op="take", slot=theirs[index]))
                        body.extend(self.compare("==", field_type))
                        if index > 0:
                            body.append(ir.Primitive(type=BOOL, op="and"))
                    if not fields:
                        body.append(ir.Const(type=BOOL, value=True))
                else:
                    body.extend(
                        ir.StackOp(op="drop", type=t) for t in reversed(other_fields)
                    )
                    body.extend(ir.LocalOp(op="drop", slot=s) for s in mine)
                    body.append(ir.Const(type=BOOL, value=False))
                inner.append((other, tuple(body)))
            arm: list[ir.Instruction] = [
                ir.LocalOp(op="set", slot=s) for s in reversed(mine)
            ]
            arm.append(ir.LocalOp(op="take", slot="$b"))
            arm.append(
                ir.Match(type=compared, inputs=(), outputs=(BOOL,), arms=tuple(inner))
            )
            outer.append((variant, tuple(arm)))
        function.body = (
            ir.LocalOp(op="set", slot="$b"),
            ir.Match(type=compared, inputs=(), outputs=(BOOL,), arms=tuple(outer)),
        )

    def array_equality(self, function: ir.Function, element: Type) -> None:
        compared = Type("array", (element,))
        function.slots.extend([("$a", compared), ("$b", compared)])
        length = self.declare(
            ir.Runtime(name="array_len", arguments=(element,)), (compared,), (U32,)
        )
        get = self.declare(
            ir.Runtime(name="array_get", arguments=(element,)),
            (compared, U32),
            (element,),
        )

        def local(op: str, slot: str) -> ir.LocalOp:
            return ir.LocalOp(op=op, slot=slot)

        condition = (
            ir.StackOp(op="dup", type=U32),
            local("get", "$a"),
            length,
            ir.Primitive(type=U32, op="lt"),
            ir.If(
                inputs=(U32,),
                outputs=(U32, BOOL),
                then_body=(
                    ir.StackOp(op="dup", type=U32),
                    local("get", "$a"),
                    ir.StackOp(op="swap"),
                    get,
                    ir.StackOp(op="over", type=U32),
                    local("get", "$b"),
                    ir.StackOp(op="swap"),
                    get,
                    *self.compare("==", element),
                ),
                else_body=(ir.Const(type=BOOL, value=False),),
            ),
        )
        step = (ir.Const(type=U32, value=1), ir.Primitive(type=U32, op="add"))
        function.body = (
            local("set", "$b"),
            local("set", "$a"),
            local("get", "$a"),
            length,
            local("get", "$b"),
            length,
            ir.Primitive(type=U32, op="eq"),
            ir.If(
                inputs=(),
                outputs=(BOOL,),
                then_body=(
                    ir.Const(type=U32, value=0),
                    ir.While(types=(U32,), condition=condition, body=step),
                    local("get", "$a"),
                    length,
                    ir.Primitive(type=U32, op="eq"),
                ),
                else_body=(ir.Const(type=BOOL, value=False),),
            ),
            local("drop", "$a"),
            local("drop", "$b"),
        )

    def variants(self, declared: Type) -> list[tuple[str, tuple[Type, ...]]]:
        return [
            (variant, self.checker.variant_fields(variant, declared))
            for variant in self.checker.variants[declared.name]
        ]

    def type_table(self, module: ir.Module) -> list[ir.TypeEntry]:
        found: dict[Type, ir.TypeEntry] = {}

        def visit(checked_type: Type) -> None:
            effect = checked_type.effect
            if effect is not None:
                for part in effect.inputs + effect.outputs:
                    visit(part)
                return
            for arg in checked_type.args:
                visit(arg)
            if checked_type.name in ir.CORE_TYPE_NAMES or checked_type in found:
                return
            variants = self.variants(checked_type)
            found[checked_type] = ir.TypeEntry(
                checked_type, tuple(variants), self.is_linear(checked_type)
            )
            for _, fields in variants:
                for field_type in fields:
                    visit(field_type)

        for function in module.functions:
            for checked_type in function.inputs + function.outputs:
                visit(checked_type)
            for _, slot_type in function.slots:
                visit(slot_type)
            for instruction in ir.walk(function.body):
                for checked_type in ir.types_of(instruction):
                    visit(checked_type)
        for declaration in module.runtime:
            for checked_type in declaration.inputs + declaration.outputs:
                visit(checked_type)
        for data in module.data:
            visit(data.type)
        return list(found.values())


@dataclass
class Local:
    name: str
    slot: str
    linear: bool


class FunctionLowering:
    def __init__(
        self,
        lowerer: Lowerer,
        root: Root,
        function: ir.Function,
        records: dict[int, Record],
        kept: int,
    ) -> None:
        self.lowerer = lowerer
        self.root = root
        self.function = function
        self.records = records
        self.kept = kept
        self.scope: list[Local] = []

    def lower_function(self, body: Body) -> None:
        instructions, _, _ = self.lower_body(body, len(self.function.inputs), True)
        self.function.body = instructions

    def lower_body(
        self, body: Body, height: int, tail: bool
    ) -> tuple[ir.Body, int, bool]:
        out: list[ir.Instruction] = []
        low = height
        ended = False
        for index, item in enumerate(body):
            item_low, ended = self.lower_item(
                item, tail and index == len(body) - 1, out
            )
            low = min(low, item_low)
        return tuple(out), low, ended

    def lower_item(
        self, item: Item, tail: bool, out: list[ir.Instruction]
    ) -> tuple[int, bool]:
        record = self.records[id(item)]
        height = len(record.before)
        location = item.location
        if isinstance(item, ConstantReference):
            literal = self.lowerer.checker.constants[item.name].value
            out.append(
                self.literal(literal, record.after[-1], location, written(item.name))
            )
            return height, False
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
            out.append(self.literal(item, record.after[-1], location, None))
            return height, False
        if isinstance(item, WordReference):
            local = self.find_local(item.name)
            if local is not None:
                op = "take" if local.linear else "get"
                out.append(
                    ir.LocalOp(
                        op=op, slot=local.slot, location=location, word=item.name
                    )
                )
                return height, False
            return self.lower_word(item, record, tail, out)
        if isinstance(item, Quotation):
            function = self.lower_quotation(item, record)
            out.append(ir.FuncRef(function=function.name, location=location))
            return height, False
        if isinstance(item, Propagate):
            return self.lower_propagate(item, record, out)
        if isinstance(item, If):
            return self.lower_if(item, record, tail, out)
        if isinstance(item, While):
            return self.lower_while(item, record, out)
        if isinstance(item, Match):
            return self.lower_match(item, record, tail, out)
        assert isinstance(item, Let), type(item).__name__
        return self.lower_let(item, record, tail, out)

    def literal(
        self,
        literal: Literal,
        literal_type: Type,
        location: Location | None,
        word: str | None,
    ) -> ir.Instruction:
        if isinstance(literal, StringLiteral):
            name = self.lowerer.string(literal.value)
            return ir.DataRef(kind="str", name=name, location=location, word=word)
        if isinstance(literal, ArrayLiteral):
            value = self.array_value(literal)
            name = self.lowerer.array(value, literal_type)
            return ir.DataRef(kind="array", name=name, location=location, word=word)
        return ir.Const(
            type=literal_type, value=literal.value, location=location, word=word
        )

    def array_value(self, literal: ArrayLiteral) -> tuple[object, ...]:
        values: list[object] = []
        for element in literal.elements:
            if isinstance(element, ConstantReference):
                element = self.lowerer.checker.constants[element.name].value
            if isinstance(element, ArrayLiteral):
                values.append(self.array_value(element))
            else:
                values.append(element.value)
        return tuple(values)

    def lower_quotation(self, item: Quotation, record: Record) -> ir.Function:
        quotation = record.after[-1]
        assert quotation.effect is not None
        function = self.root.quotation(
            quotation.effect.inputs, quotation.effect.outputs
        )
        inner = FunctionLowering(self.lowerer, self.root, function, self.records, 0)
        inner.lower_function(item.body)
        return function

    def find_local(self, name: str) -> Local | None:
        for local in reversed(self.scope):
            if local.name == name:
                return local
        return None

    def new_slot(self, name: str, slot_type: Type) -> str:
        taken = {slot for slot, _ in self.function.slots}
        slot = f"${name}"
        for number in itertools.count(1):
            if slot not in taken:
                break
            slot = f"${name}.{number}"
        self.function.slots.append((slot, slot_type))
        return slot

    def drop_slots(self, location: Location | None) -> list[ir.Instruction]:
        return [
            ir.LocalOp(op="drop", slot=local.slot, location=location)
            for local in reversed(self.scope)
            if not local.linear
        ]

    def lower_let(
        self, item: Let, record: Record, tail: bool, out: list[ir.Instruction]
    ) -> tuple[int, bool]:
        count = len(item.names)
        height = len(record.before) - count
        bound = record.before[height:]
        locals_ = [
            Local(
                name,
                self.new_slot(name, bound_type),
                self.lowerer.is_linear(bound_type),
            )
            for name, bound_type in zip(item.names, bound, strict=True)
        ]
        for local in reversed(locals_):
            out.append(
                ir.LocalOp(
                    op="set", slot=local.slot, location=item.location, word="let"
                )
            )
        self.scope.extend(locals_)
        body, low, ended = self.lower_body(item.body, height, tail)
        del self.scope[len(self.scope) - count :]
        out.extend(body)
        if not ended:
            out.extend(
                ir.LocalOp(
                    op="drop", slot=local.slot, location=item.location, word="end"
                )
                for local in reversed(locals_)
                if not local.linear
            )
        return min(height, low), ended

    def lower_if(
        self, item: If, record: Record, tail: bool, out: list[ir.Instruction]
    ) -> tuple[int, bool]:
        below = record.before[:-1]
        height = len(below)
        then_body, then_low, then_ended = self.lower_body(item.then_body, height, tail)
        else_body: ir.Body | None = None
        else_low, else_ended = height, False
        if item.else_body is not None:
            else_body, else_low, else_ended = self.lower_body(
                item.else_body, height, tail
            )
        low = min(height, then_low, else_low)
        out.append(
            ir.If(
                inputs=below[low:],
                outputs=record.after[low:],
                then_body=then_body,
                else_body=else_body,
                location=item.location,
                word="if",
            )
        )
        return low, then_ended and else_ended

    def lower_while(
        self, item: While, record: Record, out: list[ir.Instruction]
    ) -> tuple[int, bool]:
        height = len(record.before)
        condition, condition_low, _ = self.lower_body(item.condition, height, False)
        body, body_low, _ = self.lower_body(item.body, height, False)
        low = min(height, condition_low, body_low)
        out.append(
            ir.While(
                types=record.before[low:],
                condition=condition,
                body=body,
                location=item.location,
                word="while",
            )
        )
        return low, False

    def lower_match(
        self, item: Match, record: Record, tail: bool, out: list[ir.Instruction]
    ) -> tuple[int, bool]:
        scrutinee = record.before[-1]
        below = record.before[:-1]
        height = len(below)
        low = height
        ended = True
        if scrutinee.name in self.lowerer.checker.variants:
            arms: list[tuple[str, ir.Body]] = []
            for arm in item.arms:
                assert isinstance(arm.pattern, ConstructorPattern)
                variant = arm.pattern.name
                fields = self.lowerer.checker.variant_fields(variant, scrutinee)
                body, arm_low, arm_ended = self.lower_body(
                    arm.body, height + len(fields), tail
                )
                arms.append((variant, body))
                low = min(low, arm_low)
                ended = ended and arm_ended
            out.append(
                ir.Match(
                    type=scrutinee,
                    inputs=below[low:],
                    outputs=record.after[low:],
                    arms=tuple(arms),
                    location=item.location,
                    word="match",
                )
            )
            return low, ended

        literal_arms: list[tuple[ir.Value, ir.Body]] = []
        for arm in item.arms:
            assert not isinstance(arm.pattern, ConstructorPattern)
            literal = arm.pattern.literal
            if isinstance(literal, ConstantReference):
                literal = self.lowerer.checker.constants[literal.name].value
            assert not isinstance(literal, ArrayLiteral)
            body, arm_low, arm_ended = self.lower_body(arm.body, height, tail)
            literal_arms.append((literal.value, body))
            low = min(low, arm_low)
            ended = ended and arm_ended
        else_body: ir.Body | None = None
        if item.else_body is not None:
            else_body, else_low, else_ended = self.lower_body(
                item.else_body, height, tail
            )
            low = min(low, else_low)
            ended = ended and else_ended
        out.append(
            ir.LiteralMatch(
                type=scrutinee,
                inputs=below[low:],
                outputs=record.after[low:],
                arms=tuple(literal_arms),
                else_body=else_body,
                location=item.location,
                word="match",
            )
        )
        return low, ended

    def lower_propagate(
        self, item: Propagate, record: Record, out: list[ir.Instruction]
    ) -> tuple[int, bool]:
        location = item.location
        operand = record.before[-1]
        dropped = record.before[self.kept : -1]
        result = self.function.outputs[-1]

        drops: list[ir.Instruction] = [
            ir.StackOp(op="drop", type=t, location=location, word="?")
            for t in reversed(dropped)
        ]
        drops.extend(self.drop_slots(location))
        if operand.name == "Result":
            slot = self.new_slot("?", operand.args[1])
            exit_arm: ir.Body = (
                ir.LocalOp(op="set", slot=slot, location=location, word="?"),
                *drops,
                ir.LocalOp(op="take", slot=slot, location=location, word="?"),
                ir.Make(type=result, variant="err", location=location, word="?"),
                ir.Return(location=location, word="?"),
            )
            arms: tuple[tuple[str, ir.Body], ...] = (("ok", ()), ("err", exit_arm))
        else:
            exit_arm = (
                *drops,
                ir.Make(type=result, variant="none", location=location, word="?"),
                ir.Return(location=location, word="?"),
            )
            arms = (("none", exit_arm), ("some", ()))
        out.append(
            ir.Match(
                type=operand,
                inputs=dropped,
                outputs=(*dropped, record.after[-1]),
                arms=arms,
                location=location,
                word="?",
            )
        )
        return self.kept, False

    def lower_word(
        self,
        item: WordReference,
        record: Record,
        tail: bool,
        out: list[ir.Instruction],
    ) -> tuple[int, bool]:
        name = item.name
        spelled = written(name)
        location = item.location
        height = len(record.before)
        words = self.lowerer.checker.words
        if name in words:
            word = words[name]
            taken = len(word.effect.inputs)
            if word.kind == "constructor":
                out.append(
                    ir.Make(
                        type=record.after[-1],
                        variant=name,
                        location=location,
                        word=spelled,
                    )
                )
                return height - taken, False
            if word.kind == "unpack":
                out.append(
                    ir.Unpack(
                        type=record.before[-1],
                        variant=name.removesuffix("-unpack"),
                        location=location,
                        word=spelled,
                    )
                )
                return height - 1, False
            function = self.callee(item, record, word)
            low = height - len(function.inputs)
            if tail:
                out.extend(self.drop_slots(location))
            out.append(
                ir.Call(
                    function=function.name, tail=tail, location=location, word=spelled
                )
            )
            return low, tail

        base, _, modifier = name.partition(".")
        if base in CONVERSIONS:
            source = record.before[-1]
            target = CONVERSIONS[base]
            out.append(
                ir.Primitive(
                    type=source,
                    op=f"to_{target.name}",
                    modifier=modifier,
                    location=location,
                    word=name,
                )
            )
            return height - 1, False
        if name in BIT_CASTS:
            source = record.before[-1]
            op = (
                "to_bits" if name == "to-bits" else f"from_bits_{record.after[-1].name}"
            )
            out.append(ir.Primitive(type=source, op=op, location=location, word=name))
            return height - 1, False
        if name in STACK_WORDS:
            stack_word = STACK_WORDS[name]
            typed: Type | None = None
            if name in ("dup", "drop"):
                typed = record.before[-1]
            elif name == "over":
                typed = record.before[-2]
            out.append(ir.StackOp(op=name, type=typed, location=location, word=name))
            return height - stack_word.arguments, False
        return self.lower_builtin(item, record, tail, out)

    def callee(
        self, item: WordReference, record: Record, word: WordInfo
    ) -> ir.Function:
        declared = word.effect
        before, after = record.before, record.after
        taken = before[len(before) - len(declared.inputs) :]
        left = after[len(after) - len(declared.outputs) :]
        below_in = below_out = 0
        if declared.rows is not None:
            lengths = row_lengths(declared, taken, left)
            below_in, below_out = lengths[declared.rows[0]], lengths[declared.rows[1]]
        inputs = before[len(before) - len(taken) - below_in :]
        outputs = after[len(after) - len(left) - below_out :]

        quotations: dict[str, Type] = {}
        quotation_arguments(declared.inputs, taken, quotations)
        quotation_arguments(declared.outputs, left, quotations)
        assert record.variables is not None
        arguments: list[ir.TypeArgument] = [
            quotations.get(variable, record.variables[variable])
            for variable in declared.variables()
        ]
        if declared.rows is not None:
            arguments.append(inputs[:below_in])
            if declared.rows[1] != declared.rows[0]:
                arguments.append(outputs[:below_out])
        return self.lowerer.request(
            word, inputs, outputs, tuple(arguments), below_in, item.location
        )

    def lower_builtin(
        self,
        item: WordReference,
        record: Record,
        tail: bool,
        out: list[ir.Instruction],
    ) -> tuple[int, bool]:
        name = item.name
        location = item.location
        base, _, modifier = name.partition(".")
        before, after = record.before, record.after
        height = len(before)

        def emit(instruction: ir.Instruction) -> None:
            out.append(_located(instruction, location, name))

        if base == "call":
            quotation = before[-1]
            assert quotation.effect is not None
            effect = quotation.effect
            if tail:
                out.extend(self.drop_slots(location))
            emit(
                ir.CallQuotation(
                    inputs=effect.inputs, outputs=effect.outputs, tail=tail
                )
            )
            return height - 1 - len(effect.inputs), tail

        if base == "arraybuf-consume":
            quotation, buffer = before[-1], before[-2]
            assert quotation.effect is not None
            row = quotation.effect.inputs[:-1]
            call = ir.Runtime(name="arraybuf_consume", arguments=(buffer.args[0], row))
            emit(self.lowerer.declare(call, (*row, buffer, quotation), row))
            return height - 2 - len(row), False

        builtin = BUILTINS[name]
        taken = builtin.arguments
        operands = before[height - taken :]
        given = len(builtin.effect.outputs)
        results = after[len(after) - given :] if given else ()
        low = height - taken

        if base in ARITHMETIC:
            emit(ir.Primitive(type=operands[0], op=ARITHMETIC[base], modifier=modifier))
        elif base in COMPARISONS:
            for instruction in self.lowerer.compare(base, operands[0]):
                emit(instruction)
        elif base in PRIMITIVE_WORDS:
            emit(
                ir.Primitive(type=operands[0], op=underscored(base), modifier=modifier)
            )
        elif base in BOOLEAN_WORDS:
            emit(ir.Primitive(type=BOOL, op=base))
        elif base == "select":
            emit(ir.Primitive(type=operands[1], op="select"))
        elif base == "char-utf8-len":
            emit(ir.Primitive(type=CHAR, op="utf8_len"))
        elif base == "unit":
            emit(ir.Const(type=UNIT, value=None))
        elif base in PLAIN_RUNTIME_WORDS:
            call = ir.Runtime(name=underscored(base), modifier=modifier)
            emit(self.lowerer.declare(call, operands, results))
        else:
            if base == "print":
                arguments: tuple[ir.TypeArgument, ...] = (operands[0],)
            elif base in ("curry", "compose"):
                arguments = tuple(operands)
            elif base == "arraybuf-new":
                arguments = (results[0].args[0],)
            else:
                assert base.startswith(("array-", "arraybuf-")), name
                arguments = (operands[0].args[0],)
            call = ir.Runtime(
                name=underscored(base), modifier=modifier, arguments=arguments
            )
            emit(self.lowerer.declare(call, operands, results))
        return low, False


def _located(
    instruction: ir.Instruction, location: Location | None, word: str
) -> ir.Instruction:
    return replace(instruction, location=location, word=word)


def lower(program: Program) -> ir.Module:
    return Lowerer(program).compile()
