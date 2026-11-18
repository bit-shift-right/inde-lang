from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from inde.lexer import Location, quote
from inde.numeric import format_float
from inde.type_model import (
    BOOL,
    CHAR,
    FLOAT_TYPES,
    INTEGER_TYPES,
    LINEAR,
    OPAQUE_LIBRARY_TYPES,
    STR,
    STRBUF,
    UNIT,
    UNRESTRICTED,
    Type,
)

TypeArgument = Type | tuple[Type, ...]

CORE_TYPE_NAMES = frozenset(
    {t.name for t in (*INTEGER_TYPES, *FLOAT_TYPES, BOOL, CHAR, STR, UNIT, STRBUF)}
    | {t.name for t in OPAQUE_LIBRARY_TYPES}
    | {"array", "arraybuf"}
)

RUNTIME_2_CALLS = frozenset(
    {
        "write",
        "eprint",
        "ewrite",
        "show",
        "show_quoted",
        "read_line",
        "read_all",
        "args",
        "env_get",
        "read_file",
        "read_file_bytes",
        "write_file",
        "append_file",
        "write_file_bytes",
        "file_open",
        "file_create",
        "file_append",
        "file_read_line",
        "file_read_all",
        "file_read_bytes",
        "file_write",
        "file_write_bytes",
        "file_close",
        "io_error_kind",
    }
)


RUNTIME_3_CALLS = frozenset({"str_parse_f64", "str_parse_f32"})


def runtime_version_of(module: Module) -> int:
    names = {declaration.name.partition("<")[0][3:] for declaration in module.runtime}
    if names & RUNTIME_3_CALLS:
        return 3
    return 2 if names & RUNTIME_2_CALLS else 1


def ir_type(checked_type: Type) -> str:
    effect = checked_type.effect
    if effect is not None:
        assert effect.rows is None, checked_type
        prefix = "linear " if checked_type.args[0] == LINEAR else ""
        return prefix + block_effect(effect.inputs, effect.outputs)
    args = checked_type.args
    if checked_type in (LINEAR, UNRESTRICTED):
        return checked_type.name
    if checked_type.name in CORE_TYPE_NAMES:
        name = checked_type.name
    else:
        name = f"%{checked_type.name}"
    if not args:
        return name
    return f"{name}<{', '.join(ir_type(arg) for arg in args)}>"


def ir_argument(argument: TypeArgument) -> str:
    if isinstance(argument, tuple):
        return "(" + " ".join(ir_type(t) for t in argument) + ")"
    return ir_type(argument)


def ir_arguments(arguments: tuple[TypeArgument, ...]) -> str:
    if not arguments:
        return ""
    return "<" + ", ".join(ir_argument(a) for a in arguments) + ">"


NAME_STOPS = frozenset(' \t<.#"')


def function_name(word: str, arguments: tuple[TypeArgument, ...] = ()) -> str:
    plain = word and not any(character in NAME_STOPS for character in word)
    return f"${word if plain else quote_text(word)}{ir_arguments(arguments)}"


def block_effect(inputs: tuple[Type, ...], outputs: tuple[Type, ...]) -> str:
    return "[" + " ".join([*map(ir_type, inputs), "->", *map(ir_type, outputs)]) + "]"


def ir_types(types: tuple[Type, ...]) -> str:
    return "(" + " ".join(ir_type(t) for t in types) + ")"


Value = int | float | bool | str | None


def quote_text(text: str) -> str:
    return quote(text, '"')


def ir_value(value: Value, value_type: Type) -> str:
    if value_type in FLOAT_TYPES and isinstance(value, float):
        return format_float(value, value_type)
    if value_type == BOOL and isinstance(value, bool):
        return "true" if value else "false"
    if value_type == CHAR and isinstance(value, str) and len(value) == 1:
        return f"U+{ord(value):04X}"
    if value_type == STR and isinstance(value, str):
        return quote(value, '"')
    return repr(value) if isinstance(value, str) else str(value)


@dataclass(frozen=True, kw_only=True)
class Instruction:
    location: Location | None = field(default=None, compare=False)
    word: str | None = field(default=None, compare=False)

    def text(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Const(Instruction):
    type: Type
    value: Value

    def text(self) -> str:
        if self.type == UNIT:
            return "unit.const"
        return f"{ir_type(self.type)}.const {ir_value(self.value, self.type)}"


@dataclass(frozen=True, kw_only=True)
class DataRef(Instruction):
    kind: str
    name: str

    def text(self) -> str:
        return f"data.{self.kind} {self.name}"


@dataclass(frozen=True, kw_only=True)
class FuncRef(Instruction):
    function: str

    def text(self) -> str:
        return f"func.ref {self.function}"


@dataclass(frozen=True, kw_only=True)
class StackOp(Instruction):
    op: str
    type: Type | None = None

    def text(self) -> str:
        if self.type is None:
            return self.op
        return f"{self.op}.{ir_type(self.type)}"


@dataclass(frozen=True, kw_only=True)
class LocalOp(Instruction):
    op: str
    slot: str

    def text(self) -> str:
        return f"local.{self.op} {self.slot}"


@dataclass(frozen=True, kw_only=True)
class Primitive(Instruction):
    type: Type
    op: str
    modifier: str = ""

    def text(self) -> str:
        if self.op == "select":
            return f"select.{ir_type(self.type)}"
        name = f"{ir_type(self.type)}.{self.op}"
        return f"{name}.{self.modifier}" if self.modifier else name


@dataclass(frozen=True, kw_only=True)
class Runtime(Instruction):
    name: str
    modifier: str = ""
    arguments: tuple[TypeArgument, ...] = ()

    @property
    def full_name(self) -> str:
        name = f"rt.{self.name}"
        if self.modifier:
            name += f".{self.modifier}"
        return name + ir_arguments(self.arguments)

    def text(self) -> str:
        return self.full_name


@dataclass(frozen=True, kw_only=True)
class Call(Instruction):
    function: str
    tail: bool = False

    def text(self) -> str:
        return f"{'return_call' if self.tail else 'call'} {self.function}"


@dataclass(frozen=True, kw_only=True)
class CallQuotation(Instruction):
    inputs: tuple[Type, ...]
    outputs: tuple[Type, ...]
    tail: bool = False

    def text(self) -> str:
        name = "return_call.quot" if self.tail else "call.quot"
        return f"{name} {block_effect(self.inputs, self.outputs)}"


@dataclass(frozen=True, kw_only=True)
class Make(Instruction):
    type: Type
    variant: str

    def text(self) -> str:
        return f"make {ir_type(self.type)}.{self.variant}"


@dataclass(frozen=True, kw_only=True)
class Unpack(Instruction):
    type: Type
    variant: str

    def text(self) -> str:
        return f"unpack {ir_type(self.type)}.{self.variant}"


@dataclass(frozen=True, kw_only=True)
class Return(Instruction):
    def text(self) -> str:
        return "return"


Body = tuple[Instruction, ...]


@dataclass(frozen=True, kw_only=True)
class If(Instruction):
    inputs: tuple[Type, ...]
    outputs: tuple[Type, ...]
    then_body: Body
    else_body: Body | None = None

    def text(self) -> str:
        return f"if {block_effect(self.inputs, self.outputs)}"


@dataclass(frozen=True, kw_only=True)
class While(Instruction):
    types: tuple[Type, ...]
    condition: Body
    body: Body

    def text(self) -> str:
        return "while [" + " ".join(ir_type(t) for t in self.types) + "]"


@dataclass(frozen=True, kw_only=True)
class Match(Instruction):
    type: Type
    inputs: tuple[Type, ...]
    outputs: tuple[Type, ...]
    arms: tuple[tuple[str, Body], ...]

    def text(self) -> str:
        effect = block_effect(self.inputs, self.outputs)
        return f"match {ir_type(self.type)} {effect}"


@dataclass(frozen=True, kw_only=True)
class LiteralMatch(Instruction):
    type: Type
    inputs: tuple[Type, ...]
    outputs: tuple[Type, ...]
    arms: tuple[tuple[Value, Body], ...]
    else_body: Body | None = None

    def text(self) -> str:
        effect = block_effect(self.inputs, self.outputs)
        return f"match.{ir_type(self.type)} {effect}"


def walk(body: Body) -> Iterator[Instruction]:
    for instruction in body:
        yield instruction
        for inner in bodies_of(instruction):
            yield from walk(inner)


def types_of(instruction: Instruction) -> list[Type]:
    if isinstance(instruction, (Const, Make, Unpack, Primitive)):
        return [instruction.type]
    if isinstance(instruction, StackOp):
        return [] if instruction.type is None else [instruction.type]
    if isinstance(instruction, Runtime):
        found: list[Type] = []
        for argument in instruction.arguments:
            found.extend(argument if isinstance(argument, tuple) else [argument])
        return found
    if isinstance(instruction, (CallQuotation, If)):
        return [*instruction.inputs, *instruction.outputs]
    if isinstance(instruction, (Match, LiteralMatch)):
        return [instruction.type, *instruction.inputs, *instruction.outputs]
    if isinstance(instruction, While):
        return list(instruction.types)
    return []


def bodies_of(instruction: Instruction) -> list[Body]:
    if isinstance(instruction, If):
        return [instruction.then_body] + (
            [] if instruction.else_body is None else [instruction.else_body]
        )
    if isinstance(instruction, While):
        return [instruction.condition, instruction.body]
    if isinstance(instruction, Match):
        return [body for _, body in instruction.arms]
    if isinstance(instruction, LiteralMatch):
        return [body for _, body in instruction.arms] + (
            [] if instruction.else_body is None else [instruction.else_body]
        )
    return []


@dataclass(frozen=True)
class TypeEntry:
    type: Type
    variants: tuple[tuple[str, tuple[Type, ...]], ...]
    linear: bool


@dataclass(frozen=True)
class DataEntry:
    name: str
    type: Type
    value: object


@dataclass(frozen=True)
class RuntimeDeclaration:
    name: str
    inputs: tuple[Type, ...]
    outputs: tuple[Type, ...]


@dataclass
class Function:
    name: str
    inputs: tuple[Type, ...]
    outputs: tuple[Type, ...]
    slots: list[tuple[str, Type]] = field(default_factory=list)
    body: Body = ()


@dataclass
class Module:
    types: list[TypeEntry] = field(default_factory=list)
    data: list[DataEntry] = field(default_factory=list)
    runtime: list[RuntimeDeclaration] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    entry: tuple[str, str] = ("$main", "unit")
    runtime_version: int = 1

    def function(self, name: str) -> Function:
        for function in self.functions:
            if function.name == name:
                return function
        raise KeyError(name)


def format_module(module: Module, debug: bool = False) -> str:
    lines = ["module", f"  runtime {module.runtime_version}"]
    if module.types:
        lines.append("")
    for entry in module.types:
        variants = " ".join(
            f"{name}({' '.join(ir_type(f) for f in fields)})" if fields else name
            for name, fields in entry.variants
        )
        qualifier = "linear " if entry.linear else ""
        lines.append(
            f"  {qualifier}type {ir_type(entry.type)} = variant {{ {variants} }}"
        )
    if module.data:
        lines.append("")
    for data in module.data:
        lines.append(f"  data {data.name} = {format_data(data)}")
    if module.runtime:
        lines.append("")
    for declaration in module.runtime:
        lines.append(
            f"  import {declaration.name} {ir_types(declaration.inputs)} -> "
            f"{ir_types(declaration.outputs)}"
        )
    for function in module.functions:
        lines.append("")
        lines.append(
            f"  func {function.name} {ir_types(function.inputs)} -> "
            f"{ir_types(function.outputs)}"
        )
        for slot, slot_type in function.slots:
            lines.append(f"    local {slot} {ir_type(slot_type)}")
        format_body(function.body, 2, lines, debug)
        lines.append("  end")
    lines.append("")
    lines.append(f"  entry {module.entry[0]} {module.entry[1]}")
    lines.append("end")
    return "\n".join(lines) + "\n"


def format_data(data: DataEntry) -> str:
    if data.type == STR:
        assert isinstance(data.value, str)
        return "utf8 " + quote(data.value, '"')
    return f"{ir_type(data.type)} {format_elements(data.value, data.type)}"


def format_elements(value: object, array_type: Type) -> str:
    element_type = array_type.args[0]
    assert isinstance(value, tuple)
    if element_type.name == "array":
        parts = [format_elements(element, element_type) for element in value]
    else:
        parts = [ir_value(element, element_type) for element in value]
    return "[" + " ".join(parts) + "]"


SOURCE_STOPS = frozenset(' \t";')


def debug_place(location: Location) -> str:
    place = f"{location.line}:{location.column}"
    source = location.source
    if source is None:
        return place
    plain = source and not any(character in SOURCE_STOPS for character in source)
    return f"{source if plain else quote_text(source)}:{place}"


def format_body(body: Body, depth: int, lines: list[str], debug: bool) -> None:
    indent = "  " * depth
    for instruction in body:
        text = indent + instruction.text()
        if debug and instruction.location is not None:
            location = instruction.location
            text += f"  ; {debug_place(location)}"
            if instruction.word is not None:
                text += f" {instruction.word}"
        lines.append(text)
        if isinstance(instruction, If):
            format_body(instruction.then_body, depth + 1, lines, debug)
            if instruction.else_body is not None:
                lines.append(indent + "else")
                format_body(instruction.else_body, depth + 1, lines, debug)
            lines.append(indent + "end")
        elif isinstance(instruction, While):
            format_body(instruction.condition, depth + 1, lines, debug)
            lines.append(indent + "do")
            format_body(instruction.body, depth + 1, lines, debug)
            lines.append(indent + "end")
        elif isinstance(instruction, Match):
            for variant, arm in instruction.arms:
                lines.append(f"{indent}  case {variant}")
                format_body(arm, depth + 2, lines, debug)
            lines.append(indent + "end")
        elif isinstance(instruction, LiteralMatch):
            for value, arm in instruction.arms:
                lines.append(f"{indent}  case {ir_value(value, instruction.type)}")
                format_body(arm, depth + 2, lines, debug)
            if instruction.else_body is not None:
                lines.append(f"{indent}  else")
                format_body(instruction.else_body, depth + 2, lines, debug)
            lines.append(indent + "end")
