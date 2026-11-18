from __future__ import annotations

import copy
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TextIO

from inde.ast import (
    ArrayLiteral,
    Body,
    BooleanLiteral,
    CharacterLiteral,
    ConstantReference,
    ConstructorPattern,
    FloatLiteral,
    If,
    Import,
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
from inde.builtins import BIT_CASTS, BUILTINS, CONVERSIONS, STACK_WORDS, StackWord
from inde.checker import Checker, ConstantInfo, WordInfo, error_order
from inde.errors import IndeError, IndeErrors, IndeTrap
from inde.lexer import Location, quote, tokenize
from inde.modules import Loader
from inde.numeric import (
    FLOAT_UNARY,
    Failure,
    bit_not,
    bitwise,
    compare,
    copy_sign,
    count_bits,
    float_arithmetic,
    float_bits,
    float_from_bits,
    float_to_float,
    float_to_integer,
    float_unary,
    format_float,
    fused_multiply_add,
    integer_arithmetic,
    integer_to_float,
    integer_to_integer,
    integer_unary,
    shift,
)
from inde.parser import Parser
from inde.type_model import (
    BOOL,
    CHAR,
    FLOAT_TYPES,
    STR,
    STRBUF,
    U8,
    U32,
    UNIT,
    Type,
    array_of,
    from_number_type,
    result_of,
)
from inde.values import (
    Data,
    Quoted,
    Value,
    array,
    boolean,
    character,
    none,
    quoted,
    some,
    string,
)

MAX_CALL_DEPTH = 100_000
MAX_LENGTH = 2**32 - 1


UNIT_VALUE = Value(UNIT, Data("unit"))


def plain(value: Value) -> object:
    data = value.data
    if isinstance(data, Data):
        return (data.constructor, tuple(plain(field) for field in data.fields))
    if isinstance(data, tuple):
        return tuple(plain(element) for element in data)
    return data


def display(value: Value) -> str:
    if value.type == UNIT:
        return "unit"
    data = value.data
    if isinstance(data, bool):
        return "true" if data else "false"
    if isinstance(data, float):
        return format_float(data, value.type)
    if isinstance(data, (int, str)):
        return str(data)
    raise AssertionError(f"`{value.type}` has no display text")


def inspect(value: Value, nested: bool = False) -> str:
    data = value.data
    if isinstance(data, Quoted):
        return "[ ... ]" if nested else str(value.type)
    if isinstance(data, list):
        elements = " ".join(["{", *(inspect(element, True) for element in data), "}"])
        return f"{elements} arraybuf-from-array"
    if isinstance(data, Data):
        parts = [inspect(field, True) for field in data.fields] + [data.constructor]
        return " ".join(parts)
    if isinstance(data, tuple):
        return " ".join(["{", *(inspect(element, True) for element in data), "}"])
    if isinstance(data, bool):
        return display(value)
    if isinstance(data, str):
        if value.type == STRBUF:
            delim = '"'
            return f"{quote(data, delim)} strbuf-from-str"
        return quote(data, "'" if value.type.name == "char" else '"')
    suffix = "" if value.type.name in ("i32", "f32") else value.type.name
    return display(value) + suffix


def describe_operands(operands: list[Value]) -> str:
    return ", ".join(f"{operand.type} {inspect(operand)}" for operand in operands)


@dataclass
class Each:
    elements: Iterator[Value]
    parts: tuple[Body | Value, ...]


@dataclass
class Frame:
    body: Body
    locals: dict[str, Value]
    position: int = 0
    is_call: bool = False
    loop: While | None = None
    in_condition: bool = False
    push: Value | None = None
    base: int = 0
    each: Each | None = None

    def finished(self) -> bool:
        return self.position == len(self.body)


class Interpreter:
    def __init__(
        self,
        words: dict[str, WordInfo],
        constants: dict[str, ConstantInfo],
        output: TextIO | None = None,
    ) -> None:
        self.words = words
        self.constants = constants
        self.output = output or sys.stdout
        self.deepest = 0

    def execute(self, body: Body, stack: list[Value]) -> None:
        frames = [Frame(body, {}, is_call=True, base=len(stack))]
        calls = 1
        self.deepest = max(self.deepest, calls)
        while frames:
            frame = frames[-1]
            if frame.finished():
                if frame.each is not None and self.next_element(
                    frame.each, frames, stack
                ):
                    continue
                frames.pop()
                if frame.is_call:
                    calls -= 1
                if frame.push is not None:
                    stack.append(frame.push)
                if frame.loop is not None:
                    self.continue_loop(frame, frames, stack)
                continue
            item = frame.body[frame.position]
            frame.position += 1

            if isinstance(item, If):
                condition = stack.pop()
                branch = item.then_body if condition.data else item.else_body
                if branch:
                    frames.append(Frame(branch, frame.locals))
                continue
            if isinstance(item, While):
                frames.append(
                    Frame(item.condition, frame.locals, loop=item, in_condition=True)
                )
                continue
            if isinstance(item, Match):
                frames.append(Frame(self.select_arm(item, stack), frame.locals))
                continue
            if isinstance(item, Let):
                count = len(item.names)
                values = stack[len(stack) - count :]
                del stack[len(stack) - count :]
                inner = {**frame.locals, **dict(zip(item.names, values, strict=True))}
                frames.append(Frame(item.body, inner))
                continue
            if isinstance(item, WordReference) and item.name in frame.locals:
                stack.append(frame.locals[item.name])
                continue

            if isinstance(item, Quotation):
                stack.append(quoted((item.body,) if item.body else ()))
                continue
            if isinstance(item, Propagate):
                if self.propagate(stack.pop(), stack, frames):
                    calls -= 1
                continue

            if isinstance(item, WordReference) and item.name == "arraybuf-consume":
                quotation = stack.pop().data
                buffer = stack.pop().data
                assert isinstance(quotation, Quoted) and isinstance(buffer, list)
                if calls >= MAX_CALL_DEPTH:
                    raise IndeTrap("call depth exceeded", item.location)
                each = Each(iter(tuple(buffer)), quotation.parts)
                frames.append(Frame((), {}, is_call=True, base=len(stack), each=each))
                calls += 1
                self.deepest = max(self.deepest, calls)
                continue

            if isinstance(item, WordReference) and item.name == "call":
                data = stack.pop().data
                assert isinstance(data, Quoted)
                callee = data.parts
                base = len(stack)
            else:
                declared = self.declared_body(item)
                if declared is None:
                    self.run_item(item, stack)
                    continue
                callee = (declared,)
                assert isinstance(item, WordReference)
                base = len(stack) - len(self.words[item.name].effect.inputs)

            while (
                frames
                and frames[-1].finished()
                and frames[-1].loop is None
                and frames[-1].push is None
                and frames[-1].each is None
            ):
                if frames.pop().is_call:
                    calls -= 1
            if calls >= MAX_CALL_DEPTH:
                raise IndeTrap("call depth exceeded", item.location)
            frames.append(Frame((), {}, is_call=True, base=base))
            for part in reversed(callee):
                if isinstance(part, Value):
                    frames.append(Frame((), {}, push=part))
                else:
                    frames.append(Frame(part, {}))
            calls += 1
            self.deepest = max(self.deepest, calls)

    def next_element(self, each: Each, frames: list[Frame], stack: list[Value]) -> bool:
        element = next(each.elements, None)
        if element is None:
            return False
        stack.append(element)
        for part in reversed(each.parts):
            if isinstance(part, Value):
                frames.append(Frame((), {}, push=part))
            else:
                frames.append(Frame(part, {}))
        return True

    def propagate(self, value: Value, stack: list[Value], frames: list[Frame]) -> bool:
        data = value.data
        assert isinstance(data, Data)
        if data.constructor in ("ok", "some"):
            stack.append(data.fields[0])
            return False
        while not frames[-1].is_call:
            frames.pop()
        call = frames.pop()
        del stack[call.base :]
        stack.append(value)
        return True

    def continue_loop(
        self, finished: Frame, frames: list[Frame], stack: list[Value]
    ) -> None:
        loop = finished.loop
        assert loop is not None
        if not finished.in_condition:
            frames.append(
                Frame(loop.condition, finished.locals, loop=loop, in_condition=True)
            )
        elif stack.pop().data:
            frames.append(Frame(loop.body, finished.locals, loop=loop))

    def select_arm(self, item: Match, stack: list[Value]) -> Body:
        scrutinee = stack.pop()
        for arm in item.arms:
            pattern = arm.pattern
            if isinstance(pattern, ConstructorPattern):
                data = scrutinee.data
                assert isinstance(data, Data)
                if data.constructor == pattern.name:
                    stack.extend(data.fields)
                    return arm.body
            else:
                literal = pattern.literal
                if isinstance(literal, ConstantReference):
                    literal = self.constants[literal.name].value
                if self.literal_value(literal) == scrutinee:
                    return arm.body
        assert item.else_body is not None, "the checker requires an arm to match"
        return item.else_body

    def declared_body(self, item: Item) -> Body | None:
        if not isinstance(item, WordReference):
            return None
        word = self.words.get(item.name)
        if word is None or word.declaration is None:
            return None
        return word.declaration.body

    def run_item(self, item: Item, stack: list[Value]) -> None:
        if isinstance(item, ConstantReference):
            stack.append(self.literal_value(self.constants[item.name].value))
        elif isinstance(item, WordReference):
            name = item.name
            if name in self.words:
                word = self.words[name]
                if word.kind == "unpack":
                    self.unpack(stack)
                else:
                    self.construct(word, stack)
            elif name in STACK_WORDS:
                self.run_stack_word(STACK_WORDS[name], stack)
            elif name.partition(".")[0] in CONVERSIONS:
                stack.append(self.convert(name, stack.pop(), item.location))
            elif name in BIT_CASTS:
                stack.append(self.reinterpret(name, stack.pop()))
            else:
                self.run_builtin(name, stack, item.location)
        else:
            literals = (
                IntegerLiteral,
                FloatLiteral,
                CharacterLiteral,
                StringLiteral,
                BooleanLiteral,
                ArrayLiteral,
            )
            assert isinstance(item, literals), f"the checker let {item} through"
            stack.append(self.literal_value(item))

    def literal_value(self, literal: Literal) -> Value:
        if isinstance(literal, (IntegerLiteral, FloatLiteral)):
            return Value(from_number_type(literal.type), literal.value)
        if isinstance(literal, CharacterLiteral):
            return character(literal.value)
        if isinstance(literal, StringLiteral):
            return string(literal.value)
        if isinstance(literal, BooleanLiteral):
            return boolean(literal.value)
        elements = tuple(
            self.literal_value(self.constants[element.name].value)
            if isinstance(element, ConstantReference)
            else self.literal_value(element)
            for element in literal.elements
        )
        element_type = elements[0].type if elements else Type("T")
        return array(element_type, elements)

    def construct(self, constructor: WordInfo, stack: list[Value]) -> None:
        count = len(constructor.effect.inputs)
        fields = tuple(stack[len(stack) - count :])
        del stack[len(stack) - count :]
        result_type = constructor.effect.outputs[0]
        stack.append(Value(result_type, Data(constructor.name, fields)))

    def unpack(self, stack: list[Value]) -> None:
        data = stack.pop().data
        assert isinstance(data, Data)
        stack.extend(data.fields)

    def run_stack_word(self, word: StackWord, stack: list[Value]) -> None:
        taken = stack[len(stack) - word.arguments :]
        del stack[len(stack) - word.arguments :]
        stack.extend(taken[position] for position in word.result)

    def run_builtin(
        self, name: str, stack: list[Value], location: Location | None
    ) -> None:
        base, _, modifier = name.partition(".")
        count = BUILTINS[name].arguments
        operands = stack[len(stack) - count :]
        del stack[len(stack) - count :]
        if base.startswith("strbuf-"):
            stack.extend(self.strbuf_operation(base, operands))
            return
        if base in TEXT_WORDS:
            stack.extend(self.text_operation(base, operands, location))
            return
        checked = modifier == "checked"
        if base.startswith("arraybuf-"):
            stack.extend(self.buffer_operation(base, checked, operands, name, location))
            return
        if base.startswith("array-"):
            stack.extend(self.array_operation(base, checked, operands, name, location))
            return

        try:
            result = self.compute(base, "" if checked else modifier, operands)
        except Failure as failure:
            if checked:
                stack.append(none(operands[0].type))
                return
            raise IndeTrap(
                f"{failure.reason} in `{name}` ({describe_operands(operands)})",
                location,
            ) from None
        if result is not None:
            stack.append(some(result) if checked else result)

    def compute(self, base: str, mode: str, operands: list[Value]) -> Value | None:
        if base == "unit":
            return UNIT_VALUE
        if base in ("curry", "compose"):
            return self.quotation_operation(base, operands)
        first = operands[0]
        if base == "print":
            print(display(first), file=self.output)
            return None

        if base in ("==", "!=", "<", "<=", ">", ">="):
            return boolean(compare(base, plain(first), plain(operands[1])))
        if base == "select":
            return operands[1] if first.data else operands[2]
        if first.type == BOOL:
            return boolean(self.boolean_operation(base, operands))

        number_type = first.type
        if number_type in FLOAT_TYPES:
            assert isinstance(first.data, float)
            if base in FLOAT_UNARY:
                return Value(
                    number_type, float_unary(base, mode, number_type, first.data)
                )
            second = operands[1].data
            assert isinstance(second, float)
            if base == "copysign":
                return Value(number_type, copy_sign(number_type, first.data, second))
            if base == "fma":
                third = operands[2].data
                assert isinstance(third, float)
                return Value(
                    number_type,
                    fused_multiply_add(mode, number_type, first.data, second, third),
                )
            return Value(
                number_type,
                float_arithmetic(base, mode, number_type, first.data, second),
            )

        a = first.data
        assert isinstance(a, int)
        if base in ("neg", "abs"):
            return Value(number_type, integer_unary(base, mode, number_type, a))
        if base == "bit-not":
            return Value(number_type, bit_not(number_type, a))
        if base in ("count-ones", "leading-zeros", "trailing-zeros"):
            return Value(U32, count_bits(base, number_type, a))
        b = operands[1].data
        assert isinstance(b, int)
        if base in ("shl", "shr", "rotl", "rotr"):
            return Value(number_type, shift(base, mode, number_type, a, b))
        if base in ("bit-and", "bit-or", "bit-xor"):
            return Value(number_type, bitwise(base, number_type, a, b))
        return Value(number_type, integer_arithmetic(base, mode, number_type, a, b))

    def quotation_operation(self, base: str, operands: list[Value]) -> Value:
        first, second = operands
        assert isinstance(second.data, Quoted)
        if base == "curry":
            return quoted((first, *second.data.parts))
        assert isinstance(first.data, Quoted)
        return quoted(first.data.parts + second.data.parts)

    def array_operation(
        self,
        base: str,
        checked: bool,
        operands: list[Value],
        name: str,
        location: Location | None,
    ) -> list[Value]:
        elements = operands[0].data
        assert isinstance(elements, tuple)
        element_type = operands[0].type.args[0]
        if base == "array-len":
            return [Value(U32, len(elements))]
        start = operands[1].data
        assert isinstance(start, int)
        if base == "array-get":
            if start < len(elements):
                found = elements[start]
                return [some(found) if checked else found]
            if checked:
                return [none(element_type)]
            index_trap(name, f"index {start}, length {len(elements)}", location)
        assert base == "array-slice", base
        end = operands[2].data
        assert isinstance(end, int)
        if start <= end <= len(elements):
            part = array(element_type, elements[start:end])
            return [some(part) if checked else part]
        if checked:
            return [none(array_of(element_type))]
        index_trap(name, f"start {start}, end {end}, length {len(elements)}", location)

    def buffer_operation(
        self,
        base: str,
        checked: bool,
        operands: list[Value],
        name: str,
        location: Location | None,
    ) -> list[Value]:
        if base == "arraybuf-new":
            return [Value(Type("arraybuf", (Type("T"),)), [])]
        first = operands[0]
        element_type = first.type.args[0]
        if base == "arraybuf-from-array":
            elements = first.data
            assert isinstance(elements, tuple)
            return [Value(Type("arraybuf", (element_type,)), list(elements))]
        buffer = first.data
        assert isinstance(buffer, list)
        if base == "arraybuf-push":
            if len(buffer) == MAX_LENGTH:
                raise IndeTrap(f"the buffer is full in `{name}`", location)
            buffer.append(operands[1])
            return [first]
        if base == "arraybuf-pop":
            return [first, some(buffer.pop()) if buffer else none(element_type)]
        if base == "arraybuf-len":
            return [first, Value(U32, len(buffer))]
        if base == "arraybuf-snapshot":
            return [first, array(element_type, tuple(buffer))]
        if base == "arraybuf-freeze":
            return [array(element_type, tuple(buffer))]
        index = operands[1].data
        assert isinstance(index, int)
        in_range = index < len(buffer)
        if base == "arraybuf-get":
            if in_range:
                return [first, some(buffer[index]) if checked else buffer[index]]
            if checked:
                return [first, none(element_type)]
            index_trap(name, f"index {index}, length {len(buffer)}", location)
        assert base == "arraybuf-replace", base
        new = operands[2]
        result = result_of(element_type, element_type)
        if in_range:
            old = buffer[index]
            buffer[index] = new
            return [first, Value(result, Data("ok", (old,))) if checked else old]
        if checked:
            return [first, Value(result, Data("err", (new,)))]
        index_trap(name, f"index {index}, length {len(buffer)}", location)

    def text_operation(
        self, base: str, operands: list[Value], location: Location | None
    ) -> list[Value]:
        first = operands[0].data
        if base == "bytes-to-str":
            assert isinstance(first, tuple)
            raw = bytearray()
            for element in first:
                assert isinstance(element.data, int)
                raw.append(element.data)
            result = result_of(STR, U32)
            try:
                return [Value(result, Data("ok", (string(raw.decode()),)))]
            except UnicodeDecodeError as error:
                return [Value(result, Data("err", (Value(U32, error.start),)))]
        assert isinstance(first, str)
        if base == "char-utf8-len":
            return [Value(U32, len(first.encode()))]
        if base == "str-from-char":
            return [string(first)]
        encoded = first.encode()
        if base == "str-len":
            return [Value(U32, len(encoded))]
        if base == "str-to-bytes":
            return [array(U8, tuple(Value(U8, byte) for byte in encoded))]
        if base == "str-concat":
            second = operands[1].data
            assert isinstance(second, str)
            if len(encoded) + len(second.encode()) > MAX_LENGTH:
                raise IndeTrap("the string is too long in `str-concat`", location)
            return [string(first + second)]
        start = operands[1].data
        assert isinstance(start, int)
        if base == "str-char-at":
            if start < len(encoded) and is_boundary(encoded, start):
                length = utf8_length(encoded[start])
                found = encoded[start : start + length].decode()
                return [some(character(found))]
            return [none(CHAR)]
        assert base == "str-slice", base
        end = operands[2].data
        assert isinstance(end, int)
        fits = start <= end <= len(encoded)
        if fits and is_boundary(encoded, start) and is_boundary(encoded, end):
            return [some(string(encoded[start:end].decode()))]
        return [none(STR)]

    def convert(self, name: str, value: Value, location: Location | None) -> Value:
        word, _, modifier = name.partition(".")
        target = CONVERSIONS[word]
        checked = modifier == "checked"
        try:
            result = self.converted(value, target, "" if checked else modifier)
        except Failure as failure:
            if checked:
                return none(target)
            raise IndeTrap(
                f"{failure.reason} in `{name}` ({describe_operands([value])})",
                location,
            ) from None
        return some(result) if checked else result

    def reinterpret(self, name: str, value: Value) -> Value:
        target = BIT_CASTS[name][value.type]
        if name == "to-bits":
            assert isinstance(value.data, float)
            return Value(target, float_bits(value.data, value.type))
        assert isinstance(value.data, int)
        return Value(target, float_from_bits(value.data, target))

    def converted(self, value: Value, target: Type, mode: str) -> Value:
        source, data = value.type, value.data
        if source == target:
            return value
        if source == CHAR:
            assert isinstance(data, str)
            return Value(U32, ord(data))
        if target == CHAR:
            assert isinstance(data, int)
            if 0xD800 <= data <= 0xDFFF or data > 0x10FFFF:
                raise Failure("not a Unicode scalar value")
            return character(chr(data))
        if source in FLOAT_TYPES:
            assert isinstance(data, float)
            if target in FLOAT_TYPES:
                return Value(target, float_to_float(data, mode, target))
            return Value(target, float_to_integer(data, mode, target))
        assert isinstance(data, int) and not isinstance(data, bool)
        if target in FLOAT_TYPES:
            return Value(target, integer_to_float(data, target))
        return Value(target, integer_to_integer(data, mode, target))

    def strbuf_operation(self, base: str, operands: list[Value]) -> list[Value]:
        if base == "strbuf-new":
            return [Value(STRBUF, "")]
        text = operands[0].data
        assert isinstance(text, str)
        if base == "strbuf-from-str":
            return [Value(STRBUF, text)]
        if base in ("strbuf-push", "strbuf-append"):
            addition = operands[1].data
            assert isinstance(addition, str)
            return [Value(STRBUF, text + addition)]
        if base == "strbuf-len":
            return [operands[0], Value(U32, len(text.encode("utf-8")))]
        if base == "strbuf-snapshot":
            return [operands[0], string(text)]
        assert base == "strbuf-freeze", base
        return [string(text)]

    def boolean_operation(self, base: str, operands: list[Value]) -> bool:
        a = operands[0].data
        if base == "not":
            return not a
        b = operands[1].data
        if base == "and":
            return bool(a and b)
        if base == "or":
            return bool(a or b)
        assert base == "xor", base
        return a != b

    def run_main(self, error_output: TextIO | None = None) -> int:
        stack: list[Value] = []
        main = self.words["main"].declaration
        assert main is not None
        self.execute(main.body, stack)
        if not stack:
            return 0
        result = stack[0].data
        assert isinstance(result, Data)
        if result.constructor == "ok":
            return 0
        print(display(result.fields[0]), file=error_output or sys.stderr)
        return 1


TEXT_WORDS = (
    "str-len",
    "str-concat",
    "str-slice",
    "str-char-at",
    "str-to-bytes",
    "bytes-to-str",
    "char-utf8-len",
    "str-from-char",
)


def is_boundary(encoded: bytes, offset: int) -> bool:
    return offset == len(encoded) or encoded[offset] & 0xC0 != 0x80


def utf8_length(first_byte: int) -> int:
    if first_byte < 0x80:
        return 1
    if first_byte < 0xE0:
        return 2
    if first_byte < 0xF0:
        return 3
    return 4


def index_trap(name: str, details: str, location: Location | None) -> NoReturn:
    raise IndeTrap(f"index out of range in `{name}` ({details})", location)


def detached(value: Value) -> Value:
    data = value.data
    if isinstance(data, list):
        return Value(value.type, [detached(element) for element in data])
    if isinstance(data, Data):
        fields = tuple(detached(field) for field in data.fields)
        return Value(value.type, Data(data.constructor, fields))
    if isinstance(data, Quoted):
        parts = tuple(detached(p) if isinstance(p, Value) else p for p in data.parts)
        return Value(value.type, Quoted(parts))
    return value


class Session:
    def __init__(
        self,
        output: TextIO | None = None,
        checker: Checker | None = None,
        directory: Path | None = None,
        library: Path | None = None,
    ) -> None:
        self.checker = checker or Checker()
        self.interpreter = Interpreter(
            self.checker.words, self.checker.constants, output
        )
        self.stack: list[Value] = []
        self.loader = Loader(library)
        prelude = self.loader.start_session(directory or Path.cwd())
        if prelude:
            self.declare(Program(declarations=tuple(prelude)))

    def run_line(self, line: str) -> None:
        try:
            parsed = Parser(tokenize(line), self.known_types()).parse_repl_line()
        except IndeError as error:
            raise IndeErrors([error]) from None
        if isinstance(parsed, Import):
            declarations = self.loader.session_import(parsed)
            self.declare(Program(declarations=tuple(declarations)))
        elif isinstance(parsed, tuple):
            body = self.loader.session_resolve(parsed)
            assert isinstance(body, tuple)
            self.run_body(body)
        else:
            declaration = self.loader.session_resolve(parsed)
            assert not isinstance(declaration, tuple)
            self.declare(Program(declarations=(declaration,)))
            self.loader.session_declared(parsed)

    def known_types(self) -> dict[str, int]:
        known = {name: info.parameters for name, info in self.checker.types.items()}
        known.update(self.loader.session_types())
        return known

    def is_incomplete(self, text: str) -> bool:
        try:
            Parser(tokenize(text), self.known_types()).parse_repl_line()
        except IndeError as error:
            if error.message == "unterminated block comment":
                return True
            if error.location is None:
                return False
            return error.location.cursor >= len(text.rstrip())
        return False

    def declare(self, program: Program) -> None:
        trial = type(self.checker)()
        for name, table in vars(self.checker).items():
            setattr(trial, name, copy.copy(table))
        trial.errors.clear()
        trial.check_program(program)
        if trial.errors:
            raise IndeErrors(sorted(trial.errors, key=error_order))
        self.checker = trial
        self.interpreter.words = trial.words
        self.interpreter.constants = trial.constants

    def run_body(self, body: Body) -> None:
        types: list[Type] = [value.type for value in self.stack]
        self.checker.errors.clear()
        with self.checker.recover():
            self.checker.check_line(body, types)
        if self.checker.errors:
            errors = list(self.checker.errors)
            self.checker.errors.clear()
            raise IndeErrors(errors)

        stack = [detached(value) for value in self.stack]
        self.interpreter.execute(body, stack)
        self.stack = [
            Value(checked_type, value.data)
            for checked_type, value in zip(types, stack, strict=True)
        ]

    def show_stack(self) -> str:
        if not self.stack:
            return "stack: (empty)"
        return "stack: " + " | ".join(inspect(value) for value in self.stack)
