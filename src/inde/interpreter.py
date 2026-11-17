from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import TextIO

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
    Quotation,
    StringLiteral,
    While,
    WordReference,
)
from inde.builtins import BUILTINS, STACK_WORDS, StackWord
from inde.checker import Checker, ConstantInfo, WordInfo, error_order
from inde.errors import IndeError, IndeErrors, IndeTrap
from inde.lexer import Location, quote, tokenize
from inde.numeric import (
    Failure,
    bit_not,
    bitwise,
    compare,
    count_bits,
    float_arithmetic,
    float_unary,
    format_float,
    integer_arithmetic,
    integer_unary,
    shift,
)
from inde.parser import Parser
from inde.type_model import BOOL, FLOAT_TYPES, U32, UNIT, Type, from_number_type
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
    if isinstance(data, Data):
        parts = [inspect(field, True) for field in data.fields] + [data.constructor]
        return " ".join(parts)
    if isinstance(data, tuple):
        return " ".join(["{", *(inspect(element, True) for element in data), "}"])
    if isinstance(data, bool):
        return display(value)
    if isinstance(data, str):
        return quote(data, "'" if value.type.name == "char" else '"')
    suffix = "" if value.type.name in ("i32", "f32") else value.type.name
    return display(value) + suffix


def describe_operands(operands: list[Value]) -> str:
    return ", ".join(f"{operand.type} {inspect(operand)}" for operand in operands)


@dataclass
class Frame:
    body: Body
    locals: dict[str, Value]
    position: int = 0
    is_call: bool = False
    loop: While | None = None
    in_condition: bool = False
    push: Value | None = None

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
        frames = [Frame(body, {}, is_call=True)]
        calls = 1
        self.deepest = max(self.deepest, calls)
        while frames:
            frame = frames[-1]
            if frame.finished():
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

            if isinstance(item, WordReference) and item.name == "call":
                data = stack.pop().data
                assert isinstance(data, Quoted)
                callee = data.parts
            else:
                declared = self.declared_body(item)
                if declared is None:
                    self.run_item(item, stack)
                    continue
                callee = (declared,)

            while (
                frames
                and frames[-1].finished()
                and frames[-1].loop is None
                and frames[-1].push is None
            ):
                if frames.pop().is_call:
                    calls -= 1
            if calls >= MAX_CALL_DEPTH:
                raise IndeTrap("call depth exceeded", item.location)
            frames.append(Frame((), {}, is_call=True))
            for part in reversed(callee):
                if isinstance(part, Value):
                    frames.append(Frame((), {}, push=part))
                else:
                    frames.append(Frame(part, {}))
            calls += 1
            self.deepest = max(self.deepest, calls)

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

        checked = modifier == "checked"
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
        if first.type == BOOL:
            return boolean(self.boolean_operation(base, operands))

        number_type = first.type
        if number_type in FLOAT_TYPES:
            assert isinstance(first.data, float)
            if base in ("neg", "abs", "sqrt"):
                return Value(
                    number_type, float_unary(base, mode, number_type, first.data)
                )
            second = operands[1].data
            assert isinstance(second, float)
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


class Session:
    def __init__(
        self, output: TextIO | None = None, checker: Checker | None = None
    ) -> None:
        self.checker = checker or Checker()
        self.interpreter = Interpreter(
            self.checker.words, self.checker.constants, output
        )
        self.stack: list[Value] = []

    def run_line(self, line: str) -> None:
        try:
            parsed = Parser(tokenize(line), self.known_types()).parse_repl_line()
        except IndeError as error:
            raise IndeErrors([error]) from None
        if isinstance(parsed, tuple):
            self.run_body(parsed)
        else:
            self.declare(Program(declarations=(parsed,)))

    def known_types(self) -> dict[str, int]:
        return {name: info.parameters for name, info in self.checker.types.items()}

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

        stack = list(self.stack)
        self.interpreter.execute(body, stack)
        self.stack = [
            Value(checked_type, value.data)
            for checked_type, value in zip(types, stack, strict=True)
        ]

    def show_stack(self) -> str:
        if not self.stack:
            return "stack: (empty)"
        return "stack: " + " | ".join(inspect(value) for value in self.stack)
