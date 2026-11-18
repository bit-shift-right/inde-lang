from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TextIO

from inde import ir
from inde.errors import IndeTrap
from inde.interpreter import MAX_CALL_DEPTH, UNIT_VALUE, Interpreter, display
from inde.type_model import BOOL, CHAR, STR, UNIT, Effect, Type, quotation_of
from inde.values import Closure, Data, Value, array, boolean, character, string

SOURCE_OPERATIONS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "rem": "%",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "utf8_len": "char-utf8-len",
}


def source_word(op: str, modifier: str) -> str:
    base = SOURCE_OPERATIONS.get(op, op.replace("_", "-"))
    return f"{base}.{modifier}" if modifier else base


def runtime_word(call: ir.Runtime) -> str:
    if call.name.startswith("str_") and call.name[4:] in SOURCE_OPERATIONS:
        return SOURCE_OPERATIONS[call.name[4:]]
    return source_word(call.name, call.modifier)


@dataclass
class Cursor:
    body: ir.Body
    position: int = 0
    loop: ir.While | None = None
    in_condition: bool = False


@dataclass
class Activation:
    function: ir.Function | None
    cursors: list[Cursor] = field(default_factory=list)
    slots: dict[str, Value] = field(default_factory=dict)
    steps: list[str | Value] = field(default_factory=list)
    elements: Iterator[Value] | None = None
    each: Closure | None = None
    counted: bool = True
    origin: ir.Instruction | None = None


class Machine:
    def __init__(self, module: ir.Module, output: TextIO | None = None) -> None:
        self.module = module
        self.functions = {function.name: function for function in module.functions}
        self.data = {data.name: data for data in module.data}
        self.runtime = {d.name: d for d in module.runtime}
        self.output = output or sys.stdout
        self.words = Interpreter({}, {}, self.output)
        self.deepest = 0

    def run_main(self, error_output: TextIO | None = None) -> int:
        stack: list[Value] = []
        name, kind = self.module.entry
        self.run(name, stack)
        if kind == "unit":
            return 0
        result = stack[0].data
        assert isinstance(result, Data)
        if result.constructor == "ok":
            return 0
        print(display(result.fields[0]), file=error_output or sys.stderr)
        return 1

    def run(self, name: str, stack: list[Value]) -> None:
        activations: list[Activation] = []
        self.calls = 0
        self.enter(name, activations, None)
        while activations:
            activation = activations[-1]
            if activation.function is None:
                self.step(activation, activations, stack)
                continue
            cursor = activation.cursors[-1]
            if cursor.position == len(cursor.body):
                self.finish(cursor, activation, activations, stack)
                continue
            instruction = cursor.body[cursor.position]
            cursor.position += 1
            self.execute(instruction, activation, activations, stack)

    def enter(
        self,
        name: str,
        activations: list[Activation],
        instruction: ir.Instruction | None,
    ) -> None:
        if self.calls >= MAX_CALL_DEPTH:
            location = None if instruction is None else instruction.location
            raise IndeTrap("call depth exceeded", location)
        function = self.functions[name]
        activations.append(Activation(function, [Cursor(function.body)]))
        self.calls += 1
        self.deepest = max(self.deepest, self.calls)

    def leave(self, activations: list[Activation]) -> None:
        if activations.pop().counted:
            self.calls -= 1

    def finish(
        self,
        cursor: Cursor,
        activation: Activation,
        activations: list[Activation],
        stack: list[Value],
    ) -> None:
        loop = cursor.loop
        if loop is not None:
            if not cursor.in_condition:
                activation.cursors[-1] = Cursor(
                    loop.condition, loop=loop, in_condition=True
                )
                return
            if stack.pop().data:
                activation.cursors[-1] = Cursor(loop.body, loop=loop)
                return
        activation.cursors.pop()
        if not activation.cursors:
            self.leave(activations)

    def step(
        self, activation: Activation, activations: list[Activation], stack: list[Value]
    ) -> None:
        if activation.steps:
            part = activation.steps.pop(0)
            if isinstance(part, Value):
                stack.append(part)
                return
            origin = activation.origin
            if not activation.steps and activation.elements is None:
                self.leave(activations)
            self.enter(part, activations, origin)
            return
        if activation.elements is not None and activation.each is not None:
            element = next(activation.elements, None)
            if element is not None:
                stack.append(element)
                activation.steps = list(activation.each.parts)
                return
        self.leave(activations)

    def call_closure(
        self,
        closure: Closure,
        activations: list[Activation],
        instruction: ir.Instruction,
    ) -> None:
        if self.calls >= MAX_CALL_DEPTH:
            raise IndeTrap("call depth exceeded", instruction.location)
        activations.append(
            Activation(
                None, steps=list(closure.parts), counted=False, origin=instruction
            )
        )

    def execute(
        self,
        instruction: ir.Instruction,
        activation: Activation,
        activations: list[Activation],
        stack: list[Value],
    ) -> None:
        if isinstance(instruction, ir.Primitive):
            self.primitive(instruction, stack)
        elif isinstance(instruction, ir.Const):
            stack.append(self.constant(instruction.type, instruction.value))
        elif isinstance(instruction, ir.StackOp):
            self.stack_op(instruction.op, stack)
        elif isinstance(instruction, ir.LocalOp):
            self.local_op(instruction, activation, stack)
        elif isinstance(instruction, ir.Runtime):
            self.runtime_call(instruction, activations, stack)
        elif isinstance(instruction, ir.Call):
            if instruction.tail:
                self.leave(activations)
            self.enter(instruction.function, activations, instruction)
        elif isinstance(instruction, ir.CallQuotation):
            closure = stack.pop().data
            assert isinstance(closure, Closure)
            if instruction.tail:
                self.leave(activations)
            self.call_closure(closure, activations, instruction)
        elif isinstance(instruction, ir.Return):
            self.leave(activations)
        elif isinstance(instruction, ir.If):
            condition = stack.pop().data
            branch = instruction.then_body if condition else instruction.else_body
            if branch:
                activation.cursors.append(Cursor(branch))
        elif isinstance(instruction, ir.While):
            activation.cursors.append(
                Cursor(instruction.condition, loop=instruction, in_condition=True)
            )
        elif isinstance(instruction, ir.Match):
            scrutinee = stack.pop().data
            assert isinstance(scrutinee, Data)
            stack.extend(scrutinee.fields)
            for variant, body in instruction.arms:
                if variant == scrutinee.constructor:
                    activation.cursors.append(Cursor(body))
                    break
        elif isinstance(instruction, ir.LiteralMatch):
            scrutinee = stack.pop().data
            chosen = instruction.else_body
            for value, body in instruction.arms:
                if value == scrutinee and type(value) is type(scrutinee):
                    chosen = body
                    break
            assert chosen is not None
            activation.cursors.append(Cursor(chosen))
        elif isinstance(instruction, ir.Make):
            fields = self.fields(instruction.type, instruction.variant)
            count = len(fields)
            values = tuple(stack[len(stack) - count :]) if count else ()
            del stack[len(stack) - count :]
            stack.append(Value(instruction.type, Data(instruction.variant, values)))
        elif isinstance(instruction, ir.Unpack):
            data = stack.pop().data
            assert isinstance(data, Data)
            stack.extend(data.fields)
        elif isinstance(instruction, ir.DataRef):
            entry = self.data[instruction.name]
            stack.append(self.static(entry.value, entry.type))
        elif isinstance(instruction, ir.FuncRef):
            target = self.functions[instruction.function]
            closure_type = quotation_of(Effect(target.inputs, target.outputs))
            stack.append(Value(closure_type, Closure((instruction.function,))))
        else:
            raise TypeError(f"cannot run {instruction.text()}")

    def fields(self, declared: Type, variant: str) -> tuple[Type, ...]:
        for entry in self.module.types:
            if entry.type == declared:
                for name, fields in entry.variants:
                    if name == variant:
                        return fields
        raise AssertionError(f"no variant {variant} of {declared}")

    def constant(self, value_type: Type, value: ir.Value) -> Value:
        if value_type == UNIT:
            return UNIT_VALUE
        if value_type == BOOL:
            return boolean(bool(value))
        if value_type == CHAR:
            assert isinstance(value, str)
            return character(value)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        return Value(value_type, value)

    def static(self, value: object, value_type: Type) -> Value:
        if value_type == STR:
            assert isinstance(value, str)
            return string(value)
        assert isinstance(value, tuple)
        element = value_type.args[0]
        return array(element, tuple(self.static_element(v, element) for v in value))

    def static_element(self, value: object, element: Type) -> Value:
        if element == STR or element.name == "array":
            return self.static(value, element)
        assert isinstance(value, (int, float, str, bool)) or value is None
        return self.constant(element, value)

    def stack_op(self, op: str, stack: list[Value]) -> None:
        if op == "dup":
            stack.append(stack[-1])
        elif op == "drop":
            stack.pop()
        elif op == "over":
            stack.append(stack[-2])
        elif op == "swap":
            stack[-1], stack[-2] = stack[-2], stack[-1]
        else:
            stack[-3:] = [stack[-2], stack[-1], stack[-3]]

    def local_op(
        self, instruction: ir.LocalOp, activation: Activation, stack: list[Value]
    ) -> None:
        slots = activation.slots
        if instruction.op == "set":
            slots[instruction.slot] = stack.pop()
        elif instruction.op == "get":
            stack.append(slots[instruction.slot])
        elif instruction.op == "take":
            stack.append(slots.pop(instruction.slot))
        else:
            del slots[instruction.slot]

    def primitive(self, instruction: ir.Primitive, stack: list[Value]) -> None:
        op, modifier, location = (
            instruction.op,
            instruction.modifier,
            instruction.location,
        )
        if op.startswith("to_") and op != "to_bits":
            word = f"to-{op[3:]}" + (f".{modifier}" if modifier else "")
            stack.append(self.words.convert(word, stack.pop(), location))
        elif op == "to_bits" or op.startswith("from_bits_"):
            word = "to-bits" if op == "to_bits" else "from-bits"
            stack.append(self.words.reinterpret(word, stack.pop()))
        else:
            self.words.run_builtin(source_word(op, modifier), stack, location)

    def runtime_call(
        self,
        call: ir.Runtime,
        activations: list[Activation],
        stack: list[Value],
    ) -> None:
        if call.name == "curry":
            quotation = stack.pop()
            value = stack.pop()
            assert isinstance(quotation.data, Closure)
            result = self.runtime[call.full_name].outputs[0]
            stack.append(Value(result, Closure((value, *quotation.data.parts))))
        elif call.name == "compose":
            second = stack.pop()
            first = stack.pop()
            assert isinstance(first.data, Closure) and isinstance(second.data, Closure)
            result = self.runtime[call.full_name].outputs[0]
            parts = first.data.parts + second.data.parts
            stack.append(Value(result, Closure(parts)))
        elif call.name == "arraybuf_consume":
            closure = stack.pop().data
            buffer = stack.pop().data
            assert isinstance(closure, Closure) and isinstance(buffer, list)
            if self.calls >= MAX_CALL_DEPTH:
                raise IndeTrap("call depth exceeded", call.location)
            activations.append(
                Activation(
                    None, elements=iter(tuple(buffer)), each=closure, origin=call
                )
            )
            self.calls += 1
            self.deepest = max(self.deepest, self.calls)
        else:
            self.words.run_builtin(runtime_word(call), stack, call.location)
