from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from inde import ir
from inde.builtins import conversion_forms
from inde.numeric import integer_range, round_to
from inde.type_model import (
    BOOL,
    CHAR,
    F32,
    F64,
    FLOAT_TYPES,
    INTEGER_TYPES,
    LINEAR,
    NUMBER_TYPES,
    SIGNED_INTEGER_TYPES,
    STR,
    STRBUF,
    U8,
    U32,
    U64,
    UNIT,
    UNRESTRICTED,
    Effect,
    Type,
    array_of,
    option_of,
    quotation_of,
    result_of,
)

SUPPORTED_RUNTIME = 1

LIBRARY_ARITY = {"Option": 1, "Result": 2, "Pair": 2}

INTEGER_ARITHMETIC = ("", "wrap", "sat", "checked")
FLOAT_FORMS = ("", "ieee", "checked")
COMPARISONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
DISPLAYABLE = (*NUMBER_TYPES, BOOL, CHAR, STR, UNIT)
LITERAL_MATCHED = (*INTEGER_TYPES, CHAR, STR, BOOL)
SCALARS = frozenset(t.name for t in (*NUMBER_TYPES, BOOL, CHAR, STR, UNIT, STRBUF))
STACK_OPS = frozenset({"dup", "over", "drop", "swap", "rot"})
LOCAL_OPS = frozenset({"set", "get", "take", "drop"})

STRING_CALLS: dict[str, tuple[tuple[Type, ...], tuple[Type, ...]]] = {
    "str_len": ((STR,), (U32,)),
    "str_concat": ((STR, STR), (STR,)),
    "str_slice": ((STR, U32, U32), (option_of(STR),)),
    "str_char_at": ((STR, U32), (option_of(CHAR),)),
    "str_to_bytes": ((STR,), (array_of(U8),)),
    "bytes_to_str": ((array_of(U8),), (result_of(STR, U32),)),
    "str_from_char": ((CHAR,), (STR,)),
    "strbuf_new": ((), (STRBUF,)),
    "strbuf_push": ((STRBUF, CHAR), (STRBUF,)),
    "strbuf_append": ((STRBUF, STR), (STRBUF,)),
    "strbuf_len": ((STRBUF,), (STRBUF, U32)),
    "strbuf_snapshot": ((STRBUF,), (STRBUF, STR)),
    "strbuf_freeze": ((STRBUF,), (STR,)),
    "strbuf_from_str": ((STR,), (STRBUF,)),
    **{f"str_{op}": ((STR, STR), (BOOL,)) for op in COMPARISONS},
}
UNRESTRICTED_BUFFER_CALLS = frozenset(
    {"arraybuf_get", "arraybuf_snapshot", "arraybuf_freeze", "arraybuf_from_array"}
)


class IRError(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


class Invalid(Exception):
    pass


@dataclass
class State:
    stack: list[Type]
    slots: dict[str, bool]
    reachable: bool = True


@dataclass
class Context:
    function: ir.Function
    slot_types: dict[str, Type] = field(default_factory=dict)


def library_variants(
    name: str, args: tuple[Type, ...]
) -> tuple[tuple[str, tuple[Type, ...]], ...]:
    if name == "Option":
        return (("none", ()), ("some", (args[0],)))
    if name == "Result":
        return (("ok", (args[0],)), ("err", (args[1],)))
    return (("pair", (args[0], args[1])),)


def text(types: tuple[Type, ...] | list[Type]) -> str:
    return "(" + " ".join(ir.ir_type(t) for t in types) + ")"


def buffer_of(element: Type) -> Type:
    return Type("arraybuf", (element,))


class Verifier:
    def __init__(self, module: ir.Module) -> None:
        self.module = module
        self.problems: list[str] = []
        self.types: dict[Type, ir.TypeEntry] = {}
        self.functions: dict[str, ir.Function] = {}
        self.data: dict[str, ir.DataEntry] = {}
        self.runtime: dict[str, ir.RuntimeDeclaration] = {}

    def run(self) -> None:
        module = self.module
        if module.runtime_version > SUPPORTED_RUNTIME:
            self.problems.append(
                f"the module needs runtime {module.runtime_version}, but this "
                f"implementation provides runtime {SUPPORTED_RUNTIME}"
            )
        for entry in module.types:
            if entry.type in self.types:
                self.problems.append(f"type {ir.ir_type(entry.type)} is listed twice")
            self.types[entry.type] = entry
        for entry in module.types:
            self.guard(f"type {ir.ir_type(entry.type)}", self.type_entry, entry)
        for data in module.data:
            if data.name in self.data:
                self.problems.append(f"data {data.name} is listed twice")
            self.data[data.name] = data
            self.guard(f"data {data.name}", self.data_entry, data)
        for declaration in module.runtime:
            if declaration.name in self.runtime:
                self.problems.append(f"{declaration.name} is imported twice")
            self.runtime[declaration.name] = declaration
            self.guard(f"import {declaration.name}", self.runtime_entry, declaration)
        for function in module.functions:
            if function.name in self.functions:
                self.problems.append(f"function {function.name} is defined twice")
            self.functions[function.name] = function
        for function in module.functions:
            self.guard(f"in {function.name}", self.function, function)
        self.guard("entry", self.entry)
        if self.problems:
            raise IRError(self.problems)

    def guard(self, where: str, check: Callable[..., None], *arguments: object) -> None:
        try:
            check(*arguments)
        except Invalid as problem:
            self.problems.append(f"{where}: {problem}")

    def known(self, checked_type: Type) -> None:
        effect = checked_type.effect
        if effect is not None:
            if checked_type.args[0] not in (LINEAR, UNRESTRICTED) or effect.rows:
                raise Invalid(f"{checked_type} is not a quotation type")
            for part in effect.inputs + effect.outputs:
                self.known(part)
            return
        if checked_type.name in SCALARS:
            if checked_type.args:
                raise Invalid(f"{checked_type.name} takes no type arguments")
            return
        if checked_type.name in ("array", "arraybuf"):
            if len(checked_type.args) != 1:
                raise Invalid(f"{checked_type.name} takes one type argument")
            self.known(checked_type.args[0])
            if checked_type.name == "array" and self.linear(checked_type.args[0]):
                raise Invalid(
                    f"{ir.ir_type(checked_type)} holds linear elements, but the "
                    "elements of an array must be unrestricted"
                )
            return
        if checked_type not in self.types:
            raise Invalid(f"{ir.ir_type(checked_type)} is not in the type table")

    def linear(self, checked_type: Type) -> bool:
        if checked_type.effect is not None:
            return checked_type.args[0] == LINEAR
        if checked_type.name in ("strbuf", "arraybuf"):
            return True
        if checked_type.name in SCALARS or checked_type.name == "array":
            return False
        entry = self.types.get(checked_type)
        if entry is None:
            raise Invalid(f"{ir.ir_type(checked_type)} is not in the type table")
        return entry.linear

    def unrestricted(self, checked_type: Type, what: str) -> None:
        if self.linear(checked_type):
            raise Invalid(
                f"{what} needs an unrestricted type, but "
                f"{ir.ir_type(checked_type)} is linear"
            )

    def type_entry(self, entry: ir.TypeEntry) -> None:
        declared = entry.type
        if (
            declared.effect is not None
            or declared.name in SCALARS
            or declared.name
            in (
                "array",
                "arraybuf",
            )
        ):
            raise Invalid("only declared types have entries")
        names = [name for name, _ in entry.variants]
        if not names:
            raise Invalid("a type needs at least one variant")
        if len(set(names)) != len(names):
            raise Invalid("a variant is listed twice")
        for _, fields in entry.variants:
            for field_type in fields:
                self.known(field_type)
        holds_linear = any(
            self.linear(field_type)
            for _, fields in entry.variants
            for field_type in fields
        )
        if holds_linear and not entry.linear:
            raise Invalid("it holds a linear value, so it must be linear")
        if declared.name in LIBRARY_ARITY:
            if len(declared.args) != LIBRARY_ARITY[declared.name]:
                raise Invalid(
                    f"{declared.name} takes {LIBRARY_ARITY[declared.name]} "
                    "type arguments"
                )
            if entry.variants != library_variants(declared.name, declared.args):
                raise Invalid(
                    f"it does not have the variants of the core {declared.name}"
                )
            if entry.linear and not holds_linear:
                raise Invalid(
                    f"the core {declared.name} is linear only when it holds a "
                    "linear value"
                )

    def data_entry(self, data: ir.DataEntry) -> None:
        if data.type == STR:
            if not isinstance(data.value, str):
                raise Invalid("utf8 data must be text")
            return
        if data.type.name != "array":
            raise Invalid(f"static data cannot be {ir.ir_type(data.type)}")
        self.known(data.type)
        self.elements(data.value, data.type.args[0])

    def elements(self, value: object, element: Type) -> None:
        if not isinstance(value, tuple):
            raise Invalid("an array's elements must be a sequence")
        for item in value:
            if element.name == "array":
                self.elements(item, element.args[0])
            elif element == STR:
                if not isinstance(item, str):
                    raise Invalid(f"{item!r} is not a str")
            else:
                self.constant(element, item)

    def constant(self, value_type: Type, value: object) -> None:
        if value_type in INTEGER_TYPES:
            low, high = integer_range(value_type)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not low <= value <= high
            ):
                raise Invalid(f"{value!r} is not a {value_type}")
        elif value_type in FLOAT_TYPES:
            if not isinstance(value, float) or not math.isfinite(value):
                raise Invalid(f"{value!r} is not a finite {value_type}")
            if value_type == F32 and round_to(F32, value) != value:
                raise Invalid(f"{value!r} is not exactly an f32")
        elif value_type == BOOL:
            if not isinstance(value, bool):
                raise Invalid(f"{value!r} is not a bool")
        elif value_type == CHAR:
            if (
                not isinstance(value, str)
                or len(value) != 1
                or 0xD800 <= ord(value) <= 0xDFFF
            ):
                raise Invalid(f"{value!r} is not a Unicode scalar value")
        elif value_type == UNIT:
            if value is not None:
                raise Invalid("unit has no value to write")
        else:
            raise Invalid(f"there are no {ir.ir_type(value_type)} constants")

    def runtime_entry(self, declaration: ir.RuntimeDeclaration) -> None:
        for checked_type in declaration.inputs + declaration.outputs:
            self.known(checked_type)

    def runtime_signature(
        self, call: ir.Runtime
    ) -> tuple[tuple[Type, ...], tuple[Type, ...]]:
        name, modifier, arguments = call.name, call.modifier, call.arguments
        if name in STRING_CALLS:
            if modifier or arguments:
                raise Invalid(f"{call.full_name} is not in the runtime interface")
            return STRING_CALLS[name]
        types = [a for a in arguments if not isinstance(a, tuple)]
        if name == "print":
            if modifier or len(arguments) != 1 or len(types) != 1:
                raise Invalid(f"{call.full_name} is not in the runtime interface")
            if types[0] not in DISPLAYABLE:
                raise Invalid(f"{ir.ir_type(types[0])} cannot be displayed")
            return (types[0],), ()
        if name == "curry":
            return self.curry(call, types)
        if name == "compose":
            return self.compose(call, types)
        if name == "arraybuf_consume":
            return self.consume(call)
        if len(arguments) != 1 or len(types) != 1:
            raise Invalid(f"{call.full_name} is not in the runtime interface")
        element = types[0]
        self.known(element)
        array, buffer = array_of(element), buffer_of(element)
        signatures: dict[tuple[str, str], tuple[tuple[Type, ...], tuple[Type, ...]]] = {
            ("array_len", ""): ((array,), (U32,)),
            ("array_get", ""): ((array, U32), (element,)),
            ("array_get", "checked"): ((array, U32), (option_of(element),)),
            ("array_slice", ""): ((array, U32, U32), (array,)),
            ("array_slice", "checked"): ((array, U32, U32), (option_of(array),)),
            ("arraybuf_new", ""): ((), (buffer,)),
            ("arraybuf_push", ""): ((buffer, element), (buffer,)),
            ("arraybuf_pop", ""): ((buffer,), (buffer, option_of(element))),
            ("arraybuf_len", ""): ((buffer,), (buffer, U32)),
            ("arraybuf_get", ""): ((buffer, U32), (buffer, element)),
            ("arraybuf_get", "checked"): ((buffer, U32), (buffer, option_of(element))),
            ("arraybuf_replace", ""): ((buffer, U32, element), (buffer, element)),
            ("arraybuf_replace", "checked"): (
                (buffer, U32, element),
                (buffer, result_of(element, element)),
            ),
            ("arraybuf_snapshot", ""): ((buffer,), (buffer, array)),
            ("arraybuf_freeze", ""): ((buffer,), (array,)),
            ("arraybuf_from_array", ""): ((array,), (buffer,)),
        }
        if (name, modifier) not in signatures:
            raise Invalid(f"{call.full_name} is not in the runtime interface")
        if name.startswith("array_") or name in UNRESTRICTED_BUFFER_CALLS:
            self.unrestricted(element, call.full_name)
        return signatures[(name, modifier)]

    def quotation(self, checked_type: Type, call: ir.Runtime) -> Effect:
        if checked_type.effect is None:
            raise Invalid(
                f"{call.full_name} needs a quotation type, found "
                f"{ir.ir_type(checked_type)}"
            )
        self.known(checked_type)
        return checked_type.effect

    def curry(
        self, call: ir.Runtime, types: list[Type]
    ) -> tuple[tuple[Type, ...], tuple[Type, ...]]:
        if call.modifier or len(call.arguments) != 2 or len(types) != 2:
            raise Invalid(f"{call.full_name} is not in the runtime interface")
        value, quotation = types
        self.known(value)
        effect = self.quotation(quotation, call)
        if effect.inputs:
            if effect.inputs[-1] != value:
                raise Invalid(
                    f"{call.full_name} gives the quotation a {ir.ir_type(value)}, "
                    f"but it takes {text(effect.inputs)}"
                )
            curried = Effect(effect.inputs[:-1], effect.outputs)
        else:
            curried = Effect((), (value, *effect.outputs))
        linear = self.linear(value) or self.linear(quotation)
        result = quotation_of(curried, LINEAR if linear else UNRESTRICTED)
        return (value, quotation), (result,)

    def compose(
        self, call: ir.Runtime, types: list[Type]
    ) -> tuple[tuple[Type, ...], tuple[Type, ...]]:
        if call.modifier or len(call.arguments) != 2 or len(types) != 2:
            raise Invalid(f"{call.full_name} is not in the runtime interface")
        first, second = types
        before, after = self.quotation(first, call), self.quotation(second, call)
        left, needed = before.outputs, after.inputs
        if len(left) >= len(needed):
            if left[len(left) - len(needed) :] != needed:
                raise Invalid(
                    f"{call.full_name}: the first quotation leaves {text(left)}, "
                    f"but the second takes {text(needed)}"
                )
            composed = Effect(
                before.inputs, left[: len(left) - len(needed)] + after.outputs
            )
        else:
            if needed[len(needed) - len(left) :] != left:
                raise Invalid(
                    f"{call.full_name}: the first quotation leaves {text(left)}, "
                    f"but the second takes {text(needed)}"
                )
            extra = needed[: len(needed) - len(left)]
            composed = Effect(extra + before.inputs, after.outputs)
        linear = self.linear(first) or self.linear(second)
        result = quotation_of(composed, LINEAR if linear else UNRESTRICTED)
        return (first, second), (result,)

    def consume(self, call: ir.Runtime) -> tuple[tuple[Type, ...], tuple[Type, ...]]:
        arguments = call.arguments
        if (
            call.modifier
            or len(arguments) != 2
            or isinstance(arguments[0], tuple)
            or not isinstance(arguments[1], tuple)
        ):
            raise Invalid(f"{call.full_name} is not in the runtime interface")
        element, row = arguments[0], arguments[1]
        self.known(element)
        for checked_type in row:
            self.known(checked_type)
        quotation = quotation_of(Effect((*row, element), row), UNRESTRICTED)
        return (*row, buffer_of(element), quotation), row

    def function(self, function: ir.Function) -> None:
        context = Context(function)
        for checked_type in function.inputs + function.outputs:
            self.known(checked_type)
        for slot, slot_type in function.slots:
            if slot in context.slot_types:
                raise Invalid(f"slot {slot} is declared twice")
            self.known(slot_type)
            context.slot_types[slot] = slot_type
        state = State(
            list(function.inputs), {slot: False for slot in context.slot_types}
        )
        self.body(function.body, state, 0, context)
        if state.reachable:
            if tuple(state.stack) != function.outputs:
                raise Invalid(
                    f"the function ends with {text(state.stack)}, but its results "
                    f"are {text(function.outputs)}"
                )
            self.empty_slots(state, "the end of the function")

    def entry(self) -> None:
        name, kind = self.module.entry
        function = self.functions.get(name)
        if function is None:
            raise Invalid(f"there is no function {name}")
        if function.inputs:
            raise Invalid(f"{name} must take nothing")
        if kind == "unit":
            if function.outputs:
                raise Invalid(f"{name} must leave nothing for the unit form")
        elif kind == "result":
            outputs = function.outputs
            if (
                len(outputs) != 1
                or outputs[0].name != "Result"
                or outputs[0].args[0] != UNIT
                or outputs[0].args[1] not in DISPLAYABLE
            ):
                raise Invalid(
                    f"{name} must leave a %Result<unit, E> with a displayable E "
                    "for the result form"
                )
        else:
            raise Invalid(f"unknown entry form {kind}")

    def empty_slots(self, state: State, where: str) -> None:
        full = sorted(slot for slot, is_full in state.slots.items() if is_full)
        if full:
            raise Invalid(f"{', '.join(full)} must be empty at {where}")

    def body(self, body: ir.Body, state: State, floor: int, context: Context) -> None:
        for instruction in body:
            if not state.reachable:
                raise Invalid(f"`{instruction.text()}` can never run")
            try:
                self.instruction(instruction, state, floor, context)
            except Invalid as problem:
                message = str(problem)
                if message.startswith("`"):
                    raise
                where = instruction.text()
                if instruction.location is not None:
                    location = instruction.location
                    where += f" (at {location.line}:{location.column})"
                raise Invalid(f"`{where}`: {message}") from None

    def take(self, state: State, floor: int, expected: tuple[Type, ...]) -> None:
        count = len(expected)
        available = len(state.stack) - floor
        if available < count:
            found = text(state.stack[floor:])
            raise Invalid(
                f"needs {text(expected)}, but its block has only {found}"
                if floor
                else f"needs {text(expected)}, but the stack holds only {found}"
            )
        top = tuple(state.stack[len(state.stack) - count :]) if count else ()
        if top != expected:
            raise Invalid(f"needs {text(expected)}, found {text(top)}")
        del state.stack[len(state.stack) - count :]

    def apply(
        self,
        state: State,
        floor: int,
        inputs: tuple[Type, ...],
        outputs: tuple[Type, ...],
    ) -> None:
        self.take(state, floor, inputs)
        state.stack.extend(outputs)

    def instruction(
        self, instruction: ir.Instruction, state: State, floor: int, context: Context
    ) -> None:
        if isinstance(instruction, ir.Const):
            self.known(instruction.type)
            self.constant(instruction.type, instruction.value)
            state.stack.append(instruction.type)
        elif isinstance(instruction, ir.DataRef):
            data = self.data.get(instruction.name)
            if data is None:
                raise Invalid(f"there is no data {instruction.name}")
            kind = "str" if data.type == STR else "array"
            if instruction.kind != kind:
                raise Invalid(f"{instruction.name} is {kind} data")
            state.stack.append(data.type)
        elif isinstance(instruction, ir.FuncRef):
            target = self.functions.get(instruction.function)
            if target is None:
                raise Invalid(f"there is no function {instruction.function}")
            state.stack.append(
                quotation_of(Effect(target.inputs, target.outputs), UNRESTRICTED)
            )
        elif isinstance(instruction, ir.StackOp):
            self.stack_op(instruction, state, floor)
        elif isinstance(instruction, ir.LocalOp):
            self.local_op(instruction, state, floor, context)
        elif isinstance(instruction, ir.Primitive):
            inputs, outputs = self.primitive(instruction)
            self.apply(state, floor, inputs, outputs)
        elif isinstance(instruction, ir.Runtime):
            declared = self.runtime.get(instruction.full_name)
            if declared is None:
                raise Invalid("this runtime call is not imported")
            inputs, outputs = self.runtime_signature(instruction)
            if (declared.inputs, declared.outputs) != (inputs, outputs):
                raise Invalid(
                    f"it is imported as {text(declared.inputs)} -> "
                    f"{text(declared.outputs)}, but the runtime interface gives "
                    f"{text(inputs)} -> {text(outputs)}"
                )
            self.apply(state, floor, inputs, outputs)
        elif isinstance(instruction, ir.Call):
            target = self.functions.get(instruction.function)
            if target is None:
                raise Invalid(f"there is no function {instruction.function}")
            self.call(
                state, floor, target.inputs, target.outputs, instruction.tail, context
            )
        elif isinstance(instruction, ir.CallQuotation):
            quotation = state.stack[-1] if len(state.stack) > floor else None
            effect = None if quotation is None else quotation.effect
            if effect is None or (effect.inputs, effect.outputs) != (
                instruction.inputs,
                instruction.outputs,
            ):
                found = "nothing" if quotation is None else ir.ir_type(quotation)
                raise Invalid(f"needs a quotation of this effect, found {found}")
            for checked_type in instruction.inputs + instruction.outputs:
                self.known(checked_type)
            state.stack.pop()
            self.call(
                state,
                floor,
                instruction.inputs,
                instruction.outputs,
                instruction.tail,
                context,
            )
        elif isinstance(instruction, ir.Make):
            self.apply(
                state,
                floor,
                self.fields(instruction.type, instruction.variant),
                (instruction.type,),
            )
        elif isinstance(instruction, ir.Unpack):
            entry = self.entry_of(instruction.type)
            if len(entry.variants) != 1:
                raise Invalid("only a type with one variant can be unpacked")
            fields = self.fields(instruction.type, instruction.variant)
            self.apply(state, floor, (instruction.type,), fields)
        elif isinstance(instruction, ir.Return):
            outputs = context.function.outputs
            if tuple(state.stack) != outputs:
                raise Invalid(
                    f"the stack holds {text(state.stack)}, but the results are "
                    f"{text(outputs)}"
                )
            self.empty_slots(state, "a return")
            state.reachable = False
        elif isinstance(instruction, ir.If):
            self.take(state, floor, (BOOL,))
            if (
                instruction.else_body is None
                and instruction.inputs != instruction.outputs
            ):
                raise Invalid("without else, an if must leave what it takes")
            arms = [((), instruction.then_body), ((), instruction.else_body or ())]
            self.block(
                state, floor, instruction.inputs, instruction.outputs, arms, context
            )
        elif isinstance(instruction, ir.While):
            self.loop(instruction, state, floor, context)
        elif isinstance(instruction, ir.Match):
            self.match(instruction, state, floor, context)
        elif isinstance(instruction, ir.LiteralMatch):
            self.literal_match(instruction, state, floor, context)
        else:
            raise Invalid("unknown instruction")

    def stack_op(self, instruction: ir.StackOp, state: State, floor: int) -> None:
        op, typed = instruction.op, instruction.type
        if op not in STACK_OPS:
            raise Invalid("unknown stack instruction")
        if (op in ("swap", "rot")) != (typed is None):
            raise Invalid("dup, over and drop name a type, and swap and rot do not")
        if typed is not None:
            self.known(typed)
            self.unrestricted(typed, op)
        stack = state.stack
        if op == "dup":
            assert typed is not None
            self.apply(state, floor, (typed,), (typed, typed))
        elif op == "drop":
            assert typed is not None
            self.apply(state, floor, (typed,), ())
        else:
            depth = {"over": 2, "swap": 2, "rot": 3}[op]
            if len(stack) - floor < depth:
                raise Invalid(f"needs {depth} values, but its block has fewer")
            if op == "over":
                if stack[-2] != typed:
                    raise Invalid(f"copies the second value, a {ir.ir_type(stack[-2])}")
                stack.append(stack[-2])
            elif op == "swap":
                stack[-1], stack[-2] = stack[-2], stack[-1]
            else:
                stack[-3:] = [stack[-2], stack[-1], stack[-3]]

    def local_op(
        self, instruction: ir.LocalOp, state: State, floor: int, context: Context
    ) -> None:
        op, slot = instruction.op, instruction.slot
        if op not in LOCAL_OPS:
            raise Invalid("unknown local instruction")
        if slot not in context.slot_types:
            raise Invalid(f"there is no slot {slot}")
        slot_type = context.slot_types[slot]
        full = state.slots[slot]
        if op == "set":
            if full:
                raise Invalid("the slot is already full")
            self.take(state, floor, (slot_type,))
            state.slots[slot] = True
            return
        if not full:
            raise Invalid("the slot is empty")
        if op in ("get", "drop"):
            self.unrestricted(slot_type, f"local.{op}")
        if op in ("get", "take"):
            state.stack.append(slot_type)
        if op in ("take", "drop"):
            state.slots[slot] = False

    def call(
        self,
        state: State,
        floor: int,
        inputs: tuple[Type, ...],
        outputs: tuple[Type, ...],
        tail: bool,
        context: Context,
    ) -> None:
        if not tail:
            self.apply(state, floor, inputs, outputs)
            return
        count = len(inputs)
        if (
            len(state.stack) < count
            or tuple(state.stack[len(state.stack) - count :]) != inputs
        ):
            raise Invalid(f"needs {text(inputs)}, found {text(state.stack)}")
        below = tuple(state.stack[: len(state.stack) - count])
        results = context.function.outputs
        if below + outputs != results:
            raise Invalid(
                f"a tail call leaves {text(below + outputs)}, but the results are "
                f"{text(results)}"
            )
        self.empty_slots(state, "a tail call")
        state.reachable = False

    def entry_of(self, declared: Type) -> ir.TypeEntry:
        entry = self.types.get(declared)
        if entry is None:
            raise Invalid(f"{ir.ir_type(declared)} is not in the type table")
        return entry

    def fields(self, declared: Type, variant: str) -> tuple[Type, ...]:
        for name, fields in self.entry_of(declared).variants:
            if name == variant:
                return fields
        raise Invalid(f"{ir.ir_type(declared)} has no variant {variant}")

    def block(
        self,
        state: State,
        floor: int,
        inputs: tuple[Type, ...],
        outputs: tuple[Type, ...],
        arms: Sequence[tuple[tuple[Type, ...], ir.Body]],
        context: Context,
    ) -> None:
        for checked_type in inputs + outputs:
            self.known(checked_type)
        self.take(state, floor, inputs)
        below = list(state.stack)
        ends: list[dict[str, bool]] = []
        for fields, body in arms:
            arm = State([*below, *inputs, *fields], dict(state.slots))
            self.body(body, arm, len(below), context)
            if not arm.reachable:
                continue
            if arm.stack != [*below, *outputs]:
                raise Invalid(
                    f"a branch leaves {text(arm.stack[len(below) :])}, but the block "
                    f"leaves {text(outputs)}"
                )
            ends.append(arm.slots)
        if not ends:
            state.reachable = False
            return
        if any(slots != ends[0] for slots in ends[1:]):
            raise Invalid("its branches leave the slots in different states")
        state.slots = ends[0]
        state.stack.extend(outputs)

    def loop(
        self, instruction: ir.While, state: State, floor: int, context: Context
    ) -> None:
        types = instruction.types
        for checked_type in types:
            self.known(checked_type)
        self.take(state, floor, types)
        below = list(state.stack)
        condition = State([*below, *types], dict(state.slots))
        self.body(instruction.condition, condition, len(below), context)
        if not condition.reachable or condition.stack != [*below, *types, BOOL]:
            raise Invalid(
                f"the condition must leave {text((*types, BOOL))}, but it leaves "
                f"{text(condition.stack[len(below) :])}"
            )
        if condition.slots != state.slots:
            raise Invalid("the condition changes the slots")
        body = State([*below, *types], dict(state.slots))
        self.body(instruction.body, body, len(below), context)
        if not body.reachable or body.stack != [*below, *types]:
            raise Invalid(
                f"the body must leave {text(types)}, but it leaves "
                f"{text(body.stack[len(below) :])}"
            )
        if body.slots != state.slots:
            raise Invalid("the body changes the slots")
        state.stack.extend(types)

    def match(
        self, instruction: ir.Match, state: State, floor: int, context: Context
    ) -> None:
        entry = self.entry_of(instruction.type)
        self.take(state, floor, (*instruction.inputs, instruction.type))
        state.stack.extend(instruction.inputs)
        variants = [name for name, _ in entry.variants]
        arms = [name for name, _ in instruction.arms]
        if sorted(arms) != sorted(variants) or len(set(arms)) != len(arms):
            raise Invalid(
                f"its arms are {', '.join(arms) or 'none'}, but every variant must "
                f"appear exactly once: {', '.join(variants)}"
            )
        bodies = [
            (self.fields(instruction.type, name), body)
            for name, body in instruction.arms
        ]
        self.block(
            state, floor, instruction.inputs, instruction.outputs, bodies, context
        )

    def literal_match(
        self, instruction: ir.LiteralMatch, state: State, floor: int, context: Context
    ) -> None:
        matched = instruction.type
        if matched not in LITERAL_MATCHED:
            raise Invalid(f"a {ir.ir_type(matched)} cannot be matched against literals")
        self.take(state, floor, (*instruction.inputs, matched))
        state.stack.extend(instruction.inputs)
        values = [value for value, _ in instruction.arms]
        for value in values:
            if matched == STR:
                if not isinstance(value, str):
                    raise Invalid(f"{value!r} is not a str")
            else:
                self.constant(matched, value)
        if len(set(values)) != len(values):
            raise Invalid("a value has two arms")
        both_booleans = matched == BOOL and len(set(values)) == 2
        if both_booleans and instruction.else_body is not None:
            raise Invalid("both booleans have an arm, so else can never run")
        if not both_booleans and instruction.else_body is None:
            raise Invalid("it needs an else arm for the values without one")
        bodies: list[tuple[tuple[Type, ...], ir.Body]] = [
            ((), body) for _, body in instruction.arms
        ]
        if instruction.else_body is not None:
            bodies.append(((), instruction.else_body))
        self.block(
            state, floor, instruction.inputs, instruction.outputs, bodies, context
        )

    def primitive(
        self, instruction: ir.Primitive
    ) -> tuple[tuple[Type, ...], tuple[Type, ...]]:
        typed, op, modifier = instruction.type, instruction.op, instruction.modifier
        self.known(typed)
        if op == "select":
            if modifier:
                raise Invalid("select has no modifiers")
            self.unrestricted(typed, "select")
            return (BOOL, typed, typed), (typed,)
        forms, inputs, output = self.primitive_form(typed, op)
        if modifier not in forms:
            raise Invalid(
                f"{ir.ir_type(typed)}.{op} has no .{modifier} form"
                if modifier
                else f"there is no {ir.ir_type(typed)}.{op}"
            )
        result = option_of(output) if modifier == "checked" else output
        if modifier == "checked":
            self.known(result)
        return inputs, (result,)

    def primitive_form(
        self, typed: Type, op: str
    ) -> tuple[tuple[str, ...], tuple[Type, ...], Type]:
        two, one = (typed, typed), (typed,)
        if op in COMPARISONS and typed in (*NUMBER_TYPES, BOOL, CHAR):
            return ("",), two, BOOL
        if op.startswith("to_") and op != "to_bits":
            target = Type(op.removeprefix("to_"))
            forms = conversion_forms(typed, target)
            if not forms:
                raise Invalid(f"there is no conversion from {typed} to {target.name}")
            return forms, one, target
        if typed in INTEGER_TYPES:
            if op in ("add", "sub", "mul", "div", "rem"):
                return INTEGER_ARITHMETIC, two, typed
            if op in ("neg", "abs") and typed in SIGNED_INTEGER_TYPES:
                return INTEGER_ARITHMETIC, one, typed
            if op in ("bit_and", "bit_or", "bit_xor"):
                return ("",), two, typed
            if op == "bit_not":
                return ("",), one, typed
            if op in ("shl", "shr"):
                return ("", "wrap", "checked"), (typed, U32), typed
            if op in ("rotl", "rotr"):
                return ("",), (typed, U32), typed
            if op in ("count_ones", "leading_zeros", "trailing_zeros"):
                return ("",), one, U32
            if (typed, op) in ((U32, "from_bits_f32"), (U64, "from_bits_f64")):
                return ("",), one, F32 if typed == U32 else F64
        if typed in FLOAT_TYPES:
            if op in ("add", "sub", "mul", "div", "rem", "copysign"):
                return FLOAT_FORMS, two, typed
            if op in ("neg", "abs", "sqrt", "floor", "ceil", "trunc", "round"):
                return FLOAT_FORMS, one, typed
            if op == "fma":
                return FLOAT_FORMS, (typed, typed, typed), typed
            if op == "to_bits":
                return ("",), one, U32 if typed == F32 else U64
        if typed == BOOL and op in ("and", "or", "xor"):
            return ("",), two, BOOL
        if typed == BOOL and op == "not":
            return ("",), one, BOOL
        if typed == CHAR and op == "utf8_len":
            return ("",), one, U32
        raise Invalid(f"there is no {ir.ir_type(typed)}.{op}")


def verify(module: ir.Module) -> None:
    Verifier(module).run()
