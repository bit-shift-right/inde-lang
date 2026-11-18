from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction

from inde import ir
from inde.lexer import VALID_STRING_ESCAPES, Location
from inde.numeric import round_exact
from inde.type_model import (
    BOOL,
    CHAR,
    FLOAT_TYPES,
    LINEAR,
    NUMBER_TYPES,
    STR,
    UNIT,
    UNRESTRICTED,
    Effect,
    Type,
    quotation_of,
)

SCALAR_NAMES = {
    t.name: t for t in (*NUMBER_TYPES, BOOL, CHAR, STR, UNIT, Type("strbuf"))
}
STACK_OPS = frozenset({"dup", "over", "drop"})
INTEGER = re.compile(r"-?[0-9]+")
FLOAT = re.compile(r"-?[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?")
HEX = re.compile(r"[0-9A-Fa-f]{1,6}")


class IRSyntaxError(Exception):
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"line {line}: {message}")
        self.message = message
        self.line = line


@dataclass
class Line:
    number: int
    text: str
    comment: str


def split_comment(text: str) -> tuple[str, str]:
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
        elif character == "\\" and quoted:
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif character == ";" and not quoted:
            return text[:index].strip(), text[index + 1 :].strip()
    return text.strip(), ""


class Scanner:
    def __init__(self, line: Line) -> None:
        self.line = line
        self.text = line.text
        self.position = 0

    def fail(self, message: str) -> IRSyntaxError:
        return IRSyntaxError(message, self.line.number)

    def done(self) -> bool:
        self.skip()
        return self.position == len(self.text)

    def finish(self) -> None:
        if not self.done():
            raise self.fail(f"unexpected `{self.text[self.position :]}`")

    def skip(self) -> None:
        while self.position < len(self.text) and self.text[self.position] in " \t":
            self.position += 1

    def peek(self) -> str:
        return self.text[self.position] if self.position < len(self.text) else ""

    def at(self, text: str) -> bool:
        return self.text.startswith(text, self.position)

    def accept(self, text: str) -> bool:
        if self.at(text):
            self.position += len(text)
            return True
        return False

    def expect(self, text: str) -> None:
        self.skip()
        if not self.accept(text):
            found = self.text[self.position :] or "the end of the line"
            raise self.fail(f"expected `{text}`, found `{found}`")

    def token(self) -> str:
        self.skip()
        start = self.position
        while self.position < len(self.text) and self.text[self.position] not in " \t":
            self.position += 1
        if start == self.position:
            raise self.fail("expected more on this line")
        return self.text[start : self.position]

    def identifier(self) -> str:
        start = self.position
        while self.position < len(self.text) and (
            self.text[self.position].isalnum() or self.text[self.position] in "_-:~"
        ):
            self.position += 1
        if start == self.position:
            raise self.fail(f"expected a name, found `{self.text[start:]}`")
        return self.text[start : self.position]

    def string(self) -> str:
        if not self.accept('"'):
            raise self.fail("expected a quoted string")
        characters: list[str] = []
        while True:
            if self.position >= len(self.text):
                raise self.fail("the string is not closed")
            character = self.text[self.position]
            self.position += 1
            if character == '"':
                return "".join(characters)
            if character != "\\":
                characters.append(character)
                continue
            escape = self.peek()
            self.position += 1
            if escape in VALID_STRING_ESCAPES:
                characters.append(VALID_STRING_ESCAPES[escape])
            elif escape == "u" and self.accept("{"):
                end = self.text.find("}", self.position)
                if end < 0:
                    raise self.fail("the escape `\\u{` is not closed")
                digits = self.text[self.position : end]
                if not HEX.fullmatch(digits) or int(digits, 16) > 0x10FFFF:
                    raise self.fail(f"`\\u{{{digits}}}` is not a character")
                characters.append(chr(int(digits, 16)))
                self.position = end + 1
            else:
                raise self.fail(f"unknown escape `\\{escape}`")

    def type(self) -> Type:
        self.skip()
        if self.accept("linear"):
            self.skip()
            if not self.at("["):
                return LINEAR
            return self.quotation(LINEAR)
        if self.at("["):
            return self.quotation(UNRESTRICTED)
        if self.accept("%"):
            name = self.identifier()
            return Type(name, self.type_arguments())
        name = self.identifier()
        if name == "unrestricted":
            return UNRESTRICTED
        if name in ("array", "arraybuf"):
            arguments = self.type_arguments()
            if len(arguments) != 1:
                raise self.fail(f"{name} takes one type argument")
            return Type(name, arguments)
        if name in SCALAR_NAMES:
            return SCALAR_NAMES[name]
        raise self.fail(f"unknown type `{name}`")

    def type_arguments(self) -> tuple[Type, ...]:
        if not self.accept("<"):
            return ()
        arguments = [self.type()]
        while self.accept(","):
            arguments.append(self.type())
        self.expect(">")
        return tuple(arguments)

    def argument(self) -> ir.TypeArgument:
        self.skip()
        if self.at("("):
            return self.types()
        return self.type()

    def arguments(self) -> tuple[ir.TypeArgument, ...]:
        if not self.accept("<"):
            return ()
        found = [self.argument()]
        while self.accept(","):
            found.append(self.argument())
        self.expect(">")
        return tuple(found)

    def types(self) -> tuple[Type, ...]:
        self.expect("(")
        found: list[Type] = []
        while True:
            self.skip()
            if self.accept(")"):
                return tuple(found)
            found.append(self.type())

    def effect(self) -> tuple[tuple[Type, ...], tuple[Type, ...]]:
        self.expect("[")
        inputs: list[Type] = []
        while True:
            self.skip()
            if self.accept("->"):
                break
            inputs.append(self.type())
        outputs: list[Type] = []
        while True:
            self.skip()
            if self.accept("]"):
                return tuple(inputs), tuple(outputs)
            outputs.append(self.type())

    def quotation(self, qualifier: Type) -> Type:
        inputs, outputs = self.effect()
        return quotation_of(Effect(inputs, outputs), qualifier)

    def row(self) -> tuple[Type, ...]:
        self.expect("[")
        found: list[Type] = []
        while True:
            self.skip()
            if self.accept("]"):
                return tuple(found)
            found.append(self.type())

    def function_name(self) -> str:
        self.skip()
        start = self.position
        if not self.accept("$"):
            raise self.fail("expected a function name, which starts with `$`")
        if self.at('"'):
            self.string()
        else:
            while (
                self.position < len(self.text)
                and self.text[self.position] not in ir.NAME_STOPS
            ):
                self.position += 1
        self.arguments()
        while self.at(".q") or self.at("#"):
            self.position += 2 if self.at(".q") else 1
            digits = self.position
            while self.peek().isdigit():
                self.position += 1
            if digits == self.position:
                raise self.fail("expected a number in the function name")
        if self.position == start + 1:
            raise self.fail("a function name needs more than `$`")
        return self.text[start : self.position]

    def data_name(self) -> str:
        name = self.token()
        if not name.startswith("$"):
            raise self.fail(f"expected a name starting with `$`, found `{name}`")
        return name


def value_of(text: str, value_type: Type, scanner: Scanner) -> ir.Value:
    try:
        if value_type == BOOL:
            if text not in ("true", "false"):
                raise ValueError(text)
            return text == "true"
        if value_type == CHAR:
            prefix, _, code = text.partition("+")
            if prefix != "U" or not HEX.fullmatch(code):
                raise ValueError(text)
            return chr(int(code, 16))
        if value_type in FLOAT_TYPES:
            if not FLOAT.fullmatch(text):
                raise ValueError(text)
            return round_exact(Fraction(text), value_type) if text != "-0.0" else -0.0
        if value_type in NUMBER_TYPES:
            if not INTEGER.fullmatch(text):
                raise ValueError(text)
            return int(text)
    except ValueError:
        raise scanner.fail(f"`{text}` is not a {ir.ir_type(value_type)}") from None
    raise scanner.fail(f"there are no {ir.ir_type(value_type)} constants")


class Parser:
    def __init__(self, text: str) -> None:
        self.lines: list[Line] = []
        for number, raw in enumerate(text.splitlines(), start=1):
            code, comment = split_comment(raw)
            if code:
                self.lines.append(Line(number, code, comment))
        self.index = 0

    def fail(self, message: str) -> IRSyntaxError:
        number = self.lines[self.index].number if self.index < len(self.lines) else 0
        return IRSyntaxError(message, number)

    def next(self) -> Scanner:
        if self.index >= len(self.lines):
            last = self.lines[-1].number if self.lines else 1
            raise IRSyntaxError("the module ends early", last)
        scanner = Scanner(self.lines[self.index])
        self.index += 1
        return scanner

    def peek_word(self) -> str:
        if self.index >= len(self.lines):
            return ""
        return self.lines[self.index].text.split()[0]

    def module(self) -> ir.Module:
        header = self.next()
        header.expect("module")
        header.finish()
        module = ir.Module()
        version = self.next()
        version.expect("runtime")
        number = version.token()
        if not number.isdigit():
            raise version.fail("the runtime version must be a number")
        module.runtime_version = int(number)
        version.finish()
        entry: tuple[str, str] | None = None
        while True:
            word = self.peek_word()
            if word == "end":
                closing = self.next()
                closing.expect("end")
                closing.finish()
                break
            scanner = self.next()
            if word in ("type", "linear"):
                module.types.append(self.type_entry(scanner))
            elif word == "data":
                module.data.append(self.data_entry(scanner))
            elif word == "import":
                module.runtime.append(self.import_entry(scanner))
            elif word == "func":
                module.functions.append(self.function(scanner))
            elif word == "entry":
                if entry is not None:
                    raise scanner.fail("a module has one entry")
                scanner.expect("entry")
                name = scanner.function_name()
                kind = scanner.token()
                scanner.finish()
                entry = (name, kind)
            else:
                raise scanner.fail(f"unexpected `{scanner.text}`")
        if self.index < len(self.lines):
            raise self.fail("nothing may follow the module's `end`")
        if entry is None:
            raise IRSyntaxError("the module has no entry", self.lines[-1].number)
        module.entry = entry
        return module

    def type_entry(self, scanner: Scanner) -> ir.TypeEntry:
        linear = scanner.accept("linear")
        scanner.expect("type")
        scanner.skip()
        if not scanner.at("%"):
            raise scanner.fail("a type entry names a declared type, such as `%Shape`")
        declared = scanner.type()
        scanner.expect("=")
        scanner.expect("variant")
        scanner.expect("{")
        variants: list[tuple[str, tuple[Type, ...]]] = []
        while True:
            scanner.skip()
            if scanner.accept("}"):
                break
            start = scanner.position
            while scanner.peek() not in ("", " ", "(", "}"):
                scanner.position += 1
            name = scanner.text[start : scanner.position]
            if not name:
                raise scanner.fail("expected a variant name or `}`")
            fields = scanner.types() if scanner.at("(") else ()
            variants.append((name, fields))
        scanner.finish()
        return ir.TypeEntry(declared, tuple(variants), linear)

    def data_entry(self, scanner: Scanner) -> ir.DataEntry:
        scanner.expect("data")
        name = scanner.data_name()
        scanner.expect("=")
        scanner.skip()
        if scanner.accept("utf8"):
            scanner.skip()
            text = scanner.string()
            scanner.finish()
            return ir.DataEntry(name, STR, text)
        array_type = scanner.type()
        if array_type.name != "array":
            raise scanner.fail("static data is `utf8` text or an array")
        value = self.elements(scanner, array_type.args[0])
        scanner.finish()
        return ir.DataEntry(name, array_type, value)

    def elements(self, scanner: Scanner, element: Type) -> tuple[object, ...]:
        scanner.expect("[")
        found: list[object] = []
        while True:
            scanner.skip()
            if scanner.accept("]"):
                return tuple(found)
            if element.name == "array":
                found.append(self.elements(scanner, element.args[0]))
            elif element == STR:
                found.append(scanner.string())
            else:
                start = scanner.position
                while scanner.peek() not in ("", " ", "]"):
                    scanner.position += 1
                found.append(
                    value_of(scanner.text[start : scanner.position], element, scanner)
                )

    def import_entry(self, scanner: Scanner) -> ir.RuntimeDeclaration:
        scanner.expect("import")
        call = self.runtime_call(scanner)
        inputs = scanner.types()
        scanner.expect("->")
        outputs = scanner.types()
        scanner.finish()
        return ir.RuntimeDeclaration(call.full_name, inputs, outputs)

    def runtime_call(self, scanner: Scanner) -> ir.Runtime:
        scanner.expect("rt.")
        start = scanner.position
        while scanner.peek().isalnum() or scanner.peek() == "_":
            scanner.position += 1
        name = scanner.text[start : scanner.position]
        if not name:
            raise scanner.fail("expected the name of a runtime call")
        modifier = ""
        if scanner.accept("."):
            modifier = scanner.identifier()
        return ir.Runtime(name=name, modifier=modifier, arguments=scanner.arguments())

    def function(self, scanner: Scanner) -> ir.Function:
        scanner.expect("func")
        name = scanner.function_name()
        inputs = scanner.types()
        scanner.expect("->")
        outputs = scanner.types()
        scanner.finish()
        function = ir.Function(name, inputs, outputs)
        while self.peek_word() == "local":
            local = self.next()
            local.expect("local")
            slot = local.data_name()
            function.slots.append((slot, local.type()))
            local.finish()
        function.body, _ = self.body({"end"})
        return function

    def body(self, endings: set[str]) -> tuple[ir.Body, Scanner]:
        instructions: list[ir.Instruction] = []
        while True:
            if self.index >= len(self.lines):
                raise IRSyntaxError(
                    f"expected `{'` or `'.join(sorted(endings))}` before the end",
                    self.lines[-1].number,
                )
            word = self.peek_word()
            if word in endings:
                return tuple(instructions), self.next()
            if word in ("end", "else", "do", "case"):
                raise self.fail(f"`{word}` is out of place here")
            instructions.append(self.instruction(self.next()))

    def instruction(self, scanner: Scanner) -> ir.Instruction:
        location, word = self.debug(scanner.line)
        instruction = self.plain_instruction(scanner)
        if location is None:
            return instruction
        return replace(instruction, location=location, word=word)

    def debug(self, line: Line) -> tuple[Location | None, str | None]:
        if not line.comment:
            return None, None
        comment = line.comment
        source: str | None = None
        if comment.startswith('"'):
            scanner = Scanner(Line(line.number, comment, ""))
            try:
                source = scanner.string()
            except IRSyntaxError:
                return None, None
            if not scanner.accept(":"):
                return None, None
            comment = comment[scanner.position :]
        place, _, word = comment.partition(" ")
        parts = place.split(":")
        if source is None and len(parts) > 2:
            source = ":".join(parts[:-2])
            parts = parts[-2:]
        if len(parts) != 2 or source == "":
            return None, None
        row, column = parts
        if not (row.isdigit() and column.isdigit()):
            return None, None
        return Location(int(row), int(column), 0, source), word or None

    def plain_instruction(self, scanner: Scanner) -> ir.Instruction:
        text = scanner.text
        first = text.split()[0]
        if first == "if":
            scanner.expect("if")
            inputs, outputs = scanner.effect()
            scanner.finish()
            then_body, ending = self.body({"else", "end"})
            else_body: ir.Body | None = None
            if ending.text.split()[0] == "else":
                ending.expect("else")
                ending.finish()
                else_body, ending = self.body({"end"})
            ending.expect("end")
            ending.finish()
            return ir.If(
                inputs=inputs, outputs=outputs, then_body=then_body, else_body=else_body
            )
        if first == "while":
            scanner.expect("while")
            types = scanner.row()
            scanner.finish()
            condition, ending = self.body({"do"})
            ending.expect("do")
            ending.finish()
            body, ending = self.body({"end"})
            ending.expect("end")
            ending.finish()
            return ir.While(types=types, condition=condition, body=body)
        if first == "match":
            scanner.expect("match")
            matched = scanner.type()
            inputs, outputs = scanner.effect()
            scanner.finish()
            variants, _ = self.arms(lambda case: case.token())
            return ir.Match(
                type=matched,
                inputs=inputs,
                outputs=outputs,
                arms=tuple((str(name), body) for name, body in variants),
            )
        if first.startswith("match."):
            scanner.expect("match.")
            matched = scanner.type()
            inputs, outputs = scanner.effect()
            scanner.finish()

            def literal(case: Scanner) -> ir.Value:
                case.skip()
                if matched == STR:
                    return case.string()
                return value_of(case.token(), matched, case)

            literal_arms, else_body = self.arms(literal)
            return ir.LiteralMatch(
                type=matched,
                inputs=inputs,
                outputs=outputs,
                arms=tuple(literal_arms),
                else_body=else_body,
            )
        if first in ("make", "unpack"):
            scanner.expect(first)
            declared = scanner.type()
            scanner.expect(".")
            variant = scanner.token()
            scanner.finish()
            if first == "make":
                return ir.Make(type=declared, variant=variant)
            return ir.Unpack(type=declared, variant=variant)
        if first in ("data.str", "data.array"):
            scanner.expect(first)
            name = scanner.data_name()
            scanner.finish()
            return ir.DataRef(kind=first[5:], name=name)
        if first in ("func.ref", "call", "return_call"):
            scanner.expect(first)
            name = scanner.function_name()
            scanner.finish()
            if first == "func.ref":
                return ir.FuncRef(function=name)
            return ir.Call(function=name, tail=first == "return_call")
        if first in ("call.quot", "return_call.quot"):
            scanner.expect(first)
            inputs, outputs = scanner.effect()
            scanner.finish()
            return ir.CallQuotation(
                inputs=inputs, outputs=outputs, tail=first.startswith("return")
            )
        if first == "return":
            scanner.expect("return")
            scanner.finish()
            return ir.Return()
        if first.startswith("local."):
            scanner.expect("local.")
            op = scanner.identifier()
            slot = scanner.data_name()
            scanner.finish()
            return ir.LocalOp(op=op, slot=slot)
        if first in ("swap", "rot"):
            scanner.expect(first)
            scanner.finish()
            return ir.StackOp(op=first)
        if first.startswith("rt."):
            call = self.runtime_call(scanner)
            scanner.finish()
            return call
        if first == "unit.const":
            scanner.expect(first)
            scanner.finish()
            return ir.Const(type=UNIT, value=None)
        prefix = first.partition(".")[0]
        if prefix in STACK_OPS or prefix == "select":
            scanner.expect(f"{prefix}.")
            typed = scanner.type()
            scanner.finish()
            if prefix == "select":
                return ir.Primitive(type=typed, op="select")
            return ir.StackOp(op=prefix, type=typed)
        if prefix in SCALAR_NAMES:
            return self.primitive(scanner, SCALAR_NAMES[prefix])
        raise scanner.fail(f"unknown instruction `{first}`")

    def primitive(self, scanner: Scanner, typed: Type) -> ir.Instruction:
        name = scanner.token()
        parts = name.split(".")
        if len(parts) not in (2, 3) or not parts[1]:
            raise scanner.fail(f"unknown instruction `{name}`")
        op = parts[1]
        if op == "const" and len(parts) == 2:
            value = (
                scanner.string()
                if typed == STR and scanner.at('"')
                else value_of(scanner.token(), typed, scanner)
            )
            scanner.finish()
            return ir.Const(type=typed, value=value)
        scanner.finish()
        return ir.Primitive(
            type=typed, op=op, modifier=parts[2] if len(parts) == 3 else ""
        )

    def arms(
        self, pattern: Callable[[Scanner], ir.Value]
    ) -> tuple[list[tuple[ir.Value, ir.Body]], ir.Body | None]:
        arms: list[tuple[ir.Value, ir.Body]] = []
        else_body: ir.Body | None = None
        ending = self.next()
        while True:
            head = ending.text.split()[0]
            if head == "end":
                ending.expect("end")
                ending.finish()
                return arms, else_body
            if head == "else":
                if else_body is not None:
                    raise ending.fail("a match has one `else`")
                ending.expect("else")
                ending.finish()
                else_body, ending = self.body({"end"})
                continue
            if head != "case" or else_body is not None:
                raise ending.fail(
                    f"expected `case`, `else` or `end`, found `{ending.text}`"
                )
            ending.expect("case")
            value = pattern(ending)
            ending.finish()
            body, ending = self.body({"case", "else", "end"})
            arms.append((value, body))


def parse_module(text: str) -> ir.Module:
    return Parser(text).module()
