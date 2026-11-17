from __future__ import annotations

import io
from typing import TYPE_CHECKING

from inde.ast import (
    ArrayLiteral,
    BooleanLiteral,
    CharacterLiteral,
    ConstantReference,
    FloatLiteral,
    IntegerLiteral,
    Item,
    StringLiteral,
    WordReference,
)
from inde.builtins import BIT_CASTS, BUILTINS, CONVERSIONS, STACK_WORDS, StackWord
from inde.errors import IndeTrap
from inde.lexer import Location
from inde.type_model import INTEGER_TYPES
from inde.values import Value

if TYPE_CHECKING:
    from inde.checker import ConstantInfo, WordInfo

LITERALS = (
    IntegerLiteral,
    FloatLiteral,
    CharacterLiteral,
    StringLiteral,
    BooleanLiteral,
    ArrayLiteral,
)

IMPURE = ("print", "call", "curry", "compose")
IMPURE_PREFIXES = ("strbuf-", "arraybuf-")

DIVISIONS = ("/", "/.wrap", "/.sat", "%", "%.wrap", "%.sat")


class CertainFailure(Exception):
    def __init__(self, message: str, location: Location | None) -> None:
        super().__init__(message)
        self.message = message
        self.location = location


def is_foldable(name: str) -> bool:
    if name.partition(".")[0] in CONVERSIONS or name in BIT_CASTS:
        return True
    if name not in BUILTINS or name in IMPURE:
        return False
    return not name.startswith(IMPURE_PREFIXES)


def arity(name: str) -> tuple[int, int]:
    if name.partition(".")[0] in CONVERSIONS or name in BIT_CASTS:
        return 1, 1
    effect = BUILTINS[name].effect
    return len(effect.inputs), len(effect.outputs)


class Folder:
    def __init__(
        self, words: dict[str, WordInfo], constants: dict[str, ConstantInfo]
    ) -> None:
        from inde.interpreter import Interpreter

        self.constants = constants
        self.interpreter = Interpreter(words, constants, io.StringIO())
        self.known: list[Value | None] = []

    def step(self, item: Item, local: bool = False) -> None:
        if isinstance(item, LITERALS):
            self.known.append(self.interpreter.literal_value(item))
        elif isinstance(item, ConstantReference):
            literal = self.constants[item.name].value
            self.known.append(self.interpreter.literal_value(literal))
        elif isinstance(item, WordReference) and not local and item.name in STACK_WORDS:
            self.shuffle(STACK_WORDS[item.name])
        elif isinstance(item, WordReference) and not local and is_foldable(item.name):
            self.operate(item)
        else:
            self.known.clear()

    def shuffle(self, word: StackWord) -> None:
        count = word.arguments
        if len(self.known) < count:
            self.known.clear()
            self.known.extend([None] * len(word.result))
            return
        taken = self.known[len(self.known) - count :]
        del self.known[len(self.known) - count :]
        self.known.extend(taken[position] for position in word.result)

    def operate(self, item: WordReference) -> None:
        taking, leaving = arity(item.name)
        operands = self.known[len(self.known) - taking :] if taking else []
        if len(operands) == taking and all(value is not None for value in operands):
            values = [value for value in operands if value is not None]
            del self.known[len(self.known) - taking :]
            try:
                self.interpreter.run_item(item, values)
            except IndeTrap as trap:
                raise CertainFailure(
                    f"this always traps: {trap.message}", item.location
                ) from None
            self.known.extend(values)
            return
        self.check_divisor(item)
        if len(self.known) < taking:
            self.known.clear()
        else:
            del self.known[len(self.known) - taking :]
        self.known.extend([None] * leaving)

    def check_divisor(self, item: WordReference) -> None:
        if item.name not in DIVISIONS or not self.known:
            return
        divisor = self.known[-1]
        if divisor is not None and divisor.type in INTEGER_TYPES and divisor.data == 0:
            raise CertainFailure(
                f"`{item.name}` always traps here: its divisor is zero",
                item.location,
            )
