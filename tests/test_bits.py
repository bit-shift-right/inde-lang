import io
import math

import pytest

from inde.checker import Checker, check
from inde.errors import IndeError, IndeErrors
from inde.interpreter import Session
from inde.lexer import tokenize
from inde.numeric import (
    NaNBits,
    float_bits,
    float_from_bits,
    float_to_float,
    float_unary,
    round_to,
)
from inde.parser import Parser, parse
from inde.type_model import F32, F64, Type

CANONICAL_F32 = 0x7FC00000
CANONICAL_F64 = 0x7FF8000000000000


def shows(line: str) -> str:
    session = Session(io.StringIO(), Checker())
    session.run_line(line)
    return session.show_stack().removeprefix("stack: ")


def line_fails(line: str) -> str:
    checker = Checker()
    checker.check_program(parse(""))
    body = Parser(tokenize(line)).parse_repl_line()
    assert isinstance(body, tuple)
    stack: list[Type] = []
    with pytest.raises(IndeError) as err:
        checker.check_line(body, stack)
    return err.value.message


@pytest.mark.parametrize(
    ("value", "number_type", "bits"),
    [
        (1.0, F32, 0x3F800000),
        (-0.0, F32, 0x80000000),
        (0.0, F32, 0),
        (math.inf, F32, 0x7F800000),
        (-math.inf, F32, 0xFF800000),
        (round_to(F32, 0.1), F32, 0x3DCCCCCD),
        (1.0, F64, 0x3FF0000000000000),
        (-2.5, F64, 0xC004000000000000),
        (5e-324, F64, 1),
        (math.nan, F32, CANONICAL_F32),
        (math.nan, F64, CANONICAL_F64),
    ],
)
def test_float_bits(value: float, number_type: Type, bits: int) -> None:
    assert float_bits(value, number_type) == bits


@pytest.mark.parametrize(
    ("bits", "number_type", "value"),
    [
        (0x3F800000, F32, 1.0),
        (0x80000000, F32, -0.0),
        (1, F32, 1.401298464324817e-45),
        (0x7F800000, F32, math.inf),
        (0x4004000000000000, F64, 2.5),
        (0xFFF0000000000000, F64, -math.inf),
    ],
)
def test_float_from_bits(bits: int, number_type: Type, value: float) -> None:
    result = float_from_bits(bits, number_type)
    assert result == value
    assert math.copysign(1.0, result) == math.copysign(1.0, value)


@pytest.mark.parametrize(
    ("bits", "number_type"),
    [
        (CANONICAL_F32, F32),
        (0x7F800001, F32),
        (0x7FC12345, F32),
        (0xFFFFFFFF, F32),
        (CANONICAL_F64, F64),
        (0x7FF0000000000001, F64),
        (0xFFF8000000000123, F64),
    ],
)
def test_a_nan_keeps_its_bits(bits: int, number_type: Type) -> None:
    nan = float_from_bits(bits, number_type)
    assert isinstance(nan, NaNBits)
    assert math.isnan(nan)
    assert float_bits(nan, number_type) == bits


@pytest.mark.parametrize(
    ("bits", "number_type", "canonical"),
    [
        (CANONICAL_F32, F32, CANONICAL_F32),
        (0xFFC00000, F32, CANONICAL_F32),
        (0x7F800001, F32, CANONICAL_F32),
        (0xFFF8000000000123, F64, CANONICAL_F64),
    ],
)
def test_neg_and_abs_of_a_nan(bits: int, number_type: Type, canonical: int) -> None:
    for operation in ("neg", "abs"):
        nan = float_from_bits(bits, number_type)
        result = float_unary(operation, "", number_type, nan)
        assert float_bits(result, number_type) == canonical, operation


def test_converting_a_nan() -> None:
    nan = float_from_bits(0x7F800001, F32)
    assert float_bits(float_to_float(nan, "", F64), F64) == CANONICAL_F64
    wide = float_from_bits(0x7FF0000000000001, F64)
    assert float_bits(float_to_float(wide, "", F32), F32) == CANONICAL_F32


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("1.0 to-bits", "1065353216u32"),
        ("1.0f64 to-bits", "4607182418800017408u64"),
        ("1065353216u32 from-bits", "1.0"),
        ("4607182418800017408u64 from-bits", "1.0f64"),
        ("0x7F800000u32 from-bits", "inf"),
        ("0xFF800000u32 from-bits", "-inf"),
        ("0x7F800001u32 from-bits", "NaN"),
        ("1u64 from-bits", "5e-324f64"),
        ("0x7F800001u32 from-bits to-bits", "2139095041u32"),
        ("0x7F800001u32 let b in b from-bits to-bits end", "2139095041u32"),
        ("0x7F800001u32 from-bits dup ==", "false"),
        ("0x7F800001u32 from-bits 1.0 + to-bits", "2143289344u32"),
        ("0.0 0.0 /.ieee to-bits", "2143289344u32"),
        ("0.0 0.0 /.ieee neg to-bits", "2143289344u32"),
        ("0x7F800000u32 from-bits 1.0 +", "inf"),
        ("0x7F800001u32 from-bits to-f32 to-bits", "2139095041u32"),
        ("0xFFF0000000000001u64 from-bits to-f64 to-bits", "18442240474082181121u64"),
        ("0x7F800001u32 from-bits to-f64 to-bits", "9221120237041090560u64"),
    ],
)
def test_bits_in_programs(line: str, stack: str) -> None:
    assert shows(line) == stack


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("1 to-bits", "`to-bits` takes an `f32` or an `f64`, found `i32`"),
        ("1u8 from-bits", "`from-bits` takes a `u32` or a `u64`, found `u8`"),
        ("1.0 from-bits", "`from-bits` takes a `u32` or a `u64`, found `f32`"),
        ("to-bits", "`to-bits` takes 1 value, but the stack is empty"),
        (
            "none to-bits",
            (
                "`to-bits` reinterprets the value below it, but its type is not known "
                "at this point"
            ),
        ),
    ],
)
def test_bits_errors(line: str, message: str) -> None:
    assert line_fails(line) == message


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("1.0 to-bit", "unknown word `to-bit`; did you mean `to-bits`?"),
        ("1u32 frombits", "unknown word `frombits`; did you mean `from-bits`?"),
    ],
)
def test_bit_casts_are_suggested(line: str, message: str) -> None:
    assert line_fails(line) == message


def test_bit_cast_names_are_taken() -> None:
    with pytest.raises(IndeErrors) as err:
        check(parse("word to-bits ( -> ) end"))
    assert err.value.errors[0].message == (
        "`to-bits` is a built-in word and cannot be redefined"
    )


def test_a_generic_word_cannot_tell_the_width() -> None:
    with pytest.raises(IndeErrors) as err:
        check(parse("word t ( T -> u32 ) where T: Float to-bits end"))
    assert err.value.errors[0].message == (
        "`to-bits` takes an `f32` or an `f64`, found `T`"
    )
