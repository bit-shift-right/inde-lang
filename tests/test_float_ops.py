import io
import math
import struct
from fractions import Fraction

import pytest

from inde.checker import Checker
from inde.errors import IndeError
from inde.interpreter import Session
from inde.lexer import tokenize
from inde.numeric import (
    Failure,
    copy_sign,
    float_arithmetic,
    float_bits,
    float_from_bits,
    fused_multiply_add,
    ieee_remainder,
    round_exact,
    round_integral,
)
from inde.parser import Parser, parse
from inde.type_model import F32, F64, Type

CANONICAL_F32 = 0x7FC00000
MAX_F32 = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
SMALLEST_F32 = struct.unpack("<f", struct.pack("<I", 1))[0]


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


def same(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


@pytest.mark.parametrize(
    ("exact", "number_type", "value"),
    [
        (Fraction(1, 10), F64, 0.1),
        (Fraction(-1, 10), F64, -0.1),
        (Fraction(1, 10), F32, 0.10000000149011612),
        (Fraction(2**24 + 1), F32, 16777216.0),
        (Fraction(2**24 + 3), F32, 16777220.0),
        (Fraction(1, 2**150), F32, 0.0),
        (Fraction(3, 2**151), F32, SMALLEST_F32),
        (Fraction(MAX_F32) + Fraction(2**103), F32, math.inf),
        (Fraction(MAX_F32) + Fraction(2**103) - 1, F32, MAX_F32),
        (Fraction(2**1024), F64, math.inf),
    ],
)
def test_round_exact(exact: Fraction, number_type: Type, value: float) -> None:
    assert same(round_exact(exact, number_type), value)


@pytest.mark.parametrize(
    ("operation", "operand", "result"),
    [
        ("floor", 1.5, 1.0),
        ("floor", -1.5, -2.0),
        ("floor", -0.5, -1.0),
        ("floor", 0.5, 0.0),
        ("ceil", 1.5, 2.0),
        ("ceil", -1.5, -1.0),
        ("ceil", -0.5, -0.0),
        ("trunc", 2.7, 2.0),
        ("trunc", -2.7, -2.0),
        ("trunc", -0.5, -0.0),
        ("round", 2.5, 2.0),
        ("round", 3.5, 4.0),
        ("round", -2.5, -2.0),
        ("round", 2.4999999999999996, 2.0),
        ("round", -0.5, -0.0),
        ("round", 0.5, 0.0),
        ("floor", -0.0, -0.0),
        ("ceil", 0.0, 0.0),
        ("floor", 1e300, 1e300),
        ("round", 4503599627370497.0, 4503599627370497.0),
        ("floor", math.inf, math.inf),
        ("ceil", -math.inf, -math.inf),
    ],
)
def test_round_integral(operation: str, operand: float, result: float) -> None:
    assert same(round_integral(operation, operand), result)


@pytest.mark.parametrize("operation", ["floor", "ceil", "trunc", "round"])
def test_a_nan_rounds_to_the_canonical_nan(operation: str) -> None:
    nan = float_from_bits(0xFFC00001, F32)
    assert float_bits(round_integral(operation, nan), F32) == CANONICAL_F32


@pytest.mark.parametrize(
    ("a", "b", "result"),
    [
        (7.5, 2.0, 1.5),
        (-7.5, 2.0, -1.5),
        (7.5, -2.0, 1.5),
        (-7.5, -2.0, -1.5),
        (6.0, 2.0, 0.0),
        (-6.0, 2.0, -0.0),
        (-0.0, 1.0, -0.0),
        (1e300, 7.0, 1.0),
        (5.0, math.inf, 5.0),
        (-5.0, -math.inf, -5.0),
        (1.0, 0.0, math.nan),
        (math.inf, 2.0, math.nan),
        (math.nan, 2.0, math.nan),
        (2.0, math.nan, math.nan),
    ],
)
def test_ieee_remainder(a: float, b: float, result: float) -> None:
    assert same(ieee_remainder(a, b), result)


def test_a_remainder_that_is_nan() -> None:
    with pytest.raises(Failure) as failure:
        float_arithmetic("%", "", F32, 1.0, 0.0)
    assert failure.value.reason == "invalid float operation"
    assert math.isnan(float_arithmetic("%", "ieee", F32, 1.0, 0.0))
    assert math.isnan(float_arithmetic("%", "", F32, math.nan, 1.0))


@pytest.mark.parametrize(
    ("a", "b", "result"),
    [
        (1.0, -2.0, -1.0),
        (-1.0, 2.0, 1.0),
        (-3.0, 0.0, 3.0),
        (3.0, -0.0, -3.0),
        (0.0, -1.0, -0.0),
        (math.inf, -1.0, -math.inf),
        (2.0, -math.inf, -2.0),
    ],
)
def test_copy_sign(a: float, b: float, result: float) -> None:
    assert same(copy_sign(F64, a, b), result)


@pytest.mark.parametrize(
    ("bits", "number_type", "result"),
    [
        (0x7FC00000, F32, 1.0),
        (0xFFC00000, F32, -1.0),
        (0xFF800001, F32, -1.0),
        (0x7FF8000000000000, F64, 1.0),
        (0xFFF8000000000000, F64, -1.0),
    ],
)
def test_the_sign_of_a_nan(bits: int, number_type: Type, result: float) -> None:
    sign = float_from_bits(bits, number_type)
    assert same(copy_sign(number_type, 1.0, sign), result)


def test_copying_a_sign_onto_a_nan() -> None:
    nan = float_from_bits(0x7FC00001, F32)
    assert float_bits(copy_sign(F32, nan, -1.0), F32) == CANONICAL_F32


@pytest.mark.parametrize(
    ("a", "b", "c", "number_type", "result"),
    [
        (2.0, 3.0, 4.0, F32, 10.0),
        (0.1, 10.0, -1.0, F64, 5.551115123125783e-17),
        (1 + 2**-23, 1 - 2**-23, 16777218.0, F32, 16777218.0),
        (1e308, 2.0, -1e308, F64, 1e308),
        (SMALLEST_F32, 0.5, 0.0, F32, 0.0),
        (SMALLEST_F32, 0.75, 0.0, F32, SMALLEST_F32),
        (SMALLEST_F32, -0.75, 0.0, F32, -SMALLEST_F32),
        (1.0, -1.0, 1.0, F64, 0.0),
        (-0.0, 1.0, -0.0, F64, -0.0),
        (-0.0, 1.0, 0.0, F64, 0.0),
        (0.0, 1.0, -0.0, F64, 0.0),
        (-0.0, -1.0, -0.0, F64, 0.0),
        (-(2.0**-1074), 0.5, 0.0, F64, -0.0),
        (math.inf, 2.0, 1.0, F64, math.inf),
        (math.inf, -2.0, 1.0, F64, -math.inf),
        (2.0, 3.0, -math.inf, F64, -math.inf),
        (1e308, 10.0, -math.inf, F64, -math.inf),
        (math.inf, 1.0, math.inf, F64, math.inf),
    ],
)
def test_fma(a: float, b: float, c: float, number_type: Type, result: float) -> None:
    assert same(fused_multiply_add("", number_type, a, b, c), result)


@pytest.mark.parametrize(
    ("a", "b", "c", "reason"),
    [
        (MAX_F32, 2.0, 0.0, "float overflow"),
        (MAX_F32, 2.0, -MAX_F32, None),
        (math.inf, 0.0, 1.0, "invalid float operation"),
        (math.inf, 1.0, -math.inf, "invalid float operation"),
        (1.0, 1.0, math.inf, None),
        (math.nan, 0.0, 1.0, None),
    ],
)
def test_fma_failures(a: float, b: float, c: float, reason: str | None) -> None:
    if reason is None:
        fused_multiply_add("", F32, a, b, c)
        return
    with pytest.raises(Failure) as failure:
        fused_multiply_add("", F32, a, b, c)
    assert failure.value.reason == reason
    assert not math.isfinite(fused_multiply_add("ieee", F32, a, b, c))


def test_a_nan_from_fma_is_canonical() -> None:
    nan = float_from_bits(0xFFC00001, F32)
    assert float_bits(fused_multiply_add("", F32, nan, 1.0, 1.0), F32) == CANONICAL_F32
    assert float_bits(fused_multiply_add("", F32, 1.0, 1.0, nan), F32) == CANONICAL_F32


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("2.5 round", "2.0"),
        ("-0.5 ceil", "-0.0"),
        ("-1.5f64 floor", "-2.0f64"),
        ("2.7 trunc.ieee", "2.0"),
        ("2.7 trunc.checked", "2.0 some"),
    ],
)
def test_rounding_in_programs(line: str, stack: str) -> None:
    assert shows(line) == stack


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("-7.5 2.0 %", "-1.5"),
        ("7.5f64 -2.0f64 %", "1.5f64"),
        ("1.0 0.0 %.ieee", "NaN"),
        ("1.0 0.0 %.checked", "none"),
        ("1.0 2.0 %.checked", "1.0 some"),
    ],
)
def test_remainder_in_programs(line: str, stack: str) -> None:
    assert shows(line) == stack


def test_a_remainder_that_always_traps() -> None:
    assert line_fails("1.0 0.0 %") == (
        "this always traps: invalid float operation in `%` (f32 1.0, f32 0.0)"
    )


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("1.0 -2.0 copysign", "-1.0"),
        ("1.0f64 -0.0f64 copysign", "-1.0f64"),
        ("1.0 0xFFC00000u32 from-bits copysign", "-1.0"),
        ("1.0 2.0 copysign.checked", "1.0 some"),
    ],
)
def test_copysign_in_programs(line: str, stack: str) -> None:
    assert shows(line) == stack


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("2.0 3.0 4.0 fma", "10.0"),
        ("0.1f64 10.0f64 -1.0f64 fma", "5.551115123125783e-17f64"),
        (
            "0x3F800001u32 from-bits 0x3F7FFFFEu32 from-bits 16777218.0 fma",
            "16777218.0",
        ),
        ("1e38 10.0 0.0 fma.checked", "none"),
        ("1e38 10.0 0.0 fma.ieee", "inf"),
    ],
)
def test_fma_in_programs(line: str, stack: str) -> None:
    assert shows(line) == stack


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("0.1f64 10.0f64 * -1.0f64 +", "0.0f64"),
        (
            "0x3F800001u32 from-bits 0x3F7FFFFEu32 from-bits * 16777218.0 +",
            "16777220.0",
        ),
    ],
)
def test_without_fma_the_product_is_rounded(line: str, stack: str) -> None:
    assert shows(line) == stack


def test_an_fma_that_always_traps() -> None:
    assert line_fails("1e38 10.0 0.0 fma") == (
        "this always traps: float overflow in `fma` (f32 1e+38, f32 10.0, f32 0.0)"
    )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("1 floor", "`floor` takes a float, found `i32`"),
        (
            "1.0 1.0f64 copysign",
            "`copysign` takes two floats of the same type, found `f32 f64`",
        ),
        (
            "1.0 2.0 3 fma",
            "`fma` takes three floats of the same type, found `f32 f32 i32`",
        ),
        ("1.0 2.0 fma", "`fma` takes 3 values, but the stack holds only `f32 f32`"),
        (
            "1.0 round.wrap",
            (
                "`round` has no `.wrap` form; its forms are `round`, `round.ieee`, "
                "`round.checked`"
            ),
        ),
        ("1 neg.ieee", "`neg.ieee` takes a float, found `i32`"),
        (
            "1.0 2.0 %.sat",
            "`%.sat` takes two integers of the same type, found `f32 f32`",
        ),
    ],
)
def test_float_word_errors(line: str, message: str) -> None:
    assert line_fails(line) == message


def test_fma_in_generic_code() -> None:
    session = Session(io.StringIO(), Checker())
    session.run_line(
        "word lerp ( T T T -> T ) where T: Float  let a b t in t b a - a fma end end"
    )
    session.run_line("1.0 3.0 0.25 lerp 2.0f64 4.0f64 0.5f64 lerp")
    assert session.show_stack() == "stack: 1.5 | 3.0f64"
