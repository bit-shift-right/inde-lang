import math
import struct
from collections.abc import Callable
from typing import Any

import pytest

from inde import numeric
from inde.numeric import Failure
from inde.type_model import F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, Type


def as_f32(value: float) -> float:
    return float(struct.unpack("f", struct.pack("f", value))[0])


def get_failure_reason(function: Callable[..., Any], *arguments: Any) -> str:
    with pytest.raises(Failure) as failure:
        function(*arguments)
    return str(failure.value.reason)


I32_MIN, I32_MAX = -(2**31), 2**31 - 1


@pytest.mark.parametrize(
    ("number_type", "expected"),
    [
        (I8, (-128, 127)),
        (U8, (0, 255)),
        (I16, (-32768, 32767)),
        (U16, (0, 65535)),
        (I32, (I32_MIN, I32_MAX)),
        (U32, (0, 2**32 - 1)),
        (I64, (-(2**63), 2**63 - 1)),
        (U64, (0, 2**64 - 1)),
    ],
)
def test_integer_ranges(number_type: Type, expected: tuple[int, int]) -> None:
    assert numeric.integer_range(number_type) == expected


def test_plain_addition() -> None:
    assert numeric.integer_arithmetic("+", "", I32, 1, 2) == 3
    assert numeric.integer_arithmetic("+", "", U8, 200, 55) == 255
    assert numeric.integer_arithmetic("+", "", I8, -100, -28) == -128
    assert numeric.integer_arithmetic("+", "", I32, I32_MAX, 0) == I32_MAX
    assert numeric.integer_arithmetic("+", "wrap", I32, 1, 2) == 3
    assert numeric.integer_arithmetic("+", "sat", I32, 1, 2) == 3


def test_plain_addition_overflow_fails() -> None:
    assert (
        get_failure_reason(numeric.integer_arithmetic, "+", "", I32, I32_MAX, 1)
        == "integer overflow"
    )
    get_failure_reason(numeric.integer_arithmetic, "+", "", U8, 200, 56)
    get_failure_reason(numeric.integer_arithmetic, "+", "", I8, -100, -29)


@pytest.mark.parametrize(
    ("operation", "a", "b", "expected"),
    [
        ("==", 1, 1, True),
        ("!=", 1, 1, False),
        ("<", 1, 2, True),
        ("<=", 2, 2, True),
        (">", 1, 2, False),
        (">=", 1, 2, False),
        ("<", False, True, True),
        ("<", "a", "b", True),
        ("==", "abc", "abc", True),
        ("==", math.nan, math.nan, False),
        ("!=", math.nan, math.nan, True),
        ("<", math.nan, 1.0, False),
        ("==", 0.0, -0.0, True),
    ],
)
def test_comparison(operation: str, a: object, b: object, expected: bool) -> None:
    assert numeric.compare(operation, a, b) is expected


@pytest.mark.parametrize(
    ("value", "number_type", "expected"),
    [
        (0, U8, 0),
        (255, U8, 255),
        (256, U8, 0),
        (257, U8, 1),
        (-1, U8, 255),
        (127, I8, 127),
        (128, I8, -128),
        (255, I8, -1),
        (-129, I8, 127),
        (I32_MAX + 1, I32, I32_MIN),
        (2**64, U64, 0),
        (2**63, I64, -(2**63)),
        (-(2**63) - 1, I64, 2**63 - 1),
    ],
)
def test_wrap_integer(value: int, number_type: Type, expected: int) -> None:
    assert numeric.wrap_integer(value, number_type) == expected


@pytest.mark.parametrize(
    ("value", "number_type", "expected"),
    [
        (300, U8, 255),
        (-5, U8, 0),
        (100, U8, 100),
        (200, I8, 127),
        (-200, I8, -128),
        (2**40, I32, I32_MAX),
        (-(2**40), I32, I32_MIN),
    ],
)
def test_saturate_integer(value: int, number_type: Type, expected: int) -> None:
    assert numeric.saturate_integer(value, number_type) == expected


@pytest.mark.parametrize(
    ("operation", "mode", "number_type", "a", "b", "expected"),
    [
        ("-", "", I32, 5, 7, -2),
        ("*", "", I32, 6, 7, 42),
        ("-", "", U8, 7, 7, 0),
        ("*", "", I64, 2**31, 2**31, 2**62),
        ("+", "wrap", I32, I32_MAX, 1, I32_MIN),
        ("+", "wrap", U8, 200, 100, 44),
        ("-", "wrap", U8, 0, 1, 255),
        ("*", "wrap", U8, 16, 16, 0),
        ("*", "wrap", I8, 16, 8, -128),
        ("+", "sat", U8, 200, 100, 255),
        ("-", "sat", U8, 0, 1, 0),
        ("*", "sat", I8, 16, 16, 127),
        ("*", "sat", I8, -16, 16, -128),
    ],
)
def test_integer_arithmetic(
    operation: str, mode: str, number_type: Type, a: int, b: int, expected: int
) -> None:
    assert numeric.integer_arithmetic(operation, mode, number_type, a, b) == expected


@pytest.mark.parametrize(
    ("operation", "number_type", "a", "b"),
    [
        ("-", U8, 0, 1),
        ("-", I32, I32_MIN, 1),
        ("*", I8, 16, 8),
        ("*", U64, 2**32, 2**32),
    ],
)
def test_overflow_fails(operation: str, number_type: Type, a: int, b: int) -> None:
    reason = get_failure_reason(
        numeric.integer_arithmetic, operation, "", number_type, a, b
    )
    assert reason == "integer overflow"


@pytest.mark.parametrize(
    ("a", "b", "quotient", "remainder"),
    [
        (7, 2, 3, 1),
        (-7, 2, -3, -1),
        (7, -2, -3, 1),
        (-7, -2, 3, -1),
        (6, 3, 2, 0),
        (-6, 3, -2, 0),
        (0, 5, 0, 0),
        (1, 5, 0, 1),
        (-1, 5, 0, -1),
    ],
)
def test_division_truncates_toward_zero(
    a: int, b: int, quotient: int, remainder: int
) -> None:
    assert numeric.integer_arithmetic("/", "", I32, a, b) == quotient
    assert numeric.integer_arithmetic("%", "", I32, a, b) == remainder
    assert quotient * b + remainder == a


@pytest.mark.parametrize("mode", ["", "wrap", "sat"])
@pytest.mark.parametrize("operation", ["/", "%"])
def test_division_by_zero_fails_under_every_mode(operation: str, mode: str) -> None:
    reason = get_failure_reason(numeric.integer_arithmetic, operation, mode, I32, 1, 0)
    assert reason == "division by zero"


def test_minimum_divided_by_minus_one() -> None:
    divide = numeric.integer_arithmetic
    assert get_failure_reason(divide, "/", "", I32, I32_MIN, -1) == "integer overflow"
    assert divide("/", "wrap", I32, I32_MIN, -1) == I32_MIN
    assert divide("/", "sat", I32, I32_MIN, -1) == I32_MAX
    assert divide("%", "", I32, I32_MIN, -1) == 0
    assert divide("/", "", I8, -128, 1) == -128
    assert divide("/", "", U8, 255, 2) == 127


def test_neg_and_abs() -> None:
    unary = numeric.integer_unary
    assert unary("neg", "", I32, 5) == -5
    assert unary("neg", "", I32, -5) == 5
    assert unary("abs", "", I32, -5) == 5
    assert unary("abs", "", I32, 5) == 5
    assert unary("neg", "", I8, 127) == -127
    for operation in ("neg", "abs"):
        assert get_failure_reason(unary, operation, "", I8, -128) == "integer overflow"
        assert unary(operation, "wrap", I8, -128) == -128
        assert unary(operation, "sat", I8, -128) == 127


def test_round_to() -> None:
    assert numeric.round_to(F64, 0.1) == 0.1
    assert numeric.round_to(F32, 0.1) == as_f32(0.1)
    assert numeric.round_to(F32, 0.1) != 0.1
    assert numeric.round_to(F32, 1.5) == 1.5
    assert numeric.round_to(F32, 1e39) == math.inf
    assert numeric.round_to(F32, -1e39) == -math.inf
    assert numeric.round_to(F32, math.inf) == math.inf
    assert math.isnan(numeric.round_to(F32, math.nan))
    assert numeric.round_to(F32, 16777217.0) == 16777216.0


@pytest.mark.parametrize(
    ("operation", "number_type", "a", "b", "expected"),
    [
        ("+", F64, 1.5, 2.25, 3.75),
        ("-", F64, 1.5, 2.25, -0.75),
        ("*", F64, 1.5, 2.0, 3.0),
        ("/", F64, 1.0, 4.0, 0.25),
        ("+", F64, 0.1, 0.2, 0.30000000000000004),
        ("+", F32, as_f32(0.1), as_f32(0.2), as_f32(as_f32(0.1) + as_f32(0.2))),
        ("/", F32, 1.0, 3.0, as_f32(1.0 / 3.0)),
        ("*", F32, as_f32(1.1), as_f32(1.1), as_f32(as_f32(1.1) * as_f32(1.1))),
        ("/", F64, 0.0, 5.0, 0.0),
    ],
)
def test_float_arithmetic(
    operation: str, number_type: Type, a: float, b: float, expected: float
) -> None:
    assert numeric.float_arithmetic(operation, "", number_type, a, b) == expected


@pytest.mark.parametrize(
    ("operation", "number_type", "a", "b", "reason"),
    [
        ("*", F64, 1e308, 10.0, "float overflow"),
        ("+", F64, 1.7e308, 1.7e308, "float overflow"),
        ("*", F32, as_f32(3e38), 10.0, "float overflow"),
        ("/", F64, 1.0, 0.0, "division by zero"),
        ("/", F64, -1.0, 0.0, "division by zero"),
        ("/", F64, 0.0, 0.0, "invalid float operation"),
    ],
)
def test_float_failures(
    operation: str, number_type: Type, a: float, b: float, reason: str
) -> None:
    arithmetic = numeric.float_arithmetic
    assert get_failure_reason(arithmetic, operation, "", number_type, a, b) == reason


def test_ieee_mode_returns_the_raw_result() -> None:
    ieee = numeric.float_arithmetic
    assert ieee("*", "ieee", F64, 1e308, 10.0) == math.inf
    assert ieee("*", "ieee", F32, as_f32(3e38), 10.0) == math.inf
    assert ieee("/", "ieee", F64, 1.0, 0.0) == math.inf
    assert ieee("/", "ieee", F64, -1.0, 0.0) == -math.inf
    assert ieee("/", "ieee", F64, 1.0, -0.0) == -math.inf
    assert math.isnan(ieee("/", "ieee", F64, 0.0, 0.0))
    assert math.isnan(ieee("-", "ieee", F64, math.inf, math.inf))


def test_only_a_new_infinity_or_nan_fails() -> None:
    arithmetic = numeric.float_arithmetic
    assert arithmetic("+", "", F64, math.inf, 1.0) == math.inf
    assert arithmetic("*", "", F64, -math.inf, 2.0) == -math.inf
    assert arithmetic("/", "", F64, math.inf, 0.0) == math.inf
    assert math.isnan(arithmetic("+", "", F64, math.nan, 1.0))
    reason = get_failure_reason(arithmetic, "-", "", F64, math.inf, math.inf)
    assert reason == "invalid float operation"
    get_failure_reason(arithmetic, "*", "", F64, math.inf, 0.0)


def test_nan_is_canonical() -> None:
    result = numeric.float_arithmetic("-", "ieee", F64, math.inf, math.inf)
    assert struct.pack(">d", result).hex() == "7ff8000000000000"


def test_float_unary() -> None:
    unary = numeric.float_unary
    assert unary("neg", "", F64, 1.5) == -1.5
    assert math.copysign(1.0, unary("neg", "", F64, 0.0)) == -1.0
    assert unary("abs", "", F64, -1.5) == 1.5
    assert unary("abs", "", F64, -math.inf) == math.inf
    assert unary("sqrt", "", F64, 9.0) == 3.0
    assert unary("sqrt", "", F32, 2.0) == as_f32(math.sqrt(2.0))
    assert math.copysign(1.0, unary("sqrt", "", F64, -0.0)) == -1.0
    assert unary("sqrt", "", F64, math.inf) == math.inf
    assert get_failure_reason(unary, "sqrt", "", F64, -1.0) == "invalid float operation"
    assert math.isnan(unary("sqrt", "ieee", F64, -1.0))


@pytest.mark.parametrize(
    ("operation", "number_type", "a", "b", "expected"),
    [
        ("bit-and", U8, 0xF0, 0x3C, 0x30),
        ("bit-or", U8, 0xF0, 0x3C, 0xFC),
        ("bit-xor", U8, 0xF0, 0x3C, 0xCC),
        ("bit-and", I8, -1, 0x0F, 0x0F),
        ("bit-or", I8, -128, 1, -127),
        ("bit-xor", I8, -1, -1, 0),
        ("bit-xor", I32, -1, 0, -1),
    ],
)
def test_bitwise(
    operation: str, number_type: Type, a: int, b: int, expected: int
) -> None:
    assert numeric.bitwise(operation, number_type, a, b) == expected


def test_bit_not() -> None:
    assert numeric.bit_not(U8, 0x0F) == 0xF0
    assert numeric.bit_not(U8, 0) == 255
    assert numeric.bit_not(I8, 0) == -1
    assert numeric.bit_not(I32, -1) == 0


@pytest.mark.parametrize(
    ("operation", "mode", "number_type", "a", "amount", "expected"),
    [
        ("shl", "", U8, 1, 3, 8),
        ("shl", "", U8, 0xFF, 4, 0xF0),
        ("shl", "", I8, 1, 7, -128),
        ("shr", "", U8, 0x80, 7, 1),
        ("shr", "", I8, -128, 7, -1),
        ("shr", "", I8, -1, 1, -1),
        ("shr", "", U8, 0xFF, 1, 0x7F),
        ("shl", "", U8, 1, 0, 1),
        ("shl", "wrap", U8, 1, 8, 1),
        ("shl", "wrap", U8, 1, 9, 2),
        ("shr", "wrap", U32, 8, 35, 1),
        ("rotl", "", U8, 0b1000_0001, 1, 0b0000_0011),
        ("rotr", "", U8, 0b1000_0001, 1, 0b1100_0000),
        ("rotl", "", U8, 0x12, 8, 0x12),
        ("rotl", "", U8, 0x12, 12, 0x21),
        ("rotr", "", I8, 1, 1, -128),
        ("rotl", "", U8, 0x12, 0, 0x12),
    ],
)
def test_shifts(
    operation: str, mode: str, number_type: Type, a: int, amount: int, expected: int
) -> None:
    assert numeric.shift(operation, mode, number_type, a, amount) == expected


def test_shifting_by_the_width_fails() -> None:
    for operation in ("shl", "shr"):
        reason = get_failure_reason(numeric.shift, operation, "", U8, 1, 8)
        assert reason == "shift amount out of range"
        get_failure_reason(numeric.shift, operation, "", I64, 1, 64)


@pytest.mark.parametrize(
    ("operation", "number_type", "a", "expected"),
    [
        ("count-ones", U8, 0b1011, 3),
        ("count-ones", I8, -1, 8),
        ("count-ones", U32, 0, 0),
        ("leading-zeros", U8, 1, 7),
        ("leading-zeros", U8, 0, 8),
        ("leading-zeros", I8, -1, 0),
        ("leading-zeros", U32, 1, 31),
        ("trailing-zeros", U8, 0b1000, 3),
        ("trailing-zeros", U8, 0, 8),
        ("trailing-zeros", I8, -128, 7),
        ("trailing-zeros", I64, 0, 64),
    ],
)
def test_counting_bits(
    operation: str, number_type: Type, a: int, expected: int
) -> None:
    assert numeric.count_bits(operation, number_type, a) == expected


@pytest.mark.parametrize(
    ("value", "number_type", "text"),
    [
        (1.5, F32, "1.5"),
        (1.0, F32, "1.0"),
        (100.0, F32, "100.0"),
        (-2.5, F64, "-2.5"),
        (0.0, F32, "0.0"),
        (-0.0, F32, "-0.0"),
        (as_f32(0.1), F32, "0.1"),
        (0.1, F64, "0.1"),
        (as_f32(0.1) + as_f32(0.2), F64, "0.30000000447034836"),
        (0.30000000000000004, F64, "0.30000000000000004"),
        (16777216.0, F32, "16777216.0"),
        (123456.789, F64, "123456.789"),
        (0.0001, F64, "0.0001"),
        (0.00001, F64, "1e-05"),
        (0.000015, F64, "1.5e-05"),
        (1e15, F64, "1000000000000000.0"),
        (1e16, F64, "1e+16"),
        (1.5e16, F64, "1.5e+16"),
        (-1e16, F64, "-1e+16"),
        (1e100, F64, "1e+100"),
        (as_f32(3.4028235e38), F32, "3.4028235e+38"),
        (as_f32(1e-45), F32, "1e-45"),
        (5e-324, F64, "5e-324"),
        (1.7976931348623157e308, F64, "1.7976931348623157e+308"),
        (math.inf, F64, "inf"),
        (-math.inf, F32, "-inf"),
        (math.nan, F64, "NaN"),
    ],
)
def test_format_float(value: float, number_type: Type, text: str) -> None:
    assert numeric.format_float(value, number_type) == text


def test_displayed_floats_read_back_exactly() -> None:
    for value in (0.1, 1 / 3, 2 / 3, 1e-7, 123456789.125, 9.999999e15, 6.02e23):
        assert float(numeric.format_float(value, F64)) == value
        single = as_f32(value)
        assert as_f32(float(numeric.format_float(single, F32))) == single
