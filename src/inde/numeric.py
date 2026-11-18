from __future__ import annotations

import math
import re
import struct
from fractions import Fraction
from typing import Self

from inde.type_model import F32, SIGNED_INTEGER_TYPES, Type


class Failure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def bits_of(number_type: Type) -> int:
    return int(number_type.name[1:])


def is_signed(number_type: Type) -> bool:
    return number_type in SIGNED_INTEGER_TYPES


def integer_range(number_type: Type) -> tuple[int, int]:
    bits = bits_of(number_type)
    if is_signed(number_type):
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def wrap_integer(value: int, number_type: Type) -> int:
    bits = bits_of(number_type)
    value &= (1 << bits) - 1
    if is_signed(number_type) and value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


def saturate_integer(value: int, number_type: Type) -> int:
    low, high = integer_range(number_type)
    return max(low, min(high, value))


def fit_integer(exact: int, mode: str, number_type: Type) -> int:
    low, high = integer_range(number_type)
    if low <= exact <= high:
        return exact
    if mode == "wrap":
        return wrap_integer(exact, number_type)
    if mode == "sat":
        return saturate_integer(exact, number_type)
    raise Failure("integer overflow")


def truncating_divide(a: int, b: int) -> int:
    quotient = abs(a) // abs(b)
    return quotient if (a < 0) == (b < 0) else -quotient


def integer_arithmetic(
    operation: str, mode: str, number_type: Type, a: int, b: int
) -> int:
    if operation == "+":
        return fit_integer(a + b, mode, number_type)
    if operation == "-":
        return fit_integer(a - b, mode, number_type)
    if operation == "*":
        return fit_integer(a * b, mode, number_type)
    if b == 0:
        raise Failure("division by zero")
    if operation == "/":
        return fit_integer(truncating_divide(a, b), mode, number_type)
    assert operation == "%", operation
    return a - truncating_divide(a, b) * b


def integer_unary(operation: str, mode: str, number_type: Type, a: int) -> int:
    if operation == "neg":
        return fit_integer(-a, mode, number_type)
    assert operation == "abs", operation
    return fit_integer(abs(a), mode, number_type)


def bit_pattern(value: int, number_type: Type) -> int:
    return value & ((1 << bits_of(number_type)) - 1)


def bitwise(operation: str, number_type: Type, a: int, b: int) -> int:
    x, y = bit_pattern(a, number_type), bit_pattern(b, number_type)
    if operation == "bit-and":
        result = x & y
    elif operation == "bit-or":
        result = x | y
    else:
        assert operation == "bit-xor", operation
        result = x ^ y
    return wrap_integer(result, number_type)


def bit_not(number_type: Type, a: int) -> int:
    return wrap_integer(~bit_pattern(a, number_type), number_type)


def shift(operation: str, mode: str, number_type: Type, a: int, amount: int) -> int:
    bits = bits_of(number_type)
    if operation in ("rotl", "rotr"):
        amount %= bits
        if operation == "rotr":
            amount = (bits - amount) % bits
        pattern = bit_pattern(a, number_type)
        rotated = (pattern << amount) | (pattern >> (bits - amount))
        return wrap_integer(rotated, number_type)

    if amount >= bits:
        if mode != "wrap":
            raise Failure("shift amount out of range")
        amount %= bits
    if operation == "shl":
        return wrap_integer(a << amount, number_type)
    assert operation == "shr", operation
    return a >> amount


def count_bits(operation: str, number_type: Type, a: int) -> int:
    bits = bits_of(number_type)
    pattern = bit_pattern(a, number_type)
    if operation == "count-ones":
        return pattern.bit_count()
    if operation == "leading-zeros":
        return bits - pattern.bit_length()
    assert operation == "trailing-zeros", operation
    if pattern == 0:
        return bits
    return (pattern & -pattern).bit_length() - 1


def round_to(number_type: Type, value: float) -> float:
    if number_type != F32 or math.isnan(value) or math.isinf(value):
        return value
    try:
        return float(struct.unpack("f", struct.pack("f", value))[0])
    except OverflowError:
        return math.copysign(math.inf, value)


def ieee_divide(a: float, b: float) -> float:
    if b != 0.0:
        return a / b
    if a == 0.0 or math.isnan(a):
        return math.nan
    negative = math.copysign(1.0, a) != math.copysign(1.0, b)
    return -math.inf if negative else math.inf


def fit_float(result: float, mode: str, operands: tuple[float, ...]) -> float:
    if math.isnan(result):
        result = math.nan
    if mode == "ieee":
        return result
    if math.isnan(result) and not any(math.isnan(x) for x in operands):
        raise Failure("invalid float operation")
    if math.isinf(result) and not any(math.isinf(x) for x in operands):
        raise Failure("float overflow")
    return result


def float_arithmetic(
    operation: str, mode: str, number_type: Type, a: float, b: float
) -> float:
    divides_by_zero = operation == "/" and b == 0.0 and math.isfinite(a) and a != 0.0
    if divides_by_zero and mode != "ieee":
        raise Failure("division by zero")
    if operation == "+":
        exact = a + b
    elif operation == "-":
        exact = a - b
    elif operation == "*":
        exact = a * b
    elif operation == "/":
        exact = ieee_divide(a, b)
    else:
        assert operation == "%", operation
        exact = ieee_remainder(a, b)
    return fit_float(round_to(number_type, exact), mode, (a, b))


FLOAT_UNARY = ("neg", "abs", "sqrt", "floor", "ceil", "trunc", "round")


def round_integral(operation: str, a: float) -> float:
    if math.isnan(a):
        return math.nan
    if math.isinf(a):
        return a
    if operation == "floor":
        whole = math.floor(a)
    elif operation == "ceil":
        whole = math.ceil(a)
    elif operation == "trunc":
        whole = math.trunc(a)
    else:
        assert operation == "round", operation
        whole = round(a)
    return math.copysign(float(whole), a)


def ieee_remainder(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or b == 0.0:
        return math.nan
    if math.isinf(b):
        return a
    return math.fmod(a, b)


def sign_bit(value: float, number_type: Type) -> bool:
    if isinstance(value, NaNBits):
        return bool(value.bits >> (bits_of(number_type) - 1))
    return math.copysign(1.0, value) < 0


def copy_sign(number_type: Type, a: float, b: float) -> float:
    if math.isnan(a):
        return math.nan
    return -math.fabs(a) if sign_bit(b, number_type) else math.fabs(a)


def float_unary(operation: str, mode: str, number_type: Type, a: float) -> float:
    if operation in ("floor", "ceil", "trunc", "round"):
        return round_integral(operation, a)
    if math.isnan(a) and operation in ("neg", "abs"):
        return math.nan
    if operation == "neg":
        return -a
    if operation == "abs":
        return math.fabs(a)
    assert operation == "sqrt", operation
    exact = math.nan if a < 0.0 else math.sqrt(a)
    return fit_float(round_to(number_type, exact), mode, (a,))


def compare(operation: str, a: object, b: object) -> bool:
    if operation == "==":
        return a == b
    if operation == "!=":
        return a != b
    if operation == "<":
        return a < b  # type: ignore[operator, no-any-return]
    if operation == "<=":
        return a <= b  # type: ignore[operator, no-any-return]
    if operation == ">":
        return a > b  # type: ignore[operator, no-any-return]
    assert operation == ">=", operation
    return a >= b  # type: ignore[operator, no-any-return]


def shortest_digits(value: float, number_type: Type) -> tuple[str, int]:
    for precision in range(17):
        text = f"{value:.{precision}e}"
        if round_to(number_type, float(text)) == value:
            break
    mantissa, _, exponent = text.partition("e")
    digits = mantissa.replace(".", "").rstrip("0") or "0"
    return digits, int(exponent)


def format_float(value: float, number_type: Type) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    if value == 0.0:
        return f"{sign}0.0"

    digits, exponent = shortest_digits(abs(value), number_type)
    if -4 <= exponent < 16:
        if exponent < 0:
            body = "0." + "0" * (-exponent - 1) + digits
        elif exponent >= len(digits) - 1:
            body = digits + "0" * (exponent - len(digits) + 1) + ".0"
        else:
            body = digits[: exponent + 1] + "." + digits[exponent + 1 :]
        return sign + body

    mantissa = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    exponent_sign = "-" if exponent < 0 else "+"
    return f"{sign}{mantissa}e{exponent_sign}{abs(exponent):02d}"


def integer_to_integer(value: int, mode: str, target: Type) -> int:
    low, high = integer_range(target)
    if low <= value <= high:
        return value
    if mode == "wrap":
        return wrap_integer(value, target)
    if mode == "sat":
        return saturate_integer(value, target)
    raise Failure("value out of range")


def integer_to_float(value: int, target: Type) -> float:
    if target != F32 or abs(value) < 1 << 53:
        return round_to(target, float(value))
    magnitude = abs(value)
    shift = magnitude.bit_length() - 24
    kept, rest = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if rest > half or (rest == half and kept & 1):
        kept += 1
    return math.copysign(float(kept << shift), value)


def float_to_integer(value: float, mode: str, target: Type) -> int:
    low, high = integer_range(target)
    if math.isnan(value):
        if mode == "sat":
            return 0
        raise Failure("NaN has no integer value")
    if math.isinf(value):
        if mode == "sat":
            return high if value > 0 else low
        raise Failure("infinity has no integer value")
    exact = math.trunc(value)
    if low <= exact <= high:
        return exact
    if mode == "sat":
        return saturate_integer(exact, target)
    raise Failure("value out of range")


def float_to_float(value: float, mode: str, target: Type) -> float:
    if math.isnan(value):
        return math.nan
    result = round_to(target, value)
    if math.isinf(result) and not math.isinf(value) and mode != "ieee":
        raise Failure("float overflow")
    return result


class NaNBits(float):
    bits: int

    def __new__(cls, bits: int) -> Self:
        nan = super().__new__(cls, math.nan)
        nan.bits = bits
        return nan


def float_bits(value: float, number_type: Type) -> int:
    if isinstance(value, NaNBits):
        return value.bits
    if number_type == F32:
        return int(struct.unpack("<I", struct.pack("<f", value))[0])
    return int(struct.unpack("<Q", struct.pack("<d", value))[0])


def float_from_bits(bits: int, number_type: Type) -> float:
    if number_type == F32:
        value = float(struct.unpack("<f", struct.pack("<I", bits))[0])
    else:
        value = float(struct.unpack("<d", struct.pack("<Q", bits))[0])
    return NaNBits(bits) if math.isnan(value) else value


def round_exact(exact: Fraction, number_type: Type) -> float:
    if exact < 0:
        return -round_exact(-exact, number_type)
    if exact == 0:
        return 0.0
    precision, emin, emax = (24, -126, 127) if number_type == F32 else (53, -1022, 1023)
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    if exact < Fraction(2) ** e:
        e -= 1
    e = max(e, emin)
    q = round(exact * Fraction(2) ** (precision - 1 - e))
    if q == 2**precision:
        q //= 2
        e += 1
    if e > emax:
        return math.inf
    return math.ldexp(q, e - (precision - 1))


def fused_multiply_add(
    mode: str, number_type: Type, a: float, b: float, c: float
) -> float:
    operands = (a, b, c)
    if not (math.isfinite(a) and math.isfinite(b)):
        result = a * b + c
    elif not math.isfinite(c):
        result = c
    else:
        exact = Fraction(a) * Fraction(b) + Fraction(c)
        if exact != 0:
            result = round_exact(exact, number_type)
        elif (a == 0.0 or b == 0.0) and c == 0.0:
            product_negative = (math.copysign(1.0, a) < 0) != (
                math.copysign(1.0, b) < 0
            )
            negative = product_negative and math.copysign(1.0, c) < 0
            result = -0.0 if negative else 0.0
        else:
            result = 0.0
    return fit_float(result, mode, operands)


DECIMAL_DIGITS = r"[0-9](?:_?[0-9])*"
FRACTION = rf"(?:\.({DECIMAL_DIGITS}))?"
EXPONENT = rf"(?:[eE]([+-]?{DECIMAL_DIGITS}))?"
FLOAT_TEXT = re.compile(rf"([+-]?)({DECIMAL_DIGITS}){FRACTION}{EXPONENT}")
DECIMAL_REACH = 400


def parse_float(text: str, number_type: Type) -> float | str:
    shown = '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    match = FLOAT_TEXT.fullmatch(text)
    if match is None:
        return f"not a number: {shown}"
    sign, whole, fraction, exponent = (part or "" for part in match.groups())
    whole = whole.replace("_", "")
    if len(whole) > 1 and whole[0] == "0":
        return f"not a number: {shown}"
    fraction = fraction.replace("_", "")
    digits = (whole + fraction).lstrip("0")
    negative = sign == "-"
    if not digits:
        return -0.0 if negative else 0.0
    power = int(exponent.replace("_", "") or "0") - len(fraction)
    reach = len(digits) + power
    if reach > DECIMAL_REACH or reach < -DECIMAL_REACH:
        return f"number out of range: {shown}"
    exact = Fraction(int(digits)) * Fraction(10) ** power
    rounded = round_exact(exact, number_type)
    if rounded == math.inf or rounded == 0.0:
        return f"number out of range: {shown}"
    return -rounded if negative else rounded
