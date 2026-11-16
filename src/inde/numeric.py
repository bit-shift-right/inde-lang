from __future__ import annotations

import math
import struct

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
    else:
        assert operation == "/", operation
        exact = ieee_divide(a, b)
    return fit_float(round_to(number_type, exact), mode, (a, b))


def float_unary(operation: str, mode: str, number_type: Type, a: float) -> float:
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
