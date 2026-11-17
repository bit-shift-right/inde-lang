import math

import pytest

from inde.numeric import (
    Failure,
    float_to_float,
    float_to_integer,
    integer_to_float,
    integer_to_integer,
    round_to,
)
from inde.type_model import F32, F64, I8, I32, I64, U8, U64, Type

F32_MAX = 3.4028234663852886e38


@pytest.mark.parametrize(
    ("value", "mode", "target", "expected"),
    [
        (5, "", U8, 5),
        (300, "wrap", U8, 44),
        (-1, "wrap", U8, 255),
        (200, "wrap", I8, -56),
        (300, "sat", U8, 255),
        (-5, "sat", U8, 0),
        (-200, "sat", I8, -128),
        (2**40, "", I64, 2**40),
        (-(2**63), "", I64, -(2**63)),
    ],
)
def test_integer_to_integer(value: int, mode: str, target: Type, expected: int) -> None:
    assert integer_to_integer(value, mode, target) == expected


@pytest.mark.parametrize(
    ("value", "target"), [(300, U8), (-1, U8), (128, I8), (2**64, U64)]
)
def test_integer_out_of_range(value: int, target: Type) -> None:
    with pytest.raises(Failure) as failure:
        integer_to_integer(value, "", target)
    assert failure.value.reason == "value out of range"


@pytest.mark.parametrize(
    ("value", "target", "expected"),
    [
        (5, F32, 5.0),
        (-7, F64, -7.0),
        (16777217, F32, 16777216.0),
        (16777219, F32, 16777220.0),
        (2**53 + 1, F64, 2.0**53),
        (2**53 + 2**29 + 1, F32, 2.0**53 + 2**30),
        (2**64 - 1, F32, 2.0**64),
        (-(2**63), F32, -(2.0**63)),
    ],
)
def test_integer_to_float(value: int, target: Type, expected: float) -> None:
    assert integer_to_float(value, target) == expected


def test_integer_to_float_rounds_ties_to_even() -> None:
    assert integer_to_float(2**25 + 2, F32) == 2.0**25
    assert integer_to_float(2**25 + 6, F32) == 2.0**25 + 8
    big = 2**60 + 2**36
    assert integer_to_float(big, F32) == 2.0**60
    assert integer_to_float(big + 2**36, F32) == 2.0**60 + 2**37


@pytest.mark.parametrize(
    ("value", "mode", "target", "expected"),
    [
        (2.7, "", I32, 2),
        (-2.7, "", I32, -2),
        (255.9, "", U8, 255),
        (1e10, "sat", I32, 2**31 - 1),
        (-1e10, "sat", I32, -(2**31)),
        (-0.5, "", U8, 0),
        (math.nan, "sat", I32, 0),
        (math.inf, "sat", U8, 255),
        (-math.inf, "sat", I8, -128),
    ],
)
def test_float_to_integer(value: float, mode: str, target: Type, expected: int) -> None:
    assert float_to_integer(value, mode, target) == expected


@pytest.mark.parametrize(
    ("value", "target", "reason"),
    [
        (1e10, I32, "value out of range"),
        (-1.0, U8, "value out of range"),
        (256.0, U8, "value out of range"),
        (math.nan, I32, "NaN has no integer value"),
        (math.inf, I32, "infinity has no integer value"),
    ],
)
def test_float_to_integer_failures(value: float, target: Type, reason: str) -> None:
    with pytest.raises(Failure) as failure:
        float_to_integer(value, "", target)
    assert failure.value.reason == reason


def test_float_to_float() -> None:
    assert float_to_float(round_to(F32, 0.1), "", F64) == round_to(F32, 0.1)
    assert float_to_float(0.1, "", F32) == round_to(F32, 0.1)
    assert float_to_float(F32_MAX, "", F32) == F32_MAX
    assert math.isinf(float_to_float(math.inf, "", F32))
    assert math.isnan(float_to_float(math.nan, "", F32))
    assert float_to_float(1e300, "ieee", F32) == math.inf
    assert float_to_float(-1e300, "ieee", F32) == -math.inf
    with pytest.raises(Failure) as failure:
        float_to_float(1e300, "", F32)
    assert failure.value.reason == "float overflow"
