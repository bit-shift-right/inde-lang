import io

import pytest

from inde.checker import Checker
from inde.errors import IndeTrap
from inde.interpreter import Session


def shows(line: str) -> str:
    session = Session(io.StringIO(), Checker())
    session.run_line(line)
    return session.show_stack().removeprefix("stack: ")


def trap_message(line: str) -> str:
    session = Session(io.StringIO(), Checker())
    with pytest.raises(IndeTrap) as trap:
        session.run_line(line)
    return trap.value.message


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ('"héllo" str-len', "6u32"),
        ('"a\\0b" str-len', "3u32"),
        ('"" str-len', "0u32"),
        ('"foo" "bar" str-concat', '"foobar"'),
        ('"héllo" 1u32 3u32 str-slice', '"é" some'),
        ('"héllo" 1u32 2u32 str-slice', "none"),
        ('"héllo" 3u32 1u32 str-slice', "none"),
        ('"héllo" 0u32 7u32 str-slice', "none"),
        ('"héllo" 6u32 6u32 str-slice', '"" some'),
        ('"héllo" 1u32 str-char-at', "'é' some"),
        ('"héllo" 2u32 str-char-at', "none"),
        ('"héllo" 6u32 str-char-at', "none"),
        ('"a😀b" 1u32 str-char-at', "'😀' some"),
        ('"hé" str-to-bytes', "{ 104u8 195u8 169u8 }"),
        ("{ 72u8 105u8 } bytes-to-str", '"Hi" ok'),
        ("{ 0xFFu8 } bytes-to-str", "0u32 err"),
        ("{ 72u8 0xC3u8 } bytes-to-str", "1u32 err"),
        ("{ 0xEDu8 0xA0u8 0x80u8 } bytes-to-str", "0u32 err"),
        ("'é' char-utf8-len", "2u32"),
        ("'😀' char-utf8-len", "4u32"),
        ("'é' str-from-char", '"é"'),
    ],
)
def test_text_words(line: str, stack: str) -> None:
    assert shows(line) == stack


def test_walking_a_string() -> None:
    source = (
        "word count-chars ( str -> u32 ) "
        "0u32 0u32 let text count offset in "
        "count offset "
        "while dup text swap str-char-at match "
        "case some char-utf8-len let c o n in c 1u32 + o n + true end "
        "case none false end "
        "do end "
        "drop end end"
    )
    session = Session(io.StringIO(), Checker())
    session.run_line(source)
    session.run_line('"héllo😀" count-chars')
    assert session.show_stack() == "stack: 6u32"


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("5 to-f32", "5.0"),
        ("1u8 to-i32 3 +", "4"),
        ("5 to-i32", "5"),
        ("300 to-u8.wrap", "44u8"),
        ("300 to-u8.sat", "255u8"),
        ("300 to-u8.checked", "none"),
        ("200 to-u8.checked", "200u8 some"),
        ("-2.7 to-i32", "-2"),
        ("1.0e10 to-i32.sat", "2147483647"),
        ("1.0e10 to-i32.checked", "none"),
        ("16777217 to-f32", "16777216.0"),
        ("0.1 to-f64", "0.10000000149011612f64"),
        ("1.0e300f64 to-f32.ieee", "inf"),
        ("1.0e300f64 to-f32.checked", "none"),
        ("1.5f64 to-f32", "1.5"),
        ("'a' to-u32", "97u32"),
        ("97u32 to-char", "'a'"),
        ("'a' to-u32 1u32 + to-char", "'b'"),
        ("55296u32 to-char.checked", "none"),
        ("1114112u32 to-char.checked", "none"),
        ("1114111u32 to-char.checked", "'\\u{10FFFF}' some"),
        ("'a' to-char", "'a'"),
    ],
)
def test_conversions(line: str, stack: str) -> None:
    assert shows(line) == stack


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("300 let x in x to-u8 end", "value out of range in `to-u8` (i32 300)"),
        (
            "1.0e10 let x in x to-i32 end",
            "value out of range in `to-i32` (f32 10000000000.0)",
        ),
        (
            "1.0e300f64 let x in x to-f32 end",
            "float overflow in `to-f32` (f64 1e+300f64)",
        ),
        (
            "55296u32 let x in x to-char end",
            "not a Unicode scalar value in `to-char` (u32 55296u32)",
        ),
    ],
)
def test_a_failed_conversion_traps(line: str, message: str) -> None:
    assert trap_message(line) == message


def test_a_generic_conversion_runs_on_each_type() -> None:
    session = Session(io.StringIO(), Checker())
    session.run_line("word to-real ( T -> f64 ) where T: Num to-f64 end")
    session.run_line("5 to-real 2.5 to-real 7u8 to-real")
    assert session.show_stack() == "stack: 5.0f64 | 2.5f64 | 7.0f64"
