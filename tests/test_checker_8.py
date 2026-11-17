import pytest

from inde.checker import Checker, check
from inde.errors import IndeError, IndeErrors
from inde.lexer import tokenize
from inde.parser import Parser, parse
from inde.type_model import Type


def line_types(line: str) -> str:
    checker = Checker()
    checker.check_program(parse(""))
    body = Parser(tokenize(line)).parse_repl_line()
    assert isinstance(body, tuple)
    stack: list[Type] = []
    checker.check_line(body, stack)
    return " ".join(str(t) for t in stack)


def line_fails(line: str) -> str:
    with pytest.raises(IndeError) as err:
        line_types(line)
    return err.value.message


def get_failure_message(source: str) -> str:
    with pytest.raises(IndeErrors) as err:
        check(parse(source))
    messages = [error.message for error in err.value.errors]
    assert len(messages) == 1, messages
    return messages[0]


@pytest.mark.parametrize(
    ("line", "types"),
    [
        ('"abc" str-len', "u32"),
        ('"a" "b" str-concat', "str"),
        ('"abc" 0u32 2u32 str-slice', "Option[str]"),
        ('"abc" 0u32 str-char-at', "Option[char]"),
        ('"abc" str-to-bytes', "array[u8]"),
        ("{ 72u8 105u8 } bytes-to-str", "Result[str, u32]"),
        ("'a' char-utf8-len", "u32"),
        ("'a' str-from-char", "str"),
        ('"a" "b" <', "bool"),
    ],
)
def test_text_word_types(line: str, types: str) -> None:
    assert line_types(line) == types


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            '"abc" 0 str-char-at',
            "`str-char-at` takes a `str` and a `u32` byte offset, found `str i32`",
        ),
        ('"a" "b" +', "`+` takes two numbers of the same type, found `str str`"),
        (
            "{ 1 2 } bytes-to-str",
            "`bytes-to-str` takes an `array[u8]`, found `array[i32]`",
        ),
    ],
)
def test_text_word_errors(line: str, message: str) -> None:
    assert line_fails(line) == message


@pytest.mark.parametrize(
    ("line", "types"),
    [
        ("5 to-f32", "f32"),
        ("1u8 to-i32 3 +", "i32"),
        ("5 to-i32", "i32"),
        ("300 to-u8.wrap", "u8"),
        ("300 to-u8.sat", "u8"),
        ("300 to-u8.checked", "Option[u8]"),
        ("5 to-f64.ieee", "f64"),
        ("1.5 to-i64", "i64"),
        ("1.5 to-u8.sat", "u8"),
        ("1.5 to-i32.checked", "Option[i32]"),
        ("1.5 to-f64", "f64"),
        ("1.5f64 to-f32.ieee", "f32"),
        ("1.5 to-f32", "f32"),
        ("'a' to-u32", "u32"),
        ("'a' to-u32.checked", "Option[u32]"),
        ("97u32 to-char", "char"),
        ("97u32 to-char.checked", "Option[char]"),
        ("'a' to-char", "char"),
        ("'a' to-u32 1u32 + to-char", "char"),
    ],
)
def test_conversions(line: str, types: str) -> None:
    assert line_types(line) == types


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "97 to-char",
            (
                "there is no conversion from `i32` to `char`; "
                "only a `u32` converts to a `char`"
            ),
        ),
        (
            "'a' to-i32",
            (
                "there is no conversion from `char` to `i32`; "
                "a `char` converts only to `u32`"
            ),
        ),
        ("true to-i32", "there is no conversion from `bool` to `i32`"),
        ('"5" to-i32', "there is no conversion from `str` to `i32`"),
        (
            "1u8 to-char",
            (
                "there is no conversion from `u8` to `char`; "
                "only a `u32` converts to a `char`"
            ),
        ),
        (
            "65u32 to-char.wrap",
            (
                "`to-char` has no `.wrap` form when converting from `u32`; "
                "its forms are `to-char`, `to-char.checked`"
            ),
        ),
        (
            "1.5 to-i32.wrap",
            (
                "`to-i32` has no `.wrap` form when converting from `f32`; "
                "its forms are `to-i32`, `to-i32.sat`, `to-i32.checked`"
            ),
        ),
        (
            "1 to-f32.sat",
            (
                "`to-f32` has no `.sat` form when converting from `i32`; "
                "its forms are `to-f32`, `to-f32.checked`, `to-f32.ieee`"
            ),
        ),
        (
            "1 to-i64.ieee",
            (
                "`to-i64` has no `.ieee` form when converting from `i32`; its forms "
                "are `to-i64`, `to-i64.wrap`, `to-i64.sat`, `to-i64.checked`"
            ),
        ),
        ("to-i32", "`to-i32` takes 1 value, but the stack is empty"),
        (
            "none to-i32",
            (
                "`to-i32` converts the value below it, but its type is not known "
                "at this point"
            ),
        ),
    ],
)
def test_conversion_errors(line: str, message: str) -> None:
    assert line_fails(line) == message


def test_conversion_names_are_taken() -> None:
    assert get_failure_message("word to-i32 ( -> ) end") == (
        "`to-i32` is a built-in word and cannot be redefined"
    )
    assert get_failure_message("word t ( i32 -> ) let to-f32 in end end") == (
        "`to-f32` already names a word, so it cannot name a local"
    )
    assert line_fails("1 to-i3") == "unknown word `to-i3`; did you mean `to-i32`?"


@pytest.mark.parametrize(
    ("signature", "body"),
    [
        ("( T -> f64 ) where T: Num", "to-f64"),
        ("( T -> i32 ) where T: Num", "to-i32.sat"),
        ("( T -> u8 ) where T: Integer", "to-u8.wrap"),
        ("( T -> u32 ) where T: Signed", "to-u32.checked drop 0u32"),
        ("( T -> f32 ) where T: Float", "to-f32.ieee"),
    ],
)
def test_a_constrained_type_variable_converts(signature: str, body: str) -> None:
    check(parse(f"word t {signature} {body} end"))


@pytest.mark.parametrize(
    ("signature", "body", "message"),
    [
        (
            "( T -> i32 )",
            "to-i32",
            (
                "there is no conversion from `T` to `i32`; add `where T: Integer`, "
                "`where T: Float` or `where T: Num` to the signature"
            ),
        ),
        (
            "( T -> i32 ) where T: Num",
            "to-i32.wrap",
            (
                "`to-i32` has no `.wrap` form when converting from `T`; its forms "
                "are `to-i32`, `to-i32.sat`, `to-i32.checked`"
            ),
        ),
        (
            "( T -> char ) where T: Integer",
            "to-char",
            (
                "there is no conversion from `T` to `char`; only a `u32` converts "
                "to a `char`"
            ),
        ),
    ],
)
def test_what_a_type_variable_cannot_convert(
    signature: str, body: str, message: str
) -> None:
    assert get_failure_message(f"word t {signature} {body} end") == message
