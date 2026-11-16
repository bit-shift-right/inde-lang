import math
import struct

import pytest

from inde.errors import IndeError
from inde.lexer import NumberType, Token, TokenType, tokenize


def list_all_token_types(source: str) -> list[TokenType]:
    return [token.type for token in tokenize(source)[:-1]]  # Exclude END_OF_FILE


def list_all_token_texts(source: str) -> list[str]:
    return [token.text for token in tokenize(source)[:-1]]  # Exclude END_OF_FILE


def expect_single_token(source: str) -> Token:
    tokens = tokenize(source)
    assert len(tokens) == 2, f"expected one token, got {tokens[:-1]}"
    return tokens[0]


def get_failure_message(source: str) -> str:
    with pytest.raises(IndeError) as err:
        tokenize(source)

    return err.value.message


def as_f32(x: float) -> float:
    return float(struct.unpack("f", struct.pack("f", x))[0])


def test_location_counts_lines_and_columns_from_one() -> None:
    tokens = tokenize("[\n  ]")
    assert (tokens[0].location.line, tokens[0].location.column) == (1, 1)
    assert (tokens[1].location.line, tokens[1].location.column) == (2, 3)


def test_empty_source_only_contains_end_of_file() -> None:
    assert [token.type for token in tokenize("")] == [TokenType.END_OF_FILE]
    assert [token.type for token in tokenize("  \t\n ; comment")] == [
        TokenType.END_OF_FILE
    ]


def test_crlf_is_valid_line_break() -> None:
    assert list_all_token_types("(\r\n)") == [
        TokenType.LEFT_PARENTHESIS,
        TokenType.RIGHT_PARENTHESIS,
    ]


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("\r", "lone carriage return"),
        ("(   )", "printable ASCII"),  # a non-breaking space is not whitespace
    ],
)
def test_invalid_whitespace(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def test_unterminated_block_comment() -> None:
    assert "unterminated block comment" in get_failure_message(";( a b c")


def test_line_numbers_after_block_comment() -> None:
    token = tokenize(";( one\ntwo\n;) [")[0]
    assert (token.location.line, token.location.column) == (3, 4)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("(;( note ;))", ["(", ")"]),
        ("( ;( ) ;) )", ["(", ")"]),
        (";( a ;( b ;) (", ["("]),
        (";(\n multi\n line\n;) (", ["("]),
        ("; (not a block\n)", [")"]),
        (";) stray\n)", [")"]),
    ],
)
def test_block_comments(source: str, expected: list[str]) -> None:
    assert list_all_token_texts(source) == expected


def test_crlf_ends_a_word_or_number() -> None:
    assert list_all_token_texts("dup\r\nswap") == ["dup", "swap"]
    assert list_all_token_texts("5\r\n6u8\r\n") == ["5", "6u8"]
    assert list_all_token_texts("; comment\r\ndup") == ["dup"]
    token = tokenize("dup\r\nswap")[1]
    assert (token.location.line, token.location.column) == (2, 1)
    assert "lone carriage return" in get_failure_message("dup\rswap")


def test_delimiters() -> None:
    assert list_all_token_types("()[]{},:") == [
        TokenType.LEFT_PARENTHESIS,
        TokenType.RIGHT_PARENTHESIS,
        TokenType.LEFT_BRACKET,
        TokenType.RIGHT_BRACKET,
        TokenType.LEFT_BRACE,
        TokenType.RIGHT_BRACE,
        TokenType.COMMA,
        TokenType.COLON,
    ]


@pytest.mark.parametrize(
    ("source", "number", "number_type"),
    [
        ("42", 42, NumberType.I32),
        ("0", 0, NumberType.I32),
        ("-0", 0, NumberType.I32),
        ("-17", -17, NumberType.I32),
        ("+17", 17, NumberType.I32),
        ("1_000_000", 1_000_000, NumberType.I32),
        ("255u8", 255, NumberType.U8),
        ("-128i8", -128, NumberType.I8),
        ("+127i8", 127, NumberType.I8),
        ("3000000000i64", 3_000_000_000, NumberType.I64),
        ("-1_000i64", -1000, NumberType.I64),
        ("0XFF", 255, NumberType.I32),
        ("0xff", 255, NumberType.I32),
        ("-0xFF", -255, NumberType.I32),
        ("0xFFu8", 255, NumberType.U8),
        ("0xFFFFFFFFu32", 4_294_967_295, NumberType.U32),
        ("0b1010", 10, NumberType.I32),
        ("0X1f32", 7986, NumberType.I32),
    ],
)
def test_integers(source: str, number: int, number_type: NumberType) -> None:
    token = expect_single_token(source)
    assert token.type is TokenType.INTEGER
    assert token.value.number == number  # type: ignore[attr-defined]
    assert token.value.type is number_type  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("300u8", "out of range for type u8"),
        ("-1u8", "out of range for type u8"),
        ("128i8", "out of range for type i8"),
        ("-129i8", "out of range for type i8"),
        ("+128i8", "out of range for type i8"),
        ("2147483648", "out of range for type i32"),
        ("-2147483649", "out of range for type i32"),
        ("0xFFFFFFFF", "out of range for type i32"),
        ("007", "leading zero"),
        ("1__000", "underscores"),
        ("1_", "underscores"),
        ("0x_FF", "underscores"),
        ("5U8", "lowercase"),
        ("5u7", "unknown type suffix"),
        ("5sz", "reserved"),
        ("1f32", "float suffix on"),
        ("0b1f32", "only be integer types"),
        ("0b102", "invalid digit"),
        ("0x", "expected only hexadecimal digits after `0x`"),
        ("0X", "expected only hexadecimal digits after `0X`"),
        ("0B2", "expected only binary digits after `0B`"),
        ("0xFFg", "unknown type suffix"),
        ("5-3", "invalid number"),
        ("5+3", "invalid number"),
    ],
)
def test_integer_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(
    ("source", "number", "number_type"),
    [
        ("1.5", 1.5, NumberType.F32),
        ("0.5", 0.5, NumberType.F32),
        ("0.1", as_f32(0.1), NumberType.F32),
        ("1e9", 1e9, NumberType.F32),
        ("1e+3", 1000.0, NumberType.F32),
        ("1.5f64", 1.5, NumberType.F64),
        ("2.5e-3f64", 0.0025, NumberType.F64),
        ("3.4028235e38", as_f32(3.4028235e38), NumberType.F32),
        ("1e-45", as_f32(1e-45), NumberType.F32),
        ("1E9", 1e9, NumberType.F32),
        ("2.5E-3f64", 0.0025, NumberType.F64),
    ],
)
def test_floats(source: str, number: int, number_type: NumberType) -> None:
    token = expect_single_token(source)
    assert token.type is TokenType.FLOAT
    assert token.value.number == number  # type: ignore[attr-defined]
    assert token.value.type is number_type  # type: ignore[attr-defined]


def test_negative_zero_keeps_its_sign() -> None:
    number = expect_single_token("-0.0").value.number  # type: ignore[attr-defined]
    assert number == 0.0
    assert math.copysign(1.0, number) == -1.0


def test_f32_rounds_once_not_twice() -> None:
    # Should round down to 1.0
    token = expect_single_token("1.000000059604644775390625001")
    assert token.value.number == 1.0 + 2.0**-23  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("1.", "after the point"),
        ("1e", "after `e`"),
        ("1e+", "after `e+`"),
        ("1.5i32", "integer suffix on a float"),
        ("1.5.3", "invalid"),
        ("1e99", "literal overflows"),
        ("1e400f64", "overflows f64"),
        ("1e-60", "rounds to zero"),
        ("1E", "after `E`"),
        ("1.5E9F64", "lowercase"),
    ],
)
def test_float_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ('"hello"', "hello"),
        ('""', ""),
        ('"héllo"', "héllo"),
        (r'"a\nb"', "a\nb"),
        (r'"tab\there"', "tab\there"),
        (r'"a\0b"', "a\x00b"),
        (r'"say \"hi\""', 'say "hi"'),
        (r'"back\\slash"', "back\\slash"),
        ('"it\'s"', "it's"),
        ('"a;b"', "a;b"),
        (r'"\u{1F600}"', "\U0001f600"),
    ],
)
def test_strings(source: str, value: str) -> None:
    token = expect_single_token(source)
    assert token.type is TokenType.STRING
    assert token.value == value


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ('"abc', "unterminated string"),
        ('"abc\\', "unterminated string"),
        ('"a\nb"', "line break"),
        ('"a\r\nb"', "line break"),
        (r'"a\qb"', "unknown escape"),
        (r'"x\u41"', "expected {"),
        (r'"x\u{41"', "unterminated unicode"),
        (r'"x\u{}"', "one to six"),
        (r'"x\u{1234567}"', "one to six"),
        (r'"\u{D800}"', "surrogate"),
        (r'"\u{110000}"', "out of range"),
    ],
)
def test_string_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("'a'", "a"),
        ("'é'", "é"),
        ("' '", " "),
        ("'\"'", '"'),
        (r"'\n'", "\n"),
        (r"'\''", "'"),
        (r"'\\'", "\\"),
        (r"'\u{1F600}'", "\U0001f600"),
    ],
)
def test_characters(source: str, value: str) -> None:
    token = expect_single_token(source)
    assert token.type is TokenType.CHARACTER
    assert token.value == value


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("''", "empty character"),
        ("'ab'", "contain a single"),
        ("'é'", "contain a single"),
        ("'a", "unterminated character"),
        ("'\n'", "line break"),
        (r"'\q'", "unknown escape"),
        (r"'\u{D800}'", "surrogate"),
    ],
)
def test_character_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(("source", "value"), [("true", True), ("false", False)])
def test_booleans(source: str, value: bool) -> None:
    token = expect_single_token(source)
    assert token.type is TokenType.BOOLEAN
    assert token.value is value


@pytest.mark.parametrize(
    "source",
    [
        "dup",
        "str-len",
        "+",
        "-",
        "==",
        "<=",
        "empty?",
        "?x",
        "->x",
        "&&",
        "||",
        ".5",
        ".",
        "unit",
    ],
)
def test_names(source: str) -> None:
    assert expect_single_token(source).type is TokenType.NAME


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("café", "printable ASCII"),
        ("Éclair", "printable ASCII"),
        ("T١", "printable ASCII"),
        ("..aé", "printable ASCII"),
        ("a\x00b", "printable ASCII"),
        ("x\x7f", "printable ASCII"),
    ],
)
def test_name_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(
    "source",
    [
        "word",
        "end",
        "type",
        "linear",
        "where",
        "if",
        "else",
        "while",
        "do",
        "match",
        "case",
        "let",
        "in",
        "const",
    ],
)
def test_keywords(source: str) -> None:
    assert expect_single_token(source).type is TokenType.KEYWORD


@pytest.mark.parametrize("source", ["->", "?", "&", "|", "="])
def test_symbols(source: str) -> None:
    assert expect_single_token(source).type is TokenType.SYMBOL


@pytest.mark.parametrize(
    ("source", "type"),
    [
        ("T", TokenType.TYPE_VARIABLE),
        ("Q1", TokenType.TYPE_VARIABLE),
        ("Q10", TokenType.TYPE_VARIABLE),
        ("Shape", TokenType.TYPE_NAME),
        ("Rect3D", TokenType.TYPE_NAME),
        ("Io", TokenType.TYPE_NAME),
        ("IoError", TokenType.TYPE_NAME),
    ],
)
def test_type_variables_and_type_names(source: str, type: TokenType) -> None:
    assert expect_single_token(source).type is type


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("Shape_x", "letters and digits"),
        ("Foo-bar", "letters and digits"),
        ("Foo?", "letters and digits"),
        ("Max_depth", "letters and digits"),
    ],
)
def test_type_name_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(
    ("source", "type"),
    [
        ("..a", TokenType.ROW_VARIABLE),
        ("..rest", TokenType.ROW_VARIABLE),
        ("..a2", TokenType.ROW_VARIABLE),
        ("..aB", TokenType.ROW_VARIABLE),
    ],
)
def test_row_variable_names(source: str, type: TokenType) -> None:
    assert expect_single_token(source).type is type


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("..", "must have a name"),
        ("..A", "lowercase"),
        ("..1", "must start with a letter"),
        ("..a-b", "letters and digits"),
        ("...", "letters and digits"),
    ],
)
def test_row_variable_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(
    ("source", "type"),
    [
        ("IO", TokenType.CONSTANT_NAME),
        ("NEWLINE", TokenType.CONSTANT_NAME),
        ("MAX_DEPTH", TokenType.CONSTANT_NAME),
        ("TAB2", TokenType.CONSTANT_NAME),
        ("A_", TokenType.CONSTANT_NAME),
        ("Q1A", TokenType.CONSTANT_NAME),
    ],
)
def test_constant_names(source: str, type: TokenType) -> None:
    assert expect_single_token(source).type is type


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("NEW-LINE", "uppercase letters, digits and underscores"),
        ("ABC123#", "uppercase letters, digits and underscores"),
        ("AND+OR", "uppercase letters, digits and underscores"),
    ],
)
def test_constant_name_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)
