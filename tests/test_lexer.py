from typing import Any

import pytest

from inde.lexer import Token, TokenType, tokenize


def list_all_token_types(source: str) -> list[TokenType]:
    return [token.type for token in tokenize(source)[:-1]]  # Exclude EOF


def list_all_token_texts(source: str) -> list[str]:
    return [token.text for token in tokenize(source)[:-1]]  # Exclude EOF


def expect_single_token(source: str) -> Token:
    tokens = tokenize(source)
    assert len(tokens) == 2, f"expected one token, got {tokens[:-1]}"
    return tokens[0]


def get_failure_message(source: str) -> Any:
    with pytest.raises(SystemExit) as err:
        tokenize(source)

    return err.value.args[0]


### Test general lexer functionality


def test_location_counts_lines_and_columns_from_one() -> None:
    tokens = tokenize("[\n  ]")
    assert (tokens[0].location.line, tokens[0].location.column) == (1, 1)
    assert (tokens[1].location.line, tokens[1].location.column) == (2, 3)


### Test whitespace and comments


def test_empty_source_only_contains_eof() -> None:
    assert [token.type for token in tokenize("")] == [TokenType.EOF]
    assert [token.type for token in tokenize("  \t\n ; comment")] == [TokenType.EOF]


def test_crlf_is_valid_line_break() -> None:
    assert list_all_token_types("(\r\n)") == [TokenType.LPAREN, TokenType.RPAREN]


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("\r", "lone carriage return"),
        ("(   )", "printable ASCII"),  # a non-breaking space is not whitespace
        # TODO: Test for other invalid whitespace
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
        # Try lexing ( or ) token each time to check for collisions
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


### Test delimiters


def test_delimiters() -> None:
    assert list_all_token_types("()[]{},:") == [
        TokenType.LPAREN,
        TokenType.RPAREN,
        TokenType.LBRACKET,
        TokenType.RBRACKET,
        TokenType.LBRACE,
        TokenType.RBRACE,
        TokenType.COMMA,
        TokenType.COLON,
    ]


### Test names


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


### Test keywords


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
    ],
)
def test_keywords(source: str) -> None:
    assert expect_single_token(source).type is TokenType.KEYWORD
