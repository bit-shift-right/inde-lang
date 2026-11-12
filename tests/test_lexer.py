import pytest

from inde.lexer import TokenType, tokenize

PROGRAM_NAME: str = "inde_test"


def list_all_token_types(source: str) -> list[TokenType]:
    return [token.type for token in tokenize(source, PROGRAM_NAME)[:-1]]  # Exclude EOF


def list_all_token_texts(source: str) -> list[TokenType]:
    return [token.text for token in tokenize(source, PROGRAM_NAME)[:-1]]  # Exclude EOF


def expect_single_token(source: str):
    tokens = tokenize(source, PROGRAM_NAME)
    assert len(tokens) == 2, f"expected one token, got {tokens[:-1]}"
    return tokens[0]


def get_failure_message(source: str) -> str:
    with pytest.raises(SystemExit) as err:
        tokenize(source, PROGRAM_NAME)

    return err.value.args[0]


### Test general lexer functionality


def test_location_counts_lines_and_columns_from_one():
    tokens = tokenize("[\n  ]", PROGRAM_NAME)
    assert (tokens[0].location.line, tokens[0].location.column) == (1, 1)
    assert (tokens[1].location.line, tokens[1].location.column) == (2, 3)


### Test whitespace and comments


def test_empty_source_only_contains_eof():
    assert [token.type for token in tokenize("", PROGRAM_NAME)] == [TokenType.EOF]
    assert [token.type for token in tokenize("  \t\n ; comment", PROGRAM_NAME)] == [
        TokenType.EOF
    ]


def test_crlf_is_valid_line_break():
    assert list_all_token_types("(\r\n)") == [TokenType.LPAREN, TokenType.RPAREN]


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("\r", "lone carriage return"),
        # TODO: Test for other invalid whitespace
    ],
)
def test_invalid_whitespace(source, fragment):
    assert fragment in get_failure_message(source)


def test_unterminated_block_comment():
    assert "unterminated block comment" in get_failure_message(";( a b c")


def test_line_numbers_after_block_comment():
    token = tokenize(";( one\ntwo\n;) [", PROGRAM_NAME)[0]
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
def test_block_comments(source, expected):
    assert list_all_token_texts(source) == expected
