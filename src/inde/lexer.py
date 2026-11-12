from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import NoReturn

from inde import logger

VALID_WHITESPACE: tuple[str, ...] = (" ", "\t", "\n", "\r\n")


class TokenType(Enum):
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()

    # Words
    NAME = auto()
    KEYWORD = auto()

    # End of file marker
    EOF = auto()


VALID_DELIMITERS: dict[str, TokenType] = {
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    ",": TokenType.COMMA,
    ":": TokenType.COLON,
}


VALID_KEYWORDS: tuple[str, ...] = (
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
)


@dataclass
class Location:
    line: int
    column: int
    cursor: int


@dataclass
class Token:
    type: TokenType
    text: str
    location: Location
    value: object = None


class Lexer:
    def __init__(self, source: str, program_name: str) -> None:
        self.source = source
        self.program_name = program_name
        self.cursor = 0
        self.line = 1
        self.column = 1

    def fail(self, message: str, location: Location) -> NoReturn:
        raise SystemExit(
            logger.format_error(
                f'{self.program_name}:{location.line}:{location.column}: ', message
            )
        )

    def at_end(self, ahead: int = 0) -> bool:
        return (self.cursor + ahead) >= len(self.source)

    def advance(self) -> str:
        consumed = self.source[self.cursor]
        self.cursor += 1
        if consumed == "\n":
            self.column = 1
            self.line += 1
        else:
            self.column += 1
        return consumed

    def peek(self, ahead: int = 0) -> str:
        peek_cursor = self.cursor + ahead
        if not self.at_end(ahead):
            return self.source[peek_cursor]
        else:
            return ""

    def get_location(self) -> Location:
        return Location(self.line, self.column, self.cursor)

    def is_character_end_of_token(self, character: str) -> bool:
        return (
            character == ""
            or character in VALID_WHITESPACE
            or character in VALID_DELIMITERS
            or character in "'\";"
        )

    def find_end_of_token(self, token_start: Location) -> str:
        while not self.is_character_end_of_token(self.peek()):
            self.advance()
        return self.source[token_start.cursor : self.cursor]

    def emit_token(
        self, type: TokenType, token_start: Location, value: object = None
    ) -> Token:
        return Token(
            type, self.source[token_start.cursor : self.cursor], token_start, value
        )

    def find_end_of_comment(self) -> None:
        comment_start = self.get_location()

        self.advance()
        current_character = self.peek()

        if current_character == "(":
            while True:
                if self.at_end():
                    self.fail("unterminated block comment", comment_start)

                if current_character == ";" and self.peek(1) == ")":
                    self.advance()
                    self.advance()
                    return

                self.advance()
                current_character = self.peek()
        else:
            while not self.at_end() and self.peek() != "\n":
                self.advance()
            return

    def skip_whitespace_and_comments(self) -> None:
        while not self.at_end():
            current_character = self.peek()

            if current_character in VALID_WHITESPACE:
                self.advance()
            elif current_character == "\r":
                if self.peek(1) != "\n":
                    self.fail(
                        "a lone carriage return is not a valid line break",
                        self.get_location(),
                    )
                self.advance()
            elif current_character == ";":
                self.find_end_of_comment()
            else:
                return

    def lex_word(self, token_start: Location) -> Token:
        token_text = self.find_end_of_token(token_start)

        if not token_text.isascii() or not token_text.isprintable():
            self.fail("names may contain only printable ASCII characters", token_start)

        if token_text in VALID_KEYWORDS:
            return self.emit_token(TokenType.KEYWORD, token_start)

        return self.emit_token(TokenType.NAME, token_start)

    def next_token(self) -> Token:
        self.skip_whitespace_and_comments()

        token_start = self.get_location()
        if self.at_end():
            return Token(TokenType.EOF, "", token_start)

        current_character = self.peek()
        if current_character in VALID_DELIMITERS:
            self.advance()
            return self.emit_token(VALID_DELIMITERS[current_character], token_start)

        return self.lex_word(token_start)

    def get_tokens(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.type is TokenType.EOF:
                return tokens


def tokenize(source: str, program_name: str = Path(__file__).stem) -> list[Token]:
    return Lexer(source, program_name).get_tokens()


def dump_tokens(tokens: list[Token]) -> None:
    for token in tokens:
        print(
            f"{token.location.line}:{token.location.column}"
            f" | {token.type.name}: (\n"
            f"    text={token.text!r}\n"
            f"    value={token.value!r}\n"
            f")\n"
        )
    print("----- token dump")
