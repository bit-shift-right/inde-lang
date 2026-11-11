import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import NoReturn

from inde import logger

VALID_WHITESPACE = [" ", "\t", "\n", "\r\n"]


class TokenType(Enum):
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()

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
        self.cursor = 0
        self.line = 1
        self.column = 1
        self.log = logger.Logger(program_name)

    def fail(self, message: str, location: Location) -> NoReturn:
        self.log.print_error(message, f"{location.line}:{location.column}:")
        sys.exit(1)

    def at_end(self) -> bool:
        return self.cursor >= len(self.source)

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
        if not self.at_end():
            return self.source[peek_cursor]
        else:
            return ""

    def get_location(self) -> Location:
        return Location(self.line, self.column, self.cursor)

    def emit_token(
        self, type: TokenType, start: Location, value: object = None
    ) -> Token:
        return Token(type, self.source[start.cursor : self.cursor], start, value)

    def find_comment_end(self) -> None:
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
                self.fail(
                    "a lone carriage return is not a valid line break",
                    self.get_location(),
                )
            elif current_character == ";":
                self.find_comment_end()
            else:
                return

    def next_token(self) -> Token:
        self.skip_whitespace_and_comments()

        token_start = self.get_location()
        if self.at_end():
            return Token(TokenType.EOF, "", token_start)

        current_character = self.peek()
        if current_character in VALID_DELIMITERS:
            self.advance()
            return self.emit_token(VALID_DELIMITERS[current_character], token_start)

        self.fail(f"unexpected character `{current_character}`", token_start)

    def get_tokens(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.type is TokenType.EOF:
                return tokens


def tokenize(source: str, program_name: str) -> list[Token]:
    return Lexer(source, program_name).get_tokens()


def dump_tokens(tokens: list[Token]) -> None:
    for token in tokens:
        print(
            f"{token.location.line}:{token.location.column:<6}"
            f" | {token.type.name:10} "
            f"text={token.text!r}   "
            f"value={token.value!r}"
        )
