from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import NoReturn

from inde import logger


class TokenType(Enum):
    # Delimiters
    LEFT_PARENTHESIS = auto()
    RIGHT_PARENTHESIS = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    COMMA = auto()
    COLON = auto()

    # Literals
    STRING = auto()
    BOOLEAN = auto()

    # Words
    NAME = auto()
    KEYWORD = auto()
    SYMBOL = auto()
    TYPE_VARIABLE = auto()
    TYPE_NAME = auto()
    ROW_VARIABLE = auto()
    CONSTANT_NAME = auto()

    # End of file marker
    END_OF_FILE = auto()


VALID_WHITESPACE: tuple[str, ...] = (" ", "\t", "\n", "\r\n")


VALID_DELIMITERS: dict[str, TokenType] = {
    "(": TokenType.LEFT_PARENTHESIS,
    ")": TokenType.RIGHT_PARENTHESIS,
    "[": TokenType.LEFT_BRACKET,
    "]": TokenType.RIGHT_BRACKET,
    "{": TokenType.LEFT_BRACE,
    "}": TokenType.RIGHT_BRACE,
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
    "const",
)


VALID_SYMBOLS: tuple[str, ...] = ("?", "&", "|", "=", "->")


VALID_STRING_ESCAPES: dict[str, str] = {
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "0": "\0",
    "\\": "\\",
    "'": "'",
    '"': '"',
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

    def read_escape(self, token_start: Location, token_name_string: str) -> str:
        escape_start = self.get_location()
        self.advance()

        if self.at_end():
            self.fail(f"unterminated {token_name_string} literal", token_start)

        current_character = self.peek()

        if current_character == "\n" or current_character == "\r":
            self.fail(
                f"line breaks not allowed inside of {token_name_string} literals",
                token_start,
            )

        self.advance()

        if current_character in VALID_STRING_ESCAPES:
            return VALID_STRING_ESCAPES[current_character]

        self.fail(f"unknown escape sequence \\{current_character}", escape_start)

    def read_quoted(
        self, token_start: Location, delimiter: str, token_name_string: str
    ) -> list[str]:
        self.advance()
        character_buffer: list[str] = []
        while True:
            if self.at_end():
                self.fail(f"unterminated {token_name_string} literal", token_start)

            current_character = self.peek()

            if current_character == delimiter:
                self.advance()
                return character_buffer

            if current_character == "\n" or current_character == "\r":
                self.fail(
                    f"line breaks not allowed inside of {token_name_string} literals",
                    token_start,
                )

            if current_character == "\\":
                character_buffer.append(
                    self.read_escape(token_start, token_name_string)
                )

            else:
                character_buffer.append(self.advance())

    def lex_string(self, token_start: Location) -> Token:
        string_buffer = self.read_quoted(token_start, '"', "string")
        return self.emit_token(TokenType.STRING, token_start, "".join(string_buffer))

    def lex_word(self, token_start: Location) -> Token:
        token_text = self.find_end_of_token(token_start)

        if not token_text.isascii() or not token_text.isprintable():
            self.fail("names may contain only printable ASCII characters", token_start)

        if token_text in VALID_SYMBOLS:
            return self.emit_token(TokenType.SYMBOL, token_start)

        if token_text in VALID_KEYWORDS:
            return self.emit_token(TokenType.KEYWORD, token_start)

        if token_text == "true":
            return self.emit_token(TokenType.BOOLEAN, token_start, True)

        if token_text == "false":
            return self.emit_token(TokenType.BOOLEAN, token_start, False)

        if token_text.startswith(".."):
            row_var_name = token_text[2:]

            if len(row_var_name) < 1:
                self.fail("a row variable must have a name", token_start)

            if row_var_name[0].isupper():
                self.fail(
                    "a row variable name must start with a lowercase letter",
                    token_start,
                )

            if row_var_name[0].isdigit():
                self.fail("a row variable name must start with a letter", token_start)

            if not row_var_name.isalnum():
                self.fail(
                    "a row variable name may only contain letters and digits",
                    token_start,
                )

            return self.emit_token(TokenType.ROW_VARIABLE, token_start)

        if token_text[0].isupper():
            if len(token_text) == 1 or token_text[1:].isdigit():
                return self.emit_token(TokenType.TYPE_VARIABLE, token_start)

            if all(
                character.isupper() or character.isdigit() or character == "_"
                for character in token_text
            ):
                return self.emit_token(TokenType.CONSTANT_NAME, token_start)

            if token_text.isalnum():
                return self.emit_token(TokenType.TYPE_NAME, token_start)

            self.fail(
                f"`{token_text}` is not a valid name: "
                f"a type name may only contain letters and digits and "
                f"a constant name may only contain letters, digits and underscores",
                token_start,
            )

        return self.emit_token(TokenType.NAME, token_start)

    def next_token(self) -> Token:
        self.skip_whitespace_and_comments()

        token_start = self.get_location()
        if self.at_end():
            return Token(TokenType.END_OF_FILE, "", token_start)

        current_character = self.peek()
        if current_character in VALID_DELIMITERS:
            self.advance()
            return self.emit_token(VALID_DELIMITERS[current_character], token_start)
        if current_character == '"':
            return self.lex_string(token_start)

        return self.lex_word(token_start)

    def get_tokens(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.type is TokenType.END_OF_FILE:
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
