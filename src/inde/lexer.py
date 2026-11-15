import math
from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction
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
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    CHARACTER = auto()
    STRING = auto()

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


class NumberType(Enum):
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    F32 = "f32"
    F64 = "f64"

    def bits(self) -> int:
        return int(self.value[1:])

    def bytes(self) -> int:
        return int(self.bits() / 8)

    def is_float(self) -> bool:
        return self.value[0] == "f"

    def is_signed(self) -> bool:
        return self.value[0] in "if"

    def int_range(self) -> tuple[int, int]:
        bits = self.bits()
        if self.is_signed():
            return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        return 0, (1 << bits) - 1


@dataclass
class NumberValue:
    number: int | float
    type: NumberType


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


VALID_DECIMAL_DIGITS = "0123456789"
VALID_BINARY_DIGITS = "01"
VALID_HEXADECIMAL_DIGITS: str = "0123456789abcdefABCDEF"


def is_decimal_digit(character: str) -> bool:
    return character != "" and character in VALID_DECIMAL_DIGITS


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


class NumberParser:
    def __init__(
        self, number_string: str, location: Location, program_name: str
    ) -> None:
        self.number_string = number_string
        self.location = location
        self.program_name = program_name
        self.cursor = 0

    def fail(self, message: str) -> NoReturn:
        raise SystemExit(
            logger.format_error(
                f"{self.program_name}:{self.location.line}:{self.location.column}: ",
                message,
            )
        )

    def peek(self, ahead: int = 0) -> str:
        peek_cursor = self.cursor + ahead
        if peek_cursor < len(self.number_string):
            return self.number_string[peek_cursor]
        else:
            return ""

    def advance(self) -> str:
        consumed = self.number_string[self.cursor]
        self.cursor += 1
        return consumed

    def read_digits(self, valid_characters: str) -> str:
        digits = ""
        while True:
            current_character = self.peek()
            if current_character != "" and current_character in valid_characters:
                digits += self.advance()
            elif current_character == "_":
                next_character = self.peek(1)
                if (
                    digits == ""
                    or next_character == ""
                    or next_character not in valid_characters
                ):
                    self.fail("only single underscores may separate digits")
                self.advance()
            else:
                return digits

    def parse(self) -> tuple[TokenType, NumberValue]:
        negative = False
        if self.peek() in ("+", "-"):
            negative = self.advance() == "-"

        radix = 10
        if self.peek() == "0" and self.peek(1) in ("x", "b", "X", "B"):
            radix = 16 if self.peek(1) in ("x", "X") else 2
            self.advance()
            self.advance()

        if radix == 10:
            return self.parse_decimal(negative)
        return self.parse_radix(negative, radix)

    def parse_radix(self, negative: bool, radix: int) -> tuple[TokenType, NumberValue]:
        name = "hexadecimal" if radix == 16 else "binary"
        digits = self.read_digits(
            VALID_HEXADECIMAL_DIGITS if radix == 16 else VALID_BINARY_DIGITS
        )
        if digits == "":
            self.fail(
                f"expected only {name} digits after `0{'x' if radix == 16 else 'b'}`"
            )
        if radix == 2 and is_decimal_digit(self.peek()):
            self.fail(f"invalid digit `{self.peek()}` in binary literal")

        suffix = self.number_string[self.cursor :]
        suffix_type = self.suffix_type(suffix, default=NumberType.I32)
        if suffix_type.is_float():
            self.fail(f"{name} literals may only be integer types")
        value = int(digits, radix)
        return (
            TokenType.INTEGER,
            self.check_int(-value if negative else value, suffix_type, suffix),
        )

    def parse_decimal(self, negative: bool) -> tuple[TokenType, NumberValue]:
        integer_part = self.read_digits(VALID_DECIMAL_DIGITS)
        if len(integer_part) > 1 and integer_part[0] == "0":
            self.fail("decimal literals cannot contain a leading zero")

        fractional_part = ""
        exponent = ""
        is_float = False
        if self.peek() == ".":
            self.advance()
            fractional_part = self.read_digits(VALID_DECIMAL_DIGITS)
            if fractional_part == "":
                self.fail("invalid float literal; expected digits after the point")
            is_float = True
        if self.peek() in ("e", "E"):
            marker = self.advance()
            sign = self.advance() if self.peek() in ("+", "-") else ""
            digits = self.read_digits(VALID_DECIMAL_DIGITS)
            if digits == "":
                self.fail(
                    f"invalid float literal; expected digits after `{marker}{sign}`"
                )
            exponent = sign + digits
            is_float = True

        suffix = self.number_string[self.cursor :]
        type_suffix = self.suffix_type(
            suffix, default=NumberType.F32 if is_float else NumberType.I32
        )
        if type_suffix.is_float() and not is_float:
            self.fail("float suffix on an integer literal")
        if is_float and not type_suffix.is_float():
            self.fail("integer suffix on a float literal")

        if not is_float:
            value = int(integer_part)
            return (
                TokenType.INTEGER,
                self.check_int(-value if negative else value, type_suffix, suffix),
            )

        exact = Fraction(f"{integer_part}.{fractional_part or '0'}e{exponent or '0'}")
        rounded = self.round_to_float(exact, type_suffix)
        if rounded == math.inf:
            self.fail(f"float literal overflows {type_suffix.value}")
        if rounded == 0.0 and exact != 0:
            self.fail(
                f"float literal is not zero but rounds to zero in {type_suffix.value}"
            )
        return (
            TokenType.FLOAT,
            NumberValue(-rounded if negative else rounded, type_suffix),
        )

    # Round once to F32 or F64 to avoid double rounding with builtin float()
    def round_to_float(self, exact: Fraction, float_type: NumberType) -> float:
        precision, emin, emax = (24, -126, 127)
        if float_type == NumberType.F64:
            precision, emin, emax = (53, -1022, 1023)

        if exact == 0:
            return 0.0

        # Find e with 2**e <= exact < 2**(e + 1).
        e = exact.numerator.bit_length() - exact.denominator.bit_length()
        if exact < Fraction(2) ** e:
            e -= 1
        # Subnormal numbers
        e = max(e, emin)
        # Scale so the significand is an integer
        q = round(exact * Fraction(2) ** (precision - 1 - e))
        if q == 2**precision:  # Rounding carried to next power of two
            q //= 2
            e += 1
        if e > emax:
            return math.inf
        return math.ldexp(q, e - (precision - 1))

    def suffix_type(self, suffix: str, default: NumberType) -> NumberType:
        if suffix == "":
            return default
        if suffix[0] in "+-" or not suffix[0].isalpha():
            self.fail(f"invalid number literal `{self.number_string}`")
        if suffix == "sz":
            self.fail("type suffix `sz` is reserved for future use")
        names = [types.value for types in NumberType]
        if suffix in names:
            return NumberType(suffix)
        if suffix.lower() in names:
            self.fail(f"type suffixes are lowercase; write `{suffix.lower()}`")
        self.fail(
            f"unknown type suffix `{suffix}`; valid type suffixes are "
            f"{', '.join(names)}"
        )

    def check_int(
        self, value: int, type_suffix: NumberType, suffix: str
    ) -> NumberValue:
        low, high = type_suffix.int_range()
        if low <= value <= high:
            return NumberValue(value, type_suffix)
        self.fail(
            f"integer literal {self.number_string} is out of range for type "
            f"{type_suffix.value}"
        )


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
                f"{self.program_name}:{location.line}:{location.column}: ", message
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

        if current_character == "u":
            if self.peek() != "{":
                self.fail("expected { after unicode escape \\u", escape_start)

            self.advance()

            digits = ""
            while self.peek() != "" and self.peek() in VALID_HEXADECIMAL_DIGITS:
                digits += self.advance()

            if self.peek() != "}":
                self.fail("unterminated unicode escape \\u{...}", escape_start)

            self.advance()

            if not 1 <= len(digits) <= 6:
                self.fail(
                    "unicode escape \\u{...} "
                    "must contain one to six hexadecimal digits",
                    escape_start,
                )

            code = int(digits, 16)
            if 0xD800 <= code <= 0xDFFF:
                self.fail(
                    f"U+{code:X} is a surrogate, not a Unicode scalar value",
                    escape_start,
                )
            if code > 0x10FFFF:
                self.fail(
                    f"U+{code:X} is out of range; the maximum is U+10FFFF", escape_start
                )

            return chr(code)

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

    def lex_number(self, token_start: Location) -> Token:
        token_text = self.find_end_of_token(token_start)
        type, value = NumberParser(token_text, token_start, self.program_name).parse()
        return self.emit_token(type, token_start, value)

    def lex_string(self, token_start: Location) -> Token:
        string_buffer = self.read_quoted(token_start, '"', "string")
        return self.emit_token(TokenType.STRING, token_start, "".join(string_buffer))

    def lex_character(self, token_start: Location) -> Token:
        character_buffer = self.read_quoted(token_start, "'", "character")
        if len(character_buffer) == 0:
            self.fail("empty character literal", token_start)
        if len(character_buffer) > 1:
            self.fail(
                "a character literal may only contain a single scalar value; "
                f"found `{''.join(character_buffer)}`",
                token_start,
            )
        return self.emit_token(TokenType.CHARACTER, token_start, character_buffer[0])

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
                f"`{token_text}` is not a valid name; "
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
        if current_character == "'":
            return self.lex_character(token_start)
        if is_decimal_digit(current_character) or (
            current_character in ("+", "-") and is_decimal_digit(self.peek(1))
        ):
            return self.lex_number(token_start)

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
