from pathlib import Path
from typing import NoReturn

from inde import logger
from inde.ast import (
    ArrayLiteral,
    Body,
    BooleanLiteral,
    Bound,
    CharacterLiteral,
    ConstantDeclaration,
    ConstantReference,
    Constraint,
    ConstraintReference,
    ConstructorPattern,
    Declaration,
    FloatLiteral,
    If,
    IntegerLiteral,
    Item,
    Let,
    Literal,
    LiteralPattern,
    Match,
    MatchArm,
    Pattern,
    Program,
    Propagate,
    Quotation,
    QuotationType,
    StackEffect,
    StackType,
    StringLiteral,
    TypeDeclaration,
    TypeExpression,
    TypeReference,
    TypeVariable,
    Variant,
    While,
    WordDeclaration,
    WordReference,
)
from inde.lexer import (
    VALID_DELIMITERS,
    Location,
    Token,
    TokenType,
    tokenize,
)

CORE_TYPE_ARGUMENTS: dict[str, int] = {
    "i8": 0,
    "i16": 0,
    "i32": 0,
    "i64": 0,
    "u8": 0,
    "u16": 0,
    "u32": 0,
    "u64": 0,
    "f32": 0,
    "f64": 0,
    "bool": 0,
    "char": 0,
    "str": 0,
    "strbuf": 0,
    "unit": 0,
    "array": 1,
    "arraybuf": 1,
}

CORE_LIBRARY_ARGUMENTS: dict[str, int] = {"Option": 1, "Result": 2, "Pair": 2}


def collect_type_arguments(tokens: list[Token]) -> dict[str, int]:
    arguments = dict(CORE_TYPE_ARGUMENTS)
    arguments.update(CORE_LIBRARY_ARGUMENTS)
    n = len(tokens)
    for i, token in enumerate(tokens):
        if not (token.type is TokenType.KEYWORD and token.text == "type"):
            continue
        if i + 1 >= n or tokens[i + 1].type is not TokenType.TYPE_NAME:
            continue
        count = 0
        j = i + 2
        if j < n and tokens[j].type is TokenType.LEFT_BRACKET:
            j += 1
            while j < n and tokens[j].type is TokenType.TYPE_VARIABLE:
                count += 1
                j += 1
                if j < n and tokens[j].type is TokenType.COMMA:
                    j += 1
                else:
                    break
        arguments[tokens[i + 1].text] = count
    return arguments


DELIMITERS_TEXT: dict[TokenType, str] = {
    type: character for character, type in VALID_DELIMITERS.items()
}

TYPE_WORDS: dict[TokenType, str] = {
    TokenType.NAME: "name",
    TokenType.TYPE_NAME: "type name",
    TokenType.TYPE_VARIABLE: "type variable",
    TokenType.ROW_VARIABLE: "row variable",
    TokenType.CONSTANT_NAME: "constant name",
    TokenType.KEYWORD: "keyword",
    TokenType.INTEGER: "integer literal",
    TokenType.FLOAT: "float literal",
    TokenType.CHARACTER: "character literal",
    TokenType.STRING: "string literal",
    TokenType.BOOLEAN: "boolean literal",
}

LITERAL_TYPES: tuple[TokenType, ...] = (
    TokenType.INTEGER,
    TokenType.FLOAT,
    TokenType.CHARACTER,
    TokenType.STRING,
    TokenType.BOOLEAN,
)


class Parser:
    def __init__(self, tokens: list[Token], program_name: str) -> None:
        self.tokens = tokens
        self.program_name = program_name
        self.position = 0
        self.errors: list[SystemExit] = []
        self.arguments = collect_type_arguments(tokens)
        self.location = Location(0, 0, 0)

    def peek(self, ahead: int = 0) -> Token:
        return self.tokens[min(self.position + ahead, len(self.tokens) - 1)]

    def advance(self) -> Token:
        token = self.peek()
        if token.type is not TokenType.END_OF_FILE:
            self.position += 1
        return token

    def error(self, message: str, token: Token | None = None) -> SystemExit:
        _location = (token or self.peek()).location
        return SystemExit(
            logger.format_error(
                f"{self.program_name}:{_location.line}:{_location.column}: ",
                message,
            )
        )

    def fail(self, message: str, token: Token | None = None) -> NoReturn:
        raise self.error(message, token)

    def describe_token(self, token: Token) -> str:
        if token.type is TokenType.END_OF_FILE:
            return "end of file"
        if token.type is TokenType.SYMBOL or token.type in DELIMITERS_TEXT:
            return f"`{token.text}`"
        return f"{TYPE_WORDS[token.type]} `{token.text}`"

    def pluralize(self, count: int, noun: str) -> str:
        return f"{count} {noun}" if count == 1 else f"{count} {noun}s"

    def at_token(self, type: TokenType, text: str | None = None) -> bool:
        token = self.peek()
        return token.type is type and (text is None or token.text == text)

    def at_keyword(self, text: str) -> bool:
        return self.at_token(TokenType.KEYWORD, text)

    def accept_token(self, type: TokenType, text: str | None = None) -> Token | None:
        return self.advance() if self.at_token(type, text) else None

    def expect_token(
        self,
        type: TokenType,
        text: str | None = None,
        error_description: str | None = None,
    ) -> Token:
        if self.at_token(type, text):
            return self.advance()
        if error_description is None:
            if text is not None:
                error_description = f"`{text}`"
            elif type in DELIMITERS_TEXT:
                error_description = f"`{DELIMITERS_TEXT[type]}`"
            else:
                error_description = f"a {TYPE_WORDS[type]}"
        self.fail(
            f"expected {error_description}, found {{self.describe_token(self.peek())}}"
        )

    def expect_end(self, construct: str, opener: Token) -> Token:
        if self.at_keyword("end"):
            return self.advance()
        self.fail(
            f"expected `end` to close `{construct}` on line {opener.location.line}"
        )

    def expect_closing(self, type: TokenType, construct: str, opener: Token) -> Token:
        if self.at_token(type):
            return self.advance()
        self.fail(
            f"expected `{DELIMITERS_TEXT[type]}` to close `{construct}` on "
            f"line {opener.location.line}"
        )

    def at_declaration_start(self) -> bool:
        return (
            self.at_keyword("word")
            or self.at_keyword("type")
            or self.at_keyword("const")
            or (
                self.at_keyword("linear")
                and self.peek(1).type is TokenType.KEYWORD
                and self.peek(1).text == "type"
            )
        )

    def synchronize(self) -> None:
        while (
            not self.at_token(TokenType.END_OF_FILE) and not self.at_declaration_start()
        ):
            self.advance()

    def parse_program(self) -> Program:
        start = self.peek()
        declarations: list[Declaration] = []
        while not self.at_token(TokenType.END_OF_FILE):
            if not self.at_declaration_start():
                token = self.peek()
                if token.type is TokenType.KEYWORD and token.text == "end":
                    self.errors.append(self.error("`end` does not close anything here"))
                else:
                    self.errors.append(
                        self.error(
                            f"code outside a declaration, "
                            f"found {self.describe_token(token)}; "
                            "a file contains only `word` and `type` declarations"
                        )
                    )
                self.advance()
                self.synchronize()
                continue
            before = self.position
            try:
                declarations.append(self.parse_declaration())
            except SystemExit as error:
                self.errors.append(error)
                if self.position == before:
                    self.advance()
                self.synchronize()
        return Program(declarations=tuple(declarations), location=start.location)

    def parse_declaration(self) -> Declaration:
        if self.at_keyword("word"):
            return self.parse_word_declaration()
        if self.at_keyword("const"):
            return self.parse_constant_declaration()
        return self.parse_type_declaration()

    def parse_constant_declaration(self) -> ConstantDeclaration:
        start = self.expect_token(TokenType.KEYWORD, "const")
        name = self.peek()
        if name.type is TokenType.TYPE_VARIABLE:
            self.fail(
                f"`{name.text}` has the form of a type variable; choose a longer name"
            )
        if name.type in (TokenType.NAME, TokenType.TYPE_NAME):
            self.fail(f"constant names are uppercase, found `{name.text}`")
        self.expect_token(TokenType.CONSTANT_NAME, error_description="a constant name")
        self.expect_token(
            TokenType.SYMBOL, "=", error_description=f"`=` after `{name.text}`"
        )

        token = self.peek()
        if token.type is TokenType.CONSTANT_NAME:
            self.fail(
                "a constant's value is a literal, not another constant "
                f"(`{token.text}`)"
            )
        if not (token.type in LITERAL_TYPES or token.type is TokenType.LEFT_BRACE):
            self.fail(
                "a constant's value is a single literal, "
                f"found {self.describe_token(token)}"
            )
        value = self.parse_literal(consts_allowed=False)

        if self.starts_item():
            self.fail(
                "a constant's value is a single literal, "
                f"found {self.describe_token(self.peek())}"
            )
        return ConstantDeclaration(name=name.text, value=value, location=start.location)

    def parse_word_declaration(self) -> WordDeclaration:
        start = self.expect_token(TokenType.KEYWORD, "word")
        name = self.expect_word_name()
        signature = self.parse_signature()
        where = self.parse_where() if self.at_keyword("where") else ()
        body = self.parse_body()
        self.expect_end(f"definition of `{name.text}`", start)
        return WordDeclaration(
            name=name.text,
            signature=signature,
            where=where,
            body=body,
            location=start.location,
        )

    def expect_word_name(self) -> Token:
        token = self.peek()
        if token.type is TokenType.NAME:
            return self.advance()
        if token.type in (TokenType.TYPE_NAME, TokenType.TYPE_VARIABLE):
            self.fail(
                f"invalid word name `{token.text}`; "
                "word names must start with a lowercase letter or a symbol"
            )
        if token.type is TokenType.KEYWORD:
            self.fail(f"keyword `{token.text}` cannot be a word name")
        self.fail(f"expected a word name, found {self.describe_token(token)}")

    def parse_signature(self) -> StackEffect:
        opener = self.expect_token(
            TokenType.LEFT_PARENTHESIS, error_description="`(` to start the signature"
        )
        return self.parse_effect(opener, TokenType.RIGHT_PARENTHESIS)

    def parse_effect(self, opener: Token, closer: TokenType) -> StackEffect:
        inputs = self.parse_stack()
        self.expect_token(
            TokenType.SYMBOL,
            "->",
            error_description="`->` between the inputs and the outputs",
        )
        outputs = self.parse_stack()
        self.expect_token(closer)
        if (inputs.row is None) != (outputs.row is None):
            self.fail(
                "both sides of a stack effect must name a row variable",
                opener,
            )
        return StackEffect(inputs=inputs, outputs=outputs, location=opener.location)

    def parse_stack(self) -> StackType:
        start = self.peek()
        row = None
        if self.at_token(TokenType.ROW_VARIABLE):
            row = self.advance().text[2:]
        types: list[TypeExpression] = []
        while self.starts_type():
            types.append(self.parse_type())
        if self.at_token(TokenType.ROW_VARIABLE):
            self.fail("a row variable must precede the types of its stack")
        return StackType(row=row, types=tuple(types), location=start.location)

    def starts_type(self) -> bool:
        return self.peek().type in (
            TokenType.TYPE_NAME,
            TokenType.TYPE_VARIABLE,
            TokenType.NAME,
            TokenType.LEFT_BRACKET,
        ) or self.at_keyword("linear")

    def parse_type(self) -> TypeExpression:
        token = self.peek()
        if token.type is TokenType.TYPE_VARIABLE:
            self.advance()
            return TypeVariable(name=token.text, location=token.location)
        if token.type in (TokenType.TYPE_NAME, TokenType.NAME):
            return self.parse_named_type()
        if token.type is TokenType.LEFT_BRACKET or self.at_keyword("linear"):
            return self.parse_quotation_type()
        self.fail(f"expected a type, found {self.describe_token(token)}")

    def parse_quotation_type(self) -> QuotationType:
        start = self.peek()
        linear = self.accept_token(TokenType.KEYWORD, "linear") is not None
        opener = self.expect_token(
            TokenType.LEFT_BRACKET, error_description="`[` after `linear`"
        )
        effect = self.parse_effect(opener, TokenType.RIGHT_BRACKET)
        return QuotationType(effect=effect, linear=linear, location=start.location)

    def parse_named_type(self) -> TypeReference:
        token = self.advance()
        if token.type is TokenType.NAME and "->" in token.text:
            self.fail(f"`{token.text}` is a single name; put spaces around `->`", token)
        if token.type is TokenType.NAME and token.text not in CORE_TYPE_ARGUMENTS:
            self.fail(
                f"`{token.text}` is not a type; "
                "declared type names start with an uppercase letter",
                token,
            )
        argument_count = self.arguments.get(token.text, 0)
        if argument_count == 0:
            return TypeReference(name=token.text, location=token.location)

        if not self.at_token(TokenType.LEFT_BRACKET):
            self.fail(
                f"`{token.text}` "
                f"takes {self.pluralize(argument_count, 'type argument')}, "
                f"found {self.describe_token(self.peek())}; "
                f"write `{token.text}[...]`"
            )
        opener = self.advance()
        args = [self.parse_type()]
        while self.accept_token(TokenType.COMMA):
            args.append(self.parse_type())
        self.expect_closing(
            TokenType.RIGHT_BRACKET, f"type arguments of `{token.text}`", opener
        )
        if len(args) != argument_count:
            self.fail(
                f"`{token.text}` takes "
                f"{self.pluralize(argument_count, 'type argument')}, "
                f"found {len(args)}",
                token,
            )
        return TypeReference(name=token.text, args=tuple(args), location=token.location)

    def parse_where(self) -> tuple[Constraint, ...]:
        self.expect_token(TokenType.KEYWORD, "where")
        constraints = [self.parse_constraint()]
        while self.accept_token(TokenType.COMMA):
            constraints.append(self.parse_constraint())
        return tuple(constraints)

    def parse_constraint(self) -> Constraint:
        token = self.peek()
        if token.type is not TokenType.TYPE_VARIABLE:
            self.fail(
                f"expected a type variable such as `T` to constrain, "
                f"found {self.describe_token(token)}"
            )
        self.advance()
        self.expect_token(
            TokenType.COLON, error_description=f"`:` after `{token.text}`"
        )
        bounds = [self.parse_bound()]
        while self.accept_token(TokenType.SYMBOL, "&"):
            bounds.append(self.parse_bound())
        if self.at_token(TokenType.SYMBOL, "|"):
            self.fail("combine constraints with `&`, not `|`")
        return Constraint(var=token.text, bounds=tuple(bounds), location=token.location)

    def parse_bound(self) -> Bound:
        token = self.peek()
        if token.type is TokenType.TYPE_NAME:
            self.advance()
            return ConstraintReference(name=token.text, location=token.location)
        if token.type is TokenType.LEFT_BRACKET or self.at_keyword("linear"):
            return self.parse_quotation_type()
        self.fail(
            f"expected a constraint such as `Eq`, found {self.describe_token(token)}"
        )

    def parse_type_declaration(self) -> TypeDeclaration:
        start = self.peek()
        linear = self.accept_token(TokenType.KEYWORD, "linear") is not None
        self.expect_token(TokenType.KEYWORD, "type")

        name = self.peek()
        if name.type is TokenType.TYPE_VARIABLE:
            self.fail(
                f"`{name.text}` has the form of a type variable, choose a longer name"
            )
        if name.type is TokenType.NAME:
            self.fail(f"type names start with an uppercase letter, found `{name.text}`")
        self.expect_token(TokenType.TYPE_NAME, error_description="a type name")

        params: list[str] = []
        if self.at_token(TokenType.LEFT_BRACKET):
            opener = self.advance()
            params.append(
                self.expect_token(
                    TokenType.TYPE_VARIABLE,
                    error_description="a type variable such as `T`",
                ).text
            )
            while self.accept_token(TokenType.COMMA):
                params.append(
                    self.expect_token(
                        TokenType.TYPE_VARIABLE,
                        error_description="a type parameter such as `T`",
                    ).text
                )
            self.expect_closing(
                TokenType.RIGHT_BRACKET, f"parameters of `{name.text}`", opener
            )

        self.expect_token(
            TokenType.SYMBOL,
            "=",
            error_description=f"`=` before the variants of `{name.text}`",
        )
        variants = [self.parse_variant()]
        while self.accept_token(TokenType.SYMBOL, "|"):
            variants.append(self.parse_variant())
        return TypeDeclaration(
            name=name.text,
            params=tuple(params),
            variants=tuple(variants),
            linear=linear,
            location=start.location,
        )

    def parse_variant(self) -> Variant:
        name = self.peek()
        if name.type in (TokenType.TYPE_NAME, TokenType.TYPE_VARIABLE):
            self.fail(
                f"constructor names start with a lowercase letter, found `{name.text}`"
            )
        self.expect_token(TokenType.NAME, error_description="a constructor name")
        fields: list[TypeExpression] = []
        if self.accept_token(TokenType.LEFT_PARENTHESIS):
            if not self.starts_type():
                self.fail(
                    f"`{name.text}( )` lists no fields; "
                    "write `{name.text}` for a variant without fields"
                )
            while self.starts_type():
                fields.append(self.parse_type())
            self.expect_token(
                TokenType.RIGHT_PARENTHESIS,
                error_description=f"`)` to close the fields of `{name.text}`",
            )
        return Variant(name=name.text, fields=tuple(fields), location=name.location)

    def starts_item(self) -> bool:
        token = self.peek()
        if token.type in LITERAL_TYPES or token.type in (
            TokenType.NAME,
            TokenType.CONSTANT_NAME,
            TokenType.LEFT_BRACKET,
            TokenType.LEFT_BRACE,
        ):
            return True
        if token.type is TokenType.SYMBOL:
            return token.text == "?"
        if token.type is TokenType.KEYWORD:
            return token.text in ("if", "while", "match", "let")
        return False

    def parse_body(self) -> Body:
        items: list[Item] = []
        while self.starts_item():
            items.append(self.parse_item())
        return tuple(items)

    def parse_item(self) -> Item:
        token = self.peek()
        if token.type in LITERAL_TYPES or token.type is TokenType.LEFT_BRACE:
            return self.parse_literal()
        if token.type is TokenType.NAME:
            self.advance()
            return WordReference(name=token.text, location=token.location)
        if token.type is TokenType.SYMBOL and token.text == "?":
            self.advance()
            return Propagate(location=token.location)
        if token.type is TokenType.LEFT_BRACKET:
            return self.parse_quotation()
        if token.type is TokenType.CONSTANT_NAME:
            self.advance()
            return ConstantReference(name=token.text, location=token.location)
        if self.at_keyword("if"):
            return self.parse_if()
        if self.at_keyword("while"):
            return self.parse_while()
        if self.at_keyword("match"):
            return self.parse_match()
        if self.at_keyword("let"):
            return self.parse_let()
        raise AssertionError(
            f"starts_item and parse_item mismatch on {self.describe_token(token)}"
        )

    def parse_literal(self, consts_allowed: bool = True) -> Literal:
        token = self.peek()
        if token.type is TokenType.LEFT_BRACE:
            return self.parse_array_literal(consts_allowed)
        self.advance()
        if token.type is TokenType.INTEGER:
            return IntegerLiteral(
                value=token.value.number,  # type: ignore[attr-defined]
                type=token.value.type,  # type: ignore[attr-defined]
                location=token.location,
            )
        if token.type is TokenType.FLOAT:
            return FloatLiteral(
                value=token.value.number,  # type: ignore[attr-defined]
                type=token.value.type,  # type: ignore[attr-defined]
                location=token.location,
            )
        if token.type is TokenType.CHARACTER:
            return CharacterLiteral(value=str(token.value), location=token.location)
        if token.type is TokenType.STRING:
            return StringLiteral(value=str(token.value), location=token.location)
        if token.type is TokenType.BOOLEAN:
            return BooleanLiteral(value=bool(token.value), location=token.location)
        self.fail(f"expected a literal, found {self.describe_token(token)}", token)

    def parse_array_literal(self, consts_allowed: bool = True) -> ArrayLiteral:
        opener = self.expect_token(TokenType.LEFT_BRACE)
        elements: list[Literal] = []
        while not self.at_token(TokenType.RIGHT_BRACE):
            token = self.peek()
            if token.type in LITERAL_TYPES or token.type is TokenType.LEFT_BRACE:
                elements.append(self.parse_literal(consts_allowed))
            elif token.type is TokenType.END_OF_FILE:
                break
            else:
                self.fail(
                    "array literal elements must be literals, "
                    f"found {self.describe_token(token)}"
                )
        self.expect_closing(TokenType.RIGHT_BRACE, "array literal", opener)
        return ArrayLiteral(elements=tuple(elements), location=opener.location)

    def parse_quotation(self) -> Quotation:
        opener = self.expect_token(TokenType.LEFT_BRACKET)
        body = self.parse_body()
        self.expect_closing(TokenType.RIGHT_BRACKET, "quotation", opener)
        return Quotation(body=body, location=opener.location)

    def parse_if(self) -> If:
        opener = self.expect_token(TokenType.KEYWORD, "if")
        then_body = self.parse_body()
        else_body = (
            self.parse_body() if self.accept_token(TokenType.KEYWORD, "else") else None
        )
        self.expect_end("`if`", opener)
        return If(then_body=then_body, else_body=else_body, location=opener.location)

    def parse_while(self) -> While:
        opener = self.expect_token(TokenType.KEYWORD, "while")
        condition = self.parse_body()
        if not self.at_keyword("do"):
            self.fail(
                f"expected `do` after `while` that starts on "
                f"line {opener.location.line}, "
                f"found {self.describe_token(self.peek())}"
            )
        self.advance()
        body = self.parse_body()
        self.expect_end("`while`", opener)
        return While(condition=condition, body=body, location=opener.location)

    def parse_let(self) -> Let:
        opener = self.expect_token(TokenType.KEYWORD, "let")
        names: list[str] = []
        while self.at_token(TokenType.NAME):
            names.append(self.advance().text)
        if not names:
            self.fail(
                f"`let` needs at least one name, "
                f"found {self.describe_token(self.peek())}"
            )
        if not self.at_keyword("in"):
            self.fail(
                f"expected `in` after bindings of `let` that starts on "
                f"line {opener.location.line}, "
                f"found {self.describe_token(self.peek())}"
            )
        self.advance()
        body = self.parse_body()
        self.expect_end("`let`", opener)
        return Let(names=tuple(names), body=body, location=opener.location)

    def parse_match(self) -> Match:
        opener = self.expect_token(TokenType.KEYWORD, "match")
        arms: list[MatchArm] = []
        while self.at_keyword("case"):
            case = self.advance()
            pattern = self.parse_pattern()
            arms.append(
                MatchArm(
                    pattern=pattern, body=self.parse_body(), location=case.location
                )
            )
        if not arms:
            self.fail(
                f"`match` needs at least one `case`, "
                f"found {self.describe_token(self.peek())}"
            )
        else_body = (
            self.parse_body() if self.accept_token(TokenType.KEYWORD, "else") else None
        )
        self.expect_end("`match`", opener)
        return Match(arms=tuple(arms), else_body=else_body, location=opener.location)

    def parse_pattern(self) -> Pattern:
        token = self.peek()
        if token.type is TokenType.NAME:
            self.advance()
            return ConstructorPattern(name=token.text, location=token.location)
        if token.type in (
            TokenType.INTEGER,
            TokenType.CHARACTER,
            TokenType.STRING,
            TokenType.BOOLEAN,
        ):
            return LiteralPattern(literal=self.parse_literal(), location=token.location)
        if token.type is TokenType.CONSTANT_NAME:
            self.advance()
            return LiteralPattern(
                literal=ConstantReference(name=token.text, location=token.location),
                location=token.location,
            )
        if token.type is TokenType.FLOAT:
            self.fail("floats cannot be matched against literals")
        self.fail(
            f"expected a pattern after `case`, found {self.describe_token(token)}"
        )

    def parse_repl_line(self) -> Declaration | Body:
        if self.at_declaration_start():
            result: Declaration | Body = self.parse_declaration()
        else:
            result = self.parse_body()
        if not self.at_token(TokenType.END_OF_FILE):
            token = self.peek()
            if token.type is TokenType.KEYWORD and token.text == "end":
                self.fail("`end` does not close anything here")
            self.fail(f"unexpected {self.describe_token(token)}")
        return result


def parse(source: str, program_name: str = Path(__file__).stem) -> Program:
    parser = Parser(tokenize(source, program_name), program_name)
    program = parser.parse_program()
    if parser.errors:
        raise parser.errors[0]
    return program
