from typing import Any

import pytest

from inde import ast
from inde.ast import (
    ArrayLiteral,
    Body,
    BooleanLiteral,
    CharacterLiteral,
    ConstantDeclaration,
    ConstantReference,
    Constraint,
    ConstraintReference,
    ConstructorPattern,
    Declaration,
    FloatLiteral,
    If,
    Import,
    IntegerLiteral,
    Item,
    Let,
    LiteralPattern,
    Match,
    MatchArm,
    Propagate,
    Quotation,
    QuotationType,
    StackEffect,
    StackType,
    StringLiteral,
    TypeDeclaration,
    TypeReference,
    TypeVariable,
    Variant,
    While,
    WordDeclaration,
    WordReference,
)
from inde.errors import IndeError, IndeErrors
from inde.lexer import (
    NumberType,
    tokenize,
)
from inde.parser import Parser, collect_type_arguments, parse


def single_declaration(source: str) -> Declaration:
    program = parse(source)
    assert len(program.declarations) == 1
    return program.declarations[0]


def signature(text: str) -> StackEffect:
    sig = single_declaration(f"word t {text} end").signature  # type: ignore[union-attr]
    return sig


def body(text: str) -> Declaration:
    return single_declaration(f"word t ( -> ) {text} end").body  # type: ignore[union-attr, return-value]


def get_failure_message(source: str) -> str:
    with pytest.raises(IndeErrors) as err:
        parse(source)

    return err.value.errors[0].message


def list_errors(source: str) -> tuple[list[str], list[str]]:
    parser = Parser(tokenize(source))
    program = parser.parse_program()
    return (
        [declaration.name for declaration in program.declarations],
        [error.message for error in parser.errors],
    )


def i32(number: int) -> IntegerLiteral:
    return IntegerLiteral(value=number, type=NumberType.I32)


def t(name: str, *args: Any) -> TypeReference:
    return TypeReference(name=name, args=tuple(args))


def stack(*types: Any, row: str | None = None) -> StackType:
    return StackType(row=row, types=tuple(types))


def effect(inputs: StackType, outputs: StackType) -> StackEffect:
    return StackEffect(inputs=inputs, outputs=outputs)


@pytest.mark.parametrize("source", ["", "  \n ", "; only a comment", ";( a block ;)"])
def test_empty_program(source: str) -> None:
    assert parse(source).declarations == ()


def test_word_definition() -> None:
    declaration = single_declaration("word square ( i32 -> i32 ) dup * end")
    assert declaration == WordDeclaration(
        name="square",
        signature=effect(stack(t("i32")), stack(t("i32"))),
        body=(WordReference(name="dup"), WordReference(name="*")),
    )


def test_declarations_keep_their_order() -> None:
    program = parse("word b ( -> ) end word a ( -> ) end word c ( -> ) end")
    assert [declaration.name for declaration in program.declarations] == ["b", "a", "c"]


def test_spans_point_at_the_source() -> None:
    declaration = single_declaration("word square ( i32 -> i32 )\n  dup *\nend")
    assert (declaration.location.line, declaration.location.column) == (1, 1)  # type: ignore[union-attr]
    assert (declaration.body[0].location.line, declaration.body[0].location.column) == (  # type: ignore[union-attr]
        2,
        3,
    )
    assert (declaration.body[1].location.line, declaration.body[1].location.column) == (  # type: ignore[union-attr]
        2,
        7,
    )


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("word Square ( -> ) end", "lowercase"),
        ("word T ( -> ) end", "lowercase"),
        ("word let ( -> ) end", "keyword"),
        ("word 5 ( -> ) end", "expected a word name"),
        ("word a dup end", "`(` to start the signature"),
        ("word a ( -> ) dup", "expected `end`"),
        ("word a ( -> ) dup else end", "expected `end`"),
    ],
)
def test_word_definition_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def test_missing_end_names_the_word_and_line() -> None:
    message = get_failure_message("word first ( -> )\n  dup\n\nword second ( -> ) end")
    assert "`first`" in message
    assert "line 1" in message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("( -> )", effect(stack(), stack())),
        ("( i32 i32 -> i32 )", effect(stack(t("i32"), t("i32")), stack(t("i32")))),
        ("( u8 -> bool )", effect(stack(t("u8")), stack(t("bool")))),
        ("( Shape -> f32 )", effect(stack(t("Shape")), stack(t("f32")))),
        (
            "( T -> T T )",
            effect(
                stack(TypeVariable(name="T")),
                stack(TypeVariable(name="T"), TypeVariable(name="T")),
            ),
        ),
        ("( ..a -> ..a )", effect(stack(row="a"), stack(row="a"))),
        (
            "( ..a Q1 -> ..b )",
            effect(stack(TypeVariable(name="Q1"), row="a"), stack(row="b")),
        ),
        ("(i32 -> i32)", effect(stack(t("i32")), stack(t("i32")))),
    ],
)
def test_signatures(text: str, expected: StackEffect) -> None:
    assert signature(text) == expected


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("( i32 i32 )", "expected `->`"),
        ("( i32 -> i32", "expected `)`"),
        ("( ..a i32 -> i32 )", "both sides"),
        ("( i32 -> ..a )", "both sides"),
        ("( i32 ..a -> ..a )", "must precede"),
        ("( i32 -> foo )", "`foo` is not a type"),
        ("( 5 -> )", "expected `->`"),
        ("(i32->i32)", "put spaces around `->`"),
    ],
)
def test_signature_errors(text: str, fragment: str) -> None:
    assert fragment in get_failure_message(f"word t {text} end")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", i32(42)),
        ("255u8", IntegerLiteral(value=255, type=NumberType.U8)),
        ("1.5", FloatLiteral(value=1.5, type=NumberType.F32)),
        ("2.5f64", FloatLiteral(value=2.5, type=NumberType.F64)),
        ("'a'", CharacterLiteral(value="a")),
        ('"hi"', StringLiteral(value="hi")),
        ("true", BooleanLiteral(value=True)),
        ("false", BooleanLiteral(value=False)),
        ("dup", WordReference(name="dup")),
        ("+.wrap", WordReference(name="+.wrap")),
        ("str-len", WordReference(name="str-len")),
    ],
)
def test_single_items(text: str, expected: Item) -> None:
    assert body(text) == (expected,)  # type: ignore[comparison-overlap]


def test_empty_body() -> None:
    assert body("") == ()  # type: ignore[comparison-overlap]


def test_code_outside_a_declaration() -> None:
    assert "outside a declaration" in get_failure_message("1 2 + print")


def test_stray_end() -> None:
    assert "does not close anything" in get_failure_message("word a ( -> ) end end")


def test_error_per_broken_declaration() -> None:
    source = """
    word broken1 b end
    word ok1 ( -> ) dup end
    7 8 9
    word broken2 ( -> x ) end
    word ok2 ( -> ) end
    """
    names, errors = list_errors(source)
    assert names == ["ok1", "ok2"]
    assert len(errors) == 3


def test_missing_end_does_not_consume_next_word() -> None:
    names, errors = list_errors("word a ( -> ) dup\nword b ( -> ) swap end")
    assert names == ["b"]
    assert len(errors) == 1


def test_collect_type_arguments() -> None:
    arguments = collect_type_arguments(
        tokenize(
            "word w ( Tagged[i32, u8] -> ) end\n"
            "type Tagged[T, U] = tagged( T U )\n"
            "type Shape = circle( f32 )"
        )
    )
    assert arguments["Tagged"] == 2
    assert arguments["Shape"] == 0
    assert arguments["Option"] == 1
    assert arguments["array"] == 1
    assert arguments["i32"] == 0


def test_type_arguments() -> None:
    assert signature("( Option[i32] -> array[u8] )") == effect(
        stack(t("Option", t("i32"))), stack(t("array", t("u8")))
    )
    assert signature("( Result[array[u8], str] -> )") == effect(
        stack(t("Result", t("array", t("u8")), t("str"))), stack()
    )


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("( Option -> )", "write `Option[...]`"),
        ("( Result[i32] -> )", "found 1"),
        ("( array[u8 -> )", "type arguments of `array`"),
    ],
)
def test_type_argument_errors(text: str, fragment: str) -> None:
    assert fragment in get_failure_message(f"word t {text} end")


def test_quotation_types() -> None:
    assert signature("( [ i32 -> i32 ] -> linear [ ..a -> ..a ] )") == effect(
        stack(QuotationType(effect=effect(stack(t("i32")), stack(t("i32"))))),
        stack(
            QuotationType(effect=effect(stack(row="a"), stack(row="a")), linear=True)
        ),
    )

    assert signature("( i32 [ -> ] -> )") == effect(
        stack(t("i32"), QuotationType(effect=effect(stack(), stack()))), stack()
    )


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("( [ i32 -> i32 -> )", "expected `]`"),
        ("( linear i32 -> )", "`[` after `linear`"),
    ],
)
def test_quotation_type_errors(text: str, fragment: str) -> None:
    assert fragment in get_failure_message(f"word t {text} end")


def test_where_clauses() -> None:
    declaration = single_declaration(
        "word w ( T Q -> T ) where T: Eq & Unrestricted, Q: [ T -> T ] end"
    )
    assert declaration.where == (  # type: ignore[union-attr]
        Constraint(
            var="T",
            bounds=(
                ConstraintReference(name="Eq"),
                ConstraintReference(name="Unrestricted"),
            ),
        ),
        Constraint(
            var="Q",
            bounds=(
                QuotationType(
                    effect=effect(
                        stack(TypeVariable(name="T")), stack(TypeVariable(name="T"))
                    )
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("clause", "fragment"),
    [
        ("where T Eq", "`:` after `T`"),
        ("where T: Eq | Unrestricted", "`&`, not `|`"),
        ("where x: Eq", "type variable"),
    ],
)
def test_where_clause_errors(clause: str, fragment: str) -> None:
    assert fragment in get_failure_message(f"word t ( T -> T ) {clause} end")


def test_type_declarations() -> None:
    assert single_declaration(
        "type Shape = circle( f32 ) | rect( f32 f32 )"
    ) == TypeDeclaration(
        name="Shape",
        variants=(
            Variant(name="circle", fields=(t("f32"),)),
            Variant(name="rect", fields=(t("f32"), t("f32"))),
        ),
    )
    assert single_declaration("type Maybe[T] = nothing | just( T )") == TypeDeclaration(
        name="Maybe",
        params=("T",),
        variants=(
            Variant(name="nothing"),
            Variant(name="just", fields=(TypeVariable(name="T"),)),
        ),
    )
    assert single_declaration("linear type Handle = handle( u32 )") == TypeDeclaration(
        name="Handle",
        variants=(Variant(name="handle", fields=(t("u32"),)),),
        linear=True,
    )


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("type T1 = t1( i32 )", "type variable"),
        ("type shape = circle( f32 )", "start with an uppercase letter"),
        ("type Shape circle( f32 )", "`=` before the variants of `Shape`"),
        (
            "type Shape = Circle( f32 )",
            "constructor names start with a lowercase letter",
        ),
        ("type Empty = e( )", "lists no fields"),
    ],
)
def test_type_declaration_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("?", (Propagate(),)),
        ("[ 1 + ]", (Quotation(body=(i32(1), WordReference(name="+"))),)),
        ("[ [ ] ]", (Quotation(body=(Quotation(body=()),)),)),
        (
            "{ 1 { 2 } }",
            (ArrayLiteral(elements=(i32(1), ArrayLiteral(elements=(i32(2),)))),),
        ),
        ("{ }", (ArrayLiteral(elements=()),)),
        (
            "true if 1 else 2 end",
            (BooleanLiteral(value=True), If(then_body=(i32(1),), else_body=(i32(2),))),
        ),
        ("if end", (If(then_body=()),)),
        ("if else end", (If(then_body=(), else_body=()),)),
        (
            "while dup do 1 end",
            (While(condition=(WordReference(name="dup"),), body=(i32(1),)),),
        ),
        ("let x y in x end", (Let(names=("x", "y"), body=(WordReference(name="x"),)),)),
        (
            "match case circle dup case rect drop end",
            (
                Match(
                    arms=(
                        MatchArm(
                            pattern=ConstructorPattern(name="circle"),
                            body=(WordReference(name="dup"),),
                        ),
                        MatchArm(
                            pattern=ConstructorPattern(name="rect"),
                            body=(WordReference(name="drop"),),
                        ),
                    )
                ),
            ),
        ),
        (
            "match case 'a' 1 else 0 end",
            (
                Match(
                    arms=(
                        MatchArm(
                            pattern=LiteralPattern(literal=CharacterLiteral(value="a")),
                            body=(i32(1),),
                        ),
                    ),
                    else_body=(i32(0),),
                ),
            ),
        ),
        (
            "if while true do end end",
            (If(then_body=(While(condition=(BooleanLiteral(value=True),), body=()),)),),
        ),
    ],
)
def test_body_forms(text: str, expected: tuple[Any, ...]) -> None:
    assert body(text) == expected  # type: ignore[comparison-overlap]


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("word t ( -> ) if 1", "close the `if`"),
        ("word t ( -> ) [ 1 + end", "close the quotation"),
        ("word t ( -> ) { 1 2", "close the array literal"),
        ("word t ( -> ) while true 1 end end", "expected `do`"),
        ("word t ( -> ) let x 1 end end", "expected `in`"),
        ("word t ( -> ) let in 1 end end", "at least one name"),
        ("word t ( -> ) match end end", "at least one `case`"),
        ("word t ( -> ) match case 1.5 0 else 1 end end", "floats cannot"),
        ("word t ( -> ) match case [ end end", "expected a pattern"),
        ("word t ( -> ) { dup } end", "must be literals"),
    ],
)
def test_body_form_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def repl(line: str) -> Declaration | Import | Body:
    return Parser(tokenize(line)).parse_repl_line()


def test_repl_lines() -> None:
    assert repl("1 2 +") == (i32(1), i32(2), WordReference(name="+"))
    assert repl("") == ()
    assert repl("word sq ( i32 -> i32 ) dup * end") == WordDeclaration(
        name="sq",
        signature=effect(stack(t("i32")), stack(t("i32"))),
        body=(WordReference(name="dup"), WordReference(name="*")),
    )


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        ("1 end", "does not close anything"),
        ("1 )", "unexpected `)`"),
        ("word t ( -> ) end dup", "unexpected name `dup`"),
    ],
)
def test_repl_errors(line: str, fragment: str) -> None:
    with pytest.raises(IndeError) as info:
        repl(line)
    assert fragment in info.value.message


def test_constants() -> None:
    assert single_declaration("const NEWLINE = '\\n'") == ConstantDeclaration(
        name="NEWLINE", value=CharacterLiteral(value="\n")
    )
    assert single_declaration("const MAX_DEPTH = 10000u32") == ConstantDeclaration(
        name="MAX_DEPTH", value=IntegerLiteral(value=10000, type=NumberType.U32)
    )
    assert single_declaration("const PAIR = { 1 { 2 } }") == ConstantDeclaration(
        name="PAIR",
        value=ArrayLiteral(elements=(i32(1), ArrayLiteral(elements=(i32(2),)))),
    )

    assert body("NEWLINE print") == (  # type: ignore[comparison-overlap]
        ConstantReference(name="NEWLINE"),
        WordReference(name="print"),
    )
    assert body("{ NEWLINE 'a' }") == (  # type: ignore[comparison-overlap]
        ArrayLiteral(
            elements=(ConstantReference(name="NEWLINE"), CharacterLiteral(value="a"))
        ),
    )
    assert body("match case NEWLINE 1 else 0 end") == (  # type: ignore[comparison-overlap]
        Match(
            arms=(
                MatchArm(
                    pattern=LiteralPattern(literal=ConstantReference(name="NEWLINE")),
                    body=(i32(1),),
                ),
            ),
            else_body=(i32(0),),
        ),
    )

    names, errors = list_errors(
        "word broken ( -> end\nconst ONE = 1\nword ok ( -> ) ONE drop end"
    )
    assert names == ["ONE", "ok"]
    assert len(errors) == 1
    assert repl("const ONE = 1") == ConstantDeclaration(name="ONE", value=i32(1))


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("const KIB = 1024 1024 *", "single literal"),
        ("const SIZE = dup", "single literal"),
        ("const TWO = ONE", "not another constant"),
        ("const PAIR = { NEWLINE '\\t' }", "not another constant"),
        ("const PAIR = { { NEWLINE } }", "not another constant"),
        ("const Newline = '\\n'", "constant names are uppercase"),
        ("const newline = '\\n'", "constant names are uppercase"),
        ("const X1 = 5", "type variable"),
        ("const NEWLINE '\\n'", "`=` after `NEWLINE`"),
        ("const", "expected a constant name, found end of file"),
        ("word NEWLINE ( -> ) end", "found constant name `NEWLINE`"),
        ("type IO = io", "found constant name `IO`"),
    ],
)
def test_constant_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        (
            "word a ( i32 i32 ) end",
            "expected `->` between the inputs and the outputs, found `)`",
        ),
        ("word a ( -> i32", "expected `)`, found end of file"),
        ("type Empty = e( )", "write `e` for a variant without fields"),
        ("7", "only `word`, `type` and `const` declarations"),
    ],
)
def test_messages_are_fully_formatted(source: str, fragment: str) -> None:
    message = get_failure_message(source)
    assert fragment in message
    assert "{" not in message


def test_dump_omits_locations() -> None:
    text = ast.dump(parse("word sq ( i32 -> i32 ) dup * end"))
    assert "location" not in text
    assert "WordDeclaration name='sq'" in text
    assert "WordReference name='dup'" in text
