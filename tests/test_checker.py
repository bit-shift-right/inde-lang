from collections.abc import Callable
from typing import Any, TypeVar

import pytest

from inde.checker import Checker, check
from inde.errors import IndeErrors
from inde.lexer import tokenize
from inde.parser import Parser, parse
from inde.type_model import BOOL, I32, STR, U8, Effect, Type, array_of, option_of

TestFunction = TypeVar("TestFunction", bound=Callable[..., Any])


def effect_of(source: str, name: str) -> str:
    return str(check(parse(source)).words[name].effect)


def get_failure_messages(source: str, require_main: bool = False) -> list[str]:
    with pytest.raises(IndeErrors) as err:
        check(parse(source), require_main)
    return [error.message for error in err.value.errors]


def get_failure_message(source: str) -> str:
    messages = get_failure_messages(source)
    assert len(messages) == 1, messages
    return messages[0]


def body_fails(body: str, inputs: str = "", outputs: str = "") -> str:
    return get_failure_message(f"word t ( {inputs} -> {outputs} ) {body} end")


def accepts(body: str, inputs: str = "", outputs: str = "") -> None:
    check(parse(f"word t ( {inputs} -> {outputs} ) {body} end"))


def line_of(source: str) -> int:
    with pytest.raises(IndeErrors) as err:
        check(parse(source))
    location = err.value.errors[0].location
    assert location is not None
    return location.line


def test_types_print_as_they_are_written() -> None:
    assert str(I32) == "i32"
    assert str(array_of(U8)) == "array[u8]"
    assert str(Type("Result", (array_of(U8), STR))) == "Result[array[u8], str]"
    assert str(Effect((I32, I32), (BOOL,))) == "( i32 i32 -> bool )"
    assert str(Effect()) == "( -> )"


def test_types_compare_by_value() -> None:
    assert Type("i32") == I32
    assert array_of(U8) == array_of(U8)
    assert array_of(U8) != array_of(I32)
    assert option_of(I32) != I32


def test_empty_program() -> None:
    checked = check(parse(""))
    assert checked.words == {}


def test_integer_literals_and_calls() -> None:
    source = """
    word one ( -> i32 ) 1 end
    word two ( -> i32 i32 ) one one end
    word keep ( i32 -> i32 ) end
    word bytes ( -> u8 u8 ) 1u8 255u8 end
    """
    assert effect_of(source, "two") == "( -> i32 i32 )"
    assert effect_of(source, "keep") == "( i32 -> i32 )"


def test_a_word_may_be_used_before_its_definition() -> None:
    check(parse("word a ( -> i32 ) b end  word b ( -> i32 ) 7 end"))


def test_recursion() -> None:
    check(parse("word forever ( i32 -> i32 ) forever end"))


def test_a_shared_row_variable_changes_nothing() -> None:
    assert effect_of("word w ( ..a i32 -> ..a i32 ) end", "w") == "( i32 -> i32 )"


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("word t ( -> i32 ) end", "leaves `i32`, but its body leaves nothing"),
        ("word t ( -> ) 1 end", "leaves nothing, but its body leaves `i32`"),
        ("word t ( -> i32 ) 1 2 end", "but its body leaves `i32 i32`"),
        ("word t ( -> i32 ) 1u8 end", "leaves `i32`, but its body leaves `u8`"),
        ("word t ( i32 -> ) end", "leaves nothing, but its body leaves `i32`"),
    ],
)
def test_the_body_must_leave_the_declared_outputs(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def test_calls_check_their_inputs() -> None:
    source = "word need ( i32 i32 -> ) need end  word t ( -> ) {} end"
    assert "`need` takes 2 values, but the stack is empty" in get_failure_message(
        source.format("need")
    )
    assert "`need` takes 2 values, but the stack holds only `i32`" in (
        get_failure_message(source.format("1 need"))
    )
    assert "`need` takes `i32 i32`, found `i32 u8`" in get_failure_message(
        source.format("1 2u8 need")
    )


def test_a_word_cannot_reach_below_its_own_inputs() -> None:
    source = "word eat ( i32 -> ) eat end  word inner ( -> ) eat end"
    assert "the stack is empty" in get_failure_message(source)


def test_unknown_word() -> None:
    message = get_failure_message(
        "word square ( i32 -> i32 ) square end  word t ( -> ) 1 sqaure end"
    )
    assert "unknown word `sqaure`; did you mean `square`?" in message
    assert body_fails("frobnicate") == "unknown word `frobnicate`"


def test_errors_point_at_the_item() -> None:
    source = "word t ( -> )\n  1\n  nope\nend"
    assert line_of(source) == 3


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("word t ( -> ) [ 1 ] end", "quotations are not supported"),
        ("word t ( -> ) if end end", "`if` forms are not supported"),
        ("word t ( -> ) while do end end", "`while` forms are not supported"),
        ("word t ( -> ) match case 1 else end end", "`match` forms are not"),
        ("word t ( -> ) 1 let x in end end", "`let` forms are not supported"),
        ("word t ( -> ) ? end", "`?` forms are not supported"),
        ("word t ( T -> T ) end", "type variables are not supported"),
        ("word t ( [ -> ] -> ) end", "quotation types are not supported"),
        ("word t ( i32 -> i32 ) where T: Eq end", "`where` clauses are not"),
        ("word t ( ..a -> ..b ) end", "two different row variables"),
    ],
)
def test_later_slices_are_reported_as_unsupported(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def test_one_error_per_word_in_source_order() -> None:
    source = (
        "word a ( -> ) 1 end\n"
        "word fine ( -> i32 ) 1 end\n"
        "word b ( -> ) nope end\n"
        "word c ( -> i32 ) nope 1 2 3 nope end\n"
    )
    with pytest.raises(IndeErrors) as err:
        check(parse(source))
    lines = [error.location.line for error in err.value.errors]  # type: ignore[union-attr]
    assert lines == [1, 3, 4]


def test_a_broken_signature_is_reported_once() -> None:
    source = "word broken ( Nope -> ) end  word user ( -> ) broken end"
    assert get_failure_messages(source) == ["unknown type `Nope`"]


def test_unknown_type() -> None:
    assert get_failure_message("word t ( Shape -> ) end") == "unknown type `Shape`"


def test_library_constructors_wait_for_generics() -> None:
    assert "`some` cannot be used yet" in body_fails("1 some")


def test_signatures_accept_type_arguments() -> None:
    source = "word t ( Option[i32] Result[array[u8], str] -> array[array[f32]] ) t end"
    assert effect_of(source, "t") == (
        "( Option[i32] Result[array[u8], str] -> array[array[f32]] )"
    )


def test_check_body_continues_from_a_given_stack() -> None:
    checker = Checker()
    checker.check_program(parse("word one ( -> i32 ) 1 end"))
    stack: list[Type] = []
    for line in ("one one", "2"):
        body = Parser(tokenize(line)).parse_repl_line()
        assert isinstance(body, tuple)
        checker.check_body(body, stack)
    assert stack == [I32, I32, I32]


@pytest.mark.parametrize(
    ("literal", "type_text"),
    [
        ("1.5", "f32"),
        ("1.5f64", "f64"),
        ("1e9", "f32"),
        ("'a'", "char"),
        ('"hi"', "str"),
        ("true", "bool"),
        ("{ 1 2 3 }", "array[i32]"),
        ("{ 1u8 }", "array[u8]"),
        ('{ "a" "b" }', "array[str]"),
        ("{ { 1 2 } { 3 } }", "array[array[i32]]"),
    ],
)
def test_literal_types(literal: str, type_text: str) -> None:
    accepts(literal, outputs=type_text)


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("{ 1 2u8 }", "must have one type; found `i32` and `u8`"),
        ("{ 1 'a' }", "found `i32` and `char`"),
        ("{ { 1 } { 2u8 } }", "found `array[i32]` and `array[u8]`"),
        ("{ }", "the element type of `{ }`"),
    ],
)
def test_array_literal_errors(body: str, fragment: str) -> None:
    assert fragment in body_fails(body)


def test_no_implicit_conversion() -> None:
    assert "leaves `f32`, but its body leaves `i32`" in body_fails("1", outputs="f32")
    assert "leaves `f64`, but its body leaves `f32`" in body_fails("1.5", outputs="f64")


@pytest.mark.parametrize(
    ("body", "inputs", "outputs"),
    [
        ("1 2 +", "", "i32"),
        ("1.5 2.5 *", "", "f32"),
        ("1u8 2u8 -", "", "u8"),
        ("+", "i64 i64", "i64"),
        ("1 2 + 3 *", "", "i32"),
        ("1 2 <", "", "bool"),
        ("1.5 2.5 ==", "", "bool"),
        ("'a' 'b' <", "", "bool"),
        ('"a" "b" ==', "", "bool"),
        ("true false and not", "", "bool"),
        ("1 2 +.wrap", "", "i32"),
        ("1 2 +.checked", "", "Option[i32]"),
        ("1.0 0.0 /.ieee", "", "f32"),
        ("1.0 0.0 /.checked", "", "Option[f32]"),
        ("7 2 %", "", "i32"),
        ("5 neg abs", "", "i32"),
        ("9.0 sqrt", "", "f32"),
        ("0xF0u8 0x3Cu8 bit-and", "", "u8"),
        ("1u8 3u32 shl", "", "u8"),
        ("0b1011u8 count-ones", "", "u32"),
        ("42 print", "", ""),
        ('"hi" print 1.5 print true print', "", ""),
    ],
)
def test_builtins(body: str, inputs: str, outputs: str) -> None:
    accepts(body, inputs, outputs)


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("1 2.0 +", "`+` takes two numbers of the same type, found `i32 f32`"),
        ("1 2u8 +", "found `i32 u8`"),
        ("1 +", "`+` takes 2 values, but the stack holds only `i32`"),
        ("+", "`+` takes 2 values, but the stack is empty"),
        ("true 1 and", "`and` takes two `bool` values, found `bool i32`"),
        ("1 not", "`not` takes a `bool`, found `i32`"),
        ("1.5 2.5 %", "`%` takes two integers of the same type"),
        ("1.5 2.5 +.wrap", "`+.wrap` takes two integers of the same type"),
        ("1 2 +.ieee", "`+.ieee` takes two floats of the same type"),
        ("1u8 neg", "`neg` takes a signed integer or a float, found `u8`"),
        ("1u8 3 shl", "`shl` takes an integer and a `u32` amount"),
        ('"a" "b" <', "two ordered values of the same type"),
        ("{ 1 } print", "`print` takes a value that can be displayed"),
        ("print", "`print` takes 1 value, but the stack is empty"),
    ],
)
def test_builtin_errors(body: str, fragment: str) -> None:
    assert fragment in body_fails(body)


def test_comparison_result_is_not_a_number() -> None:
    assert "`+` takes two numbers of the same type, found `bool i32`" in body_fails(
        "1 2 < 1 +"
    )


@pytest.mark.parametrize(
    ("body", "inputs", "outputs"),
    [
        ("dup", "i32", "i32 i32"),
        ("drop", "i32", ""),
        ("swap", "i32 f32", "f32 i32"),
        ("over", "i32 f32", "i32 f32 i32"),
        ("rot", "i32 f32 bool", "f32 bool i32"),
        ("dup *", "i32", "i32"),
        ("over over + rot rot - *", "i32 i32", "i32"),
        ('1 "a" swap drop', "", "str"),
    ],
)
def test_stack_words(body: str, inputs: str, outputs: str) -> None:
    accepts(body, inputs, outputs)


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("dup", "`dup` takes 1 value, but the stack is empty"),
        ("1 swap", "`swap` takes 2 values, but the stack holds only `i32`"),
        ("1 2 rot", "`rot` takes 3 values"),
    ],
)
def test_stack_word_errors(body: str, fragment: str) -> None:
    assert fragment in body_fails(body)


def test_linear_core_types_cannot_be_copied_or_discarded() -> None:
    assert "`dup` cannot copy a value of the linear type `strbuf`" in body_fails(
        "dup", "strbuf", "strbuf strbuf"
    )
    assert "`drop` cannot discard a value of the linear type `strbuf`" in body_fails(
        "drop", "strbuf"
    )
    assert "`over` cannot copy a value of the linear type `arraybuf[u8]`" in (
        body_fails("over", "arraybuf[u8] i32", "arraybuf[u8] i32 arraybuf[u8]")
    )
    accepts("swap", "strbuf i32", "i32 strbuf")
    accepts("rot", "strbuf i32 i32", "i32 i32 strbuf")
    accepts("over", "i32 strbuf", "i32 strbuf i32")


def test_constants() -> None:
    source = """
    word newline ( -> char ) NEWLINE end
    const NEWLINE = '\\n'
    const MAX_DEPTH = 10000u32
    const DIGITS = { '0' '1' '2' }
    word depth ( -> u32 ) MAX_DEPTH end
    word digits ( -> array[char] ) DIGITS end
    word both ( -> array[char] ) { NEWLINE '\\t' } end
    """
    checked = check(parse(source))
    assert str(checked.constants["MAX_DEPTH"].type) == "u32"
    assert str(checked.constants["DIGITS"].type) == "array[char]"


@pytest.mark.parametrize(
    ("source", "messages"),
    [
        (
            "word t ( -> i32 ) LIMIT end",
            ["unknown constant `LIMIT`"],
        ),
        (
            "const LIMIT = 5  word t ( -> i32 ) LIMTI end",
            ["unknown constant `LIMTI`; did you mean `LIMIT`?"],
        ),
        (
            "const LIMIT = 5  const LIMIT = 6",
            ["constant `LIMIT` is already defined"],
        ),
        (
            "const LIMIT = 5u8  word t ( -> i32 ) LIMIT end",
            ["`t` declares that it leaves `i32`, but its body leaves `u8`"],
        ),
        (
            "const NEWLINE = '\\n'  word t ( -> ) { NEWLINE 1 } drop end",
            [
                (
                    "the elements of an array literal must have one type; "
                    "found `char` and `i32`"
                )
            ],
        ),
        (
            "const MIXED = { 1 2u8 }  word t ( -> ) MIXED drop end",
            [
                (
                    "the elements of an array literal must have one type; "
                    "found `i32` and `u8`"
                )
            ],
        ),
    ],
)
def test_constant_errors(source: str, messages: list[str]) -> None:
    assert get_failure_messages(source) == messages


SHAPES = "type Shape = circle( f32 ) | rect( f32 f32 ) | dot\n"


def test_constructors_are_words() -> None:
    assert effect_of(SHAPES, "circle") == "( f32 -> Shape )"
    assert effect_of(SHAPES, "rect") == "( f32 f32 -> Shape )"
    assert effect_of(SHAPES, "dot") == "( -> Shape )"
    check(
        parse(
            SHAPES
            + "word unit-circle ( -> Shape ) 1.0 circle end\n"
            + "word square ( f32 -> Shape ) dup rect end\n"
            + "word two ( -> Shape Shape ) dot dot end\n"
        )
    )


def test_types_may_be_used_before_they_are_declared() -> None:
    source = (
        "word w ( -> Line ) 1.0 2.0 point 3.0 4.0 point line end\n"
        "type Line = line( Point Point )\n"
        "type Point = point( f32 f32 )\n"
    )
    assert effect_of(source, "line") == "( Point Point -> Line )"


def test_type_arguments() -> None:
    source = (
        "type Tagged[T] = tagged( str T )\n"
        "word a ( array[u8] -> array[u8] ) end\n"
        "word b ( Option[i32] Result[array[u8], str] -> ) drop drop end\n"
        "word c ( Tagged[i32] -> Tagged[i32] ) end\n"
        "word d ( -> Option[i32] ) 1 2 +.checked end\n"
    )
    assert effect_of(source, "b") == "( Option[i32] Result[array[u8], str] -> )"
    assert check(parse(source)).types["Tagged"].parameters == 1


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("type Box = box( Missing )", "unknown type `Missing`"),
        (
            SHAPES + "word t ( -> Shape ) 1 circle end",
            "`circle` takes `f32`, found `i32`",
        ),
        (SHAPES + "word t ( -> f32 ) 1.0 circle end", "but its body leaves `Shape`"),
        (SHAPES + SHAPES, "type `Shape` is already defined on line 1"),
        (
            "type Option = nothing",
            "type `Option` is already defined by the core library",
        ),
        ("type Two[T, T] = two( T T )", "type parameter `T` appears more than once"),
        (
            "type Tagged[T] = tagged( str T )\nword t ( -> ) 1 tagged end",
            "`tagged` cannot be used yet: the constructors of `Tagged` need generics",
        ),
    ],
)
def test_type_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def test_linear_types() -> None:
    source = (
        "linear type Handle = handle( u32 )\n"
        "type Session = session( Handle str )\n"
        "type Pool = pool( Session ) | empty\n"
        "type Label = label( str )\n"
        "type Buffered = buffered( strbuf )\n"
    )
    assert check(parse(source)).linear_types == {
        "Handle",
        "Session",
        "Pool",
        "Buffered",
    }
    for name in ("Handle", "Session", "Pool", "Buffered"):
        message = get_failure_message(source + f"word t ( {name} -> ) drop end")
        assert f"`drop` cannot discard a value of the linear type `{name}`" in message
        message = get_failure_message(
            source + f"word t ( {name} -> {name} {name} ) dup end"
        )
        assert f"`dup` cannot copy a value of the linear type `{name}`" in message
    check(parse(source + "word t ( Label -> Label Label ) dup end"))
    check(parse(source + "word t ( Handle i32 -> i32 Handle ) swap end"))


def test_a_recursive_type_is_not_linear_by_accident() -> None:
    source = "type Chain = link( i32 Chain ) | stop\n"
    assert check(parse(source)).linear_types == set()


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        (
            "word a ( -> ) end\nword a ( -> i32 ) 1 end",
            "`a` is already defined on line 1",
        ),
        ("word dup ( i32 -> i32 i32 ) end", "`dup` is a built-in word and cannot be"),
        ("word + ( i32 i32 -> i32 ) end", "`+` is a built-in word"),
        ("word some ( -> ) end", "`some` is a built-in word"),
        (
            "type Shape = circle( f32 )\nword circle ( -> ) end",
            "`circle` is already defined on line 1",
        ),
        ("type Aa = same\ntype Bb = same", "`same` is already defined on line 1"),
        ("type Flag = print", "`print` is a built-in word"),
    ],
)
def test_each_name_is_defined_once(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def test_the_first_definition_wins() -> None:
    source = (
        "word a ( -> i32 ) 1 end\n"
        "word a ( -> f32 ) 1.5 end\n"
        "word user ( -> i32 ) a end\n"
    )
    assert get_failure_messages(source) == ["`a` is already defined on line 1"]


@pytest.mark.parametrize(
    "source",
    [
        "word main ( -> ) end",
        "word main ( -> Result[unit, str] ) main end",
        "word main ( -> Result[unit, i32] ) main end",
    ],
)
def test_valid_main(source: str) -> None:
    check(parse(source), require_main=True)


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("word main ( i32 -> ) drop end", "`main` must have the signature"),
        ("word main ( -> i32 ) 1 end", "found `( -> i32 )`"),
        (
            "word main ( -> Result[i32, str] ) main end",
            "`main` must have the signature",
        ),
        ("word main ( -> Result[unit, array[u8]] ) main end", "must be displayable"),
        ("word helper ( -> ) end", "the program has no `main`"),
        ("", "the program has no `main`"),
    ],
)
def test_invalid_main(source: str, fragment: str) -> None:
    messages = get_failure_messages(source, require_main=True)
    assert len(messages) == 1
    assert fragment in messages[0]


def test_main_is_only_required_on_request() -> None:
    check(parse("word helper ( -> ) end"))
    assert "`main` must have" in get_failure_message("word main ( i32 -> ) drop end")


def test_types_with_parameters_can_be_linear() -> None:
    source = (
        "linear type Handle = handle( u32 )\n"
        "type Holder[T] = holder( strbuf T )\n"
        "type Wrap[T] = wrap( Handle T )\n"
        "type Nested[T] = nested( Wrap[T] )\n"
        "type Once[T] = once( linear [ T -> T ] )\n"
        "type Tagged[T] = tagged( str T )\n"
    )
    assert check(parse(source)).linear_types == {
        "Handle",
        "Holder",
        "Wrap",
        "Nested",
        "Once",
    }
    for name in ("Holder", "Wrap", "Nested", "Once"):
        message = get_failure_message(source + f"word t ( {name}[i32] -> ) drop end")
        assert f"cannot discard a value of the linear type `{name}[i32]`" in message
    check(parse(source + "word t ( Tagged[i32] -> ) drop end"))
    assert "linear type `Tagged[strbuf]`" in get_failure_message(
        source + "word t ( Tagged[strbuf] -> ) drop end"
    )


@pytest.mark.parametrize(
    ("source", "messages"),
    [
        ("type Box[T] = box( Missing T )", ["unknown type `Missing`"]),
        ("type Box[T] = box( array[Missing] T )", ["unknown type `Missing`"]),
        ("type Box[T] = box( T U )", ["`U` is not a parameter of `Box`"]),
        ("type Box[T] = box( [ T -> U ] )", ["`U` is not a parameter of `Box`"]),
        (
            "type Box[T] = full( Missing T ) | empty\nword t ( -> ) empty full end",
            ["unknown type `Missing`"],
        ),
    ],
)
def test_fields_of_types_with_parameters_are_checked(
    source: str, messages: list[str]
) -> None:
    assert get_failure_messages(source) == messages


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("word t ( array[strbuf] -> array[strbuf] ) end", "but `strbuf` is linear"),
        ("word t ( -> Option[array[arraybuf[u8]]] ) t end", "`arraybuf[u8]` is linear"),
        (
            "linear type Handle = handle( u32 )\nword t ( array[Handle] -> ) t end",
            "the elements of an `array` must be unrestricted, but `Handle` is linear",
        ),
        (
            "linear type Handle = handle( u32 )\ntype Bag = bag( array[Handle] )",
            "but `Handle` is linear (in `bag`)",
        ),
    ],
)
def test_array_elements_must_be_unrestricted(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def test_arrays_of_unrestricted_elements() -> None:
    check(parse("type Label = label( str )\nword t ( array[Label] -> ) drop end"))
    check(parse("word t ( arraybuf[strbuf] -> arraybuf[strbuf] ) end"))
