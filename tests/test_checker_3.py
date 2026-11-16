import pytest

from inde.checker import check
from inde.errors import IndeErrors
from inde.parser import parse

HANDLE = "linear type Handle = handle( u32 )\n"
SHAPES = "type Shape = circle( f32 ) | rect( f32 f32 ) | dot\n"
LIST = "type List[T] = nil | cons( T List[T] )\n"


def effect_of(source: str, name: str) -> str:
    return str(check(parse(source)).words[name].effect)


def get_failure_messages(source: str) -> list[str]:
    with pytest.raises(IndeErrors) as err:
        check(parse(source))
    return [error.message for error in err.value.errors]


def get_failure_message(source: str) -> str:
    messages = get_failure_messages(source)
    assert len(messages) == 1, messages
    return messages[0]


def body_fails(
    body: str, inputs: str = "", outputs: str = "", declarations: str = ""
) -> str:
    return get_failure_message(
        f"{declarations}word t ( {inputs} -> {outputs} ) {body} end"
    )


def accepts(
    body: str, inputs: str = "", outputs: str = "", declarations: str = ""
) -> None:
    check(parse(f"{declarations}word t ( {inputs} -> {outputs} ) {body} end"))


def test_builtins_are_generic_words_now() -> None:
    accepts("1 2 + 1.5 2.5 + drop drop")
    accepts("1u8 2u8 +.wrap 3u8 * drop")
    assert "`+` takes two numbers of the same type, found `i32 f32`" in body_fails(
        "1 2.0 +"
    )
    assert "`+` takes two numbers of the same type, found `str str`" in body_fails(
        '"a" "b" +'
    )
    assert "`shl` takes an integer and a `u32` amount, found `u8 i32`" in body_fails(
        "1u8 3 shl"
    )


def test_strings_are_ordered() -> None:
    accepts('"a" "b" <', outputs="bool")
    assert "two ordered values of the same type" in body_fails("{ 1 } { 2 } <")


def test_unit() -> None:
    accepts("unit", outputs="unit")
    accepts("unit print")
    accepts("unit unit ==", outputs="bool")


def test_the_two_sides_of_an_if_are_unified() -> None:
    accepts("if { } else { 1 2 } end", "bool", "array[i32]")
    accepts("if { 1u8 } else { } end", "bool", "array[u8]")
    assert "`array[i32]` before `else`, `array[f32]` after it" in body_fails(
        "if { 1 } else { 1.5 } end", "bool", "array[i32]"
    )


def test_an_empty_array_takes_its_type_from_the_signature() -> None:
    accepts("{ }", outputs="array[str]")
    accepts("{ { } { 1 } }", outputs="array[array[i32]]")


def test_what_is_still_unsupported() -> None:
    assert "quotation types are not supported" in get_failure_message(
        "word t ( [ -> ] -> ) end"
    )
    assert "two different row variables" in get_failure_message(
        "word t ( ..a -> ..b ) end"
    )


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("word some ( -> ) end", "`some` is a built-in word and cannot be redefined"),
        ("type Mine = none", "`none` is a built-in word and cannot be redefined"),
        (
            "type Option = nothing",
            "type `Option` is already defined by the core library",
        ),
        ("type Box[T] = box( U )", "`U` is not a parameter of `Box`"),
        ("type Box[T] = box( Missing T )", "unknown type `Missing`"),
        ("word pair-unpack ( -> ) end", "`pair-unpack` is a built-in word"),
        (
            "type Point = point( i32 i32 )\nword point-unpack ( -> ) end",
            "`point-unpack` is already defined on line 1",
        ),
        ("type Plain = plain( T )", "`T` is not a parameter of `Plain`"),
    ],
)
def test_declaration_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def test_checked_words_return_an_option() -> None:
    accepts("1 2 +.checked", outputs="Option[i32]")
    accepts("unit unit ==", outputs="bool")


@pytest.mark.parametrize(
    ("source", "name", "text"),
    [
        ("word same ( T -> T ) end", "same", "( T -> T )"),
        ("word flip ( T U -> U T ) swap end", "flip", "( T U -> U T )"),
        ("word third ( T U V -> U V T ) rot end", "third", "( T U V -> U V T )"),
        (
            "word keep ( Option[T] Q1 -> Q1 Option[T] ) swap end",
            "keep",
            "( Option[T] Q1 -> Q1 Option[T] )",
        ),
        (
            "word f ( T -> T ) where T: Eq end",
            "f",
            "( T -> T ) where T: Eq",
        ),
        (
            "word f ( T U -> U T ) where T: Eq & Unrestricted, U: Num swap end",
            "f",
            "( T U -> U T ) where T: Eq & Unrestricted, U: Num",
        ),
        (
            "word f ( T -> T ) where T: Eq & Eq, T: Eq end",
            "f",
            "( T -> T ) where T: Eq",
        ),
    ],
)
def test_generic_signatures(source: str, name: str, text: str) -> None:
    assert effect_of(source, name) == text


def test_a_type_variable_is_only_equal_to_itself() -> None:
    assert "leaves `U`, but its body leaves `T`" in get_failure_message(
        "word t ( T -> U ) end"
    )
    assert "leaves `i32`, but its body leaves `T`" in get_failure_message(
        "word t ( T -> i32 ) end"
    )
    assert "`+` takes two numbers of the same type, found `T i32`" in (
        get_failure_message("word t ( T -> T ) 1 + end")
    )


def test_calling_a_generic_word() -> None:
    source = (
        "word flip ( T U -> U T ) swap end\n"
        'word a ( -> str i32 ) 1 "x" flip end\n'
        "word b ( -> f32 bool ) true 1.5 flip end\n"
        "word c ( T U -> T U ) flip flip end\n"
    )
    check(parse(source))


def test_each_call_works_out_its_own_types() -> None:
    declarations = "word same ( T T -> T T ) end\n"
    accepts('1 2 same "a" "b" same', "", "i32 i32 str str", declarations)
    assert "`same` takes `T T`, found `i32 f32`" in body_fails(
        "1 2.5 same", "", "i32 f32", declarations
    )
    assert "`same` takes 2 values, but the stack holds only `i32`" in body_fails(
        "1 same", "", "", declarations
    )


def test_a_generic_word_may_call_itself() -> None:
    check(parse("word spin ( T U -> U T ) swap spin swap end"))


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        (
            "word t ( i32 -> i32 ) where T: Eq end",
            "`T` has a constraint but does not appear in the signature of `t`",
        ),
        (
            "word t ( T -> T ) where U: Eq end",
            "`U` has a constraint but does not appear in the signature of `t`",
        ),
        ("word t ( T -> T ) where T: Nope end", "unknown constraint `Nope`"),
        (
            "word t ( T -> T ) where T: Displya end",
            "unknown constraint `Displya`; did you mean `Display`?",
        ),
        (
            "word t ( T -> T ) where T: [ -> ] end",
            "quotation constraints are not supported",
        ),
    ],
)
def test_where_clause_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


@pytest.mark.parametrize(
    "source",
    [
        "word twice ( T -> T T ) where T: Unrestricted dup end",
        "word lose ( T -> ) where T: Unrestricted drop end",
        "word square ( T -> T ) where T: Num dup * end",
        "word show ( T -> ) where T: Display print end",
        "word same ( T T -> bool ) where T: Eq == end",
        "word same ( T T -> bool ) where T: PartialEq == end",
        "word less ( T T -> bool ) where T: Ord < end",
        "word halve ( T -> T ) where T: Integer 1u32 shr end",
        "word flip ( T -> T ) where T: Signed neg end",
        "word root ( T -> T ) where T: Float sqrt end",
        "word keep ( T T -> T bool ) where T: Eq & Unrestricted over == end",
        "word both ( T U -> ) where T: Display, U: Display print print end",
        "word wrap ( T -> array[T] ) where T: Unrestricted drop { } end",
    ],
)
def test_a_constraint_allows_what_it_promises(source: str) -> None:
    check(parse(source))


@pytest.mark.parametrize(
    "source",
    [
        "word t ( T T -> T ) where T: Integer + end",
        "word t ( T T -> bool ) where T: Integer == end",
        "word t ( T T -> bool ) where T: Integer < end",
        "word t ( T -> ) where T: Integer print end",
        "word t ( T -> T T ) where T: Integer dup end",
        "word t ( T T -> T ) where T: Signed bit-and neg end",
        "word t ( T T -> T ) where T: Float + end",
        "word t ( T T -> bool ) where T: Float < end",
        "word t ( T -> T T ) where T: Num dup end",
        "word t ( T T -> bool ) where T: Num == end",
        "word t ( T T -> bool ) where T: Eq != end",
        "word t ( T T -> bool ) where T: Ord == end",
        "word t ( T T -> bool ) where T: Ord >= end",
    ],
)
def test_a_constraint_brings_others_with_it(source: str) -> None:
    check(parse(source))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "word t ( T -> T T ) dup end",
            (
                "`dup` cannot copy a value of type `T`, which may be linear; "
                "add `where T: Unrestricted` to the signature"
            ),
        ),
        (
            "word t ( T -> ) drop end",
            (
                "`drop` cannot discard a value of type `T`, which may be linear; "
                "add `where T: Unrestricted` to the signature"
            ),
        ),
        (
            "word t ( T U -> T U T ) where U: Unrestricted over end",
            (
                "`over` cannot copy a value of type `T`, which may be linear; "
                "add `where T: Unrestricted` to the signature"
            ),
        ),
        (
            "word t ( T -> ) print end",
            (
                "`print` takes a value that can be displayed, found `T`; "
                "add `where T: Display` to the signature"
            ),
        ),
        (
            "word t ( T T -> T ) + end",
            (
                "`+` takes two numbers of the same type, found `T T`; "
                "add `where T: Num` to the signature"
            ),
        ),
        (
            "word t ( T T -> bool ) where T: Display == end",
            (
                "`==` takes two values of the same type, found `T T`; "
                "add `where T: PartialEq` to the signature"
            ),
        ),
        (
            "word t ( T -> T T ) where T: Eq dup end",
            (
                "`dup` cannot copy a value of type `T`, which may be linear; "
                "add `where T: Unrestricted` to the signature"
            ),
        ),
        (
            "word t ( T T -> bool ) where T: PartialEq < end",
            (
                "`<` takes two ordered values of the same type, found `T T`; "
                "add `where T: PartialOrd` to the signature"
            ),
        ),
        (
            "word t ( T -> T ) where T: Num neg end",
            "`neg` takes a signed integer or a float, found `T`",
        ),
        (
            "word t ( T T -> T ) where T: Num +.wrap end",
            (
                "`+.wrap` takes two integers of the same type, found `T T`; "
                "add `where T: Integer` to the signature"
            ),
        ),
        (
            "word t ( T -> array[T] ) drop { } end",
            (
                "the elements of an `array` must be unrestricted, but `T` may be "
                "linear; add `where T: Unrestricted` (in `t`)"
            ),
        ),
    ],
)
def test_a_body_may_assume_nothing_about_a_type_variable(
    source: str, message: str
) -> None:
    assert message in get_failure_messages(source)


def test_callers_must_meet_the_constraints() -> None:
    declarations = (
        "word square ( T -> T ) where T: Num dup * end\n"
        "word show ( T -> ) where T: Display print end\n"
        "word twice ( T -> T T ) where T: Unrestricted dup end\n" + HANDLE + SHAPES
    )
    accepts("3 square 1.5 square 2u8 square", "", "i32 f32 u8", declarations)
    accepts('1 show "a" show true show 1.5 show', "", "", declarations)
    assert (
        body_fails('"a" square', "", "str", declarations)
        == "`str` does not satisfy `Num`, which `square` requires of its `T`"
    )
    assert (
        body_fails("dot show", "", "", declarations)
        == "`Shape` does not satisfy `Display`, which `show` requires of its `T`"
    )
    assert "`Handle` does not satisfy `Unrestricted`, which `twice` requires" in (
        body_fails("twice", "Handle", "Handle Handle", declarations)
    )
    assert "`strbuf` does not satisfy `Unrestricted`" in body_fails(
        "twice", "strbuf", "strbuf strbuf", declarations
    )


def test_a_constraint_is_passed_on() -> None:
    source = (
        "word square ( T -> T ) where T: Num dup * end\n"
        "word fourth ( T -> T ) where T: Integer square square end\n"
        "word broken ( T -> T ) where T: Display square end\n"
    )
    assert get_failure_messages(source) == [
        (
            "`T` does not satisfy `Num`, which `square` requires of its `T`; "
            "add `where T: Num` to the signature"
        )
    ]


@pytest.mark.parametrize(
    ("body", "inputs", "declarations"),
    [
        ("==", "array[i32] array[i32]", ""),
        ("==", "array[array[str]] array[array[str]]", ""),
        ("==", "Colour Colour", "type Colour = red | green\n"),
        ("==", "Point Point", "type Point = point( i32 i32 )\n"),
        (
            "!=",
            "Line Line",
            "type Point = point( i32 i32 )\ntype Line = line( Point Point )\n",
        ),
        ("==", "Chain Chain", "type Chain = link( i32 Chain ) | stop\n"),
        ("==", "Handle Handle", HANDLE),
    ],
)
def test_equality_is_derived(body: str, inputs: str, declarations: str) -> None:
    accepts(body, inputs, "bool", declarations)


@pytest.mark.parametrize(
    ("inputs", "declarations", "found"),
    [
        ("Shape Shape", SHAPES, "`Shape Shape`"),
        ("array[f32] array[f32]", "", "`array[f32] array[f32]`"),
        ("strbuf strbuf", "", "`strbuf strbuf`"),
        (
            "Outer Outer",
            SHAPES + "type Outer = outer( i32 Shape )\n",
            "`Outer Outer`",
        ),
        (
            "Chain Chain",
            "type Chain = link( f32 Chain ) | stop\n",
            "`Chain Chain`",
        ),
    ],
)
def test_what_is_not_eq(inputs: str, declarations: str, found: str) -> None:
    message = body_fails("==", inputs, "bool", declarations)
    assert message == f"`==` takes two values of the same type, found {found}"


def test_floats_are_only_partially_equal() -> None:
    declarations = (
        "word same ( T T -> bool ) where T: Eq == end\n"
        "word alike ( T T -> bool ) where T: PartialEq == end\n"
    )
    accepts("1.5 2.5 ==", "", "bool")
    accepts("1.5 2.5 alike", "", "bool", declarations)
    assert "`f32` does not satisfy `Eq`, which `same` requires of its `T`" in (
        body_fails("1.5 2.5 same", "", "bool", declarations)
    )


def test_main_may_return_any_displayable_error() -> None:
    check(parse("word main ( -> Result[unit, unit] ) main end"), require_main=True)
    with pytest.raises(IndeErrors) as err:
        check(parse("word main ( -> Result[unit, array[u8]] ) main end"))
    assert "must be displayable" in err.value.errors[0].message


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("none", "( -> Option[T] )"),
        ("some", "( T -> Option[T] )"),
        ("ok", "( T -> Result[T, E] )"),
        ("err", "( E -> Result[T, E] )"),
        ("pair", "( A B -> Pair[A, B] )"),
        ("pair-unpack", "( Pair[A, B] -> A B )"),
    ],
)
def test_the_core_library(name: str, text: str) -> None:
    assert effect_of("", name) == text


def test_constructors_of_declared_types() -> None:
    source = LIST + "type Tagged[T] = tagged( str T )\n"
    assert effect_of(source, "nil") == "( -> List[T] )"
    assert effect_of(source, "cons") == "( T List[T] -> List[T] )"
    assert effect_of(source, "tagged") == "( str T -> Tagged[T] )"
    assert effect_of(source, "tagged-unpack") == "( Tagged[T] -> str T )"
    assert "nil-unpack" not in check(parse(source)).words


@pytest.mark.parametrize(
    ("body", "outputs"),
    [
        ("5 some", "Option[i32]"),
        ('"a" some some', "Option[Option[str]]"),
        ("none", "Option[f32]"),
        ("1 ok", "Result[i32, str]"),
        ('"bad" err', "Result[i32, str]"),
        ("unit ok", "Result[unit, i32]"),
        ('1 "a" pair', "Pair[i32, str]"),
        ('1 "a" pair pair-unpack', "i32 str"),
        ("1 2 pair pair-unpack +", "i32"),
        ("1 2 3 nil cons cons cons", "List[i32]"),
        ("nil", "List[str]"),
        ("{ 1 } some", "Option[array[i32]]"),
        ("{ } some", "Option[array[u8]]"),
    ],
)
def test_building_values(body: str, outputs: str) -> None:
    accepts(body, "", outputs, LIST)


@pytest.mark.parametrize(
    ("body", "outputs", "fragment"),
    [
        (
            "5 some",
            "Option[f32]",
            "leaves `Option[f32]`, but its body leaves `Option[i32]`",
        ),
        ("none", "i32", "leaves `i32`, but its body leaves `Option[_]`"),
        (
            '1 "a" nil cons cons',
            "List[i32]",
            "`cons` takes `T List[T]`, found `i32 List[str]`",
        ),
        ("1 ok", "Result[str, str]", "but its body leaves `Result[i32, _]`"),
        ("1 pair-unpack", "", "`pair-unpack` takes `Pair[A, B]`, found `i32`"),
        ("some", "", "`some` takes 1 value, but the stack is empty"),
    ],
)
def test_building_values_errors(body: str, outputs: str, fragment: str) -> None:
    assert fragment in body_fails(body, "", outputs, LIST)


def test_branches_agree_on_an_option() -> None:
    accepts("if none else 5 some end", "bool", "Option[i32]")
    accepts("if 5 some else none end", "bool", "Option[i32]")
    accepts("if none else none end", "bool", "Option[str]")
    assert "`Option[i32]` before `else`, `Option[str]` after it" in body_fails(
        'if 5 some else "a" some end', "bool", "Option[i32]"
    )


def test_generic_words_over_generic_types() -> None:
    source = (
        "word wrap ( T -> Option[T] ) some end\n"
        "word both ( T -> Pair[T, T] ) where T: Unrestricted dup pair end\n"
        "word swap-pair ( Pair[A, B] -> Pair[B, A] ) pair-unpack swap pair end\n"
        "word a ( -> Option[i32] ) 5 wrap end\n"
        'word b ( -> Pair[str, str] ) "x" both end\n'
        'word c ( -> Pair[str, i32] ) 1 "x" pair swap-pair end\n'
    )
    check(parse(source))


def test_unpack_words() -> None:
    declarations = "type Point = point( i32 i32 )\n" + HANDLE + SHAPES
    assert effect_of(declarations, "point-unpack") == "( Point -> i32 i32 )"
    accepts("point-unpack +", "Point", "i32", declarations)
    accepts("handle-unpack", "Handle", "u32", declarations)
    assert "unknown word `circle-unpack`" in body_fails(
        "circle-unpack", "Shape", "f32", declarations
    )


def test_a_broken_type_with_parameters_is_reported_once() -> None:
    source = (
        "type Box[T] = full( Missing T ) | empty\nword t ( -> ) empty full drop end\n"
    )
    assert get_failure_messages(source) == ["unknown type `Missing`"]


def test_linear_types_with_parameters() -> None:
    source = (
        HANDLE
        + "type Holder[T] = holder( strbuf T )\n"
        + "type Wrap[T] = wrap( Handle T )\n"
        + "type Nested[T] = nested( Wrap[T] )\n"
        + "type Tagged[T] = tagged( str T )\n"
    )
    assert check(parse(source)).linear_types == {"Handle", "Holder", "Wrap", "Nested"}
    assert "cannot discard a value of the linear type `Option[Handle]`" in (
        body_fails("some drop", "Handle", "", source)
    )
    accepts("some drop", "i32", "", source)
    accepts("some", "Handle", "Option[Handle]", source)


def test_library_names_cannot_name_locals() -> None:
    assert "`some` already names a word" in body_fails("let some in end", "i32")


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "fragment"),
    [
        ("match case some end", "Option[i32]", "i32", "is missing `case none`"),
        (
            "match case ok case error drop 0 end",
            "Result[i32, str]",
            "i32",
            "`error` is not a constructor of `Result[i32, str]`; did you mean `err`?",
        ),
    ],
)
def test_a_match_must_still_name_every_constructor(
    body: str, inputs: str, outputs: str, fragment: str
) -> None:
    assert fragment in body_fails(body, inputs, outputs)


@pytest.mark.parametrize(
    ("body", "inputs", "outputs"),
    [
        ("match case none 0 case some end", "Option[i32]", "i32"),
        ("match case some 1 + case none 0 end", "Option[i32]", "i32"),
        ('match case none "" case some end', "Option[str]", "str"),
        ("match case ok case err drop 0 end", "Result[i32, str]", "i32"),
        ("match case ok drop 0 case err end", "Result[str, i32]", "i32"),
        ("match case pair + end", "Pair[i32, i32]", "i32"),
        ("match case pair swap pair end", "Pair[i32, str]", "Pair[str, i32]"),
        (
            "match case none 0 case some match case none 1 case some end end",
            "Option[Option[i32]]",
            "i32",
        ),
        ("match case nil 0 case cons drop end", "List[i32]", "i32"),
        ("match case none none case some 1 + some end", "Option[i32]", "Option[i32]"),
    ],
)
def test_match(body: str, inputs: str, outputs: str) -> None:
    accepts(body, inputs, outputs, LIST)


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "fragment"),
    [
        (
            "match case none 0 case some 1.5 + end",
            "Option[i32]",
            "i32",
            "`+` takes two numbers of the same type, found `i32 f32`",
        ),
        (
            "match case none 0 case some end",
            "Option[str]",
            "i32",
            "`i32` after `case none`, `str` after `case some`",
        ),
    ],
)
def test_match_errors(body: str, inputs: str, outputs: str, fragment: str) -> None:
    assert fragment in body_fails(body, inputs, outputs)


def test_generic_words_that_match() -> None:
    source = LIST + (
        "word or-else ( Option[T] T -> T ) where T: Unrestricted\n"
        "  swap match case none case some swap drop end\n"
        "end\n"
        "word length ( List[T] -> i32 ) where T: Unrestricted\n"
        "  match case nil 0 case cons length swap drop 1 + end\n"
        "end\n"
        "word first ( Pair[A, B] -> A ) where B: Unrestricted\n"
        "  match case pair drop end\n"
        "end\n"
        "word a ( -> i32 ) 5 some 0 or-else end\n"
        'word b ( -> str ) none "none" or-else end\n'
        "word c ( -> i32 ) 1 2 3 nil cons cons cons length end\n"
        'word d ( -> i32 ) 1 "x" pair first end\n'
    )
    check(parse(source))


def test_a_field_keeps_its_variable_inside_a_generic_word() -> None:
    assert "`+` takes two numbers of the same type, found `T i32`" in (
        get_failure_message(
            "word t ( Option[T] -> i32 ) where T: Unrestricted\n"
            "  match case none 0 case some 1 + end\n"
            "end"
        )
    )


def test_equality_of_types_with_parameters() -> None:
    accepts("==", "Option[i32] Option[i32]", "bool")
    accepts("5 some none ==", "", "bool")
    accepts("==", "List[str] List[str]", "bool", LIST)
    accepts("==", "Result[i32, str] Result[i32, str]", "bool")
    assert "found `Option[f32] Option[f32]`" in body_fails(
        "==", "Option[f32] Option[f32]", "bool"
    )
    assert "found `List[f32] List[f32]`" in body_fails(
        "==", "List[f32] List[f32]", "bool", LIST
    )


def test_main_returning_a_result() -> None:
    source = 'word main ( -> Result[unit, str] ) true if unit ok else "bad" err end end'
    check(parse(source), require_main=True)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "{ } drop",
            (
                "cannot work out the element type of this `{ }`; "
                "nothing here says what it is"
            ),
        ),
        (
            "none drop",
            "cannot work out the type `T` of this `none`; nothing here says what it is",
        ),
        (
            "none none == drop",
            "cannot work out the type `T` of this `none`; nothing here says what it is",
        ),
        (
            "1 ok match case ok drop case err drop end",
            "cannot work out the type `E` of this `ok`; nothing here says what it is",
        ),
        (
            "nil match case nil case cons drop drop end",
            "cannot work out the type `T` of this `nil`; nothing here says what it is",
        ),
    ],
)
def test_a_type_that_is_never_found(body: str, message: str) -> None:
    assert body_fails(body, declarations=LIST) == message


def test_the_error_points_at_the_value() -> None:
    source = "word t ( -> )\n  1 drop\n  { } drop\nend"
    with pytest.raises(IndeErrors) as err:
        check(parse(source))
    location = err.value.errors[0].location
    assert location is not None
    assert (location.line, location.column) == (3, 3)


def test_a_constant_has_one_type() -> None:
    assert get_failure_messages("const EMPTY = { }\nword t ( -> ) EMPTY drop end") == [
        "cannot work out the element type of this `{ }`; nothing here says what it is"
    ]
    check(parse("const NESTED = { { } { 1 } }"))


def test_a_constraint_waits_for_its_type() -> None:
    accepts("none dup", "", "Option[i32] Option[i32]")
    assert (
        body_fails("none dup", "", "Option[Handle] Option[Handle]", HANDLE)
        == "`dup` cannot copy a value of the linear type `Option[Handle]`"
    )
    assert "cannot work out the element type of this `{ }`" in body_fails(
        "{ } { } ==", "", "bool"
    )


def test_a_late_constraint_on_a_builtin() -> None:
    declarations = "word make ( -> T ) make end\n"
    accepts("make make +", "", "i32", declarations)
    assert (
        body_fails("make make +", "", "str", declarations)
        == "`+` takes two numbers of the same type, found `str str`"
    )


def test_a_local_whose_type_arrives_later() -> None:
    accepts("none let x in x end", "", "Option[i32]")
    assert "linear locals are not supported" in body_fails(
        "none let x in x end", "", "Option[Handle]", HANDLE
    )
