import pytest

from inde.checker import check
from inde.errors import IndeErrors
from inde.parser import parse

SHAPES = "type Shape = circle( f32 ) | rect( f32 f32 ) | dot\n"
HANDLE = "linear type Handle = handle( u32 )\n"


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


@pytest.mark.parametrize(
    ("body", "inputs", "outputs"),
    [
        ("if 1 else 2 end", "bool", "i32"),
        ("if else end", "bool", ""),
        ("if end", "bool", ""),
        ("if 1 + end", "i32 bool", "i32"),
        ("if drop 0 end", "i32 bool", "i32"),
        ("if + else drop end", "i32 i32 bool", "i32"),
        ("0 == if 1 else 2 end", "i32", "i32"),
        ("if if 1 else 2 end else drop 3 end", "bool bool", "i32"),
        ("if drop 1 else if 2 else 3 end end", "bool bool", "i32"),
        ("if 1 2 else 3 4 end +", "bool", "i32"),
        ('if "yes" else "no" end', "bool", "str"),
    ],
)
def test_if(body: str, inputs: str, outputs: str) -> None:
    accepts(body, inputs, outputs)


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "fragment"),
    [
        ("if end", "", "", "`if` takes 1 value, but the stack is empty"),
        ("if end", "i32", "", "`if` takes a `bool`, found `i32`"),
        (
            "if 1 else 2.0 end",
            "bool",
            "i32",
            (
                "the branches of `if` leave different stacks: "
                "`i32` before `else`, `f32` after it"
            ),
        ),
        (
            "if 1 else end",
            "bool",
            "i32",
            "`i32` before `else`, nothing after it",
        ),
        (
            "if 1 2 else 3 end",
            "bool",
            "i32",
            "`i32 i32` before `else`, `i32` after it",
        ),
        (
            "if 1 end",
            "bool",
            "",
            (
                "an `if` without `else` must leave the stack as it found it, "
                "but its branch turns nothing into `i32`"
            ),
        ),
        ("if drop end", "i32 bool", "", "its branch turns `i32` into nothing"),
        ("if drop 1.5 end", "i32 bool", "i32", "turns `i32` into `f32`"),
        ("if nope else 1 end", "bool", "i32", "unknown word `nope`"),
        ("if 1 else nope end", "bool", "i32", "unknown word `nope`"),
        ("if 1 else 2 end", "bool", "", "but its body leaves `i32`"),
    ],
)
def test_if_errors(body: str, inputs: str, outputs: str, fragment: str) -> None:
    assert fragment in body_fails(body, inputs, outputs)


def test_only_the_changed_part_of_the_stack_is_shown() -> None:
    message = body_fails("if 1 else 2.5 end", "str char bool", "str char i32")
    assert "`i32` before `else`, `f32` after it" in message


def test_recursion_through_if() -> None:
    check(
        parse(
            "word countdown ( i32 -> ) "
            "dup 0 == if drop else 1 - countdown end end\n"
            "word factorial ( i32 -> i32 ) "
            "dup 1 <= if drop 1 else dup 1 - factorial * end end"
        )
    )


def test_if_with_linear_values() -> None:
    accepts("if else end", "Handle bool", "Handle", HANDLE)
    accepts("if 1 swap else 2 swap end", "Handle bool", "i32 Handle", HANDLE)
    assert "cannot discard a value of the linear type `Handle`" in body_fails(
        "if drop else end", "Handle bool", "", HANDLE
    )


def test_an_error_in_a_branch_points_inside_the_branch() -> None:
    source = "word t ( bool -> i32 )\n  if\n    1\n  else\n    nope\n  end\nend"
    with pytest.raises(IndeErrors) as err:
        check(parse(source))
    location = err.value.errors[0].location
    assert location is not None
    assert (location.line, location.column) == (5, 5)


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "fragment"),
    [
        (
            "match case 0 else end",
            "",
            "",
            "`match` takes 1 value, but the stack is empty",
        ),
        ("match case 0 1 else 2 end", "f32", "i32", "a `f32` cannot be matched"),
        (
            "match case 0 1 else 2 end",
            "array[i32]",
            "i32",
            "a value of type `array[i32]` cannot be matched",
        ),
    ],
)
def test_what_cannot_be_matched(
    body: str, inputs: str, outputs: str, fragment: str
) -> None:
    assert fragment in body_fails(body, inputs, outputs)


def test_matching_on_a_broken_type_is_reported_once() -> None:
    source = (
        "type Broken = bad( Missing ) | fine\n"
        "word t ( Broken -> ) match case bad drop case fine end end\n"
    )
    assert get_failure_messages(source) == ["unknown type `Missing`"]


@pytest.mark.parametrize(
    ("body", "inputs", "outputs"),
    [
        ("while dup 0 > do 1 - end", "i32", "i32"),
        ("while false do end", "", ""),
        ("while true do end", "", ""),
        ("while over over < do 1 + end", "i32 i32", "i32 i32"),
        ("while dup 0 > do 1 - dup print end drop", "i32", ""),
        ("while dup 0 > do while false do end 1 - end", "i32", "i32"),
        ("while dup 0 > do dup 5 == if 1 - else 2 - end end", "i32", "i32"),
        ("0 swap while dup 0 > do swap over + swap 1 - end drop", "i32", "i32"),
    ],
)
def test_while(body: str, inputs: str, outputs: str) -> None:
    accepts(body, inputs, outputs)


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "fragment"),
    [
        (
            "while do end",
            "",
            "",
            (
                "the condition of `while` must add one `bool` to the stack, "
                "but it turns nothing into nothing"
            ),
        ),
        ("while 1 do end", "", "", "but it turns nothing into `i32`"),
        ("while 0 > do end", "i32", "i32", "but it turns `i32` into `bool`"),
        ("while true true do end", "", "", "turns nothing into `bool bool`"),
        (
            "while true do 1 end",
            "",
            "",
            (
                "the body of `while` must leave the stack as it found it, "
                "but it turns nothing into `i32`"
            ),
        ),
        ("while dup 0 > do drop end", "i32", "", "turns `i32` into nothing"),
        ("while dup 0 > do drop 1.5 end", "i32", "i32", "turns `i32` into `f32`"),
        ("while nope do end", "", "", "unknown word `nope`"),
        ("while true do nope end", "", "", "unknown word `nope`"),
    ],
)
def test_while_errors(body: str, inputs: str, outputs: str, fragment: str) -> None:
    assert fragment in body_fails(body, inputs, outputs)


def test_a_loop_leaves_the_stack_as_it_was() -> None:
    assert "but its body leaves `i32`" in body_fails(
        "while dup 0 > do 1 - end", "i32", ""
    )


def test_a_linear_value_is_threaded_through_a_loop() -> None:
    accepts("while false do end", "Handle", "Handle", HANDLE)
    accepts("0 while dup 3 < do 1 + end drop", "Handle", "Handle", HANDLE)


@pytest.mark.parametrize(
    ("body", "inputs", "outputs"),
    [
        ("let n in n n * end", "i32", "i32"),
        ("let x y in y x end", "i32 f32", "f32 i32"),
        ("let x in end", "i32", ""),
        ("let x in 1 end", "str", "i32"),
        ("let a b c in c b a end", "i32 f32 str", "str f32 i32"),
        ("let x in 2 let y in x y + end end", "i32", "i32"),
        ("let x in x 0 > if x else 0 x - end end", "i32", "i32"),
        ("let n in 0 while dup n < do 1 + end end", "i32", "i32"),
        ("1 let x in x end 2 let x in x end +", "", "i32"),
        ("let x in x end", "i32 str", "i32 str"),
    ],
)
def test_let(body: str, inputs: str, outputs: str) -> None:
    accepts(body, inputs, outputs)


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "fragment"),
    [
        ("let x in end", "", "", "`let` takes 1 value, but the stack is empty"),
        ("let x y in end", "i32", "", "`let` takes 2 values"),
        ("let x in x end x", "i32", "i32 i32", "unknown word `x`"),
        ("let x in 1 let x in end end", "i32", "", "`x` is already a local here"),
        ("let x x in end", "i32 i32", "", "`x` is already a local here"),
        ("let dup in end", "i32", "", "`dup` already names a word"),
        ("let + in end", "i32", "", "`+` already names a word"),
        ("let t in end", "i32", "", "`t` already names a word"),
        ("let some in end", "i32", "", "`some` already names a word"),
        ("let x in x x + end", "i32", "f32", "but its body leaves `i32`"),
        ("let x in x 1.5 + end", "i32", "i32", "found `i32 f32`"),
    ],
)
def test_let_errors(body: str, inputs: str, outputs: str, fragment: str) -> None:
    assert fragment in body_fails(body, inputs, outputs)


def test_a_local_cannot_share_a_name_with_a_constructor() -> None:
    assert "`circle` already names a word" in body_fails(
        "let circle in end", "i32", "", SHAPES
    )


def test_locals_are_not_words_elsewhere() -> None:
    source = (
        "word a ( i32 -> i32 ) let x in x end end\n"
        "word b ( i32 -> i32 ) let x in x x + end end\n"
        "word c ( -> i32 ) x end\n"
    )
    assert get_failure_messages(source) == ["unknown word `x`"]


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "declarations"),
    [
        ("match case 0 10 case 1 20 else 30 end", "i32", "i32", ""),
        ("match case 0 1 else 2 end", "i32", "i32", ""),
        ("match case 1u8 true else false end", "u8", "bool", ""),
        ("match case 'a' 1 case 'b' 2 else 0 end", "char", "i32", ""),
        ('match case "yes" true case "no" false else false end', "str", "bool", ""),
        ("match case true 1 case false 0 end", "bool", "i32", ""),
        ("match case true 1 else 0 end", "bool", "i32", ""),
        ("match case LIMIT 1 else 0 end", "i32", "i32", "const LIMIT = 5\n"),
        (
            "match case NEWLINE 1 case 'x' 2 else 0 end",
            "char",
            "i32",
            "const NEWLINE = '\\n'\n",
        ),
        ("match case 0 drop 0 else 1 + end", "i32 i32", "i32", ""),
        ("dup match case 0 drop else print end", "i32", "", ""),
        ("match case 0 else end", "i32", "", ""),
        (
            "match case 0 match case 'a' 1 else 2 end else drop 3 end",
            "char i32",
            "i32",
            "",
        ),
    ],
)
def test_match_on_literals(
    body: str, inputs: str, outputs: str, declarations: str
) -> None:
    accepts(body, inputs, outputs, declarations)


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "fragment"),
    [
        (
            "match case 0 1 end",
            "i32",
            "i32",
            "a `match` on a `i32` needs an `else` arm for the values without a `case`",
        ),
        ("match case true 1 end", "bool", "i32", "needs an `else` arm"),
        (
            "match case true 1 case false 0 else 2 end",
            "bool",
            "i32",
            "both `true` and `false` have a `case`, so `else` can never run",
        ),
        (
            "match case 1 0 else 1 end",
            "u8",
            "i32",
            "`case 1` is a `i32`, but the value being matched is a `u8`",
        ),
        ("match case 'a' 0 else 1 end", "i32", "i32", "`case 'a'` is a `char`"),
        ('match case "a" 0 else 1 end', "char", "i32", '`case "a"` is a `str`'),
        ("match case true 0 else 1 end", "i32", "i32", "`case true` is a `bool`"),
        (
            "match case 1 0 case 1 1 else 2 end",
            "i32",
            "i32",
            "the value of `case 1` already has a `case`",
        ),
        (
            "match case true 0 case true 1 else 2 end",
            "bool",
            "i32",
            "already has a `case`",
        ),
        (
            "match case dot 0 else 1 end",
            "i32",
            "i32",
            "`dot` is a constructor, but a `i32` is matched against literals",
        ),
        (
            "match case 0 1 else 2.5 end",
            "i32",
            "i32",
            (
                "the arms of `match` leave different stacks: "
                "`i32` after `case 0`, `f32` after `else`"
            ),
        ),
        (
            "match case 0 1 case 1 else 2 end",
            "i32",
            "i32",
            "`i32` after `case 0`, nothing after `case 1`",
        ),
        ("match case 0 nope else 1 end", "i32", "i32", "unknown word `nope`"),
        ("match case 0 1 else nope end", "i32", "i32", "unknown word `nope`"),
    ],
)
def test_match_on_literals_errors(
    body: str, inputs: str, outputs: str, fragment: str
) -> None:
    assert fragment in body_fails(body, inputs, outputs)


def test_constant_patterns() -> None:
    declarations = "const LIMIT = 5\nconst SMALL = 5u8\nconst FIVE = 5\n"
    assert "`case SMALL` is a `u8`, but the value being matched is a `i32`" in (
        body_fails("match case SMALL 0 else 1 end", "i32", "i32", declarations)
    )
    assert "the value of `case FIVE` already has a `case`" in body_fails(
        "match case LIMIT 0 case FIVE 1 else 2 end", "i32", "i32", declarations
    )
    assert "the value of `case LIMIT` already has a `case`" in body_fails(
        "match case 5 0 case LIMIT 1 else 2 end", "i32", "i32", declarations
    )
    assert "unknown constant `NOPE`" in body_fails(
        "match case NOPE 0 else 1 end", "i32", "i32", declarations
    )


@pytest.mark.parametrize(
    ("body", "inputs", "outputs"),
    [
        ("match case circle dup * case rect * case dot 0.0 end", "Shape", "f32"),
        ("match case dot 0.0 case rect * case circle end", "Shape", "f32"),
        (
            "match case circle drop 1 case rect drop drop 2 case dot 3 end",
            "Shape",
            "i32",
        ),
        ("match case circle drop case rect drop drop case dot end", "Shape", ""),
        ("match case circle + case rect + + case dot end", "f32 Shape", "f32"),
        ("match case circle circle case rect rect case dot dot end", "Shape", "Shape"),
    ],
)
def test_match_on_constructors(body: str, inputs: str, outputs: str) -> None:
    accepts(body, inputs, outputs, SHAPES)


@pytest.mark.parametrize(
    ("body", "outputs", "fragment"),
    [
        (
            "match case circle drop 1 case rect drop drop 2 end",
            "i32",
            "this `match` on a `Shape` is missing `case dot`",
        ),
        ("match case dot 1 end", "i32", "is missing `case circle`, `case rect`"),
        (
            "match case circle drop 1 case rect drop drop 2 case dot 3 else 4 end",
            "i32",
            "a `match` on constructors has no `else`",
        ),
        (
            (
                "match case circle drop 1 case circle drop 2 case rect drop drop 3 "
                "case dot 4 end"
            ),
            "i32",
            "`case circle` appears more than once",
        ),
        (
            "match case circel drop 1 case rect drop drop 2 case dot 3 end",
            "i32",
            "`circel` is not a constructor of `Shape`; did you mean `circle`?",
        ),
        (
            "match case 1 1 case rect drop drop 2 case dot 3 end",
            "i32",
            (
                "`case 1` is not a constructor, but a `Shape` is matched against "
                "its constructors"
            ),
        ),
        (
            "match case circle 1 case rect drop drop 2 case dot 3 end",
            "i32",
            (
                "the arms of `match` leave different stacks: "
                "`f32 i32` after `case circle`, `i32` after `case rect`"
            ),
        ),
        (
            "match case circle drop 1 case rect drop drop 2 case dot 3.5 end",
            "i32",
            "`i32` after `case circle`, `f32` after `case dot`",
        ),
        (
            "match case circle drop 1 case rect + 2 case dot 3 end",
            "i32",
            "`i32` after `case circle`, `f32 i32` after `case rect`",
        ),
    ],
)
def test_match_on_constructors_errors(body: str, outputs: str, fragment: str) -> None:
    assert fragment in body_fails(body, "Shape", outputs, SHAPES)


def test_another_types_constructor() -> None:
    declarations = SHAPES + "type Colour = red | green\n"
    assert "`red` is not a constructor of `Shape`" in body_fails(
        "match case red 1 case rect drop drop 2 case dot 3 end",
        "Shape",
        "i32",
        declarations,
    )


def test_matching_takes_a_linear_value_apart() -> None:
    accepts("match case handle end", "Handle", "u32", HANDLE)
    declarations = HANDLE + "type Session = session( Handle str ) | closed\n"
    accepts(
        "match case session drop case closed 0u32 handle end",
        "Session",
        "Handle",
        declarations,
    )
    assert "cannot discard a value of the linear type `Handle`" in body_fails(
        "match case session drop drop case closed end", "Session", "", declarations
    )


def test_recursive_type() -> None:
    source = (
        "type Chain = link( i32 Chain ) | stop\n"
        "word total ( Chain -> i32 )\n"
        "  match\n"
        "    case link total +\n"
        "    case stop 0\n"
        "  end\n"
        "end\n"
    )
    check(parse(source))


def test_patterns_in_messages_are_written_as_literals() -> None:
    message = body_fails("match case '\\n' 1 case '\\n' 2 else 0 end", "char", "i32")
    assert message == "the value of `case '\\n'` already has a `case`"
    message = body_fails('match case "a\\"b\\t" 1 else 0 end', "i32", "i32")
    assert '`case "a\\"b\\t"` is a `str`' in message


@pytest.mark.parametrize(
    ("body", "inputs", "fragment"),
    [
        ("let total in totl end", "i32", "unknown word `totl`; did you mean `total`?"),
        (
            "let total in 1 let count in cuont end end",
            "i32",
            "unknown word `cuont`; did you mean `count`?",
        ),
        (
            "let count in true if cuont else 0 end end",
            "i32",
            "did you mean `count`?",
        ),
        ("let total in end totl", "i32", "unknown word `totl`; did you mean `rotl`?"),
    ],
)
def test_a_misspelt_local_is_suggested(body: str, inputs: str, fragment: str) -> None:
    assert fragment in body_fails(body, inputs, "i32")
