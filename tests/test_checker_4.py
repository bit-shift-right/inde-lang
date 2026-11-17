import pytest

from inde.checker import Checker, check
from inde.errors import IndeError, IndeErrors
from inde.lexer import tokenize
from inde.parser import Parser, parse
from inde.type_model import Type

DIP = (
    "word dip ( ..a T Q -> ..b T ) where Q: [ ..a -> ..b ]\n"
    "  swap [ ] curry compose call\n"
    "end\n"
)


def effect_of(source: str, name: str = "t") -> str:
    return str(check(parse(source)).words[name].effect)


def get_failure_message(source: str) -> str:
    with pytest.raises(IndeErrors) as err:
        check(parse(source))
    messages = [error.message for error in err.value.errors]
    assert len(messages) == 1, messages
    return messages[0]


def word(body: str, inputs: str, outputs: str, where: str, declarations: str) -> str:
    where = f" where {where}" if where else ""
    return f"{declarations}word t ( {inputs} -> {outputs} ){where} {body} end"


def body_fails(
    body: str,
    inputs: str = "",
    outputs: str = "",
    declarations: str = "",
    where: str = "",
) -> str:
    return get_failure_message(word(body, inputs, outputs, where, declarations))


def accepts(
    body: str,
    inputs: str = "",
    outputs: str = "",
    declarations: str = "",
    where: str = "",
) -> None:
    check(parse(word(body, inputs, outputs, where, declarations)))


def line_types(line: str, declarations: str = "") -> str:
    checker = Checker()
    checker.check_program(parse(declarations))
    body = Parser(tokenize(line)).parse_repl_line()
    assert isinstance(body, tuple)
    stack: list[Type] = []
    checker.check_line(body, stack)
    return " ".join(str(t) for t in stack)


def line_fails(line: str, declarations: str = "") -> str:
    with pytest.raises(IndeError) as err:
        line_types(line, declarations)
    return err.value.message


@pytest.mark.parametrize(
    ("source", "text"),
    [
        ("word t ( [ i32 -> i32 ] -> ) drop end", "( [ i32 -> i32 ] -> )"),
        ("word t ( [ -> ] -> [ -> ] ) end", "( [ -> ] -> [ -> ] )"),
        (
            "word t ( linear [ -> str ] -> linear [ -> str ] ) end",
            "( linear [ -> str ] -> linear [ -> str ] )",
        ),
        (
            "word t ( Q -> Q ) where Q: [ i32 -> bool ] end",
            "( Q -> Q ) where Q: [ i32 -> bool ]",
        ),
        (
            "word t ( Q -> Q ) where Q: [ -> ] & Unrestricted end",
            "( Q -> Q ) where Q: [ -> ] & Unrestricted",
        ),
        (
            "word t ( ..a Q -> ..a Q ) where Q: [ ..a -> ..a ] end",
            "( ..a Q -> ..a Q ) where Q: [ ..a -> ..a ]",
        ),
        (
            "word t ( Option[[ T -> T ]] -> Option[[ T -> T ]] ) end",
            "( Option[[ T -> T ]] -> Option[[ T -> T ]] )",
        ),
    ],
)
def test_quotation_types_in_signatures(source: str, text: str) -> None:
    assert effect_of(source) == text


def test_two_different_rows_must_be_kept_apart() -> None:
    assert body_fails("", "..a", "..b") == (
        "`t` declares that it leaves `..b`, but its body leaves `..a`"
    )
    assert body_fails("drop", "..a i32", "..b") == (
        "`t` declares that it leaves `..b`, but its body leaves `..a`"
    )


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        (
            "word t ( Q -> Q ) where Q: linear [ -> ] end",
            (
                "a quotation constraint accepts either qualifier, "
                "so it is written without `linear`"
            ),
        ),
        (
            "word t ( Q -> Q ) where Q: [ -> ] & [ i32 -> ] end",
            "`Q` has two quotation constraints",
        ),
        (
            "word t ( -> ) where Q: [ -> ] end",
            "`Q` has a constraint but does not appear in the signature of `t`",
        ),
        ("word t ( [ -> Missing ] -> ) drop end", "unknown type `Missing`"),
    ],
)
def test_quotation_type_errors(source: str, fragment: str) -> None:
    assert fragment in get_failure_message(source)


def test_a_quotation_type_in_a_declaration_names_no_rows() -> None:
    assert get_failure_message("type Op = op( [ ..a i32 -> ..a ] )") == (
        "a quotation type in a type declaration cannot name its rows"
    )


def test_unrestricted_quotations_can_be_copied() -> None:
    accepts("dup drop drop", "[ i32 -> i32 ]")
    accepts("dup", "Q", "Q Q", where="Q: [ -> ] & Unrestricted")


def test_linear_quotations_cannot() -> None:
    assert body_fails("dup drop drop", "linear [ -> ]") == (
        "`dup` cannot copy a value of the linear type `linear [ -> ]`"
    )
    assert body_fails("drop", "linear [ -> ]") == (
        "`drop` cannot discard a value of the linear type `linear [ -> ]`"
    )


def test_a_quotation_constraint_says_nothing_about_linearity() -> None:
    assert get_failure_message("word t ( Q -> ) where Q: [ -> ] drop end") == (
        "`drop` cannot discard a value of type `Q`, which may be linear; "
        "add `where Q: Unrestricted` to the signature"
    )


def test_the_qualifier_is_part_of_the_type() -> None:
    assert body_fails("", "linear [ -> ]", "[ -> ]") == (
        "`t` declares that it leaves `[ -> ]`, but its body leaves `linear [ -> ]`"
    )
    assert body_fails("", "[ -> i32 ]", "[ -> f32 ]") == (
        "`t` declares that it leaves `[ -> f32 ]`, but its body leaves `[ -> i32 ]`"
    )


def test_a_type_holding_a_linear_quotation_is_linear() -> None:
    source = "type Once = once( linear [ -> ] )\nword t ( Once -> ) drop end"
    assert get_failure_message(source) == (
        "`drop` cannot discard a value of the linear type `Once`"
    )
    accepts("dup", "Plain", "Plain Plain", "type Plain = plain( [ -> ] )\n")


def test_calling_a_word_that_takes_a_quotation() -> None:
    declarations = "word keep-it ( [ i32 -> i32 ] -> [ i32 -> i32 ] ) end\n"
    accepts("keep-it", "[ i32 -> i32 ]", "[ i32 -> i32 ]", declarations)
    assert body_fails("keep-it", "[ i32 -> f32 ]", "[ i32 -> i32 ]", declarations) == (
        "`keep-it` takes `[ i32 -> i32 ]`, found `[ i32 -> f32 ]`"
    )


def test_a_quotation_constraint_may_mention_another() -> None:
    for where in ("Q: [ P -> ], P: [ -> ]", "P: [ -> ], Q: [ P -> ]"):
        source = f"word t ( P Q -> ) where {where} call end"
        assert effect_of(source) == "( P Q -> ) where P: [ -> ], Q: [ P -> ]"


@pytest.mark.parametrize(
    "where", ["Q: [ Q -> ]", "Q: [ P -> ], P: [ Q -> ]", "Q: [ -> Option[Q] ]"]
)
def test_a_quotation_constraint_cannot_refer_back(where: str) -> None:
    assert get_failure_message(f"word t ( Q -> ) where {where} drop end") == (
        "the quotation constraint on `Q` refers back to `Q`"
    )


@pytest.mark.parametrize(
    ("line", "types"),
    [
        ("[ ]", "[ -> ]"),
        ("[ 1 ]", "[ -> i32 ]"),
        ("[ 1 + ]", "[ i32 -> i32 ]"),
        ("[ 1.5 * 2.0 + ]", "[ f32 -> f32 ]"),
        ('[ "a" ] [ 1 ]', "[ -> str ] [ -> i32 ]"),
        ("[ 0 > ]", "[ i32 -> bool ]"),
        ("[ if 1 else 2 end ]", "[ bool -> i32 ]"),
        ("[ dup 0 > if 1 - end ]", "[ i32 -> i32 ]"),
        ("[ while dup 0 > do 1 - end ]", "[ i32 -> i32 ]"),
        ("[ let x in x x * end 1 + ]", "[ i32 -> i32 ]"),
        ("[ [ 1 + ] ]", "[ -> [ i32 -> i32 ] ]"),
        ("[ 1 some ]", "[ -> Option[i32] ]"),
    ],
)
def test_the_type_of_a_quotation_is_its_effect(line: str, types: str) -> None:
    assert line_types(line) == types


def test_a_quotation_word_has_its_type() -> None:
    assert effect_of("word t ( -> [ i32 -> i32 ] ) [ 1 + ] end") == (
        "( -> [ i32 -> i32 ] )"
    )
    assert body_fails("[ 1.5 + ]", "", "[ i32 -> i32 ]") == (
        "`t` declares that it leaves `[ i32 -> i32 ]`, "
        "but its body leaves `[ f32 -> f32 ]`"
    )


def test_a_quotation_can_use_words() -> None:
    declarations = "word double ( i32 -> i32 ) 2 * end\n"
    assert line_types("[ double double ]", declarations) == "[ i32 -> i32 ]"
    accepts("[ t ] drop", "", "")


def test_branches_must_leave_the_same_quotation() -> None:
    accepts("if [ 1 ] else [ 2 ] end", "bool", "[ -> i32 ]")
    assert body_fails("if [ 1 ] else [ 2 3 ] end drop", "bool") == (
        "the branches of `if` leave different stacks: "
        "`[ -> i32 ]` before `else`, `[ -> i32 i32 ]` after it"
    )


def test_every_type_must_still_be_found() -> None:
    assert line_fails("[ drop ]") == (
        "cannot work out the type of a value that `drop` takes; "
        "nothing here says what it is"
    )
    assert body_fails("[ dup ] drop") == (
        "cannot work out the type of a value that `dup` takes; "
        "nothing here says what it is"
    )


def test_a_quotation_cannot_see_locals() -> None:
    message = (
        "`x` is a local, and locals are not visible inside a quotation; "
        "push `x` before the quotation and `curry` it"
    )
    assert body_fails("1 let x in [ x ] drop end") == message
    assert body_fails("1 let x in [ [ x 1 + ] ] drop end") == message
    accepts("[ 1 let x in x end ] drop")


def test_unknown_words_inside_a_quotation() -> None:
    assert body_fails("[ frobnicate ] drop") == "unknown word `frobnicate`"


def test_quotations_have_no_equality() -> None:
    assert body_fails("[ ] [ ] == drop") == (
        "`==` takes two values of the same type, found `[ -> ] [ -> ]`; "
        "quotations have no equality or order"
    )
    assert body_fails("[ ] print") == (
        "`print` takes a value that can be displayed, found `[ -> ]`; "
        "quotations cannot be displayed"
    )


@pytest.mark.parametrize(
    ("line", "types"),
    [
        ("5 [ 1 + ] call", "i32"),
        ("[ ] call", ""),
        ("1 2 [ swap ] call", "i32 i32"),
        ('1 [ "a" ] call', "i32 str"),
        ("[ 1 2 ] call +", "i32"),
        ("[ dup * ] 3 swap call", "i32"),
        ("2 [ [ 1 + ] call ] call", "i32"),
        ("true [ if 1 else 2 end ] call", "i32"),
    ],
)
def test_call(line: str, types: str) -> None:
    assert line_types(line) == types


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "[ 1 + ] call",
            (
                "the quotation given to `call` takes `i32` from the rest of the stack, "
                "found nothing"
            ),
        ),
        (
            "5 [ 1.0 + ] call",
            (
                "the quotation given to `call` takes `f32` from the rest of the stack, "
                "found `i32`"
            ),
        ),
    ],
)
def test_call_errors(line: str, message: str) -> None:
    assert line_fails(line) == message


def test_call_needs_a_quotation() -> None:
    assert line_fails("1 call") == "`call` takes a quotation, found `i32`"
    assert line_fails("call") == "`call` takes 1 value, but the stack is empty"


def test_calling_a_quotation_given_to_a_word() -> None:
    accepts("call", "i32 [ i32 -> i32 ]", "i32")
    accepts("call", "i32 Q", "i32", where="Q: [ i32 -> i32 ]")
    accepts("call", "linear [ -> ]")
    assert effect_of(
        "word apply ( ..a Q -> ..b ) where Q: [ ..a -> ..b ] call end", "apply"
    ) == ("( ..a Q -> ..b ) where Q: [ ..a -> ..b ]")


def test_a_word_that_names_its_rows() -> None:
    declarations = "word apply ( ..a Q -> ..b ) where Q: [ ..a -> ..b ] call end\n"
    accepts("1 [ 1 + ] apply", "", "i32", declarations)
    accepts('1 2 [ + "x" ] apply drop', "", "i32", declarations)
    assert body_fails("[ 1 + ] apply", "", "i32", declarations) == (
        "`apply` takes `i32` from the rest of the stack, found nothing"
    )


def test_a_body_cannot_reach_below_its_row() -> None:
    source = "word t ( ..a Q -> ..b ) where Q: [ ..a -> ..b ] 1 swap call end"
    assert get_failure_message(source) == (
        "the quotation given to `call` takes `..a` from the rest of the stack, "
        "found `..a i32`"
    )


def test_a_word_shows_what_its_quotation_must_do() -> None:
    declarations = (
        "word twice ( Q -> ) where Q: [ -> ] & Unrestricted dup call call end\n"
    )
    accepts('[ "hello" print ] twice', "", "", declarations)
    assert body_fails("1 [ dup ] twice", "", "i32", declarations) == (
        "`twice` takes `Q` where `Q: [ -> ]`, found `[ _ -> _ _ ]`"
    )


@pytest.mark.parametrize(
    ("line", "types"),
    [
        ("4 3 [ * ] curry", "i32 [ i32 -> i32 ]"),
        ("4 3 [ * ] curry call", "i32"),
        ("1 [ ] curry", "[ -> i32 ]"),
        ('"a" [ print ] curry', "[ -> ]"),
        ("[ 1 + ] [ 2 * ] compose", "[ i32 -> i32 ]"),
        ("5 [ 1 + ] dup compose call", "i32"),
        ("[ 1 ] [ drop 'a' ] compose", "[ -> char ]"),
        ("[ 1 2 ] [ + ] compose", "[ -> i32 ]"),
        ("3 10 [ - ] curry [ 2 * ] compose call", "i32"),
        ("[ 1 [ + ] curry ] call", "[ i32 -> i32 ]"),
    ],
)
def test_curry_and_compose(line: str, types: str) -> None:
    assert line_types(line) == types


def test_dip() -> None:
    assert effect_of(DIP, "dip") == "( ..a T Q -> ..b T ) where Q: [ ..a -> ..b ]"
    accepts("1 2 [ 10 * ] dip +", "", "i32", DIP)


def test_capturing_something_linear_makes_a_linear_quotation() -> None:
    assert effect_of("word t ( strbuf -> linear [ -> strbuf ] ) [ ] curry end") == (
        "( strbuf -> linear [ -> strbuf ] )"
    )
    assert body_fails("[ ] curry", "strbuf", "[ -> strbuf ]") == (
        "`t` declares that it leaves `[ -> strbuf ]`, "
        "but its body leaves `linear [ -> strbuf ]`"
    )
    assert body_fails("[ ] curry dup drop drop", "strbuf") == (
        "`dup` cannot copy a value of the linear type `linear [ -> strbuf ]`"
    )
    accepts("[ ] compose", "linear [ -> ]", "linear [ -> ]")
    accepts("swap compose", "linear [ -> ] [ -> ]", "linear [ -> ]")
    accepts("[ ] curry", "[ -> ]", "[ -> [ -> ] ]")


def test_a_quotation_variable_may_be_linear() -> None:
    accepts("[ ] compose", "Q", "linear [ -> ]", where="Q: [ -> ]")
    accepts("[ ] compose", "Q", "[ -> ]", where="Q: [ -> ] & Unrestricted")


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "[ 1 + ] [ 2.0 + ] compose",
            (
                "`compose` takes two quotations where the first leaves what the "
                "second takes, but the first leaves `i32` and the second takes `f32`"
            ),
        ),
        (
            '[ "a" ] [ 1 + ] compose',
            (
                "`compose` takes two quotations where the first leaves what the "
                "second takes, but the first leaves `str` and the second takes `i32`"
            ),
        ),
        (
            "1 [ ] compose",
            (
                "`compose` takes two quotations where the first leaves what the "
                "second takes, found `i32 [ -> ]`"
            ),
        ),
        (
            '"a" [ 1 + ] curry',
            (
                "`curry` takes a value and a quotation that takes it, "
                "found `str [ i32 -> i32 ]`"
            ),
        ),
        (
            "[ ] 1 curry",
            ("`curry` takes a value and a quotation that takes it, found `[ -> ] i32`"),
        ),
    ],
)
def test_curry_and_compose_errors(line: str, message: str) -> None:
    assert line_fails(line) == message


def test_the_qualifier_can_wait() -> None:
    keep = (
        DIP + "word keep ( ..a T Q -> ..b T ) where Q: [ ..a T -> ..b ], "
        "T: Unrestricted\n  over [ call ] dip\nend\n"
    )
    accepts("5 [ 1 + ] keep +", "", "i32", keep)
    times = (
        DIP + "word times ( ..a Q i32 -> ..a ) where Q: [ ..a -> ..a ] & Unrestricted\n"
        "  dup 0 > if\n"
        "    1 - swap dup rot [ [ call ] dip ] dip times\n"
        "  else drop drop end\nend\n"
    )
    accepts("1 [ 2 * ] 10 times", "", "i32", times)


def test_a_qualifier_that_is_never_found() -> None:
    assert body_fails("[ [ call ] curry ] drop") == (
        "`curry` cannot tell whether what it captures is linear, "
        "because its type is never worked out"
    )


def test_a_late_qualifier_must_agree() -> None:
    source = "word t ( strbuf -> [ -> strbuf ] )\n  [ [ ] curry ] call\nend"
    assert get_failure_message(source) == (
        "the quotation `curry` builds here holds something linear, so it is a "
        "linear quotation, but it is used as an unrestricted one"
    )
