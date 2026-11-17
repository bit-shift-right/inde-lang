import pytest

from inde.checker import Checker, check
from inde.errors import IndeError, IndeErrors
from inde.lexer import tokenize
from inde.parser import Parser, parse
from inde.type_model import Type

FAULT = "type Fault = fault | broken( i32 )\n"
DIVIDE = (
    "word divide ( i32 i32 -> Result[i32, str] )\n"
    '  dup 0 == if drop drop "division by zero" err else / ok end\n'
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


@pytest.mark.parametrize(
    ("body", "inputs", "outputs"),
    [
        ("+.checked ? some", "i32 i32", "Option[i32]"),
        ("+.checked ? 2 *.checked ? some", "i32 i32", "Option[i32]"),
        ("divide ? ok", "i32 i32", "Result[i32, str]"),
        ("divide ? 1 + divide ? ok", "i32 i32 i32", "Result[i32, str]"),
        ("if 1 2 +.checked ? else 3 end some", "bool", "Option[i32]"),
        ("0 while dup 10 < do 1 +.checked ? end some", "", "Option[i32]"),
        (
            "match case some ? some case none none end",
            "Option[Option[i32]]",
            "Option[i32]",
        ),
        ("1 some let o in o ? end some", "", "Option[i32]"),
    ],
)
def test_where_question_mark_is_used(body: str, inputs: str, outputs: str) -> None:
    accepts(body, inputs, outputs, DIVIDE)


@pytest.mark.parametrize(
    ("outputs", "returns"),
    [
        ("", "nothing"),
        ("i32", "`i32`"),
        ("Option[i32] i32", "`Option[i32] i32`"),
        ("T", "`T`"),
    ],
)
def test_the_word_must_return_a_result_or_an_option(outputs: str, returns: str) -> None:
    assert body_fails("1 some ? drop", "", outputs) == (
        f"`?` needs `t` to return a `Result` or an `Option`, but it returns {returns}"
    )


def test_question_mark_needs_a_value() -> None:
    assert body_fails("?", "", "Option[i32]") == (
        "`?` takes 1 value, but the stack is empty"
    )


def test_not_inside_a_quotation() -> None:
    assert body_fails("[ 1 some ? ] call some", "", "Option[i32]") == (
        "`?` is not allowed inside a quotation"
    )
    assert body_fails("[ [ 1 some ? ] ] drop none", "", "Option[i32]") == (
        "`?` is not allowed inside a quotation"
    )


def test_not_outside_a_word() -> None:
    checker = Checker()
    checker.check_program(parse(""))
    body = Parser(tokenize("1 some ?")).parse_repl_line()
    assert isinstance(body, tuple)
    stack: list[Type] = []
    with pytest.raises(IndeError) as err:
        checker.check_line(body, stack)
    assert err.value.message == "`?` can only be used in the body of a word"


def test_only_on_the_stack_the_word_was_given() -> None:
    assert get_failure_message(
        "word t ( ..a Q -> ..b Option[i32] ) where Q: [ ..a -> ..b ]\n"
        "  call 1 2 +.checked ? some\n"
        "end"
    ) == (
        "`?` cannot exit `t` early, because `t` turns `..a` into `..b`, "
        "and an early exit would leave `..a`"
    )
    accepts(
        "call 1 2 +.checked ? some",
        "..a Q",
        "..a Option[i32]",
        where="Q: [ ..a -> ..a ]",
    )
    assert get_failure_message(
        "word t ( ..a Q R -> ..a Option[i32] )\n"
        "  where Q: [ ..a -> ..b ], R: [ ..b -> ..a ]\n"
        "  let q r in q call 1 2 +.checked ? r call some end\n"
        "end"
    ) == (
        "`?` cannot exit `t` early here, because the stack below `t`'s own "
        "values is no longer the one it was called with"
    )


def test_question_mark_leaves_the_value() -> None:
    accepts("1 some ? 1 + some", "", "Option[i32]")
    accepts('"x" ok ? "y" == ok', "", "Result[bool, Fault]", FAULT)
    assert body_fails('1 some ? "a" + some', "", "Option[i32]") == (
        "`+` takes two numbers of the same type, found `i32 str`"
    )


@pytest.mark.parametrize(
    ("body", "outputs", "message"),
    [
        (
            "1 2 +.checked ? ok",
            "Result[i32, str]",
            (
                "`?` here takes a `Result`, because `t` returns `Result[i32, str]`, "
                "found `Option[i32]`; `Option` and `Result` cannot be mixed"
            ),
        ),
        (
            '"x" ok ? some',
            "Option[str]",
            (
                "`?` here takes an `Option`, because `t` returns `Option[str]`, "
                "found `Result[str, _]`; `Option` and `Result` cannot be mixed"
            ),
        ),
        (
            "fault err ? ok",
            "Result[i32, str]",
            (
                "the error types differ: `?` found `Result[_, Fault]`, but `t` "
                "returns errors of type `str`; convert with `match`"
            ),
        ),
        ("5 ? some", "Option[i32]", "`?` takes an `Option`, found `i32`"),
        ('"x" ? ok', "Result[i32, str]", "`?` takes a `Result`, found `str`"),
    ],
)
def test_what_question_mark_takes(body: str, outputs: str, message: str) -> None:
    assert body_fails(body, "", outputs, FAULT) == message


def test_what_question_mark_leaves_is_worked_out() -> None:
    accepts("divide ? drop unit ok", "i32 i32", "Result[unit, str]", DIVIDE)
    accepts("none ? some", "", "Option[T]")


def test_the_value_type_may_differ() -> None:
    accepts('"x" ok ? drop 5 ok', "", "Result[i32, Fault]", FAULT)
    accepts("ok ? drop 5 ok", "str", "Result[i32, E]")


def test_values_on_the_stack_are_dropped() -> None:
    accepts('"a" 1 2 +.checked ? drop drop none', "", "Option[i32]")
    assert body_fails("1 2 +.checked ? drop some", "strbuf", "Option[strbuf]") == (
        "the linear `strbuf` on the stack would be discarded if `?` exits early; "
        "handle the failure with `match` instead"
    )


def test_a_type_variable_on_the_stack() -> None:
    assert body_fails("1 2 +.checked ? drop some", "T", "Option[T]") == (
        "a value of type `T` on the stack would be discarded if `?` exits early, "
        "and `T` may be linear; add `where T: Unrestricted` to the signature"
    )
    accepts("1 2 +.checked ? drop some", "T", "Option[T]", where="T: Unrestricted")
    assert body_fails(
        "1 2 +.checked ? drop some", "Q", "Option[Q]", where="Q: [ -> ]"
    ) == (
        "a value of type `Q` on the stack would be discarded if `?` exits early, "
        "and `Q` may be linear; add `where Q: Unrestricted` to the signature"
    )


def test_a_type_found_later() -> None:
    assert body_fails("none 1 2 +.checked ? drop", "", "Option[strbuf]") == (
        "the linear `Option[strbuf]` on the stack would be discarded if `?` "
        "exits early; handle the failure with `match` instead"
    )


def test_unused_locals_are_dropped() -> None:
    accepts("let n in 1 2 +.checked ? n + some end", "i32", "Option[i32]")
    accepts(
        "let b in b strbuf-freeze 1 2 +.checked ? drop some end",
        "strbuf",
        "Option[str]",
    )
    assert body_fails(
        "let b in 1 2 +.checked ? drop b strbuf-freeze some end",
        "strbuf",
        "Option[str]",
    ) == (
        "the linear local `b` would be discarded if this `?` exits early; "
        "use it before the `?`, or handle the failure with `match`"
    )


def test_a_local_used_on_another_path() -> None:
    assert body_fails(
        "let c b in c if b strbuf-freeze some else 1 some ? drop "
        "b strbuf-freeze some end end",
        "bool strbuf",
        "Option[str]",
    ) == (
        "the linear local `b` would be discarded if this `?` exits early; "
        "use it before the `?`, or handle the failure with `match`"
    )


def test_a_local_of_a_type_variable() -> None:
    assert body_fails("let x in 1 some ? drop x some end", "T", "Option[T]") == (
        "the local `x` would be discarded if this `?` exits early; use it before "
        "the `?`, or handle the failure with `match`; `x` has type `T`, which may "
        "be linear; add `where T: Unrestricted` to the signature"
    )
