import pytest

from inde.checker import Checker, check
from inde.errors import IndeError, IndeErrors
from inde.lexer import tokenize
from inde.parser import Parser, parse
from inde.type_model import Type


def line_types(line: str) -> str:
    checker = Checker()
    checker.check_program(parse(""))
    body = Parser(tokenize(line)).parse_repl_line()
    assert isinstance(body, tuple)
    stack: list[Type] = []
    checker.check_line(body, stack)
    return " ".join(str(t) for t in stack)


def line_fails(line: str) -> str:
    with pytest.raises(IndeError) as err:
        line_types(line)
    return err.value.message


def get_failure_message(source: str) -> str:
    with pytest.raises(IndeErrors) as err:
        check(parse(source))
    messages = [error.message for error in err.value.errors]
    assert len(messages) == 1, messages
    return messages[0]


@pytest.mark.parametrize(
    ("line", "types"),
    [
        ("{ 10 20 30 } array-len", "u32"),
        ("{ 10 20 30 } 1u32 array-get", "i32"),
        ("{ 10 20 30 } 5u32 array-get.checked", "Option[i32]"),
        ("{ 10 20 30 } 1u32 3u32 array-slice", "array[i32]"),
        ("{ 10 20 30 } 1u32 3u32 array-slice.checked", "Option[array[i32]]"),
        ("arraybuf-new 1 arraybuf-push", "arraybuf[i32]"),
        ("arraybuf-new 1 arraybuf-push arraybuf-pop", "arraybuf[i32] Option[i32]"),
        ("arraybuf-new 1 arraybuf-push arraybuf-len", "arraybuf[i32] u32"),
        ("arraybuf-new 1 arraybuf-push 0u32 arraybuf-get", "arraybuf[i32] i32"),
        (
            "arraybuf-new 1 arraybuf-push 0u32 arraybuf-get.checked",
            "arraybuf[i32] Option[i32]",
        ),
        ("arraybuf-new 1 arraybuf-push 0u32 9 arraybuf-replace", "arraybuf[i32] i32"),
        (
            "arraybuf-new 1 arraybuf-push 0u32 9 arraybuf-replace.checked",
            "arraybuf[i32] Result[i32, i32]",
        ),
        ("arraybuf-new 1 arraybuf-push arraybuf-snapshot", "arraybuf[i32] array[i32]"),
        ("arraybuf-new 1 arraybuf-push arraybuf-freeze", "array[i32]"),
        ("{ 1 2 } arraybuf-from-array", "arraybuf[i32]"),
        ("{ 1 2 } arraybuf-from-array [ drop ] arraybuf-consume", ""),
        ("0 { 1 2 } arraybuf-from-array [ + ] arraybuf-consume", "i32"),
        (
            (
                "arraybuf-new strbuf-new arraybuf-push "
                "[ strbuf-freeze print ] arraybuf-consume"
            ),
            "",
        ),
        (
            "arraybuf-new strbuf-new arraybuf-push arraybuf-pop",
            "arraybuf[strbuf] Option[strbuf]",
        ),
        ("{ 1 2 } { 1 2 } ==", "bool"),
    ],
)
def test_array_word_types(line: str, types: str) -> None:
    assert line_types(line) == types


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "{ 1 2 } 0 array-get",
            "`array-get` takes an `array` and a `u32` index, found `array[i32] i32`",
        ),
        (
            "arraybuf-new strbuf-new arraybuf-push arraybuf-freeze",
            (
                "`arraybuf-freeze` takes an `arraybuf` of unrestricted elements, "
                "found `arraybuf[strbuf]`"
            ),
        ),
        (
            "arraybuf-new strbuf-new arraybuf-push 0u32 arraybuf-get",
            (
                "`arraybuf-get` takes an `arraybuf` of unrestricted elements and a "
                "`u32` index, found `arraybuf[strbuf] u32`"
            ),
        ),
        (
            "arraybuf-new 1 arraybuf-push dup",
            "`dup` cannot copy a value of the linear type `arraybuf[i32]`",
        ),
        (
            "arraybuf-new 1 arraybuf-push drop",
            "`drop` cannot discard a value of the linear type `arraybuf[i32]`",
        ),
        (
            "arraybuf-new arraybuf-freeze",
            (
                "cannot work out the type `T` of this `arraybuf-new`; "
                "nothing here says what it is"
            ),
        ),
        (
            "{ 1.0 } { 1.0 } ==",
            (
                "`==` takes two values of the same type, found `array[f32] array[f32]`"
                "; `array[f32]` has no equality, because it holds `f32` values, whose "
                "equality is only partial"
            ),
        ),
    ],
)
def test_array_word_errors(line: str, message: str) -> None:
    assert line_fails(line) == message


def test_consume_needs_an_unrestricted_quotation() -> None:
    assert line_fails(
        "arraybuf-new strbuf-new arraybuf-push "
        "strbuf-new [ strbuf-freeze drop strbuf-freeze drop ] curry arraybuf-consume"
    ) == (
        "`arraybuf-consume` takes an `arraybuf` and a quotation to run on each "
        "element, found `arraybuf[strbuf] linear [ strbuf -> ]`; a linear "
        "quotation can be called only once"
    )
    assert get_failure_message(
        "word t ( arraybuf[T] Q -> ) where Q: [ T -> ] arraybuf-consume end"
    ) == (
        "`arraybuf-consume` takes an `arraybuf` and a quotation to run on each "
        "element, found `arraybuf[T] Q`; add `where Q: Unrestricted` to the signature"
    )
    check(
        parse(
            "word t ( arraybuf[T] Q -> ) where Q: [ T -> ] & Unrestricted "
            "arraybuf-consume end"
        )
    )


def test_consume_must_leave_the_stack_as_it_was() -> None:
    assert line_fails("{ 1 2 } arraybuf-from-array [ ] arraybuf-consume") == (
        "`arraybuf-consume` takes an `arraybuf` and a quotation to run on each "
        "element, found `arraybuf[i32] [ -> ]`"
    )
