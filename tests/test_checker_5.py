import pytest

from inde.checker import check
from inde.errors import IndeErrors
from inde.parser import parse

HANDLE = "linear type Handle = handle( u32 )\n"


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


def location_of(source: str) -> tuple[int, int]:
    with pytest.raises(IndeErrors) as err:
        check(parse(source))
    location = err.value.errors[0].location
    assert location is not None
    return location.line, location.column


@pytest.mark.parametrize(
    ("body", "outputs"),
    [
        ("strbuf-new", "strbuf"),
        ("strbuf-new 'a' strbuf-push", "strbuf"),
        ('strbuf-new "abc" strbuf-append', "strbuf"),
        ("strbuf-new strbuf-len", "strbuf u32"),
        ("strbuf-new strbuf-snapshot", "strbuf str"),
        ("strbuf-new strbuf-freeze", "str"),
        ('"abc" strbuf-from-str', "strbuf"),
    ],
)
def test_strbuf_words(body: str, outputs: str) -> None:
    accepts(body, "", outputs)


def test_a_strbuf_is_linear() -> None:
    assert body_fails("strbuf-new dup", "", "strbuf strbuf") == (
        "`dup` cannot copy a value of the linear type `strbuf`"
    )
    assert body_fails("strbuf-new drop") == (
        "`drop` cannot discard a value of the linear type `strbuf`"
    )
    assert body_fails("strbuf-new 1 strbuf-push", "", "strbuf") == (
        "`strbuf-push` takes a `strbuf` and a `char`, found `strbuf i32`"
    )


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "declarations"),
    [
        ("strbuf-new let b in b strbuf-freeze end", "", "str", ""),
        ("let b in b 'x' strbuf-push end", "strbuf", "strbuf", ""),
        ("let h in h end", "Handle", "Handle", HANDLE),
        ("let h in h handle-unpack end", "Handle", "u32", HANDLE),
        ("let buffer n in buffer end", "strbuf i32", "strbuf", ""),
        ("let a b in b a end", "strbuf strbuf", "strbuf strbuf", ""),
        ("let x in x end", "T", "T", ""),
        ("let f in f call end", "linear [ -> ]", "", ""),
        ("let b in 1 let n in b n end end", "strbuf", "strbuf i32", ""),
        ("none let x in x end", "", "Option[Handle]", HANDLE),
    ],
)
def test_a_linear_local_used_once(
    body: str, inputs: str, outputs: str, declarations: str
) -> None:
    accepts(body, inputs, outputs, declarations)


def test_unrestricted_locals_are_as_before() -> None:
    accepts("let n in end", "i32")
    accepts("let n in n n n + + end", "i32", "i32")
    accepts("let x in x x end", "T", "T T", where="T: Unrestricted")


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "declarations", "message"),
    [
        (
            "strbuf-new let b in end",
            "",
            "",
            "",
            "the linear local `b` is never used",
        ),
        (
            "strbuf-new let b in b b end",
            "",
            "strbuf strbuf",
            "",
            "the linear local `b` is used twice",
        ),
        (
            "let h in h h h end",
            "Handle",
            "Handle Handle Handle",
            HANDLE,
            "the linear local `h` is used 3 times",
        ),
        (
            "let buffer n in n end",
            "strbuf i32",
            "i32",
            "",
            "the linear local `buffer` is never used",
        ),
        (
            "let f in end",
            "linear [ -> ]",
            "",
            "",
            "the linear local `f` is never used",
        ),
    ],
)
def test_a_linear_local_used_otherwise(
    body: str, inputs: str, outputs: str, declarations: str, message: str
) -> None:
    assert body_fails(body, inputs, outputs, declarations) == message


def test_the_error_points_at_the_extra_use() -> None:
    source = "word t ( strbuf -> strbuf strbuf )\n  let b in\n    b\n    b\n  end\nend"
    assert location_of(source) == (4, 5)
    source = "word t ( strbuf -> )\n  let b in\n  end\nend"
    assert location_of(source) == (2, 3)


@pytest.mark.parametrize(
    ("inputs", "outputs", "where", "body", "message"),
    [
        (
            "T",
            "",
            "",
            "let x in end",
            (
                "the local `x` is never used; `x` has type `T`, which may be linear; "
                "add `where T: Unrestricted` to the signature"
            ),
        ),
        (
            "T",
            "T T",
            "",
            "let x in x x end",
            (
                "the local `x` is used twice; `x` has type `T`, which may be linear; "
                "add `where T: Unrestricted` to the signature"
            ),
        ),
        (
            "Option[T]",
            "",
            "",
            "let o in end",
            (
                "the local `o` is never used; `o` has type `Option[T]`, which may be "
                "linear; add `where T: Unrestricted` to the signature"
            ),
        ),
        (
            "Q",
            "",
            "Q: [ -> ]",
            "let f in end",
            (
                "the local `f` is never used; `f` has type `Q`, which may be linear; "
                "add `where Q: Unrestricted` to the signature"
            ),
        ),
    ],
)
def test_a_type_variable_may_be_linear(
    inputs: str, outputs: str, where: str, body: str, message: str
) -> None:
    assert body_fails(body, inputs, outputs, where=where) == message


def test_whether_a_local_is_linear_may_be_found_later() -> None:
    assert body_fails("5u32 handle [ let x in end ] call", "", "", HANDLE) == (
        "the linear local `x` is never used"
    )
    accepts("5u32 handle [ let x in x handle-unpack end ] call", "", "u32", HANDLE)
    accepts("5 [ let x in end ] call")


def test_a_local_of_an_unknown_type_is_reported_once() -> None:
    assert body_fails("none let x in end") == (
        "cannot work out the type `T` of this `none`; nothing here says what it is"
    )


@pytest.mark.parametrize(
    ("body", "inputs", "outputs"),
    [
        (
            'let c b in c if b strbuf-freeze else b drop-it "x" end end',
            "bool strbuf",
            "str",
        ),
        ("let c b in c if b else b 'x' strbuf-push end end", "bool strbuf", "strbuf"),
        (
            (
                "let o b in o match case some drop b strbuf-freeze "
                "case none b strbuf-freeze end end"
            ),
            "Option[i32] strbuf",
            "str",
        ),
        (
            "let n b in n match case 1 b case 2 b 'x' strbuf-push else b end end",
            "i32 strbuf",
            "strbuf",
        ),
        (
            "let c b in c if c if b else b end else b end end",
            "bool strbuf",
            "strbuf",
        ),
    ],
)
def test_every_branch_uses_a_linear_local_alike(
    body: str, inputs: str, outputs: str
) -> None:
    declarations = "word drop-it ( strbuf -> ) strbuf-freeze drop end\n"
    accepts(body, inputs, outputs, declarations)


def test_branches_that_need_not_agree() -> None:
    accepts("let c b in b c if else end end", "bool strbuf", "strbuf")
    accepts("let c n in c if n else 0 end end", "bool i32", "i32")


@pytest.mark.parametrize(
    ("body", "inputs", "outputs", "message"),
    [
        (
            'let c b in c if b strbuf-freeze else "x" end end',
            "bool strbuf",
            "str",
            (
                "the branches of `if` use the linear local `b` differently: "
                "once before `else`, never after it"
            ),
        ),
        (
            "let c b in c if b strbuf-freeze drop end end",
            "bool strbuf",
            "",
            (
                "the branches of `if` use the linear local `b` differently: "
                "once in its body, never when the body is skipped"
            ),
        ),
        (
            (
                "let o b in o match case some drop b strbuf-freeze "
                'case none "none" end end'
            ),
            "Option[i32] strbuf",
            "str",
            (
                "the arms of `match` use the linear local `b` differently: "
                "once in `case some`, never in `case none`"
            ),
        ),
        (
            (
                "let n b in n match case 1 b strbuf-freeze "
                'case 2 "two" else b strbuf-freeze end end'
            ),
            "i32 strbuf",
            "str",
            (
                "the arms of `match` use the linear local `b` differently: "
                "once in `case 1`, never in `case 2`, once in `else`"
            ),
        ),
    ],
)
def test_branches_that_disagree(
    body: str, inputs: str, outputs: str, message: str
) -> None:
    assert body_fails(body, inputs, outputs) == message


def test_a_branch_that_uses_it_twice() -> None:
    assert body_fails(
        "let c b in c if b b else b b end end", "bool strbuf", "strbuf strbuf"
    ) == ("the linear local `b` is used twice")


def test_branches_and_a_type_variable() -> None:
    assert body_fails(
        "let c x in c if x some else none end end", "bool T", "Option[T]"
    ) == (
        "the branches of `if` use the local `x` differently: once before `else`, "
        "never after it; `x` has type `T`, which may be linear; "
        "add `where T: Unrestricted` to the signature"
    )


LOOP = '"" false while dup do drop drop {use} false end drop'


def test_a_linear_local_cannot_be_used_inside_a_loop() -> None:
    assert body_fails(
        "let b in " + LOOP.format(use="b strbuf-freeze") + " end", "strbuf", "str"
    ) == (
        "the linear local `b` is used inside a loop, which may run more than once; "
        "pass the value through the loop on the stack instead"
    )


def test_nor_in_the_condition() -> None:
    assert body_fails(
        "let b in while b strbuf-freeze drop false do end end", "strbuf"
    ) == (
        "the linear local `b` is used inside a loop, which may run more than once; "
        "pass the value through the loop on the stack instead"
    )


def test_what_a_loop_may_do_with_locals() -> None:
    accepts("let n in 0 while dup n < do 1 + end end", "i32", "i32")
    accepts(
        "let b in 0 while dup 3 < do 1 + end drop b strbuf-freeze end",
        "strbuf",
        "str",
    )
    accepts("while false do let b in b end end", "strbuf", "strbuf")
    accepts(
        "while false do strbuf-new let b in b strbuf-freeze drop end end",
    )


def test_a_loop_and_a_type_variable() -> None:
    assert body_fails(
        "let x in while false do x x == drop end end", "T", where="T: Eq"
    ) == (
        "the local `x` is used inside a loop, which may run more than once; "
        "`x` has type `T`, which may be linear; "
        "add `where T: Unrestricted` to the signature"
    )
