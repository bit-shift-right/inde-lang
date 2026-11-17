import io

import pytest

from inde.checker import Checker, check
from inde.interpreter import Interpreter, Session
from inde.parser import parse


def printed(body: str, declarations: str = "") -> list[str]:
    checked = check(parse(f"{declarations}\nword main ( -> ) {body} end"), True)
    output = io.StringIO()
    status = Interpreter(checked.words, checked.constants, output).run_main()
    assert status == 0
    return output.getvalue().splitlines()


def session_after(*lines: str) -> Session:
    session = Session(io.StringIO(), Checker())
    for line in lines:
        session.run_line(line)
    return session


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("strbuf-new strbuf-freeze print", [""]),
        ("strbuf-new 'h' strbuf-push 'i' strbuf-push strbuf-freeze print", ["hi"]),
        (
            'strbuf-new "ab" strbuf-append "cd" strbuf-append strbuf-freeze print',
            ["abcd"],
        ),
        ('"abc" strbuf-from-str strbuf-freeze print', ["abc"]),
        ('"abc" strbuf-from-str strbuf-len print strbuf-freeze print', ["3", "abc"]),
        ('"héllo" strbuf-from-str strbuf-len print strbuf-freeze drop', ["6"]),
        (
            (
                '"ab" strbuf-from-str strbuf-snapshot swap "c" strbuf-append '
                "strbuf-freeze print print"
            ),
            ["abc", "ab"],
        ),
    ],
)
def test_strbuf_words(body: str, expected: list[str]) -> None:
    assert printed(body) == expected


def test_a_strbuf_through_locals() -> None:
    body = (
        "strbuf-new let b in "
        '  b "x = " strbuf-append 42 let n in n end drop "42" strbuf-append '
        "end strbuf-freeze print"
    )
    assert printed(body) == ["x = 42"]


def test_a_strbuf_built_in_a_loop() -> None:
    declarations = (
        "word stars ( i32 -> str )\n"
        "  strbuf-new swap\n"
        "  while dup 0 > do\n"
        "    1 - let b n in b '*' strbuf-push n end\n"
        "  end\n"
        "  drop strbuf-freeze\n"
        "end\n"
    )
    assert printed("3 stars print 0 stars print", declarations) == ["***", ""]


def test_the_repl_shows_a_strbuf_as_source() -> None:
    session = session_after("strbuf-new", '"hi" strbuf-append')
    assert session.show_stack() == 'stack: "hi" strbuf-from-str'
    session.run_line("strbuf-freeze")
    assert session.show_stack() == 'stack: "hi"'
