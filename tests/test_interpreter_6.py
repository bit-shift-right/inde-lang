import io

import pytest

from inde.checker import Checker, check
from inde.interpreter import Interpreter, Session
from inde.parser import parse

WORDS = (
    "word add ( i32 i32 -> Option[i32] ) +.checked ? some end\n"
    "word divide ( i32 i32 -> Result[i32, str] )\n"
    '  dup 0 == if drop drop "division by zero" err else / ok end\n'
    "end\n"
    "word calc ( i32 i32 i32 -> Result[i32, str] ) divide ? swap divide ? 1 + ok end\n"
    "word deep ( i32 -> Option[i32] )\n"
    "  dup 0 > if 1 - deep ? 1 + some else drop none end\n"
    "end\n"
    "word looped ( i32 -> Option[i32] )\n"
    "  0 swap while dup 0 > do 1 - swap 1000000000 +.checked ? swap end drop some\n"
    "end\n"
    "word tail ( i32 -> Option[i32] ) 100 swap add end\n"
    "word via ( -> Option[i32] ) 5 [ 2147483647 add ] call end\n"
    "word twice ( i32 -> Option[i32] ) 1 swap add ? 2 add end\n"
)


def session_after(*lines: str) -> Session:
    session = Session(io.StringIO(), Checker())
    for line in lines:
        session.run_line(line)
    return session


def run(source: str) -> tuple[list[str], int, str]:
    checked = check(parse(source), require_main=True)
    output, errors = io.StringIO(), io.StringIO()
    interpreter = Interpreter(checked.words, checked.constants, output)
    status = interpreter.run_main(errors)
    return output.getvalue().splitlines(), status, errors.getvalue()


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("1 2 add", "3 some"),
        ("2147483647 1 add", "none"),
        ("7 2147483647 1 add", "7 | none"),
        ("7 100 2 calc", "8 ok"),
        ("7 100 0 calc", '"division by zero" err'),
        ("7 0 2 calc", "1 ok"),
        ("3 deep", "none"),
        ("1 looped", "1000000000 some"),
        ("5 looped", "none"),
        ("1 tail", "101 some"),
        ("9 2147483647 tail", "9 | none"),
        ("via", "none"),
        ("1 twice", "4 some"),
        ("2147483647 twice", "none"),
        ("4 2147483646 twice", "4 | none"),
    ],
)
def test_question_mark(line: str, stack: str) -> None:
    words = [f"word {text}" for text in WORDS.split("word ")[1:]]
    session = session_after(*words, line)
    assert session.show_stack() == f"stack: {stack}"


def test_early_exit_skips_the_rest_of_the_word() -> None:
    source = WORDS + (
        "word noisy ( i32 -> Option[i32] )\n"
        '  "before" print 2147483647 add ? "after" print some\n'
        "end\n"
        "word report ( Option[i32] -> )\n"
        '  match case some print case none "none" print end\n'
        "end\n"
        "word main ( -> ) 0 noisy report 1 noisy report end\n"
    )
    lines, status, _ = run(source)
    assert status == 0
    assert lines == ["before", "after", "2147483647", "before", "none"]


def test_main_can_use_question_mark() -> None:
    source = WORDS + (
        "word main ( -> Result[unit, str] )\n"
        "  10 2 divide ? print\n"
        "  1 0 divide ? print\n"
        "  unit ok\n"
        "end\n"
    )
    lines, status, errors = run(source)
    assert lines == ["5"]
    assert status == 1
    assert errors == "division by zero\n"


def test_an_early_exit_through_many_calls() -> None:
    source = WORDS + (
        "word main ( -> )\n"
        '  50000 deep match case some print case none "none" print end\n'
        "end\n"
    )
    lines, status, _ = run(source)
    assert (lines, status) == (["none"], 0)
