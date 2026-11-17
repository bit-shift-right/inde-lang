import io

import pytest

from inde.checker import Checker, check
from inde.interpreter import Interpreter, Session
from inde.parser import parse

DIP = (
    "word dip ( ..a T Q -> ..b T ) where Q: [ ..a -> ..b ]\n"
    "  swap [ ] curry compose call\n"
    "end\n"
)
TIMES = (
    "word times ( ..a Q i32 -> ..a ) where Q: [ ..a -> ..a ] & Unrestricted\n"
    "  dup 0 > if\n"
    "    1 - swap dup rot [ [ call ] dip ] dip times\n"
    "  else\n"
    "    drop drop\n"
    "  end\n"
    "end\n"
)


def run(source: str) -> tuple[list[str], int, Interpreter]:
    checked = check(parse(source), require_main=True)
    output, errors = io.StringIO(), io.StringIO()
    interpreter = Interpreter(checked.words, checked.constants, output)
    status = interpreter.run_main(errors)
    return output.getvalue().splitlines(), status, interpreter


def printed(body: str, declarations: str = "") -> list[str]:
    lines, status, _ = run(f"{declarations}\nword main ( -> ) {body} end")
    assert status == 0
    return lines


def session_after(*lines: str) -> Session:
    session = Session(io.StringIO(), Checker())
    for line in lines:
        session.run_line(line)
    return session


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("5 [ 1 + ] call print", ["6"]),
        ("5 [ 1 + ] dup compose call print", ["7"]),
        ("4 3 [ * ] curry call print", ["12"]),
        ("[ ] call", []),
        ('[ "a" print ] call', ["a"]),
        ('[ "a" print ] dup call call', ["a", "a"]),
        ('"hi" [ print ] curry call', ["hi"]),
        ("1 2 [ swap ] call print print", ["1", "2"]),
        ("3 [ 1 + ] [ 2 * ] compose call print", ["8"]),
        ("3 [ 2 * ] [ 1 + ] compose call print", ["7"]),
        ("3 10 [ - ] curry [ 2 * ] compose call print", ["-14"]),
        ("2 [ 10 swap - ] 5 [ + ] curry compose call print", ["13"]),
        ("1 [ [ 1 + ] call ] call print", ["2"]),
        ("[ 1 [ + ] curry ] call 41 swap call print", ["42"]),
        ("true [ if 1 else 2 end ] call print", ["1"]),
    ],
)
def test_calling_quotations(body: str, expected: list[str]) -> None:
    assert printed(body) == expected


def test_the_parts_run_in_order() -> None:
    body = (
        '[ "a" print ] [ "b" print ] compose "c" [ print ] curry compose '
        '[ "d" print ] compose call'
    )
    assert printed(body) == ["a", "b", "c", "d"]


def test_curry_captures_the_value_at_that_moment() -> None:
    assert printed("1 [ ] curry 2 swap call print print") == ["1", "2"]
    assert printed("1 [ ] curry 2 [ ] curry compose call print print") == ["2", "1"]


def test_locals_are_still_there_after_a_call() -> None:
    body = "10 let x in 1 [ 1 + ] call x + print end"
    assert printed(body) == ["12"]
    body = "10 let x in 1 [ 2 let y in y y * end + ] call x + print end"
    assert printed(body) == ["15"]


def test_combinators() -> None:
    assert printed("1 2 [ 10 * ] dip print print", DIP) == ["2", "10"]
    assert printed("1 [ 2 * ] 10 times print", DIP + TIMES) == ["1024"]
    assert printed('[ "x" print ] 3 times', DIP + TIMES) == ["x", "x", "x"]


def test_quotations_in_values() -> None:
    declarations = "type Op = op( str [ i32 i32 -> i32 ] )\n"
    body = (
        '"plus" [ + ] op op-unpack swap print 3 4 rot call print '
        "[ * ] some match case some 6 7 rot call print case none end"
    )
    assert printed(body, declarations) == ["plus", "7", "42"]


def test_a_call_in_tail_position_does_not_grow_the_stack() -> None:
    declarations = (
        "word count-down ( i32 -> i32 )\n"
        "  dup 0 > if 1 - [ count-down ] call end\n"
        "end\n"
    )
    _, status, interpreter = run(
        declarations + "word main ( -> ) 200000 count-down print end"
    )
    assert status == 0
    assert interpreter.deepest < 10


def test_a_captured_value_is_not_lost_by_a_tail_call() -> None:
    declarations = "word one ( -> i32 ) 1 end\n"
    assert printed("[ one ] 5 [ + ] curry compose call print", declarations) == ["6"]


def test_the_repl_shows_a_quotation_by_its_type() -> None:
    session = session_after("4 3 [ * ] curry")
    assert session.show_stack() == "stack: 4 | [ i32 -> i32 ]"
    session.run_line("call")
    assert session.show_stack() == "stack: 12"
    session = session_after("[ 1 + ] some")
    assert session.show_stack() == "stack: [ ... ] some"


def test_the_repl_keeps_quotations_between_lines() -> None:
    session = session_after("[ 2 * ]", "[ 1 + ]", "compose", "5 swap call")
    assert session.show_stack() == "stack: 11"
    session = session_after(DIP.replace("\n", " "), "1 2 [ 10 * ]", "dip")
    assert session.show_stack() == "stack: 10 | 2"


def test_an_unknown_row_left_on_the_repl_stack_stays_its_own() -> None:
    run = "word run ( ..a [ ..a -> ..b ] -> ..b ) call end"
    session = session_after(run, "[ run ]", "[ 1 ] [ 2 ]", "5 [ + ] curry")
    types = [str(value.type) for value in session.stack]
    assert types[0] == "[ ..? [ ..? -> ..? ] -> ..? ]"
    session.run_line("drop drop drop [ 7 ] swap call")
    assert session.show_stack() == "stack: 7"
