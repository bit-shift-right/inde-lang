import io

import pytest

from inde.checker import Checker, check
from inde.errors import IndeTrap
from inde.interpreter import Interpreter, Session
from inde.parser import parse

FILL = (
    "word fill ( i32 -> arraybuf[i32] )\n"
    "  arraybuf-new swap\n"
    "  while dup 0 > do\n"
    "    let buffer n in buffer n arraybuf-push n 1 - end\n"
    "  end\n"
    "  drop\n"
    "end\n"
)


def session_after(*lines: str) -> Session:
    session = Session(io.StringIO(), Checker())
    for line in lines:
        session.run_line(line)
    return session


def shows(line: str) -> str:
    return session_after(line).show_stack().removeprefix("stack: ")


def run(source: str) -> tuple[list[str], Interpreter]:
    checked = check(parse(source), require_main=True)
    output = io.StringIO()
    interpreter = Interpreter(checked.words, checked.constants, output)
    assert interpreter.run_main() == 0
    return output.getvalue().splitlines(), interpreter


def trap_message(line: str) -> str:
    session = session_after()
    with pytest.raises(IndeTrap) as trap:
        session.run_line(line)
    return trap.value.message


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("{ 10 20 30 } array-len", "3u32"),
        ("{ 1 } 0u32 0u32 array-slice array-len", "0u32"),
        ("{ 10 20 30 } 1u32 array-get", "20"),
        ("{ 10 20 30 } 2u32 array-get.checked", "30 some"),
        ("{ 10 20 30 } 5u32 array-get.checked", "none"),
        ("{ 10 20 30 } 1u32 3u32 array-slice", "{ 20 30 }"),
        ("{ 10 20 30 } 1u32 1u32 array-slice", "{ }"),
        ("{ 10 20 30 } 0u32 2u32 array-slice.checked", "{ 10 20 } some"),
        ("{ 10 20 30 } 2u32 1u32 array-slice.checked", "none"),
        ("{ 10 20 30 } 1u32 4u32 array-slice.checked", "none"),
        ('{ "a" "b" } 1u32 array-get', '"b"'),
    ],
)
def test_array_words(line: str, stack: str) -> None:
    assert shows(line) == stack


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "{ 1 2 } 5u32 let a i in a i array-get end",
            "index out of range in `array-get` (index 5, length 2)",
        ),
        (
            "{ 1 2 } 2u32 1u32 let a s e in a s e array-slice end",
            "index out of range in `array-slice` (start 2, end 1, length 2)",
        ),
        (
            "{ 1 2 } 0u32 3u32 let a s e in a s e array-slice end",
            "index out of range in `array-slice` (start 0, end 3, length 2)",
        ),
    ],
)
def test_a_bad_index_traps(line: str, message: str) -> None:
    assert trap_message(line) == message


@pytest.mark.parametrize(
    ("line", "stack"),
    [
        ("arraybuf-new 1 arraybuf-push 2 arraybuf-push arraybuf-freeze", "{ 1 2 }"),
        ("arraybuf-new 7 arraybuf-push", "{ 7 } arraybuf-from-array"),
        (
            "arraybuf-new 7 arraybuf-push arraybuf-pop",
            "{ } arraybuf-from-array | 7 some",
        ),
        (
            "arraybuf-new 1 arraybuf-push arraybuf-pop swap arraybuf-pop",
            "1 some | { } arraybuf-from-array | none",
        ),
        (
            "{ 1 2 3 } arraybuf-from-array arraybuf-len",
            "{ 1 2 3 } arraybuf-from-array | 3u32",
        ),
        (
            "{ 1 2 3 } arraybuf-from-array 1u32 arraybuf-get",
            "{ 1 2 3 } arraybuf-from-array | 2",
        ),
        (
            "{ 1 2 3 } arraybuf-from-array 5u32 arraybuf-get.checked",
            "{ 1 2 3 } arraybuf-from-array | none",
        ),
        (
            "arraybuf-new 1 arraybuf-push 0u32 9 arraybuf-replace",
            "{ 9 } arraybuf-from-array | 1",
        ),
        (
            "arraybuf-new 1 arraybuf-push 0u32 9 arraybuf-replace.checked",
            "{ 9 } arraybuf-from-array | 1 ok",
        ),
        (
            "arraybuf-new 1 arraybuf-push 5u32 9 arraybuf-replace.checked",
            "{ 1 } arraybuf-from-array | 9 err",
        ),
        (
            "{ 1 2 } arraybuf-from-array arraybuf-snapshot swap 3 arraybuf-push",
            "{ 1 2 } | { 1 2 3 } arraybuf-from-array",
        ),
        (
            '{ "a" } arraybuf-from-array "b" arraybuf-push arraybuf-freeze',
            '{ "a" "b" }',
        ),
    ],
)
def test_arraybuf_words(line: str, stack: str) -> None:
    assert shows(line) == stack


def test_a_buffer_of_linear_elements() -> None:
    line = (
        'arraybuf-new "a" strbuf-from-str arraybuf-push arraybuf-pop swap arraybuf-pop'
    )
    session = session_after(line)
    assert session.show_stack() == (
        'stack: "a" strbuf-from-str some | { } arraybuf-from-array | none'
    )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "{ 1 2 } arraybuf-from-array 2u32 arraybuf-get",
            "index out of range in `arraybuf-get` (index 2, length 2)",
        ),
        (
            "arraybuf-new 0u32 1 arraybuf-replace",
            "index out of range in `arraybuf-replace` (index 0, length 0)",
        ),
    ],
)
def test_a_bad_buffer_index_traps(line: str, message: str) -> None:
    assert trap_message(line) == message


def test_a_buffer_built_in_a_loop() -> None:
    lines, _ = run(FILL + "word main ( -> ) 5 fill arraybuf-freeze array-len print end")
    assert lines == ["5"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("{ 1 2 3 } arraybuf-from-array [ print ] arraybuf-consume", ["1", "2", "3"]),
        ("arraybuf-new [ 1 + print ] arraybuf-consume", []),
        ("0 { 1 2 3 } arraybuf-from-array [ + ] arraybuf-consume print", ["6"]),
        (
            (
                'arraybuf-new "a" strbuf-from-str arraybuf-push "b" strbuf-from-str '
                "arraybuf-push [ strbuf-freeze print ] arraybuf-consume"
            ),
            ["a", "b"],
        ),
        (
            (
                '"x" { 1 2 } arraybuf-from-array [ swap dup print swap print ] '
                "arraybuf-consume drop"
            ),
            ["x", "1", "x", "2"],
        ),
        (
            (
                '{ 1 2 } arraybuf-from-array "!" [ swap print print ] curry '
                "arraybuf-consume"
            ),
            ["1", "!", "2", "!"],
        ),
    ],
)
def test_consume(body: str, expected: list[str]) -> None:
    lines, _ = run(f"word main ( -> ) {body} end")
    assert lines == expected


def test_consume_runs_words_and_tail_calls() -> None:
    source = FILL + (
        "word add ( i32 i32 -> i32 ) + end\n"
        "word main ( -> ) 0 1000 fill [ add ] arraybuf-consume print end\n"
    )
    lines, interpreter = run(source)
    assert lines == ["500500"]
    assert interpreter.deepest < 10


def test_a_question_mark_inside_a_consume() -> None:
    source = FILL + (
        "word add ( i32 i32 -> Option[i32] ) +.checked ? some end\n"
        "word step ( Option[i32] i32 -> Option[i32] )\n"
        "  swap match case some add case none drop none end\n"
        "end\n"
        'word show ( Option[i32] -> ) match case some print case none "overflow" '
        "print end end\nword main ( -> )\n"
        "  0 some 10 fill [ step ] arraybuf-consume show\n"
        "  2147483000 some 3 fill [ 1000 * step ] arraybuf-consume show\n"
        "end\n"
    )
    lines, _ = run(source)
    assert lines == ["55", "overflow"]


def test_a_long_consume() -> None:
    source = (
        FILL + "word main ( -> ) 0 20000 fill [ drop 1 + ] arraybuf-consume print end\n"
    )
    lines, interpreter = run(source)
    assert lines == ["20000"]
    assert interpreter.deepest < 10


def test_a_line_that_traps_leaves_the_buffer_as_it_was() -> None:
    session = session_after("arraybuf-new 3 arraybuf-push")
    with pytest.raises(IndeTrap):
        session.run_line("4 arraybuf-push 9u32 arraybuf-get")
    assert session.show_stack() == "stack: { 3 } arraybuf-from-array"
    session.run_line("4 arraybuf-push")
    assert session.show_stack() == "stack: { 3 4 } arraybuf-from-array"


def test_also_inside_other_values() -> None:
    session = session_after("arraybuf-new 3 arraybuf-push some")
    with pytest.raises(IndeTrap):
        session.run_line(
            "match case some 4 arraybuf-push 9u32 arraybuf-get drop some case none "
            "none end"
        )
    assert session.show_stack() == "stack: { 3 } arraybuf-from-array some"
