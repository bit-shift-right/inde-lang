import copy
import io
from typing import Any

import pytest

from inde.checker import check
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Interpreter, Session, display, inspect
from inde.parser import parse
from inde.repl import run_repl
from inde.type_model import F32, I32, U8, Type, option_of
from inde.values import Data, Value, array, boolean, character, none, some, string

SHAPES = "type Shape = circle( f32 ) | rect( f32 f32 ) | dot\n"


def run(source: str) -> tuple[str, int]:
    checked = check(parse(source), require_main=True)
    output = io.StringIO()
    interpreter = Interpreter(checked.words, checked.constants, output)
    status = interpreter.run_main()
    return output.getvalue(), status


def printed(body: str, declarations: str = "") -> list[str]:
    output, status = run(f"{declarations}\nword main ( -> ) {body} end")
    assert status == 0
    return output.splitlines()


def get_trap_message(body: str, declarations: str = "") -> str:
    with pytest.raises(IndeTrap) as trap:
        run(f"{declarations}\nword main ( -> ) {body} end")
    return trap.value.message


def session_after(*lines: str) -> Any:
    session = Session(io.StringIO())
    for line in lines:
        session.run_line(line)
    return session


def integer(value: int, number_type: Type = I32) -> Value:
    return Value(number_type, value)


def test_values_keep_their_types() -> None:
    assert integer(5) == integer(5)
    assert integer(5) != integer(5, U8)
    assert some(integer(5)).type == option_of(I32)
    assert none(I32) == Value(option_of(I32), Data("none"))


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (integer(42), "42"),
        (integer(-7), "-7"),
        (integer(255, U8), "255"),
        (boolean(True), "true"),
        (boolean(False), "false"),
        (character("a"), "a"),
        (string("hi there"), "hi there"),
        (string(""), ""),
    ],
)
def test_display(value: Value, text: str) -> None:
    assert display(value) == text


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (integer(42), "42"),
        (integer(42, U8), "42u8"),
        (boolean(True), "true"),
        (character("a"), "'a'"),
        (character("\n"), "'\\n'"),
        (character("'"), "'\\''"),
        (string("hi"), '"hi"'),
        (string('say "hi"\n'), '"say \\"hi\\"\\n"'),
        (string("it's"), '"it\'s"'),
        (string("\x01"), '"\\u{1}"'),
        (array(I32, (integer(1), integer(2))), "{ 1 2 }"),
        (Value(Type("Shape"), Data("dot")), "dot"),
        (some(integer(5)), "5 some"),
        (none(I32), "none"),
        (
            Value(Type("Pair"), Data("pair", (integer(1), string("a")))),
            '1 "a" pair',
        ),
    ],
)
def test_inspect(value: Value, text: str) -> None:
    assert inspect(value) == text


def test_print_integers_and_text() -> None:
    assert printed('1 print 255u8 print "hello" print true print') == [
        "1",
        "255",
        "hello",
        "true",
    ]
    assert printed("'a' print") == ["a"]


def test_addition() -> None:
    assert printed("1 2 + print") == ["3"]
    assert printed("1 2 + 3 + 4 + print") == ["10"]
    assert printed("100u8 100u8 + print") == ["200"]


def test_calls() -> None:
    declarations = """
    word three ( -> i32 ) 1 2 + end
    word add-three ( i32 -> i32 ) three + end
    word show ( i32 -> ) print end
    """
    assert printed("10 add-three show three show", declarations) == ["13", "3"]


def test_words_run_in_order() -> None:
    declarations = 'word a ( -> ) "a" print end  word b ( -> ) a "b" print a end'
    assert printed('b "main" print', declarations) == ["a", "b", "a", "main"]


def test_comparison_and_booleans() -> None:
    assert printed("1 2 < print 2 2 == print 1 2 != print") == ["true"] * 3
    assert printed("true false and print true false or print") == ["false", "true"]
    assert printed("true true xor print false not print") == ["false", "true"]
    assert printed("\"a\" \"a\" == print 'a' 'b' < print") == ["true", "true"]


def test_overflow_traps() -> None:
    message = get_trap_message("2147483647 1 let a b in a b + print end")
    assert message == "integer overflow in `+` (i32 2147483647, i32 1)"
    assert get_trap_message("200u8 100u8 let a b in a b + print end") == (
        "integer overflow in `+` (u8 200u8, u8 100u8)"
    )


def test_a_trap_stops_the_program_at_once() -> None:
    source = (
        'word main ( -> ) "before" print '
        '2147483647 1 let a b in a b + print end "after" print end'
    )
    checked = check(parse(source))
    output = io.StringIO()
    with pytest.raises(IndeTrap) as trap:
        Interpreter(checked.words, checked.constants, output).run_main()
    assert output.getvalue() == "before\n"
    assert trap.value.location is not None
    assert (trap.value.location.line, trap.value.location.column) == (1, 61)


def test_checked_addition_returns_an_option() -> None:
    session = session_after("1 2 +.checked", "2147483647 1 +.checked")
    assert session.stack == [some(integer(3)), none(I32)]


def test_exit_status() -> None:
    assert run("word main ( -> ) end") == ("", 0)
    assert run("word main ( -> ) 1 print end") == ("1\n", 0)


def test_main_returning_a_result() -> None:
    checked = check(parse("word main ( -> Result[unit, str] ) main end"))
    interpreter = Interpreter(checked.words, checked.constants, io.StringIO())
    result_type = checked.words["main"].effect.outputs[0]
    errors = io.StringIO()
    for value, status, text in [
        (Value(result_type, Data("ok", (Value(Type("unit"), Data("unit")),))), 0, ""),
        (Value(result_type, Data("err", (string("bad input"),))), 1, "bad input\n"),
    ]:
        interpreter.execute = lambda body, stack, v=value: stack.append(v)  # type: ignore[method-assign, misc]
        assert interpreter.run_main(errors) == status
        assert errors.getvalue() == text


def test_a_tail_call_does_not_grow_the_call_stack() -> None:
    words = "\n".join(f"word w{i} ( -> ) {i} print w{i + 1} end" for i in range(50))
    checked = check(parse(words + "\nword w50 ( -> ) end\nword main ( -> ) w0 end"))
    interpreter = Interpreter(checked.words, checked.constants, io.StringIO())
    assert interpreter.run_main() == 0
    assert interpreter.deepest == 1


def test_calls_that_are_not_tail_calls_nest() -> None:
    words = "\n".join(f"word w{i} ( -> ) w{i + 1} {i} print end" for i in range(50))
    checked = check(parse(words + "\nword w50 ( -> ) end\nword main ( -> ) w0 end"))
    output = io.StringIO()
    interpreter = Interpreter(checked.words, checked.constants, output)
    interpreter.run_main()
    assert interpreter.deepest == 51
    assert output.getvalue().splitlines() == [str(i) for i in range(49, -1, -1)]


def test_ten_thousand_nested_calls() -> None:
    count = 10_000
    words = "\n".join(
        f"word w{i} ( i32 -> i32 ) w{i + 1} 1 + end" for i in range(count)
    )
    source = (
        f"{words}\nword w{count} ( i32 -> i32 ) end\nword main ( -> ) 0 w0 print end"
    )
    assert run(source) == (f"{count}\n", 0)


def test_endless_recursion_traps() -> None:
    declarations = "word down ( -> ) down 1 print end"
    assert get_trap_message("down", declarations) == "call depth exceeded"


def test_session_keeps_the_stack_between_lines() -> None:
    session = session_after("1 2", "+", "10 +")
    assert session.stack == [integer(13)]
    assert session.show_stack() == "stack: 13"
    assert Session().show_stack() == "stack: (empty)"


def test_session_declarations() -> None:
    session = session_after(
        "word three ( -> i32 ) 1 2 + end",
        "word six ( -> i32 ) three three + end",
        "six three",
    )
    assert session.stack == [integer(6), integer(3)]


def test_session_rejects_a_bad_declaration_completely() -> None:
    session = session_after("word good ( -> i32 ) 1 end")
    with pytest.raises(IndeErrors):
        session.run_line("word bad ( -> i32 ) nope end")
    with pytest.raises(IndeErrors):
        session.run_line("word worse ( -> i32 ) end")
    session.run_line("word bad ( -> i32 ) good end")
    session.run_line("word worse ( -> i32 ) good end")
    session.run_line("bad worse +")
    assert session.stack == [integer(2)]


def test_session_reports_syntax_errors() -> None:
    session = session_after("1")
    for line in ("1 end", "'unterminated", "word w ( -> "):
        with pytest.raises(IndeErrors):
            session.run_line(line)
    assert session.stack == [integer(1)]


def test_session_names_are_defined_once() -> None:
    session = session_after("word one ( -> i32 ) 1 end")
    with pytest.raises(IndeErrors) as failure:
        session.run_line("word one ( -> i32 ) 2 end")
    assert "already defined" in failure.value.errors[0].message
    session.run_line("one")
    assert session.stack == [integer(1)]


def test_stack_words() -> None:
    assert printed("1 2 swap print print") == ["1", "2"]
    assert printed("7 dup + print") == ["14"]
    assert printed("1 2 drop print") == ["1"]
    assert printed("1 2 over print print print") == ["1", "2", "1"]
    assert printed("1 2 3 rot print print print") == ["1", "3", "2"]
    assert printed('1 "a" swap print print') == ["1", "a"]


def test_constants() -> None:
    declarations = "const LIMIT = 10  const GREETING = \"hello\"  const LETTER = 'x'"
    assert printed("LIMIT 1 + print GREETING print LETTER print", declarations) == [
        "11",
        "hello",
        "x",
    ]


def test_array_literals() -> None:
    session = session_after("const TWO = 2", "{ 1 TWO 3 }", "{ { 1u8 } }")
    bytes_value = array(U8, (integer(1, U8),))
    assert session.stack == [
        array(I32, (integer(1), integer(2), integer(3))),
        array(bytes_value.type, (bytes_value,)),
    ]
    assert session.show_stack() == "stack: { 1 2 3 } | { { 1u8 } }"


def test_constructors() -> None:
    session = session_after(
        "type Shape = circle( f32 ) | rect( f32 f32 ) | dot",
        "2.0 circle",
        "1.0 2.0 rect dot",
    )
    shape = Type("Shape")
    assert session.stack == [
        Value(shape, Data("circle", (Value(F32, 2.0),))),
        Value(shape, Data("rect", (Value(F32, 1.0), Value(F32, 2.0)))),
        Value(shape, Data("dot")),
    ]
    assert session.show_stack() == "stack: 2.0 circle | 1.0 2.0 rect | dot"


def test_nested_constructors() -> None:
    session = session_after(
        "type Point = point( i32 i32 )",
        "type Line = line( Point Point )",
        "1 2 point 3 4 point line",
    )
    assert session.show_stack() == "stack: 1 2 point 3 4 point line"


def test_arithmetic_and_modifiers() -> None:
    assert printed("6 7 * print 5 7 - print") == ["42", "-2"]
    assert printed("200u8 100u8 +.wrap print 200u8 100u8 +.sat print") == ["44", "255"]
    assert (
        get_trap_message("0u8 1u8 let a b in a b - print end")
        == "integer overflow in `-` (u8 0u8, u8 1u8)"
    )
    session = session_after("0u8 1u8 -.checked", "3 4 *.checked")
    assert session.stack == [none(U8), some(integer(12))]


def test_division() -> None:
    assert printed("-7 2 / print -7 2 % print") == ["-3", "-1"]
    assert (
        get_trap_message("1 0 let a b in a b / print end")
        == "division by zero in `/` (i32 1, i32 0)"
    )
    assert (
        get_trap_message("1 0 let a b in a b %.wrap print end")
        == "division by zero in `%.wrap` (i32 1, i32 0)"
    )
    assert printed("-5 abs print 5 neg print") == ["5", "-5"]
    session = session_after("1 0 /.checked", "-128i8 neg.checked", "7 2 /.checked")
    assert session.stack == [none(I32), none(Type("i8")), some(integer(3))]


def test_float_arithmetic() -> None:
    session = session_after("1.5 2.0 *", "1.0 0.0 /.checked", "1.0 0.0 /.ieee")
    assert session.stack == [Value(F32, 3.0), none(F32), Value(F32, float("inf"))]
    assert (
        get_trap_message("1.0 0.0 let a b in a b / print end")
        == "division by zero in `/` (f32 1.0, f32 0.0)"
    )
    assert "float overflow in `*`" in get_trap_message(
        "3e38 10.0 let a b in a b * print end"
    )
    assert "invalid float operation in `sqrt`" in get_trap_message(
        "-1.0 let x in x sqrt print end"
    )


def test_bits() -> None:
    assert printed("0xF0u8 0x3Cu8 bit-and print 0b1011u8 count-ones print") == [
        "48",
        "3",
    ]
    assert printed("1u8 3u32 shl print 0u8 bit-not print") == ["8", "255"]
    assert get_trap_message("1u8 8u32 let a b in a b shl print end") == (
        "shift amount out of range in `shl` (u8 1u8, u32 8u32)"
    )
    session = session_after("1u8 8u32 shl.checked", "1u8 9u32 shl.wrap")
    assert session.stack == [none(U8), integer(2, U8)]


def test_printing_floats() -> None:
    assert printed("1.5 print 0.1 print 1.0 print") == ["1.5", "0.1", "1.0"]
    assert printed("1e16f64 print 0.00001 print") == ["1e+16", "1e-05"]
    assert printed("0.1f64 0.2f64 + print 0.1 0.2 + print") == [
        "0.30000000000000004",
        "0.3",
    ]
    assert printed("9.0 sqrt print 2.0 sqrt print") == ["3.0", "1.4142135"]
    session = session_after("1.5", "1.5f64")
    assert session.show_stack() == "stack: 1.5 | 1.5f64"


def test_a_line_that_does_not_check_changes_nothing() -> None:
    session = session_after("1 2")
    with pytest.raises(IndeErrors) as failure:
        session.run_line('3 4 + "a" +')
    assert "`+` takes two numbers" in failure.value.errors[0].message
    assert session.stack == [integer(1), integer(2)]
    session.run_line("+")
    assert session.stack == [integer(3)]


def test_a_line_that_does_not_check_prints_nothing() -> None:
    output = io.StringIO()
    session = Session(output)
    with pytest.raises(IndeErrors):
        session.run_line('"printed too early" print nope')
    assert output.getvalue() == ""


def test_a_line_is_checked_against_the_current_stack() -> None:
    session = session_after('"text"')
    with pytest.raises(IndeErrors) as failure:
        session.run_line("1 +")
    assert "found `str i32`" in failure.value.errors[0].message
    with pytest.raises(IndeErrors) as failure:
        Session().run_line("+")
    assert "the stack is empty" in failure.value.errors[0].message


def test_a_trap_restores_the_stack() -> None:
    session = session_after("1 2")
    with pytest.raises(IndeTrap):
        session.run_line("+ 2147483647 +")
    assert session.stack == [integer(1), integer(2)]
    session.run_line("+")
    assert session.stack == [integer(3)]


def test_repl_transcript() -> None:
    lines = [
        "1 2 +",
        "",
        "word double ( i32 -> i32 ) 2 + end",
        "double",
        "qqq",
        "2147483647 +",
        '"done" print',
    ]
    output, errors = io.StringIO(), io.StringIO()
    run_repl(
        io.StringIO("\n".join(lines) + "\n"), output, errors, session=Session(output)
    )
    assert output.getvalue().splitlines() == [
        "stack: 3",
        "stack: 3",
        "stack: 5",
        "done",
        "stack: 5",
    ]
    assert errors.getvalue().splitlines() == [
        "repl:1:1: error: unknown word `qqq`",
        "repl:1:12: trap: integer overflow in `+` (i32 5, i32 2147483647)",
    ]


def run_transcript(lines: list[str]) -> tuple[list[str], list[str]]:
    output, errors = io.StringIO(), io.StringIO()
    run_repl(io.StringIO("\n".join(lines) + "\n"), output, errors, Session(output))
    return output.getvalue().splitlines(), errors.getvalue().splitlines()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("word sq ( i32 -> i32 )", True),
        ("word sq ( i32 -> i32 )\n  dup *", True),
        ("word sq ( i32 -> i32 )\n  dup *\nend", False),
        ("word", True),
        ("word sq (", True),
        ("type Shape =", True),
        ("type Shape = circle( f32 ) |", True),
        ("type Shape = circle( f32 )", False),
        ("const LIMIT =", True),
        ("{ 1 2", True),
        ("1 2 +   ; a comment", False),
        ("word sq ( i32 -> i32 ) ; a comment", True),
        (";( a block comment\nstill going", True),
        (";( a block comment ;)", False),
        ("", False),
        ("1 )", False),
        ("1 end", False),
        ("word 5", False),
        ('"unterminated', False),
        ("word sq ( i32 -> i32 ) end end", False),
    ],
)
def test_is_incomplete(text: str, expected: bool) -> None:
    assert Session().is_incomplete(text) is expected


def test_repl_reads_a_declaration_over_several_lines() -> None:
    output, errors = run_transcript(
        [
            "word square ( i32 -> i32 )",
            "  dup *",
            "end",
            "7 square",
            "type Shape =",
            "  circle( f32 ) |",
            "  dot",
            "2.0 circle",
            "{ 1",
            "  2 }",
        ]
    )
    assert errors == []
    assert output == [
        "stack: (empty)",
        "stack: 49",
        "stack: 49",
        "stack: 49 | 2.0 circle",
        "stack: 49 | 2.0 circle | { 1 2 }",
    ]


def test_repl_blank_line_submits_unfinished_input() -> None:
    output, errors = run_transcript(["word square ( i32 -> i32 )", "", "1 2 +"])
    assert errors == [
        "repl:3:1: error: expected `end` to close the definition of `square` on line 1"
    ]
    assert output == ["stack: 3"]


def test_repl_reports_unfinished_input_at_the_end() -> None:
    output, errors = run_transcript(["1", "word square ( i32 -> i32 )"])
    assert output == ["stack: 1"]
    assert len(errors) == 1
    assert "expected `end`" in errors[0]


def test_repl_error_lines_count_from_the_first_line_of_the_input() -> None:
    _, errors = run_transcript(["word bad ( -> )", "  1", "  qqq", "end"])
    assert errors == ["repl:3:3: error: unknown word `qqq`"]


def test_declaring_does_not_slow_down_with_the_number_of_declarations() -> None:
    session = Session(io.StringIO())
    for i in range(400):
        session.run_line(f"word w{i} ( i32 -> i32 ) 1 + end")
    session.run_line("0 w0 w399")
    assert session.stack == [integer(2)]


def test_a_failed_declaration_leaves_every_table_alone() -> None:
    session = session_after("type Shape = circle( f32 )", "const LIMIT = 5")
    before = {name: copy.copy(table) for name, table in vars(session.checker).items()}
    for line in (
        "type Other = circle( f32 )",
        "type Broken = broken( Missing ) | fine",
        "linear type Handle = handle( Missing )",
        "const LIMIT = 6",
        "const MIXED = { 1 2u8 }",
        "word bad ( Shape -> ) drop drop end",
    ):
        with pytest.raises(IndeErrors):
            session.run_line(line)
        assert vars(session.checker) == before, line
