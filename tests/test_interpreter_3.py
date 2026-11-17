import io

import pytest

from inde.checker import Checker, check
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Interpreter, Session
from inde.parser import parse
from inde.type_model import I32, STR, Type, option_of

LIST = "type List[T] = nil | cons( T List[T] )\n"


def run(source: str) -> tuple[list[str], int, str]:
    checked = check(parse(source), require_main=True)
    output, errors = io.StringIO(), io.StringIO()
    interpreter = Interpreter(checked.words, checked.constants, output)
    status = interpreter.run_main(errors)
    return output.getvalue().splitlines(), status, errors.getvalue()


def printed(body: str, declarations: str = "") -> list[str]:
    lines, status, _ = run(f"{declarations}\nword main ( -> ) {body} end")
    assert status == 0
    return lines


def session_after(*lines: str) -> Session:
    session = Session(io.StringIO(), Checker())
    for line in lines:
        session.run_line(line)
    return session


def test_unit() -> None:
    assert printed("unit print unit unit == print") == ["unit", "true"]
    assert session_after("unit").show_stack() == "stack: unit"


def test_strings_are_ordered() -> None:
    assert printed('"apple" "banana" < print "b" "a" < print "a" "a" <= print') == [
        "true",
        "false",
        "true",
    ]


def test_equality_ignores_how_a_value_was_typed() -> None:
    assert printed("1 0 /.checked 2147483647 1 +.checked == print") == ["true"]
    assert printed("4 2 /.checked 1 1 +.checked == print") == ["true"]
    assert printed("4 2 /.checked 1 0 /.checked == print") == ["false"]


def test_generic_words() -> None:
    declarations = (
        "word twice ( T -> T T ) where T: Unrestricted dup end\n"
        "word square ( T -> T ) where T: Num dup * end\n"
        "word larger ( T T -> T ) where T: Ord & Unrestricted\n"
        "  over over < if swap drop else drop end\n"
        "end\n"
    )
    assert printed("5 twice + print", declarations) == ["10"]
    assert printed('"a" twice print print', declarations) == ["a", "a"]
    assert printed(
        "3 square print 1.5 square print 2u8 square print", declarations
    ) == [
        "9",
        "2.25",
        "4",
    ]
    assert printed('3 7 larger print "b" "a" larger print', declarations) == ["7", "b"]


def test_a_generic_word_still_traps_in_the_callers_type() -> None:
    declarations = "word square ( T -> T ) where T: Num dup * end\n"
    checked = check(parse(f"{declarations}word main ( -> ) 200u8 square print end"))
    interpreter = Interpreter(checked.words, checked.constants, io.StringIO())
    with pytest.raises(IndeTrap) as trap:
        interpreter.run_main()
    assert trap.value.message == "integer overflow in `*` (u8 200u8, u8 200u8)"


def test_equality_of_declared_types() -> None:
    declarations = "type Point = point( i32 i32 )\ntype Colour = red | green\n"
    assert printed("1 2 point 1 2 point == print", declarations) == ["true"]
    assert printed("1 2 point 2 1 point == print", declarations) == ["false"]
    assert printed("red green != print red red == print", declarations) == [
        "true",
        "true",
    ]
    assert printed("{ 1 2 } { 1 2 } == print { 1 } { 1 2 } == print") == [
        "true",
        "false",
    ]


def test_unpack_words() -> None:
    assert printed("1 2 pair pair-unpack + print") == ["3"]
    assert printed('1 "a" pair pair-unpack print print') == ["a", "1"]
    declarations = "linear type Handle = handle( u32 )\n"
    assert printed("7u32 handle handle-unpack print", declarations) == ["7"]


def test_values_in_the_repl() -> None:
    session = session_after("5 some", '"a" 2.5 pair')
    assert session.show_stack() == 'stack: 5 some | "a" 2.5 pair'
    assert [str(value.type) for value in session.stack] == [
        "Option[i32]",
        "Pair[str, f32]",
    ]
    session.run_line("pair-unpack")
    assert session.show_stack() == 'stack: 5 some | "a" | 2.5'


def test_match_on_options() -> None:
    declarations = (
        "word or-zero ( Option[i32] -> i32 ) match case none 0 case some end end\n"
    )
    assert printed("5 some or-zero print none or-zero print", declarations) == [
        "5",
        "0",
    ]


def test_results() -> None:
    declarations = (
        "word safe-div ( i32 i32 -> Result[i32, str] )\n"
        '  dup 0 == if drop drop "division by zero" err else / ok end\n'
        "end\n"
        "word show ( Result[i32, str] -> ) match case ok print case err print end end\n"
    )
    assert printed("7 2 safe-div show 1 0 safe-div show", declarations) == [
        "3",
        "division by zero",
    ]


def test_checked_arithmetic_can_be_used_now() -> None:
    body = (
        '2147483647 1 +.checked match case none "overflow" print case some print end '
        '1 2 +.checked match case none "overflow" print case some print end'
    )
    assert printed(body) == ["overflow", "3"]


def test_generic_words_over_generic_types() -> None:
    declarations = LIST + (
        "word or-else ( Option[T] T -> T ) where T: Unrestricted\n"
        "  swap match case none case some swap drop end\n"
        "end\n"
        "word length ( List[T] -> i32 ) where T: Unrestricted\n"
        "  match case nil 0 case cons length swap drop 1 + end\n"
        "end\n"
        "word sum ( List[i32] -> i32 ) match case nil 0 case cons sum + end end\n"
    )
    assert printed(
        '5 some 0 or-else print none "none" or-else print', declarations
    ) == [
        "5",
        "none",
    ]
    assert printed("1 2 3 nil cons cons cons length print", declarations) == ["3"]
    assert printed('"a" "b" nil cons cons length print', declarations) == ["2"]
    assert printed("1 2 3 nil cons cons cons sum print", declarations) == ["6"]


def test_equality_of_types_with_parameters() -> None:
    assert printed("5 some 5 some == print 5 some none == print") == ["true", "false"]
    assert printed("none 5 some != print") == ["true"]
    declarations = LIST
    assert printed("1 2 nil cons cons 1 2 nil cons cons == print", declarations) == [
        "true"
    ]
    assert printed("1 nil cons nil == print", declarations) == ["false"]


def test_main_returning_a_result() -> None:
    assert run("word main ( -> Result[unit, str] ) unit ok end") == ([], 0, "")
    assert run('word main ( -> Result[unit, str] ) "it broke" err end') == (
        [],
        1,
        "it broke\n",
    )
    assert run("word main ( -> Result[unit, i32] ) 1 print 42 err end") == (
        ["1"],
        1,
        "42\n",
    )


def test_the_repl_knows_the_exact_type_of_each_value() -> None:
    session = session_after("true if none else 5 some end")
    assert session.stack[0].type == option_of(I32)
    session.run_line("match case none 0 case some 1 + end")
    assert session.show_stack() == "stack: 0"

    session = session_after('1 0 == if { } else { "a" } end')
    assert session.stack[0].type == Type("array", (STR,))


def test_the_repl_needs_every_type() -> None:
    session = session_after("1")
    for line in ("none", "{ }", "unit ok"):
        with pytest.raises(IndeErrors) as failure:
            session.run_line(line)
        assert "cannot work out" in failure.value.errors[0].message
    assert session.show_stack() == "stack: 1"
    session.run_line("none 5 some ==")
    assert session.show_stack() == "stack: 1 | false"


def test_a_type_holding_an_array_of_its_parameter() -> None:
    declarations = (
        "type Bag[T] = bag( array[T] ) | empty\n"
        "word count ( Bag[T] -> i32 ) where T: Unrestricted\n"
        "  match case bag drop 1 case empty 0 end\n"
        "end\n"
        "word nothing ( -> Bag[str] ) empty end\n"
    )
    assert printed('{ "a" } bag count print nothing count print', declarations) == [
        "1",
        "0",
    ]
    session = session_after("type Bag[T] = bag( array[T] ) | empty", "{ 1 2 } bag")
    assert session.show_stack() == "stack: { 1 2 } bag"


def test_repl_types_with_parameters_from_earlier_lines() -> None:
    session = session_after(
        "type Box[T] = box( T )",
        "type Crate[T] = crate( Box[T] Box[T] )",
        "word open ( Box[T] -> T ) box-unpack end",
        "1 box 2 box crate crate-unpack open swap open +",
    )
    assert session.show_stack() == "stack: 3"
    assert not session.is_incomplete("word w ( Box[i32] -> i32 ) open end")
    with pytest.raises(IndeErrors) as failure:
        session.run_line("word w ( Box -> ) drop end")
    assert "`Box` takes 1 type argument" in failure.value.errors[0].message
