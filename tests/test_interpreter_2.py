import io
from typing import Any

import pytest

from inde.checker import check
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Interpreter, Session
from inde.parser import parse
from inde.repl import run_repl

SHAPES = "type Shape = circle( f32 ) | rect( f32 f32 ) | dot\n"


def run_program(source: str) -> tuple[list[str], Any]:
    checked = check(parse(source), require_main=True)
    output = io.StringIO()
    interpreter = Interpreter(checked.words, checked.constants, output)
    assert interpreter.run_main() == 0
    return output.getvalue().splitlines(), interpreter


def printed(body: str, declarations: str = "") -> list[str]:
    lines, _ = run_program(f"{declarations}\nword main ( -> ) {body} end")
    return lines


def get_trap_message(body: str, declarations: str = "") -> str:
    checked = check(parse(f"{declarations}\nword main ( -> ) {body} end"))
    interpreter = Interpreter(checked.words, checked.constants, io.StringIO())
    with pytest.raises(IndeTrap) as trap:
        interpreter.run_main()
    return trap.value.message


def session_after(*lines: str) -> Any:
    session = Session(io.StringIO())
    for line in lines:
        session.run_line(line)
    return session


def test_if_runs_one_branch() -> None:
    assert printed('true if "yes" print else "no" print end') == ["yes"]
    assert printed('false if "yes" print else "no" print end') == ["no"]
    assert printed('1 2 < if "less" else "more" end print') == ["less"]


def test_if_without_else() -> None:
    assert printed('true if "ran" print end "after" print') == ["ran", "after"]
    assert printed('false if "ran" print end "after" print') == ["after"]


def test_empty_branches() -> None:
    assert printed('true if else "no" print end "after" print') == ["after"]
    assert printed('false if "yes" print else end "after" print') == ["after"]


def test_nested_if() -> None:
    declarations = (
        "word sign ( i32 -> str )\n"
        '  dup 0 < if drop "negative" else 0 == if "zero" else "positive" end end\n'
        "end\n"
    )
    assert printed("-5 sign print 0 sign print 7 sign print", declarations) == [
        "negative",
        "zero",
        "positive",
    ]


def test_code_after_a_branch_runs() -> None:
    assert printed("true if 1 else 2 end 10 + print") == ["11"]


def test_short_circuit_idiom() -> None:
    declarations = 'word loud ( -> bool ) "evaluated" print true end\n'
    assert printed("false if loud else false end print", declarations) == ["false"]
    assert printed("true if loud else false end print", declarations) == [
        "evaluated",
        "true",
    ]


def test_recursion_that_ends() -> None:
    declarations = (
        "word factorial ( i32 -> i32 )\n"
        "  dup 1 <= if drop 1 else dup 1 - factorial * end\n"
        "end\n"
    )
    assert printed("10 factorial print 1 factorial print", declarations) == [
        "3628800",
        "1",
    ]


def test_a_trap_inside_a_branch() -> None:
    assert get_trap_message("true if 1 0 / print end") == (
        "division by zero in `/` (i32 1, i32 0)"
    )


def test_a_call_at_the_end_of_a_branch_is_a_tail_call() -> None:
    source = (
        "word count ( i32 i32 -> i32 )\n"
        "  dup 0 == if drop else 1 - swap 1 + swap count end\n"
        "end\n"
        "word main ( -> ) 0 50000 count print end\n"
    )
    lines, interpreter = run_program(source)
    assert lines == ["50000"]
    assert interpreter.deepest <= 2


def test_a_call_that_is_not_last_in_its_branch_nests() -> None:
    source = (
        "word depth ( i32 -> i32 )\n"
        "  dup 0 == if else 1 - depth 1 + end\n"
        "end\n"
        "word main ( -> ) 500 depth print end\n"
    )
    lines, interpreter = run_program(source)
    assert lines == ["500"]
    assert interpreter.deepest == 502


def test_runaway_recursion_through_if_still_traps() -> None:
    declarations = "word down ( i32 -> i32 ) dup 0 < if else 1 + down 1 + end end\n"
    assert get_trap_message("0 down print", declarations) == "call depth exceeded"


def test_repl_reads_a_form_over_several_lines() -> None:
    output, errors = io.StringIO(), io.StringIO()
    lines = ["true if", "  1", "else", "  2", "end"]
    run_repl(io.StringIO("\n".join(lines) + "\n"), output, errors, Session(output))
    assert errors.getvalue() == ""
    assert output.getvalue().splitlines() == ["stack: 1"]


def test_repl_checks_a_form_before_running_it() -> None:
    session = session_after("1")
    with pytest.raises(IndeErrors):
        session.run_line('true if "printed too early" print 2 end')
    assert session.show_stack() == "stack: 1"


def test_while() -> None:
    assert printed("3 while dup 0 > do dup print 1 - end drop") == ["3", "2", "1"]
    assert printed('while false do "never" print end "after" print') == ["after"]
    assert printed("0 while dup 3 < do 1 + end print") == ["3"]


def test_the_condition_runs_once_more_than_the_body() -> None:
    declarations = 'word check ( i32 -> i32 bool ) "check" print dup 2 < end\n'
    assert printed('0 while check do "body" print 1 + end drop', declarations) == [
        "check",
        "body",
        "check",
        "body",
        "check",
    ]


def test_nested_loops() -> None:
    body = (
        "0 0 while dup 3 < do "
        "0 while dup 4 < do rot 1 + rot rot 1 + end drop "
        "1 + end drop print"
    )
    assert printed(body) == ["12"]


def test_a_long_loop_does_not_grow_the_frame_stack() -> None:
    source = (
        "word sum-to ( i32 -> i32 )\n"
        "  0 swap\n"
        "  while dup 0 > do swap over + swap 1 - end\n"
        "  drop\n"
        "end\n"
        "word main ( -> ) 20000 sum-to print end\n"
    )
    lines, interpreter = run_program(source)
    assert lines == ["200010000"]
    assert interpreter.deepest == 2


def test_a_call_at_the_end_of_a_loop_body_is_not_a_tail_call() -> None:
    declarations = "word step ( i32 -> i32 ) dup print 1 - end\n"
    assert printed("3 while dup 0 > do step end drop", declarations) == [
        "3",
        "2",
        "1",
    ]


def test_a_call_at_the_end_of_a_condition_is_not_a_tail_call() -> None:
    declarations = "word positive ( i32 -> i32 bool ) dup 0 > end\n"
    assert printed("2 while positive do dup print 1 - end drop", declarations) == [
        "2",
        "1",
    ]


def test_if_inside_a_loop() -> None:
    body = "0 while dup 6 < do dup 2 % 0 == if dup print end 1 + end drop"
    assert printed(body) == ["0", "2", "4"]


def test_a_trap_inside_a_loop() -> None:
    assert get_trap_message("3 while true do 10 over / drop 1 - end drop") == (
        "division by zero in `/` (i32 10, i32 0)"
    )


def test_endless_recursion_inside_a_loop_traps() -> None:
    declarations = "word spin ( -> ) while true do spin end end\n"
    assert get_trap_message("spin", declarations) == "call depth exceeded"


def test_let() -> None:
    assert printed("7 let n in n n * print end") == ["49"]
    assert printed('1 "a" let x y in y print x print end') == ["a", "1"]
    assert printed("1 2 3 let a b c in a print b print c print end") == [
        "1",
        "2",
        "3",
    ]


def test_the_body_can_leave_values() -> None:
    assert printed("3 4 let x y in x y + x y * end print print") == ["12", "7"]


def test_an_unused_local_is_dropped() -> None:
    assert printed('1 2 let x in end print "after" print') == ["1", "after"]


def test_nested_scopes() -> None:
    body = "1 let x in 2 let y in x y + print end x print end"
    assert printed(body) == ["3", "1"]


def test_locals_are_visible_in_nested_blocks() -> None:
    assert printed('5 let n in n 3 > if "big" print n print end end') == ["big", "5"]
    assert printed("3 let n in 0 while dup n < do dup print 1 + end drop end") == [
        "0",
        "1",
        "2",
    ]


def test_each_call_has_its_own_locals() -> None:
    declarations = (
        "word show ( i32 -> ) let n in n 0 > if n 1 - show end n print end end\n"
    )
    assert printed("3 show", declarations) == ["0", "1", "2", "3"]


def test_a_callee_does_not_see_the_callers_locals() -> None:
    source = (
        "word peek ( -> i32 ) x end\nword main ( -> ) 1 let x in peek print end end"
    )
    with pytest.raises(IndeErrors) as failure:
        check(parse(source))
    assert "unknown word `x`" in failure.value.errors[0].message


def test_a_call_at_the_end_of_a_let_body_is_a_tail_call() -> None:
    source = (
        "word count ( i32 i32 -> i32 )\n"
        "  let total left in\n"
        "    left 0 == if total else total 1 + left 1 - count end\n"
        "  end\n"
        "end\n"
        "word main ( -> ) 0 50000 count print end\n"
    )
    lines, interpreter = run_program(source)
    assert lines == ["50000"]
    assert interpreter.deepest <= 2


def test_let_in_the_repl() -> None:
    session = session_after("2 3", "let x y in x y * x y + end")
    assert session.show_stack() == "stack: 6 | 5"
    with pytest.raises(IndeErrors):
        session.run_line("x")


def test_match_on_integers() -> None:
    declarations = (
        "word name ( i32 -> str )\n"
        '  match case 0 "zero" case 1 "one" else "many" end\n'
        "end\n"
    )
    assert printed("0 name print 1 name print 7 name print", declarations) == [
        "zero",
        "one",
        "many",
    ]


def test_match_on_other_literals() -> None:
    assert printed("'b' match case 'a' 1 case 'b' 2 else 0 end print") == ["2"]
    assert printed('"no" match case "yes" 1 case "no" 2 else 0 end print') == ["2"]
    assert printed('"maybe" match case "yes" 1 else 0 end print') == ["0"]
    assert printed("false match case true 1 case false 2 end print") == ["2"]
    assert printed("200u8 match case 200u8 1 else 0 end print") == ["1"]


def test_match_on_constants() -> None:
    declarations = "const LIMIT = 5\nconst NEWLINE = '\\n'\n"
    assert printed("5 match case LIMIT 1 else 0 end print", declarations) == ["1"]
    assert printed("4 match case LIMIT 1 else 0 end print", declarations) == ["0"]
    assert printed("'\\n' match case NEWLINE 1 else 0 end print", declarations) == ["1"]


def test_only_one_arm_runs() -> None:
    body = '1 match case 0 "zero" print case 1 "one" print else "other" print end'
    assert printed(body) == ["one"]


def test_the_scrutinee_is_consumed() -> None:
    assert printed("9 1 match case 1 else end print") == ["9"]


def test_match_on_constructors() -> None:
    declarations = SHAPES + (
        "word area ( Shape -> f32 )\n"
        "  match\n"
        "    case circle dup * 3.0 *\n"
        "    case rect *\n"
        "    case dot 0.0\n"
        "  end\n"
        "end\n"
    )
    body = "2.0 circle area print 2.0 3.5 rect area print dot area print"
    assert printed(body, declarations) == ["12.0", "7.0", "0.0"]


def test_fields_arrive_first_field_deepest() -> None:
    declarations = "type Tagged = tagged( i32 str )\n"
    assert printed('1 "a" tagged match case tagged print print end', declarations) == [
        "a",
        "1",
    ]


def test_recursive_data() -> None:
    declarations = (
        "type Chain = link( i32 Chain ) | stop\n"
        "word total ( Chain -> i32 ) match case link total + case stop 0 end end\n"
    )
    assert printed("1 2 3 stop link link link total print", declarations) == ["6"]


def test_a_call_at_the_end_of_an_arm_is_a_tail_call() -> None:
    source = (
        "word count ( i32 i32 -> i32 )\n"
        "  dup match\n"
        "    case 0 drop\n"
        "    else 1 - swap 1 + swap count\n"
        "  end\n"
        "end\n"
        "word main ( -> ) 0 50000 count print end\n"
    )
    lines, interpreter = run_program(source)
    assert lines == ["50000"]
    assert interpreter.deepest <= 2


def test_everything_together() -> None:
    source = SHAPES + (
        "const SIDES = 4\n"
        "word describe ( Shape -> )\n"
        "  match\n"
        '    case circle let r in r 1.0 > if "big circle" else "small circle" end end\n'
        '    case rect == if "square" else "rectangle" end\n'
        '    case dot "dot"\n'
        "  end\n"
        "  print\n"
        "end\n"
        "word main ( -> )\n"
        "  0 while dup SIDES < do\n"
        "    dup match\n"
        "      case 0 2.0 circle describe\n"
        "      case 1 0.5 circle describe\n"
        "      case 2 1.0 1.0 rect describe\n"
        "      else dot describe\n"
        "    end\n"
        "    1 +\n"
        "  end drop\n"
        "end\n"
    )
    lines, _ = run_program(source)
    assert lines == ["big circle", "small circle", "square", "dot"]


def test_repl_refuses_a_word_named_like_an_earlier_local() -> None:
    session = session_after("word double ( i32 -> i32 ) let n in n n + end end")
    with pytest.raises(IndeErrors) as failure:
        session.run_line("word n ( -> i32 ) 1 end")
    assert [error.message for error in failure.value.errors] == [
        "`n` is a local in `double`, so it cannot name a word"
    ]
    assert failure.value.errors[0].location is not None
    assert failure.value.errors[0].location.column == 1
    session.run_line("4 double")
    assert session.show_stack() == "stack: 8"


def test_repl_locals_of_a_line_or_a_failed_word_do_not_reserve_names() -> None:
    session = session_after("1 let n in n end")
    with pytest.raises(IndeErrors):
        session.run_line("word broken ( i32 -> ) let m in nope end end")
    session.run_line("word n ( -> i32 ) 2 end")
    session.run_line("word m ( -> i32 ) 3 end")
    session.run_line("n m")
    assert session.show_stack() == "stack: 1 | 2 | 3"


def test_inspect_still_quotes_text() -> None:
    session = session_after('"say \\"hi\\"\\n" \'\\\'\' \'"\' "it\'s"')
    assert session.show_stack() == (
        'stack: "say \\"hi\\"\\n" | \'\\\'\' | \'"\' | "it\'s"'
    )
