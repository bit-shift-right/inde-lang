import io
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from inde import __main__ as command
from inde import ir
from inde.checker import check
from inde.errors import IndeTrap
from inde.interpreter import Interpreter
from inde.lowering import lower
from inde.machine import Machine
from inde.parser import parse
from inde.verifier import verify

Outcome = tuple[str, object, object, str]


def outcome_of(run: object, output: io.StringIO) -> Outcome:
    errors = io.StringIO()
    assert callable(run)
    try:
        status = run(errors)
    except IndeTrap as trap:
        location = trap.location and (trap.location.line, trap.location.column)
        return ("trap", trap.message, location, output.getvalue())
    return ("status", status, errors.getvalue(), output.getvalue())


def interpreted(source: str) -> Outcome:
    checked = check(parse(source), require_main=True)
    output = io.StringIO()
    interpreter = Interpreter(checked.words, checked.constants, output)
    return outcome_of(interpreter.run_main, output)


def compiled(source: str) -> ir.Module:
    module = lower(parse(source))
    verify(module)
    return module


def machine_run(module: ir.Module) -> Outcome:
    output = io.StringIO()
    return outcome_of(Machine(module, output).run_main, output)


def both(source: str) -> Outcome:
    expected = interpreted(source)
    assert machine_run(compiled(source)) == expected
    return expected


def printed(body: str, declarations: str = "") -> list[str]:
    result = both(f"{declarations}\nword main ( -> ) {body} end")
    assert result[:3] == ("status", 0, ""), result
    return result[3].splitlines()


def trap_of(body: str, declarations: str = "") -> tuple[object, object]:
    result = both(f"{declarations}\nword main ( -> ) {body} end")
    assert result[0] == "trap", result
    return result[1], result[2]


SHAPES = "type Shape = circle( f32 ) | rect( f32 f32 ) | dot\n"
COMBINATORS = """\
word dip ( ..a T Q -> ..b T ) where Q: [ ..a -> ..b ]
  swap [ ] curry compose call
end
word keep ( ..a T Q -> ..b T ) where Q: [ ..a T -> ..b ], T: Unrestricted
  over [ call ] dip
end
word times ( ..a Q i32 -> ..a ) where Q: [ ..a -> ..a ] & Unrestricted
  dup 0 > if 1 - swap dup rot [ [ call ] dip ] dip times else drop drop end
end
"""


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("1 2 + print 7 2 / print -7 2 % print", ["3", "3", "-1"]),
        ("200u8 100u8 +.wrap print 200u8 100u8 +.sat print", ["44", "255"]),
        ("1.5 2.25 * print 2.5 round print -0.5 ceil print", ["3.375", "2.0", "-0.0"]),
        ("2147483647 1 +.checked match case some print case none 0 print end", ["0"]),
        ("300 to-u8.wrap print 3.7 to-i32 print 'a' to-u32 print", ["44", "3", "97"]),
        ("1.0 to-bits print 1065353216u32 from-bits print", ["1065353216", "1.0"]),
        ("1 3u32 shl print 255u8 count-ones print", ["8", "8"]),
        (
            "true false xor print 1 2 < print 'a' 'b' >= print",
            ["true", "true", "false"],
        ),
        ("true 10 20 select print", ["10"]),
        ('"héllo" str-len print "ab" "cd" str-concat print', ["6", "abcd"]),
        ('"hello" 1u32 3u32 str-slice match case some print case none end', ["el"]),
        (
            (
                "strbuf-new 'a' strbuf-push \"bc\" strbuf-append strbuf-len print "
                "strbuf-freeze print"
            ),
            ["3", "abc"],
        ),
        ("unit print 1.5f64 print 'x' print", ["unit", "1.5", "x"]),
    ],
)
def test_values_and_primitives(body: str, expected: list[str]) -> None:
    assert printed(body) == expected


def test_data_and_matching() -> None:
    body = (
        "2.0 circle area print 3.0 4.0 rect area print dot area print "
        "'b' classify print 'z' classify print "
        '"yes" match case "yes" 1 case "no" 2 else 3 end print '
        "true match case true 1 case false 0 end print"
    )
    declarations = (
        SHAPES + "word area ( Shape -> f32 )\n"
        "  match case circle dup * 3.0 * case rect * case dot 0.0 end\n"
        "end\n"
        "word classify ( char -> i32 ) match case 'a' 1 case 'b' 2 else 0 end end\n"
    )
    assert printed(body, declarations) == ["12.0", "12.0", "0.0", "2", "0", "1", "1"]


def test_equality() -> None:
    declarations = "type Tree = leaf | node( Tree i32 Tree )\n"
    body = (
        "leaf 1 leaf node leaf 1 leaf node == print "
        "leaf 1 leaf node leaf 2 leaf node == print "
        "{ { 1 } { } } { { 1 } { } } == print { 1 2 } { 1 3 } != print "
        '"a" "a" == print unit unit != print 1 some none == print'
    )
    assert printed(body, declarations) == [
        "true",
        "false",
        "true",
        "true",
        "true",
        "false",
        "false",
    ]


def test_locals_and_loops() -> None:
    declarations = (
        "word sum-to ( i32 -> i32 )\n"
        "  0 swap while dup 0 > do swap over + swap 1 - end drop\n"
        "end\n"
        "word hypot ( f32 f32 -> f32 ) let x y in x x * y y * + sqrt end end\n"
    )
    assert printed(
        "4 sum-to print 3.0 4.0 hypot print 1 let x in end", declarations
    ) == [
        "10",
        "5.0",
    ]


def test_quotations() -> None:
    body = (
        "1 2 [ 10 * ] dip + print "
        "5 [ 1 + ] keep + print "
        "0 [ 2 + ] 5 times print "
        "[ 1 + ] [ 2 * ] compose 5 swap call print "
        "3 [ + ] curry 4 swap call print"
    )
    assert printed(body, COMBINATORS) == ["12", "11", "10", "12", "7"]


def test_consume() -> None:
    body = (
        "0 { 1 2 3 } arraybuf-from-array [ + ] arraybuf-consume print "
        'arraybuf-new "a" strbuf-from-str arraybuf-push "b" strbuf-from-str '
        "arraybuf-push strbuf-new swap [ strbuf-freeze strbuf-append ] "
        "arraybuf-consume strbuf-freeze print"
    )
    assert printed(body) == ["6", "ab"]


def test_buffers() -> None:
    body = (
        "arraybuf-new 1 arraybuf-push 2 arraybuf-push arraybuf-pop "
        "match case some print case none end "
        "0u32 9 arraybuf-replace print arraybuf-freeze 0u32 array-get print "
        "{ 1 2 3 } 1u32 3u32 array-slice array-len print "
        "{ 1 } 5u32 array-get.checked match case some print case none 0 print end"
    )
    assert printed(body) == ["2", "1", "9", "2", "0"]


def test_question_mark() -> None:
    declarations = (
        "word half ( i32 -> Result[i32, str] )\n"
        '  dup 2 % 0 == if 2 / ok else drop "odd" err end\n'
        "end\n"
        "word quarter ( i32 -> Result[i32, str] )\n"
        '  "note" swap half ? swap drop half\n'
        "end\n"
        "word show ( Result[i32, str] -> ) match case ok print case err print end end\n"
    )
    assert printed("8 quarter show 6 quarter show 3 quarter show", declarations) == [
        "2",
        "odd",
        "odd",
    ]


def test_main_returning_a_result() -> None:
    assert both("word main ( -> Result[unit, str] ) unit ok end") == (
        "status",
        0,
        "",
        "",
    )
    assert both('word main ( -> Result[unit, str] ) "bad input" err end') == (
        "status",
        1,
        "bad input\n",
        "",
    )


@pytest.mark.parametrize(
    ("body", "message", "location"),
    [
        (
            "2147483647 let n in n 1 + drop end",
            "integer overflow in `+` (i32 2147483647, i32 1)",
            (2, 42),
        ),
        (
            "0 let d in 1 d / drop end",
            "division by zero in `/` (i32 1, i32 0)",
            (2, 33),
        ),
        (
            "{ 1 2 } let a in a 5u32 array-get drop end",
            "index out of range in `array-get` (index 5, length 2)",
            (2, 42),
        ),
        (
            "300 let n in n to-u8 drop end",
            "value out of range in `to-u8` (i32 300)",
            (2, 33),
        ),
        (
            "1.0 let x in x 0.0 / drop end",
            "division by zero in `/` (f32 1.0, f32 0.0)",
            (2, 37),
        ),
    ],
)
def test_traps(body: str, message: str, location: tuple[int, int]) -> None:
    assert trap_of(body) == (message, location)


def test_output_before_a_trap_is_kept() -> None:
    result = both('word main ( -> ) "before" print 0 let d in 1 d / drop end end')
    assert result == (
        "trap",
        "division by zero in `/` (i32 1, i32 0)",
        (1, 48),
        "before\n",
    )


def test_tail_calls_do_not_grow_the_call_stack() -> None:
    source = (
        "word count ( i32 i32 -> i32 )\n"
        "  let total n in n 0 == if total else total 1 + n 1 - count end end\n"
        "end\n"
        "word main ( -> ) 0 100000 count print end"
    )
    machine = Machine(compiled(source), io.StringIO())
    assert machine.run_main() == 0
    assert machine.deepest <= 3


def test_call_depth() -> None:
    source = (
        "word deep ( i32 -> i32 ) dup 0 == if else 1 - deep 1 + end end\n"
        "word main ( -> ) 200000 deep print end"
    )
    assert both(source)[:3] == ("trap", "call depth exceeded", (1, 47))


def test_debug_information_only_affects_diagnostics() -> None:
    source = "word main ( -> ) 7 print 0 let d in 1 d / drop end end"
    module = compiled(source)

    def stripped(body: ir.Body) -> ir.Body:
        result = []
        for instruction in body:
            instruction = replace(instruction, location=None)
            for name in ("then_body", "else_body", "condition", "body"):
                inner = getattr(instruction, name, None)
                if inner is not None:
                    instruction = replace(instruction, **{name: stripped(inner)})  # type: ignore[arg-type]
            result.append(instruction)
        return tuple(result)

    for function in module.functions:
        function.body = stripped(function.body)
    assert machine_run(module) == (
        "trap",
        "division by zero in `/` (i32 1, i32 0)",
        None,
        "7\n",
    )


def test_run_ir_on_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    program = tmp_path / "p.inde"
    program.write_text(
        'word main ( -> ) "hi" print 0 let d in 1 d / drop end end', encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["inde", "--run-ir", str(program)])
    assert command.main() == 2
    captured = capsys.readouterr()
    assert captured.out == "hi\n"
    assert "trap: division by zero in `/` (i32 1, i32 0)" in captured.err
