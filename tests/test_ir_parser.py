import io
import sys
from pathlib import Path

import pytest

from inde import __main__ as command
from inde import ir
from inde.ir_parser import IRSyntaxError, parse_module
from inde.lowering import lower
from inde.machine import Machine
from inde.parser import parse
from inde.type_model import I32
from inde.verifier import IRError, verify

PROGRAMS = {
    "values": (
        'word main ( -> ) "héllo\\n\\t\\"q\\"" print 1.5 print -0.0 print '
        "2147483647 print -128i8 print 'a' print '\\u{1F600}' print true print "
        "200u8 100u8 +.wrap print 1.0f64 to-bits print end"
    ),
    "data": (
        "type Shape = circle( f32 ) | rect( f32 f32 ) | dot\n"
        "word area ( Shape -> f32 )\n"
        "  match case circle dup * 3.0 * case rect * case dot 0.0 end\n"
        "end\n"
        "word main ( -> ) 2.0 circle area print { { 1 2 } { } } array-len print "
        '{ "a" "b" } 1u32 array-get print '
        "3 some match case some print case none end end"
    ),
    "control": (
        "word sum-to ( i32 -> i32 )\n"
        "  0 swap while dup 0 > do swap over + swap 1 - end drop\n"
        "end\n"
        "word classify ( char -> i32 ) match case 'a' 1 case 'b' 2 else 0 end end\n"
        "word main ( -> ) 4 sum-to print 'b' classify print "
        '"no" match case "yes" 1 else 0 end print '
        "3.0 4.0 let x y in x x * y y * + sqrt print end end"
    ),
    "quotations": (
        "word dip ( ..a T Q -> ..b T ) where Q: [ ..a -> ..b ]\n"
        "  swap [ ] curry compose call\n"
        "end\n"
        "word <=> ( i32 i32 -> i32 ) - end\n"
        "word main ( -> ) 1 2 [ 10 * ] dip + print 5 3 <=> print "
        "0 { 1 2 3 } arraybuf-from-array [ + ] arraybuf-consume print end"
    ),
    "results": (
        "word half ( i32 -> Result[i32, str] )\n"
        '  dup 2 % 0 == if 2 / ok else drop "odd" err end\n'
        "end\n"
        "word main ( -> Result[unit, str] ) 8 half ? print unit ok end"
    ),
}

HAND_WRITTEN = """\
module
  runtime 1

  import rt.print<i32> (i32) -> ()

  ; factorial with a loop and two locals
  func $factorial (i32) -> (i32)
    local $n i32
    local $total i32
    local.set $n
    i32.const 1
    local.set $total
    while []
      local.get $n
      i32.const 1
      i32.gt
    do
      local.take $total
      local.get $n
      i32.mul
      local.set $total
      local.take $n
      i32.const 1
      i32.sub
      local.set $n
    end
    local.drop $n
    local.take $total
  end

  func $main () -> ()
    i32.const 5  ; 9:3
    call $factorial  ; 9:5 factorial
    rt.print<i32>
  end

  entry $main unit
end
"""


def lowered(source: str) -> ir.Module:
    module = lower(parse(source))
    verify(module)
    return module


def run(module: ir.Module) -> str:
    output = io.StringIO()
    assert Machine(module, output).run_main(io.StringIO()) in (0, 1)
    return output.getvalue()


@pytest.mark.parametrize("name", sorted(PROGRAMS))
@pytest.mark.parametrize("debug", [False, True])
def test_round_trip(name: str, debug: bool) -> None:
    module = lowered(PROGRAMS[name])
    text = ir.format_module(module, debug=debug)
    parsed = parse_module(text)
    assert ir.format_module(parsed, debug=debug) == text
    if debug:
        assert parsed == module
    verify(parsed)
    assert run(parsed) == run(module)


def test_debug_comments_restore_locations_and_words() -> None:
    module = parse_module(
        ir.format_module(lowered("word main ( -> ) 1 2 + print end"), debug=True)
    )
    body = module.functions[0].body
    assert [
        (i.location and i.location.line, i.location and i.location.column) for i in body
    ] == [(1, 18), (1, 20), (1, 22), (1, 24)]
    assert [i.word for i in body] == [None, None, "+", "print"]


def test_comments_and_blank_lines_are_ignored() -> None:
    text = wrapped("    ; nothing here\n    unit.const  ; a note\n    drop.unit\n")
    plain = wrapped("    unit.const\n    drop.unit\n")
    assert parse_module("; header\n\n" + text) == parse_module(plain)


@pytest.mark.parametrize(
    "word", ["<=>", "a.b", "x#y", 'say"hi"', "plain-word", "with space"]
)
def test_quoted_function_names(word: str) -> None:
    name = ir.function_name(word)
    specialized = ir.function_name(word, (I32,))
    text = (
        "module\n  runtime 1\n\n"
        f"  func {name} () -> ()\n  end\n\n"
        f"  func {specialized}.q0 () -> ()\n    call {name}\n  end\n\n"
        f"  func {name}#1 () -> ()\n    return_call {specialized}.q0\n  end\n\n"
        f"  entry {name}#1 unit\nend\n"
    )
    module = parse_module(text)
    assert [f.name for f in module.functions] == [
        name,
        f"{specialized}.q0",
        f"{name}#1",
    ]
    assert ir.format_module(module) == text
    verify(module)


def test_names_that_need_quoting() -> None:
    assert ir.function_name("plain-word") == "$plain-word"
    assert ir.function_name("<=>") == '$"<=>"'
    assert ir.function_name("a.b") == '$"a.b"'
    assert ir.function_name('say"hi"') == '$"say\\"hi\\""'


def test_hand_written_module() -> None:
    module = parse_module(HAND_WRITTEN)
    verify(module)
    assert run(module) == "120\n"
    call = module.functions[1].body[1]
    assert isinstance(call, ir.Call) and call.word == "factorial"


def wrapped(body: str) -> str:
    return (
        "module\n  runtime 1\n  func $main () -> ()\n"
        f"{body}  end\n  entry $main unit\nend\n"
    )


@pytest.mark.parametrize(
    ("text", "line", "message"),
    [
        (wrapped("    i32.const x\n"), 4, "x"),
        (wrapped("    i32.const 1 2\n"), 4, "unexpected `2`"),
        (wrapped("    f32.const 1.5.5\n"), 4, "1.5.5"),
        (wrapped("    char.const 'a'\n"), 4, ""),
        (wrapped("    char.const U+110000\n"), 4, "U+110000"),
        (wrapped("    i32.const 1_000\n"), 4, "1_000"),
        (wrapped("    i32.const +1\n"), 4, "+1"),
        (wrapped("    f64.const 1/2\n"), 4, "1/2"),
        (wrapped("    f64.const nan\n"), 4, "nan"),
        ('module\n  runtime 1\n  data $s0 = utf8 "\\u{zz}"\nend\n', 3, "\\u{zz}"),
        ('module\n  runtime 1\n  data $s0 = utf8 "\\u{110000}"\nend\n', 3, ""),
        (wrapped("    frob\n"), 4, "frob"),
        (wrapped("    local.get missing\n"), 4, ""),
        (wrapped("    drop.%\n"), 4, ""),
        (wrapped("    if\n      i32.const 1\n    end\n"), 4, "`[`"),
        (wrapped("    if [->]\n      unit.const\n"), 7, ""),
        ('module\n  runtime 1\n  data $s0 = utf8 "open\nend\n', 3, ""),
        ('module\n  runtime 1\n  data $s0 = utf8 "\\q"\nend\n', 3, ""),
        ("module\n  runtime 1\n  func $main () -> ()\n    unit.const\n", 4, "end"),
        ("modul\nend\n", 1, ""),
        ("module\n  runtime 1\n  type %T = variant { a b(i32)\nend\n", 3, "`}`"),
        ("module\n  runtime 1\n  type %T = variant { a(i32\nend\n", 3, ""),
        ("module\n  runtime 1\n  data $a = array<i32> [1 2\nend\n", 3, ""),
        ("module\n  runtime one\nend\n", 2, "number"),
        ("module\n  runtime 1\nend\n", 3, "no entry"),
        (wrapped("").replace("unit\n", "unit\n  entry $main unit\n"), 6, "one entry"),
        (wrapped("") + "extra\n", 7, "follow"),
        ("module\n  runtime 1\n  bogus\nend\n", 3, "bogus"),
    ],
)
def test_syntax_errors(text: str, line: int, message: str) -> None:
    with pytest.raises(IRSyntaxError) as failure:
        parse_module(text)
    assert failure.value.line == line
    assert message in failure.value.message
    assert str(failure.value).startswith(f"line {line}: ")


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        ("    i32.frob\n", "there is no i32.frob"),
        ("    u8.const 300\n    drop.u8\n", "300 is not a u8"),
        ("    unit.const\n    drop.%Nope\n", "%Nope is not in the type table"),
        ("    call $missing\n", "there is no function $missing"),
        ("    i32.const 1\n", "its results are ()"),
    ],
)
def test_meaning_is_left_to_the_verifier(body: str, problem: str) -> None:
    module = parse_module(wrapped(body))
    with pytest.raises(IRError) as failure:
        verify(module)
    assert any(problem in line for line in failure.value.problems)


def test_unrecognized_comments_are_not_debug_information() -> None:
    module = parse_module(wrapped("    unit.const  ; one:two\n    drop.unit  ; 3:4\n"))
    first, second = module.functions[0].body
    assert first.location is None and first.word is None
    assert second.location is not None and second.location.line == 3


def test_run_ir_accepts_ir_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    program = tmp_path / "factorial.ir"
    program.write_text(HAND_WRITTEN, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["inde", "--run-ir", str(program)])
    assert command.main() == 0
    assert capsys.readouterr().out == "120\n"


@pytest.mark.parametrize(
    ("text", "status", "error"),
    [
        (wrapped("    i32.const 1\n"), 1, "the IR is invalid"),
        (wrapped("    i32.cnst 1\n"), 1, ":4: error:"),
    ],
)
def test_bad_ir_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    text: str,
    status: int,
    error: str,
) -> None:
    program = tmp_path / "bad.ir"
    program.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["inde", "--run-ir", str(program)])
    assert command.main() == status
    assert error in capsys.readouterr().err


def test_ir_files_can_be_reformatted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    program = tmp_path / "factorial.ir"
    program.write_text(HAND_WRITTEN, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["inde", "--ir-debug", str(program)])
    assert command.main() == 0
    assert parse_module(capsys.readouterr().out) == parse_module(HAND_WRITTEN)
