import io
from pathlib import Path

import pytest

from inde.checker import check
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Interpreter, Session
from inde.lowering import lower
from inde.machine import Machine
from inde.modules import load_program
from inde.parser import parse
from inde.system import System
from inde.verifier import verify

POINT = "type Point = point( x: f32 y: f32 )\n"
LEXER = "type Lexer = lexer( text: str offset: u32 tokens: arraybuf[str] )\n"
BOX = "type Box[T] = box( value: T count: i32 )\n"


def outcome(run: object, out: io.StringIO) -> tuple[object, str]:
    assert callable(run)
    try:
        status = run()
    except IndeTrap as trap:
        status = f"trap: {trap.message}"
    return status, out.getvalue()


def printed(body: str, declarations: str = "") -> list[str]:
    source = f"{declarations}\nword main ( -> ) {body} end"
    checked = check(parse(source), require_main=True)
    module = lower(parse(source))
    verify(module)
    out, machine_out = io.StringIO(), io.StringIO()
    interpreter = Interpreter(
        checked.words, checked.constants, system=System(stdout=out)
    )
    first = outcome(interpreter.run_main, out)
    machine = Machine(module, system=System(stdout=machine_out))
    second = outcome(machine.run_main, machine_out)
    assert first == second
    assert first[0] == 0, first
    return first[1].splitlines()


def failures(source: str) -> str:
    with pytest.raises(IndeErrors) as failure:
        check(parse(source), require_main=False)
    return str(failure.value)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("1.0 2.0 point point-y@ print point-x@ print drop", ["2.0", "1.0"]),
        (
            (
                "1.0 2.0 point 5.0 point-x! [ 2.0 * ] change-point-y "
                "point-unpack print print"
            ),
            ["4.0", "5.0"],
        ),
        (
            "1.0 2.0 point 9.0 exchange-point-x print point-x@ print drop",
            ["1.0", "9.0"],
        ),
        ("1.0 2.0 point match case point + print end", ["3.0"]),
        (
            "10.0 1.0 2.0 point [ over + ] change-point-x point-x@ print drop drop",
            ["11.0"],
        ),
    ],
)
def test_point(body: str, expected: list[str]) -> None:
    assert printed(body, POINT) == expected


def test_a_linear_record() -> None:
    declarations = (
        LEXER + "word peek ( Lexer -> Lexer Option[char] )\n"
        "  lexer-text@ swap lexer-offset@ rot swap str-char-at\n"
        "end\n"
        "word advance ( Lexer -> Lexer ) [ 1u32 + ] change-lexer-offset end\n"
        "word emit ( Lexer str -> Lexer ) [ arraybuf-push ] curry "
        "change-lexer-tokens end\n"
    )
    body = (
        '"ab" 0u32 arraybuf-new lexer '
        "peek match case some str-from-char emit case none end advance "
        "lexer-offset@ print "
        "arraybuf-new exchange-lexer-tokens arraybuf-freeze array-len print "
        "lexer-unpack arraybuf-freeze array-len print drop drop"
    )
    assert printed(body, declarations) == ["1", "1", "0"]


def test_a_generic_record() -> None:
    body = (
        '"v" 3 box box-value@ print 7 box-count! box-count@ print drop '
        'strbuf-new 1 box [ "z" strbuf-append ] change-box-value '
        "box-unpack drop strbuf-freeze print"
    )
    assert printed(body, BOX) == ["v", "7", "z"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            (
                "type Holder = holder( buf: strbuf )\n"
                "word t ( Holder -> Holder strbuf ) holder-buf@ end"
            ),
            (
                "`holder-buf@` is not defined, because the field `buf` holds a linear "
                "`strbuf`; use `change-holder-buf` or `exchange-holder-buf`"
            ),
        ),
        (
            (
                "type Holder = holder( buf: strbuf )\n"
                "word t ( Holder strbuf -> Holder ) holder-buf! end"
            ),
            "`holder-buf!` is not defined",
        ),
        (
            BOX + "word t ( Box[strbuf] -> Box[strbuf] strbuf ) box-value@ end",
            (
                "`strbuf` does not satisfy `Unrestricted`, which `box-value@` requires "
                "of its `T`"
            ),
        ),
        (
            "type Pair2 = pair2( a: i32 i32 )",
            "name every field of `pair2` or none",
        ),
        (
            "type Shape = circle( r: f32 ) | dot",
            "only a type with one constructor may name its fields",
        ),
        ("type Twice = twice( a: i32 a: i32 )", "`twice` has two fields named `a`"),
        ("type Gap = gap( a: )", "expected the type of the field `a`"),
        (
            POINT + "word point-x@ ( -> ) end",
            "`point-x@` is already defined",
        ),
    ],
)
def test_record_errors(source: str, message: str) -> None:
    assert message in failures(source)


def test_record_words_are_ordinary_words() -> None:
    program = parse(POINT)
    names = [declaration.name for declaration in program.declarations]
    assert names == [
        "Point",
        "point-x@",
        "point-x!",
        "change-point-x",
        "exchange-point-x",
        "point-y@",
        "point-y!",
        "change-point-y",
        "exchange-point-y",
    ]
    checked = check(program)
    assert str(checked.words["point-x@"].effect) == "( Point -> Point f32 )"
    assert str(checked.words["change-point-y"].effect) == (
        "( ..a Point Q -> ..a Point ) where Q: [ ..a f32 -> ..a f32 ]"
    )


def test_quotation_names_avoid_the_parameters() -> None:
    checked = check(parse("type Holds[Q] = holds( item: Q )"))
    assert "P: [ ..a Q -> ..a Q ]" in str(checked.words["change-holds-item"].effect)


def write(directory: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")


def test_field_words_follow_the_constructor_across_modules(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "shapes.inde": (
                "pub type Point = point( x: f32 y: f32 )\n"
                "pub opaque type Secret = secret( code: i32 )\n"
                "pub word make-secret ( i32 -> Secret ) secret end\n"
            ),
            "main.inde": (
                "import shapes\n"
                "word main ( -> ) 1.0 2.0 point point-y@ print drop end\n"
            ),
        },
    )
    program = load_program(tmp_path / "main.inde")
    out = io.StringIO()
    checked = check(program, require_main=True)
    Interpreter(checked.words, checked.constants, system=System(stdout=out)).run_main()
    assert out.getvalue() == "2.0\n"
    (tmp_path / "main.inde").write_text(
        "import shapes\nword main ( -> ) 1 make-secret secret-code@ drop drop end\n",
        encoding="utf-8",
    )
    with pytest.raises(IndeErrors) as failure:
        load_program(tmp_path / "main.inde")
    assert "word `secret-code@` is private to module `shapes`" in str(failure.value)


def test_records_in_the_repl(tmp_path: Path) -> None:
    session = Session(io.StringIO(), directory=tmp_path)
    session.run_line("type Point = point( x: f32 y: f32 )")
    session.run_line("1.0 2.0 point 3.0 point-x! point-x@")
    assert session.show_stack() == "stack: 3.0 2.0 point | 3.0"


def test_records_keep_derived_equality() -> None:
    declarations = "type Size = size( w: i32 h: i32 )\n"
    body = "1 2 size 1 2 size == print 1 2 size 3 size-h! 1 2 size != print"
    assert printed(body, declarations) == ["true", "true"]
