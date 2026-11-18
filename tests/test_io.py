import io
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from inde import __main__ as command
from inde import ir
from inde.checker import check
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Interpreter, Session
from inde.lowering import lower
from inde.machine import Machine
from inde.parser import parse
from inde.repl import run_repl
from inde.system import System
from inde.verifier import IRError, verify

Outcome = tuple[object, str, str]


def system_for(
    stdin: bytes, args: tuple[str, ...], env: dict[str, str]
) -> tuple[System, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    system = System(stdout=out, stderr=err, stdin=io.BytesIO(stdin), args=args, env=env)
    return system, out, err


def outcome(run: Callable[[], int], out: io.StringIO, err: io.StringIO) -> Outcome:
    try:
        status: object = run()
    except IndeTrap as trap:
        status = f"trap: {trap.message}"
    return status, out.getvalue(), err.getvalue()


def both(
    source: str,
    stdin: bytes = b"",
    args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    setup: Callable[[], None] | None = None,
) -> Outcome:
    checked = check(parse(source), require_main=True)
    module = lower(parse(source))
    verify(module)
    results = []
    for kind in ("interpreter", "machine"):
        if setup is not None:
            setup()
        system, out, err = system_for(stdin, args, env or {})
        if kind == "interpreter":
            interpreter = Interpreter(checked.words, checked.constants, system=system)
            results.append(outcome(interpreter.run_main, out, err))
        else:
            results.append(outcome(Machine(module, system=system).run_main, out, err))
    assert results[0] == results[1]
    return results[0]


def lines(
    body: str,
    declarations: str = "",
    args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> list[str]:
    source = f"{declarations}\nword main ( -> ) {body} end"
    status, out, err = both(source, args=args, env=env)
    assert (status, err) == (0, ""), (status, err)
    return out.splitlines()


SHOW_RESULT = (
    "word shown ( Result[str, IoError] -> )\n"
    "  match case ok print case err print end\n"
    "end\n"
)
SHOW_DONE = (
    "word done ( Result[unit, IoError] -> ) match case ok drop case err print end end\n"
)
CLOSE = "word close ( File -> ) file-close match case ok drop case err print end end\n"


def test_standard_output_and_error() -> None:
    status, out, err = both(
        'word main ( -> ) 1 write "a" write 2.5 print "e" ewrite true eprint '
        "unit print end"
    )
    assert (status, out, err) == (0, "1a2.5\nunit\n", "etrue\n")


def test_writing_to_standard_error_flushes_standard_output() -> None:
    events: list[str] = []

    class Recorder(io.StringIO):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def write(self, text: str) -> int:
            events.append(f"{self.name}:{text}")
            return super().write(text)

        def flush(self) -> None:
            events.append(f"{self.name}:flush")

    system = System(stdout=Recorder("out"), stderr=Recorder("err"))
    source = 'word main ( -> ) "a" write "b" eprint end'
    checked = check(parse(source), require_main=True)
    Interpreter(checked.words, checked.constants, system=system).run_main()
    assert events[:3] == ["out:a", "out:flush", "err:b\n"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("12 show str-len print", ["2"]),
        ("-0.0 show print 1.0e20 show print", ["-0.0", "1e+20"]),
        ('"a\\tb\\"" show-quoted print', ['"a\\tb\\""']),
        ("'\\n' show-quoted print 'x' show print", ["'\\n'", "x"]),
        (
            "5u8 show-quoted print unit show print false show-quoted print",
            ["5", "unit", "false"],
        ),
    ],
)
def test_show(body: str, expected: list[str]) -> None:
    assert lines(body) == expected


def test_reading_standard_input() -> None:
    body = (
        'read-line ? match case some show-quoted print case none "eof" print end '
        'read-line ? match case some show-quoted print case none "eof" print end '
        "read-all ? show-quoted print "
        'read-line ? match case some print case none "eof" print end '
        "read-all ? show-quoted print unit ok"
    )
    source = f"word main ( -> Result[unit, IoError] ) {body} end"
    status, out, _ = both(source, stdin=b"one\r\ntwo\nthree\nfour")
    assert status == 0
    assert out.splitlines() == ['"one"', '"two"', '"three\\nfour"', "eof", '""']


def test_invalid_standard_input() -> None:
    source = "word main ( -> Result[unit, IoError] ) read-line ? drop unit ok end"
    assert both(source, stdin=b"\xff\n") == (
        1,
        "",
        "standard input is not valid UTF-8\n",
    )
    kind = (
        "word main ( -> ) read-all match case ok print case err io-error-kind "
        "invalid-text == print end end"
    )
    assert both(kind, stdin=b"a\xc3") == (0, "true\n", "")


def test_arguments_and_environment() -> None:
    body = (
        "args array-len print args 1u32 array-get print "
        '"HOME" env-get match case some print case none "none" print end '
        '"MISSING" env-get match case some print case none "none" print end '
        '"" env-get match case some print case none "none" print end '
        '"A=B" env-get match case some print case none "none" print end'
    )
    env = {"HOME": "/home/me", "A": "x"}
    assert lines(body, args=("first", "second"), env=env) == [
        "2",
        "second",
        "/home/me",
        "none",
        "none",
        "none",
    ]


def test_invalid_utf8_in_arguments_is_replaced() -> None:
    raw = b"a\xffb".decode("utf-8", "surrogateescape")
    assert lines("args 0u32 array-get print", args=(raw,)) == ["a�b"]


def test_whole_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    body = (
        '"notes.txt" "one\\ntwo" write-file done '
        '"notes.txt" "\\nthree" append-file done '
        '"notes.txt" read-file shown '
        '"notes.txt" read-file-bytes match case ok array-len print case err print end '
        '"raw.bin" { 104u8 105u8 } write-file-bytes done '
        '"raw.bin" read-file shown '
        '"bad.bin" { 255u8 } write-file-bytes done '
        '"bad.bin" read-file shown '
        '"bad.bin" read-file-bytes match case ok array-len print case err print end'
    )
    assert lines(body, SHOW_RESULT + SHOW_DONE) == [
        "one",
        "two",
        "three",
        "13",
        "hi",
        "bad.bin: not valid UTF-8 text",
        "1",
    ]
    assert (tmp_path / "notes.txt").read_text() == "one\ntwo\nthree"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('"missing.txt" read-file shown', ["missing.txt: file not found"]),
        ('"sub" read-file shown', ["sub: is a directory"]),
        ('"notes.txt/x" read-file shown', ["notes.txt/x: not a directory"]),
        ('"no/such/dir.txt" "x" write-file done', ["no/such/dir.txt: file not found"]),
        ('"a\\u{0}b" read-file shown', ["a\0b: invalid path"]),
    ],
)
def test_file_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, expected: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    assert lines(body, SHOW_RESULT + SHOW_DONE) == expected


def test_error_kinds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    declarations = (
        "word kind ( Result[str, IoError] -> )\n"
        '  match case ok drop "ok" print case err io-error-kind\n'
        "    match\n"
        '      case not-found "not found" print\n'
        '      case is-a-directory "directory" print\n'
        '      case invalid-text "text" print\n'
        '      case permission-denied "denied" print\n'
        '      case already-exists "exists" print\n'
        '      case not-a-directory "not a directory" print\n'
        '      case other-error "other" print\n'
        "    end\n"
        "  end\n"
        "end\n"
    )
    body = (
        '"missing" read-file kind "sub" read-file kind '
        '"missing" read-file match case ok drop case err io-error-kind not-found == '
        "print end"
    )
    assert lines(body, declarations) == ["not found", "directory", "true"]


def test_open_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    declarations = (
        CLOSE + SHOW_DONE + "word each-line ( File -> File )\n"
        "  file-read-line match\n"
        "    case ok match case some show-quoted print each-line case none end\n"
        "    case err print\n"
        "  end\n"
        "end\n"
        "word count ( File u32 -> File )\n"
        "  file-read-bytes match case ok array-len print case err print end\n"
        "end\n"
        "word rest ( File -> File )\n"
        "  file-read-all match case ok show-quoted print case err print end\n"
        "end\n"
    )
    body = (
        '"log.txt" file-create ? "first\\r\\nsecond" file-write done close '
        '"log.txt" file-append ? "\\nthird" file-write done '
        "{ 10u8 52u8 } file-write-bytes done close "
        '"log.txt" file-open ? each-line close '
        '"log.txt" file-open ? 3u32 count rest 10u32 count close '
        "unit ok"
    )
    source = declarations + f"word main ( -> Result[unit, IoError] ) {body} end"
    status, out, err = both(source)
    assert (status, err) == (0, ""), err
    assert out.splitlines() == [
        '"first"',
        '"second"',
        '"third"',
        '"4"',
        "3",
        '"st\\r\\nsecond\\nthird\\n4"',
        "0",
    ]


def test_wrong_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    body = (
        '"a.txt" file-create ? file-read-all match case ok print case err print end '
        "close "
        '"a.txt" file-open ? "x" file-write match case ok drop case err print end '
        "close unit ok"
    )
    source = CLOSE + f"word main ( -> Result[unit, IoError] ) {body} end"
    assert both(source) == (
        0,
        (
            "a.txt: the file is not open for reading\n"
            "a.txt: the file is not open for writing\n"
        ),
        "",
    )


def test_main_reports_an_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = (
        'word main ( -> Result[unit, IoError] ) "x" print "gone.txt" read-file ? '
        "print unit ok end"
    )
    assert both(source) == (1, "x\n", "gone.txt: file not found\n")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            (
                'word main ( -> ) "a.txt" file-create match case ok drop case err drop '
                "end end"
            ),
            "linear",
        ),
        (
            (
                'word main ( -> Result[unit, IoError] ) "a.txt" file-open ? '
                '"x" file-write ? drop file-close ? unit ok end'
            ),
            "?",
        ),
        (
            (
                'word main ( -> ) "a.txt" file-open match case ok dup drop drop '
                "case err drop end end"
            ),
            "dup",
        ),
        ("word f ( array[File] -> ) drop end word main ( -> ) end", "File"),
        ("word File ( -> ) end word main ( -> ) end", ""),
        ("type File = file word main ( -> ) end", "File"),
        ("word not-found ( -> ) end word main ( -> ) end", "not-found"),
    ],
)
def test_files_are_linear(source: str, message: str) -> None:
    with pytest.raises(IndeErrors) as failure:
        check(parse(source), require_main=True)
    assert message in str(failure.value)


def test_io_is_never_folded() -> None:
    module = lower(parse('word main ( -> ) args array-len print "x" show print end'))
    text = ir.format_module(module)
    assert "rt.args" in text
    assert text.startswith("module\n  runtime 2\n")


def test_the_runtime_version_follows_the_calls_used() -> None:
    assert lower(parse("word main ( -> ) 1 print end")).runtime_version == 1
    module = lower(parse('word main ( -> ) "x" eprint end'))
    assert module.runtime_version == 2
    module.runtime_version = 1
    with pytest.raises(IRError) as failure:
        verify(module)
    assert any("needs runtime 2" in problem for problem in failure.value.problems)


def test_the_io_error_kind_type_is_checked() -> None:
    module = lower(
        parse(
            'word main ( -> ) "x" read-file match case ok drop case err '
            "io-error-kind not-found == print end end"
        )
    )
    verify(module)
    entry = next(e for e in module.types if e.type.name == "IoErrorKind")
    module.types[module.types.index(entry)] = ir.TypeEntry(
        entry.type, entry.variants[:-1], entry.linear
    )
    with pytest.raises(IRError) as failure:
        verify(module)
    assert any("core IoErrorKind" in problem for problem in failure.value.problems)


def test_failed_output_ends_the_program() -> None:
    class Broken(io.StringIO):
        def write(self, text: str) -> int:
            raise BrokenPipeError(32, "Broken pipe")

    source = 'word main ( -> ) "x" print end'
    checked = check(parse(source), require_main=True)
    system = System(stdout=Broken(), stderr=io.StringIO())
    with pytest.raises(IndeTrap) as failure:
        Interpreter(checked.words, checked.constants, system=system).run_main()
    assert failure.value.message == "output failed"


def test_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    program = tmp_path / "count.inde"
    program.write_text(
        "word main ( -> Result[unit, IoError] )\n"
        '  args 0u32 array-get dup read-file ? str-len swap write ": " write print\n'
        "  args array-len print unit ok\n"
        "end\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    for flag in ([], ["--run-ir"]):
        monkeypatch.setattr(
            sys, "argv", ["inde", *flag, str(program), "notes.txt", "--extra"]
        )
        assert command.main() == 0
        assert capsys.readouterr().out == "notes.txt: 5\n2\n"
        monkeypatch.setattr(sys, "argv", ["inde", *flag, str(program), "gone.txt"])
        assert command.main() == 1
        assert capsys.readouterr().err == "gone.txt: file not found\n"
    monkeypatch.setattr(sys, "argv", ["inde", "--ir", str(program), "notes.txt"])
    assert command.main() == 1
    assert capsys.readouterr().err.startswith("usage:")


def test_reading_input_in_the_repl() -> None:
    output = io.StringIO()
    run_repl(io.StringIO("read-line\nhello there\n"), output, io.StringIO())
    assert output.getvalue() == 'stack: "hello there" some ok\n'


def test_values_in_the_repl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    session = Session(io.StringIO(), directory=tmp_path)
    session.run_line('"gone.txt" read-file')
    session.run_line('"new.txt" file-create')
    assert session.show_stack() == (
        'stack: <IoError "gone.txt: file not found"> err | <File "new.txt"> ok'
    )


def test_reading_bytes_with_a_large_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "small.txt").write_bytes(b"x" * 70000)
    body = (
        '"small.txt" file-open ? 4294967295u32 file-read-bytes '
        "match case ok array-len print case err print end close unit ok"
    )
    source = CLOSE + f"word main ( -> Result[unit, IoError] ) {body} end"
    assert both(source) == (0, "70000\n", "")
