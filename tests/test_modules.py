import io
import sys
from pathlib import Path

import pytest

from inde import __main__ as command
from inde import ir
from inde.ast import Program
from inde.checker import check
from inde.errors import IndeErrors
from inde.interpreter import Interpreter, Session
from inde.ir_parser import parse_module
from inde.lowering import lower
from inde.machine import Machine
from inde.modules import load_program
from inde.parser import parse
from inde.verifier import verify

GEOMETRY = """\
pub type Shape = circle( f32 ) | rect( f32 f32 )

pub word area ( Shape -> f32 )
  match case circle square PI * case rect * end
end

word square ( f32 -> f32 ) dup * end

const PI = 3.14159
"""

SHAPES = """\
import geometry

word main ( -> ) 2.0 circle area print 3.0 4.0 rect area print end
"""


def write(directory: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def program_in(
    tmp_path: Path, files: dict[str, str], main: str = "main.inde"
) -> Program:
    write(tmp_path, files)
    return load_program(tmp_path / main, library=tmp_path / "lib")


def run(tmp_path: Path, files: dict[str, str], main: str = "main.inde") -> str:
    program = program_in(tmp_path, files, main)
    output = io.StringIO()
    checked = check(program, require_main=True)
    assert Interpreter(checked.words, checked.constants, output).run_main() == 0
    module = lower(program)
    verify(module)
    machine_output = io.StringIO()
    assert Machine(module, machine_output).run_main() == 0
    assert machine_output.getvalue() == output.getvalue()
    return output.getvalue()


def errors(tmp_path: Path, files: dict[str, str], main: str = "main.inde") -> list[str]:
    with pytest.raises(IndeErrors) as failure:
        program = program_in(tmp_path, files, main)
        check(program, require_main=True)
    found = []
    for error in failure.value.errors:
        location = error.location
        where = ""
        if location is not None:
            where = f"{Path(location.source).name if location.source else main}:"
            where += f"{location.line}: "
        found.append(where + error.message)
    return found


def test_a_program_of_two_modules(tmp_path: Path) -> None:
    files = {"geometry.inde": GEOMETRY, "main.inde": SHAPES}
    assert run(tmp_path, files) == "12.56636\n12.0\n"


def test_declarations_of_other_modules_are_prefixed_in_the_ir(tmp_path: Path) -> None:
    module = lower(
        program_in(tmp_path, {"geometry.inde": GEOMETRY, "main.inde": SHAPES})
    )
    names = [function.name for function in module.functions]
    assert names == ["$geometry:area", "$geometry:square", "$main"]
    text = ir.format_module(module)
    assert "type %geometry:Shape = variant { geometry:circle(f32) " in text
    assert "make %geometry:Shape.geometry:circle" in text


def test_debug_comments_name_the_file(tmp_path: Path) -> None:
    module = lower(
        program_in(tmp_path, {"geometry.inde": GEOMETRY, "main.inde": SHAPES})
    )
    text = ir.format_module(module, debug=True)
    assert f"call $geometry:square  ; {tmp_path / 'geometry.inde'}:4:21 square" in text
    assert "call $geometry:area  ; 3:29 area" in text
    parsed = parse_module(text)
    assert ir.format_module(parsed, debug=True) == text
    call = parsed.functions[0].body[0]
    assert isinstance(call, ir.Match)
    location = call.arms[0][1][0].location
    assert location is not None and location.source == str(tmp_path / "geometry.inde")


def test_quoted_sources_in_debug_comments() -> None:
    text = (
        "module\n  runtime 1\n\n  func $main () -> ()\n"
        '    unit.const  ; "a b;c.inde":2:3 unit\n'
        "    drop.unit  ; c:\\\\x.inde:4:5\n"
        "  end\n\n  entry $main unit\nend\n"
    )
    first, second = parse_module(text).functions[0].body
    assert first.location is not None and first.location.source == "a b;c.inde"
    assert (first.location.line, first.location.column, first.word) == (2, 3, "unit")
    assert second.location is not None and second.location.source == "c:\\\\x.inde"
    assert ir.format_module(parse_module(text), debug=True) == text


def test_generics_types_and_constants_cross_modules(tmp_path: Path) -> None:
    files = {
        "boxes.inde": (
            "pub type Box[T] = box( T )\n"
            "pub word unbox ( Box[T] -> T ) match case box end end\n"
            "pub const LIMIT = 3\n"
            "pub word twice ( T -> T T ) where T: Unrestricted dup end\n"
        ),
        "main.inde": (
            "import boxes\n"
            "word big? ( i32 -> bool ) match case LIMIT true else false end end\n"
            "word main ( -> ) 7 box unbox print { 1 2 } box unbox array-len print "
            "LIMIT big? print 2 big? print { LIMIT 4 } array-len print "
            '"x" twice str-concat print end\n'
        ),
    }
    assert run(tmp_path, files) == "7\n2\ntrue\nfalse\n2\nxx\n"


def test_imports_are_shared_and_checked_once(tmp_path: Path) -> None:
    files = {
        "base.inde": "pub word one ( -> i32 ) 1 end\n",
        "left.inde": "import base\npub word two ( -> i32 ) one one + end\n",
        "right.inde": "import base\npub word three ( -> i32 ) one 2 + end\n",
        "main.inde": (
            "import left\nimport right\nword main ( -> ) two three + print end\n"
        ),
    }
    assert run(tmp_path, files) == "5\n"
    program = program_in(tmp_path, files)
    names = [d.name for d in program.declarations]
    assert names == ["base:one", "left:two", "right:three", "main"]


def test_private_names_and_locals_do_not_leak(tmp_path: Path) -> None:
    files = {
        "tools.inde": (
            "word helper ( i32 -> i32 ) 1 + end\n"
            "pub word bump ( i32 -> i32 ) let n in n helper end end\n"
            "word main ( -> ) end\n"
        ),
        "main.inde": (
            "import tools\n"
            "word helper ( i32 -> i32 ) 10 + end\n"
            "word n ( -> i32 ) 5 end\n"
            "word main ( -> ) n bump helper print end\n"
        ),
    }
    assert run(tmp_path, files) == "16\n"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (
            {"main.inde": "import geometry\nword main ( -> ) 2.0 square drop end"},
            ["main.inde:2: word `square` is private to module `geometry`"],
        ),
        (
            {"main.inde": "import geometry\nword main ( -> ) PI drop end"},
            ["main.inde:2: constant `PI` is private to module `geometry`"],
        ),
        (
            {
                "main.inde": (
                    "import geometry\nword area ( -> ) end\nword main ( -> ) end"
                )
            },
            ["main.inde:2: `area` is already provided by the import of `geometry`"],
        ),
        (
            {"main.inde": ("import geometry\ntype Shape = blob\nword main ( -> ) end")},
            [
                (
                    "main.inde:2: type `Shape` is already provided by the import of "
                    "`geometry`"
                )
            ],
        ),
        (
            {"main.inde": "import geometry\nword main ( -> ) 1 let area in end end"},
            ["main.inde:2: `area` already names a word, so it cannot name a local"],
        ),
        (
            {
                "a.inde": "pub word size ( -> i32 ) 1 end",
                "b.inde": "pub word size ( -> i32 ) 2 end",
                "main.inde": "import a\nimport b\nword main ( -> ) size print end",
            },
            ["main.inde:3: `size` is provided by both `a` and `b`"],
        ),
        (
            {
                "c.inde": "import d\npub word x ( -> ) end",
                "d.inde": "import c\npub word y ( -> ) end",
                "main.inde": "import c\nword main ( -> ) end",
            },
            ["d.inde:1: import cycle `c` \u2192 `d` \u2192 `c`"],
        ),
        (
            {
                "points.inde": (
                    "type Point = point( i32 i32 )\n"
                    "pub word origin ( -> Point ) 0 0 point end\n"
                    "pub type Line = line( Point Point )\n"
                    "pub word apply ( Q -> ) where Q: [ Point -> ] drop end\n"
                ),
                "main.inde": "import points\nword main ( -> ) end",
            },
            [
                (
                    "points.inde:2: the public word `origin` mentions the private type "
                    "`Point`"
                ),
                (
                    "points.inde:3: the public type `Line` mentions the private type "
                    "`Point`"
                ),
                (
                    "points.inde:4: the public word `apply` mentions the private type "
                    "`Point`"
                ),
            ],
        ),
        (
            {
                "a.inde": "pub word size ( -> i32 ) 1 end",
                "f.inde": "import a\npub word g ( -> ) end",
                "main.inde": "import f\nword main ( -> ) size print end",
            },
            [
                (
                    "main.inde:2: unknown word `size`; it is in module `a`, which this "
                    "module does not import"
                )
            ],
        ),
        (
            {"main.inde": "import missing\nword main ( -> ) end"},
            [
                (
                    "main.inde:1: no module named `missing` (looked beside this file "
                    "and in the standard library)"
                )
            ],
        ),
        (
            {"main.inde": "word main ( -> ) end\nimport geometry"},
            ["main.inde:2: imports come before every declaration"],
        ),
        (
            {"main.inde": "import geometry\nimport geometry\nword main ( -> ) end"},
            [("main.inde:2: `geometry` is imported twice; the first is on line 1")],
        ),
        (
            {"main.inde": "import main\nword main ( -> ) end"},
            ["main.inde:1: a module cannot import itself"],
        ),
        (
            {
                "g.inde": "pub word go ( -> ) helper end",
                "main.inde": (
                    "import g\nword helper ( -> ) end\nword main ( -> ) go end"
                ),
            },
            ["g.inde:1: unknown word `helper`"],
        ),
        (
            {
                "h.inde": "pub word print ( -> ) end\ntype Option = o",
                "main.inde": "import h\nword main ( -> ) end",
            },
            [
                "h.inde:1: `print` is a built-in word and cannot be redefined",
                "h.inde:2: type `Option` is already defined by the core library",
            ],
        ),
        (
            {"main.inde": "import Geometry\nword main ( -> ) end"},
            [
                (
                    "main.inde:1: a module name is lowercase letters, digits and `-`, "
                    "starting with a letter; found type name `Geometry`"
                )
            ],
        ),
        (
            {"main.inde": "pub 1\nword main ( -> ) end"},
            [
                (
                    "main.inde:1: `pub` comes before a `word`, `type` or `const` "
                    "declaration, found integer literal `1`"
                )
            ],
        ),
        (
            {"bad.inde": "pub word w ( -> i32 ) 1.0 end", "main.inde": "import bad"},
            [
                (
                    "bad.inde:1: `bad:w` declares that it leaves `i32`, but its body "
                    "leaves `f32`"
                ),
            ],
        ),
    ],
)
def test_module_errors(
    tmp_path: Path, files: dict[str, str], expected: list[str]
) -> None:
    files = {"geometry.inde": GEOMETRY, **files}
    assert errors(tmp_path, files)[: len(expected)] == expected


def test_unused_ambiguity_is_allowed(tmp_path: Path) -> None:
    files = {
        "a.inde": "pub word size ( -> i32 ) 1 end",
        "b.inde": "pub word size ( -> i32 ) 2 end",
        "main.inde": "import a\nimport b\nword main ( -> ) end",
    }
    assert run(tmp_path, files) == ""


def test_main_of_another_module_is_an_ordinary_word(tmp_path: Path) -> None:
    files = {
        "other.inde": "pub word main ( -> ) 1 print end",
        "main.inde": "import other\nword main ( -> ) end",
    }
    assert errors(tmp_path, files) == [
        "main.inde:2: `main` is already provided by the import of `other`"
    ]
    files["other.inde"] = (
        "word main ( i32 -> ) drop end\npub word hi ( -> ) 2 print end"
    )
    files["main.inde"] = "import other\nword main ( -> ) hi end"
    assert run(tmp_path, files) == "2\n"


def test_the_prelude_is_imported_everywhere(tmp_path: Path) -> None:
    files = {
        "lib/prelude.inde": (
            "pub word twice ( i32 -> i32 ) 2 * end\nword hidden ( -> ) end\n"
        ),
        "lib/text.inde": "pub word shout ( i32 -> i32 ) twice 1 + end\n",
        "main.inde": "import text\nword main ( -> ) 5 shout twice print end\n",
    }
    assert run(tmp_path, files) == "22\n"
    files["main.inde"] = "word twice ( -> ) end\nword main ( -> ) hidden end\n"
    assert errors(tmp_path, files) == [
        "main.inde:1: `twice` is already provided by the import of `prelude`",
        "main.inde:2: word `hidden` is private to module `prelude`",
    ]


def test_a_local_module_comes_before_the_library(tmp_path: Path) -> None:
    files = {
        "lib/text.inde": "pub word which ( -> i32 ) 1 end\n",
        "lib/uses.inde": "import text\npub word inner ( -> i32 ) which end\n",
        "text.inde": "pub word which ( -> i32 ) 2 end\n",
        "main.inde": (
            "import text\nimport uses\nword main ( -> ) which print inner print end\n"
        ),
    }
    assert run(tmp_path, files) == "2\n1\n"
    names = [d.name for d in program_in(tmp_path, files).declarations]
    assert names == ["text:which", "text~2:which", "uses:inner", "main"]


def test_single_files_are_unchanged(tmp_path: Path) -> None:
    files = {"main.inde": "word main ( -> ) 1 2 + print end\n"}
    assert run(tmp_path, files) == "3\n"
    program = program_in(tmp_path, files)
    assert [d.name for d in program.declarations] == ["main"]


def test_running_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(
        tmp_path,
        {
            "geometry.inde": GEOMETRY
            + "pub word broken ( f32 -> f32 ) 0 let z in 1 z / drop end end\n",
            "main.inde": SHAPES.replace("end\n", "1.0 broken drop end\n"),
        },
    )
    for flag in ([], ["--run-ir"]):
        monkeypatch.setattr(sys, "argv", ["inde", *flag, str(tmp_path / "main.inde")])
        assert command.main() == 2
        captured = capsys.readouterr()
        assert captured.out == "12.56636\n12.0\n"
        assert captured.err.startswith(f"{tmp_path / 'geometry.inde'}:10:")
        assert "division by zero" in captured.err


def test_errors_name_their_files_on_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(
        tmp_path,
        {"main.inde": "import geometry\nword main ( -> ) 1 end\n"},
    )
    write(tmp_path, {"geometry.inde": GEOMETRY + "pub word bad ( -> ) 1 end\n"})
    monkeypatch.setattr(sys, "argv", ["inde", str(tmp_path / "main.inde")])
    assert command.main() == 1
    err = capsys.readouterr().err.splitlines()
    assert err[0].startswith(f"{tmp_path / 'main.inde'}:2:")
    assert err[1].startswith(f"{tmp_path / 'geometry.inde'}:10:")


def test_imports_in_the_repl(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "geometry.inde": GEOMETRY,
            "a.inde": "pub word size ( -> i32 ) 1 end",
            "lib/prelude.inde": "pub word twice ( i32 -> i32 ) 2 * end",
        },
    )
    output = io.StringIO()
    session = Session(output, directory=tmp_path, library=tmp_path / "lib")
    session.run_line("4 twice")
    session.run_line("import geometry")
    session.run_line("2.0 circle area print")
    assert output.getvalue() == "12.56636\n"
    for line, message in [
        ("square", "word `square` is private to module `geometry`"),
        ("word area ( -> ) end", "`area` is already provided by the import of"),
        ("import geometry", "`geometry` is already imported"),
        ("import nothing", "no module named `nothing`"),
    ]:
        with pytest.raises(IndeErrors) as failure:
            session.run_line(line)
        assert failure.value.errors[0].message.startswith(message)
    session.run_line("word size ( -> i32 ) 9 end")
    with pytest.raises(IndeErrors) as failure:
        session.run_line("import a")
    assert failure.value.errors[0].message == (
        "`size` from `a` is already defined in this session"
    )
    session.run_line("size")
    assert [value.data for value in session.stack] == [8, 9]


def test_parsing_imports_and_visibility() -> None:
    program = parse(
        "import geometry\nimport text-tools\n"
        "pub word w ( -> ) end\nword v ( -> ) end\n"
        "pub type Tree = leaf\npub const MAX = 1\nlinear type Lock = lock\n"
    )
    assert [i.name for i in program.imports] == ["geometry", "text-tools"]
    assert [d.public for d in program.declarations] == [True, False, True, True, False]
