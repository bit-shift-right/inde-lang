import io
from pathlib import Path

import pytest

from inde.checker import check
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Interpreter
from inde.lowering import lower
from inde.machine import Machine
from inde.modules import LIBRARY, load_program
from inde.system import System
from inde.verifier import IRError, verify

MODULES = ("math", "text", "arrays", "io", "maps")


def run(tmp_path: Path, source: str, stdin: bytes = b"") -> tuple[object, str, str]:
    path = tmp_path / "main.inde"
    path.write_text(source, encoding="utf-8")
    program = load_program(path)
    checked = check(program, require_main=True)
    module = lower(program)
    verify(module)
    results = []
    for kind in ("interpreter", "machine"):
        out, err = io.StringIO(), io.StringIO()
        system = System(stdout=out, stderr=err, stdin=io.BytesIO(stdin))
        try:
            if kind == "interpreter":
                interpreter = Interpreter(
                    checked.words, checked.constants, system=system
                )
                status: object = interpreter.run_main()
            else:
                status = Machine(module, system=system).run_main()
        except IndeTrap as trap:
            status = f"trap: {trap.message}"
        results.append((status, out.getvalue(), err.getvalue()))
    assert results[0] == results[1]
    return results[0]


def printed(
    tmp_path: Path, imports: str, body: str, declarations: str = ""
) -> list[str]:
    header = "".join(f"import {name}\n" for name in imports.split())
    source = f"{header}{declarations}\nword main ( -> ) {body} end\n"
    status, out, err = run(tmp_path, source)
    assert (status, err) == (0, ""), (status, err)
    return out.splitlines()


def test_every_module_checks_and_verifies(tmp_path: Path) -> None:
    header = "".join(f"import {name}\n" for name in MODULES)
    assert printed(tmp_path, "", "", header) == []
    for name in (*MODULES, "prelude"):
        assert (LIBRARY / f"{name}.inde").is_file()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("1 2 nip print", ["2"]),
        ("1 2 tuck print print print", ["2", "1", "2"]),
        ("3 4 dup2 + print + print", ["7", "7"]),
        ("1 2 [ 10 * ] dip + print", ["12"]),
        ('"a" [ str-len ] keep print print', ["a", "1"]),
        ("7 [ 1 + ] [ 2 * ] bi + print", ["22"]),
        ("0 [ 2 + ] 5u32 times print 1 [ 1 + ] 0u32 times print", ["10", "1"]),
        ("none 9 option-or print 3 some 9 option-or print", ["9", "3"]),
        ("4 some [ 1 + ] option-map 0 option-or print", ["5"]),
        (
            (
                '{ 1 } 5u32 array-get.checked "missing" option-to-result '
                "match case ok print case err print end"
            ),
            ["missing"],
        ),
        ('"boom" err 1 result-or print 2 good 0 result-or print', ["1", "2"]),
        ("2 good [ 3 * ] result-map 0 result-or print", ["6"]),
        (
            (
                '"bad" fail [ str-len ] result-map-err '
                "match case ok print case err print end"
            ),
            ["3"],
        ),
        (
            (
                "5 good result-to-option 0 option-or print "
                '"x" fail result-to-option 0 option-or print'
            ),
            ["5", "0"],
        ),
    ],
)
def test_prelude(tmp_path: Path, body: str, expected: list[str]) -> None:
    declarations = (
        "word good ( i32 -> Result[i32, str] ) ok end\n"
        "word fail ( str -> Result[i32, str] ) err end\n"
    )
    assert printed(tmp_path, "", body, declarations) == expected


def test_prelude_words_are_taken(tmp_path: Path) -> None:
    with pytest.raises(IndeErrors) as failure:
        printed(tmp_path, "", "", "word dip ( -> ) end")
    assert "already provided by the import of `prelude`" in str(failure.value)


def python_floor(a: int, b: int) -> tuple[int, int]:
    return a // b, a % b


def python_euclid(a: int, b: int) -> tuple[int, int]:
    q = a // b if b > 0 else -(a // -b)
    return q, a - b * q


def test_math(tmp_path: Path) -> None:
    pairs = [(7, 2), (-7, 2), (7, -2), (-7, -2), (6, 3), (-6, 3), (0, -5), (1, 7)]
    body = " ".join(
        f"{a} {b} floor-div print {a} {b} floor-mod print "
        f"{a} {b} euclid-div print {a} {b} euclid-mod print"
        for a, b in pairs
    )
    expected = []
    for a, b in pairs:
        expected += [str(n) for n in (*python_floor(a, b), *python_euclid(a, b))]
    assert printed(tmp_path, "math", body) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("3 5 min print 3 5 max print 1.5 -2.0 min print", ["3", "5", "-2.0"]),
        (
            "12 0 10 clamp print -3 0 10 clamp print 4 0 10 clamp print",
            ["10", "0", "4"],
        ),
        ("'a' 'b' max print", ["b"]),
        (
            (
                "-128i8 127i8 floor-div print -128i8 127i8 floor-mod print "
                "200u8 7u8 floor-div print 200u8 7u8 euclid-mod print"
            ),
            ["-2", "126", "28", "4"],
        ),
        (
            (
                "4 even? print -3 odd? print 0u64 even? print 255u8 odd? print "
                "-6 even? print 7i8 even? print"
            ),
            ["true", "true", "true", "true", "true", "false"],
        ),
    ],
)
def test_math_words(tmp_path: Path, body: str, expected: list[str]) -> None:
    assert printed(tmp_path, "math", body) == expected


def test_math_traps_like_division(tmp_path: Path) -> None:
    source = (
        "import math\nword main ( -> ) 1 0 let a b in a b floor-div print end end\n"
    )
    assert run(tmp_path, source)[0] == "trap: division by zero in `/` (i32 1, i32 0)"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            (
                "'a' ascii-to-upper print 'Z' ascii-to-lower print "
                "'é' ascii-to-upper print"
            ),
            ["A", "z", "é"],
        ),
        (
            (
                "'5' ascii-digit? print 'x' ascii-digit? print ' ' ascii-space? print "
                "'\\t' ascii-space? print 'q' ascii-letter? print "
                "'_' ascii-letter? print "
                "'Q' ascii-upper? print 'Q' ascii-lower? print"
            ),
            ["true", "false", "true", "true", "true", "false", "true", "false"],
        ),
        (
            (
                '"hello world" "wor" str-find match case some print case none end '
                '"hello" "z" str-find match case some print case none "none" print end '
                '"héllo" "llo" str-find match case some print case none end '
                '"abc" "" str-find match case some print case none end'
            ),
            ["6", "none", "3", "0"],
        ),
        (
            (
                '"hello" "ell" str-contains? print "hello" "he" str-starts-with? print '
                '"hello" "llo" str-ends-with? print "lo" "hello" str-ends-with? print '
                '"" "" str-starts-with? print'
            ),
            ["true", "true", "true", "false", "true"],
        ),
        (
            (
                '"a,b,,c" "," str-split dup array-len print "|" str-join print '
                '"abc" "" str-split "|" str-join print '
                '"a--b--" "--" str-split array-len print '
                '"" "," str-split array-len print'
            ),
            ["4", "a|b||c", "abc", "3", "1"],
        ),
        (
            (
                '"one\\r\\ntwo\\n\\nthree\\n" str-lines dup array-len print '
                '"|" str-join print "" str-lines array-len print '
                '"x" str-lines array-len print "\\n" str-lines array-len print'
            ),
            ["4", "one|two||three", "0", "1", "1"],
        ),
        (
            '{ } "," str-join show-quoted print { "a" } "," str-join print',
            ['""', "a"],
        ),
        (
            (
                '"  hi there \\t\\n" str-trim show-quoted print '
                '"  hi " str-trim-start show-quoted print '
                '"  hi " str-trim-end show-quoted print '
                '"   " str-trim show-quoted print'
            ),
            ['"hi there"', '"hi "', '"  hi"', '""'],
        ),
        (
            (
                '"a-b-c" "-" "+=" str-replace print "aaa" "aa" "b" str-replace print '
                '"abc" "" "x" str-replace print'
            ),
            ["a+=b+=c", "ba", "abc"],
        ),
        (
            '"ab" 3u32 str-repeat print "ab" 0u32 str-repeat show-quoted print',
            ["ababab", '""'],
        ),
        (
            (
                '"héllo" str-chars dup array-len print str-from-chars print '
                '"Hé llo" str-ascii-upper print "ÀBC" str-ascii-lower print'
            ),
            ["5", "héllo", "Hé LLO", "Àbc"],
        ),
    ],
)
def test_text(tmp_path: Path, body: str, expected: list[str]) -> None:
    assert printed(tmp_path, "text", body) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("123", "123"),
        ("+42", "42"),
        ("-0", "0"),
        ("-9223372036854775808", "-9223372036854775808"),
        ("9223372036854775807", "9223372036854775807"),
        ("9223372036854775808", 'integer out of range: "9223372036854775808"'),
        ("-9223372036854775809", 'integer out of range: "-9223372036854775809"'),
        ("", 'not an integer: ""'),
        ("-", 'not an integer: "-"'),
        ("12a", 'not an integer: "12a"'),
        (" 1", 'not an integer: " 1"'),
        ("１", 'not an integer: "１"'),
    ],
)
def test_parsing_integers(tmp_path: Path, text: str, expected: str) -> None:
    body = f'"{text}" str-parse-int match case ok print case err print end'
    assert printed(tmp_path, "text", body) == [expected]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("0 { 1 2 3 } [ + ] array-each print", ["6"]),
        ("{ 1 2 3 } [ 10 * ] array-map [ print ] array-each", ["10", "20", "30"]),
        ('{ 1 2 } [ show "!" str-concat ] array-map "," str-join print', ["1!,2!"]),
        (
            "{ 1 2 3 4 5 } [ 2 % 1 == ] array-filter [ print ] array-each",
            ["1", "3", "5"],
        ),
        (
            (
                '{ "a" "b" "c" } strbuf-new [ strbuf-append ] array-fold '
                "strbuf-freeze print"
            ),
            ["abc"],
        ),
        ("{ 1 2 3 } 0 [ + ] array-fold print", ["6"]),
        (
            (
                "{ 1 2 3 } [ 2 > ] array-any? print { 1 2 3 } [ 0 > ] array-all? print "
                "{ 1 2 3 } [ 5 > ] array-any? print { } [ 5 > ] array-all? print"
            ),
            ["true", "true", "false", "true"],
        ),
        (
            (
                "{ 4 5 6 } [ 4 > ] array-find match case some print case none end "
                "{ 4 } [ 4 > ] array-find "
                'match case some print case none "none" print end'
            ),
            ["5", "none"],
        ),
        (
            "{ 1 } 0u32 0u32 array-slice array-empty? print { 1 } array-empty? print",
            ["true", "false"],
        ),
        (
            (
                "{ 1 2 } { 3 } array-concat 9 array-append array-reverse "
                "[ print ] array-each"
            ),
            ["9", "3", "2", "1"],
        ),
        (
            (
                "2u32 5u32 array-range [ print ] array-each "
                "5u32 2u32 array-range array-len print"
            ),
            ["2", "3", "4", "0"],
        ),
        (
            (
                '{ 5 3 9 1 3 7 } array-sort [ show ] array-map " " str-join print '
                "{ 1 } 0u32 0u32 array-slice array-sort array-len print"
            ),
            ["1 3 3 5 7 9", "0"],
        ),
        (
            '{ "ccc" "a" "bb" "dd" "e" } [ str-len ] array-sort-by " " str-join print',
            ["a e bb dd ccc"],
        ),
        (
            (
                '{ 1 1 2 3 3 3 1 } array-unique [ show ] array-map "," str-join print '
                "{ 1 } 0u32 0u32 array-slice array-unique array-len print"
            ),
            ["1,2,3,1", "0"],
        ),
        ("{ 1 2 3 } [ 100 swap - ] array-map [ print ] array-each", ["99", "98", "97"]),
    ],
)
def test_arrays(tmp_path: Path, body: str, expected: list[str]) -> None:
    assert printed(tmp_path, "arrays text", body) == expected


def test_sorting_many_values(tmp_path: Path) -> None:
    values = [(n * 7919) % 1009 for n in range(300)]
    literal = " ".join(str(v) for v in values)
    body = f'{{ {literal} }} array-sort [ show ] array-map " " str-join print'
    assert printed(tmp_path, "arrays text", body) == [
        " ".join(str(v) for v in sorted(values))
    ]


def test_quotations_may_use_the_stack_below(tmp_path: Path) -> None:
    body = (
        'strbuf-new { "a" "b" } [ strbuf-append ] array-each strbuf-freeze print '
        "10 { 1 2 } [ over + ] array-map [ print ] array-each drop"
    )
    assert printed(tmp_path, "arrays", body) == ["ab", "11", "12"]


def test_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    body = (
        '"w.txt" { "alpha" "beta" "gamma" } write-lines ? print '
        '"w.txt" file-lines ? array-len print '
        '"w.txt" [ "delta\\n" file-write ] with-file-append ? print '
        '"w.txt" [ file-read-all [ str-len ] result-map ] with-file-open ? print '
        '"gone.txt" [ "x" file-write ] with-file-open '
        "match case ok print case err print end "
        '"w.txt" [ "x" file-write ] with-file-open '
        "match case ok print case err print end "
        '"new.txt" [ "made" file-write ] with-file-create ? print '
        '"new.txt" read-file ? print '
        '"e.txt" { } write-lines ? drop "e.txt" read-file ? str-len print '
        "unit ok"
    )
    source = f"import io\nword main ( -> Result[unit, IoError] ) {body} end\n"
    status, out, err = run(tmp_path, source)
    assert (status, err) == (0, "")
    assert out.splitlines() == [
        "unit",
        "3",
        "unit",
        "23",
        "gone.txt: file not found",
        "w.txt: the file is not open for writing",
        "unit",
        "made",
        "0",
    ]


def test_the_example_program(tmp_path: Path) -> None:
    source = (
        "import text\nimport arrays\n\n"
        "word main ( -> Result[unit, IoError] )\n"
        '  read-line ? "" option-or\n'
        '  " " str-split [ str-len 0u32 > ] array-filter\n'
        "  array-sort array-unique [ str-len ] array-sort-by array-reverse\n"
        "  [ print ] array-each\n"
        "  unit ok\n"
        "end\n"
    )
    status, out, _ = run(tmp_path, source, b"the cat  sat on the mat\n")
    assert status == 0
    assert out.split() == ["the", "sat", "mat", "cat", "on"]


def test_imports_of_library_modules_are_not_transitive(tmp_path: Path) -> None:
    with pytest.raises(IndeErrors) as failure:
        printed(tmp_path, "io", '"a b" " " str-split drop')
    assert "it is in module `text`, which this module does not import" in str(
        failure.value
    )


MAP = 'map-new "b" 2 map-insert "a" 1 map-insert "c" 3 map-insert "a" 10 map-insert'


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (f"{MAP} map-len print", ["3"]),
        (
            f'{MAP} "a" map-get 0 option-or print {MAP} "z" map-get 0 option-or print',
            ["10", "0"],
        ),
        (
            f'{MAP} "c" map-contains? print {MAP} "z" map-contains? print',
            ["true", "false"],
        ),
        (f'{MAP} map-keys " " str-join print', ["a b c"]),
        (f"{MAP} map-values [ print ] array-each", ["10", "2", "3"]),
        (f'{MAP} "b" map-remove map-keys " " str-join print', ["a c"]),
        (f'{MAP} "z" map-remove map-len print', ["3"]),
        (
            (
                f'{MAP} "c" 0 [ 5 + ] map-update "q" 0 [ 5 + ] map-update '
                '"c" map-get 0 option-or print'
            ),
            ["8"],
        ),
        (
            f'{MAP} "q" 0 [ 5 + ] map-update "q" map-get 0 option-or print',
            ["5"],
        ),
        (
            (
                "{ 1 2 1 } map-new [ 0u32 [ 1u32 + ] map-update ] array-fold "
                'map-entries [ pair-unpack swap write " " write print ] array-each'
            ),
            ["1 2", "2 1"],
        ),
        (
            (
                f"{MAP} map-entries map-from-entries {MAP} map-equal? print "
                f'{MAP} "a" 1 map-insert {MAP} map-equal? print'
            ),
            ["true", "false"],
        ),
        (
            (
                'map-new 3 "c" map-insert 1 "a" map-insert map-entries '
                "[ pair-unpack print print ] array-each"
            ),
            ["a", "1", "c", "3"],
        ),
    ],
)
def test_maps(tmp_path: Path, body: str, expected: list[str]) -> None:
    assert printed(tmp_path, "maps arrays text", body) == expected


def test_a_large_map(tmp_path: Path) -> None:
    declarations = (
        "word next ( u32 -> u32 ) 1103515245u32 *.wrap 12345u32 +.wrap end\n"
        "word fill ( Map[u32, u32] u32 u32 -> Map[u32, u32] )\n"
        "  let m seed n in\n"
        "    n 0u32 == if m else\n"
        "      seed next let s in m s 500u32 % s map-insert s n 1u32 - fill end\n"
        "    end\n"
        "  end\n"
        "end\n"
        "word strip ( Map[u32, u32] u32 -> Map[u32, u32] )\n"
        "  let m i in i 500u32 < if m i map-remove i 2u32 + strip else m end end\n"
        "end\n"
        "word ascending? ( array[u32] u32 -> bool )\n"
        "  let keys i in\n"
        "    i 1u32 + keys array-len >= if true else\n"
        "      keys i array-get keys i 1u32 + array-get < if\n"
        "        keys i 1u32 + ascending?\n"
        "      else false end\n"
        "    end\n"
        "  end\n"
        "end\n"
        "word sorted-in ( u32 Map[u32, u32] -> Map[u32, u32] )\n"
        "  let i m in\n"
        "    i 600u32 < if i 1u32 + m i i map-insert sorted-in else m end\n"
        "  end\n"
        "end\n"
    )
    body = (
        "map-new 7u32 1000u32 fill dup map-len print "
        "0u32 strip dup map-len print map-keys dup "
        "[ 2u32 % 0u32 == ] array-any? print 0u32 ascending? print "
        "0u32 map-new sorted-in map-len print"
    )
    seed, keys = 7, set()
    for _ in range(1000):
        seed = (seed * 1103515245 + 12345) % 2**32
        keys.add(seed % 500)
    odd = len([key for key in keys if key % 2])
    assert printed(tmp_path, "maps arrays", body, declarations) == [
        str(len(keys)),
        str(odd),
        "false",
        "true",
        "600",
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            (
                "set-new 3 set-add 1 set-add 3 set-add dup set-len print "
                "set-items [ print ] array-each"
            ),
            ["2", "1", "3"],
        ),
        (
            (
                '{ "b" "a" "b" } set-from-array dup "a" set-contains? print '
                '"a" set-remove "a" set-contains? print'
            ),
            ["true", "false"],
        ),
        (
            (
                "{ 1 2 3 } set-from-array { 2 3 4 } set-from-array set-union "
                "set-items [ print ] array-each"
            ),
            ["1", "2", "3", "4"],
        ),
        (
            (
                "{ 1 2 3 } set-from-array { 2 3 4 } set-from-array set-intersection "
                "set-items [ print ] array-each"
            ),
            ["2", "3"],
        ),
        (
            (
                "{ 1 2 3 } set-from-array { 2 3 4 } set-from-array set-difference "
                "set-items [ print ] array-each"
            ),
            ["1"],
        ),
        (
            (
                "{ 1 2 } set-from-array { 2 1 2 } set-from-array set-equal? print "
                "{ 1 } set-from-array { 2 } set-from-array set-equal? print"
            ),
            ["true", "false"],
        ),
    ],
)
def test_sets(tmp_path: Path, body: str, expected: list[str]) -> None:
    assert printed(tmp_path, "maps arrays", body) == expected


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("leaf drop", "`leaf` belongs to the opaque type `Map`, so only module `maps`"),
        (
            "map-new match case leaf case node drop drop drop drop drop end",
            "`leaf` belongs to the opaque type `Map`",
        ),
        ("set-new set-unpack drop", "`set-unpack` belongs to the opaque type `Set`"),
        (
            "map-new 1 2 map-insert dup == drop",
            (
                "`maps:Map[i32, i32]` is opaque, so only module `maps` can compare "
                "its values"
            ),
        ),
    ],
)
def test_maps_are_opaque(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(IndeErrors) as failure:
        printed(tmp_path, "maps", body)
    assert message in str(failure.value)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('"{} + {} = {}" { "1" "2" "3" } str-format print', ["1 + 2 = 3"]),
        ('"{{x}} {} {}" { "a" } str-format print', ["{x} a {}"]),
        ('"é{}}" { "b" "c" } str-format print', ["éb}"]),
        ('"{" { } str-format print "}" { } str-format print', ["{", "}"]),
        ('"{}" { "{}" } str-format print', ["{}"]),
        (
            (
                "\"7\" 3u32 '0' str-pad-left print \"ab\" 4u32 '.' str-pad-right print "
                "\"abcdef\" 3u32 ' ' str-pad-left print "
                "\"é\" 3u32 'ü' str-pad-left print"
            ),
            ["007", "ab..", "abcdef", "üüé"],
        ),
        (
            '"héllo" str-reverse print "" str-reverse show-quoted print',
            ["olléh", '""'],
        ),
    ],
)
def test_more_text(tmp_path: Path, body: str, expected: list[str]) -> None:
    assert printed(tmp_path, "text", body) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            (
                "{ 1 2 3 4 } 2u32 array-take array-len print "
                "{ 1 2 3 4 } 9u32 array-take array-len print"
            ),
            ["2", "4"],
        ),
        (
            (
                "{ 1 2 3 4 } 3u32 array-drop [ print ] array-each "
                "{ 1 2 } 5u32 array-drop array-len print"
            ),
            ["4", "0"],
        ),
        (
            (
                "{ 5 6 } array-first 0 option-or print "
                "{ 5 6 } array-last 0 option-or print "
                "{ 5 } 0u32 0u32 array-slice array-last 0 option-or print"
            ),
            ["5", "6", "0"],
        ),
        (
            (
                "{ 4 5 6 } 6 array-index-of match case some print case none end "
                "{ 4 } 9 array-index-of "
                'match case some print case none "none" print end'
            ),
            ["2", "none"],
        ),
        ("{ 1 2 3 4 5 } [ 2 % 0 == ] array-count print", ["2"]),
        (
            (
                '{ 1 2 3 } { "a" "b" } array-zip [ pair-unpack swap write print ] '
                "array-each"
            ),
            ["1a", "2b"],
        ),
        ("{ { 1 2 } { } { 3 } } array-flatten [ print ] array-each", ["1", "2", "3"]),
        ("{ 3 1 2 } [ > ] array-sort-with [ print ] array-each", ["3", "2", "1"]),
        (
            (
                '{ "bb" "a" "cc" "d" } [ str-len swap str-len < ] array-sort-with '
                '" " str-join print'
            ),
            ["bb cc a d"],
        ),
    ],
)
def test_more_arrays(tmp_path: Path, body: str, expected: list[str]) -> None:
    assert printed(tmp_path, "arrays text", body) == expected


@pytest.mark.parametrize(
    ("text", "as_f64", "as_f32"),
    [
        ("1.5", "1.5", "1.5"),
        ("-2e10", "-20000000000.0", "-20000000000.0"),
        ("0.1", "0.1", "0.1"),
        ("42", "42.0", "42.0"),
        ("1_000.5", "1000.5", "1000.5"),
        ("-0", "-0.0", "-0.0"),
        ("16777217", "16777217.0", "16777216.0"),
        ("1e-46", "1e-46", 'number out of range: "1e-46"'),
        ("3.4028236e38", "3.4028236e+38", 'number out of range: "3.4028236e38"'),
        ("1e400", 'number out of range: "1e400"', 'number out of range: "1e400"'),
        ("1e99999999999", 'number out of range: "1e99999999999"', None),
        (" 1", 'not a number: " 1"', None),
        ("01.5", 'not a number: "01.5"', None),
        ("1.", 'not a number: "1."', None),
        (".5", 'not a number: ".5"', None),
        ("inf", 'not a number: "inf"', None),
        ("1.5f32", 'not a number: "1.5f32"', None),
    ],
)
def test_parsing_floats(
    tmp_path: Path, text: str, as_f64: str, as_f32: str | None
) -> None:
    body = (
        f'"{text}" str-parse-f64 match case ok print case err print end '
        f'"{text}" str-parse-f32 match case ok print case err print end'
    )
    expected = [as_f64, as_f32 or as_f64]
    assert printed(tmp_path, "", body) == expected


def test_float_parsing_needs_runtime_3(tmp_path: Path) -> None:
    path = tmp_path / "main.inde"
    path.write_text('word main ( -> ) "1" str-parse-f64 drop end', encoding="utf-8")
    module = lower(load_program(path))
    assert module.runtime_version == 3
    module.runtime_version = 2
    with pytest.raises(IRError) as failure:
        verify(module)
    assert any("needs runtime 3" in problem for problem in failure.value.problems)


def test_the_word_count_example(tmp_path: Path) -> None:
    source = (
        "import text\nimport arrays\nimport maps\n\n"
        "word count ( Map[str, u32] str -> Map[str, u32] ) "
        "0u32 [ 1u32 + ] map-update end\n\n"
        "word main ( -> Result[unit, IoError] )\n"
        '  read-all ? str-lines [ " " str-split ] array-map array-flatten\n'
        "  [ str-len 0u32 > ] array-filter\n"
        "  map-new [ count ] array-fold map-entries\n"
        "  [ pair-unpack nip ] array-sort-by array-reverse 3u32 array-take\n"
        "  [\n"
        "    pair-unpack show 4u32 ' ' str-pad-left swap\n"
        '    { } swap array-append swap array-append "{}  {}" swap str-format print\n'
        "  ] array-each\n"
        "  unit ok\n"
        "end\n"
    )
    stdin = b"the cat and the dog\nthe bird and a cat\nthe end\n"
    status, out, _ = run(tmp_path, source, stdin)
    assert status == 0
    assert out.splitlines() == ["the     4", "cat     2", "and     2"]
