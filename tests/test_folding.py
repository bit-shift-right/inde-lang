import pytest

from inde.checker import Checker, check
from inde.errors import IndeError, IndeErrors
from inde.folding import Folder
from inde.interpreter import inspect
from inde.lexer import tokenize
from inde.parser import Parser, parse
from inde.type_model import Type

DECLARATIONS = (
    "const ZERO = 0\n"
    "const LIMIT = 2147483647\n"
    "const PRIMES = { 2 3 5 7 }\n"
    "word five ( -> i32 ) 5 end\n"
)


def known_after(line: str, locals_: tuple[str, ...] = ()) -> list[str]:
    checker = Checker()
    checker.check_program(parse(DECLARATIONS))
    folder = Folder(checker.words, checker.constants)
    body = Parser(tokenize(line)).parse_repl_line()
    assert isinstance(body, tuple)
    for item in body:
        name = getattr(item, "name", None)
        folder.step(item, local=name in locals_)
    return ["?" if value is None else inspect(value) for value in folder.known]


def line_fails(line: str) -> str:
    checker = Checker()
    checker.check_program(parse(DECLARATIONS))
    body = Parser(tokenize(line)).parse_repl_line()
    assert isinstance(body, tuple)
    stack: list[Type] = []
    with pytest.raises(IndeError) as err:
        checker.check_line(body, stack)
    return err.value.message


def accepts(source: str) -> None:
    check(parse(DECLARATIONS + source))


def get_failure_message(source: str) -> str:
    with pytest.raises(IndeErrors) as err:
        check(parse(DECLARATIONS + source))
    messages = [error.message for error in err.value.errors]
    assert len(messages) == 1, messages
    return messages[0]


@pytest.mark.parametrize(
    ("line", "known"),
    [
        ("1 2", ["1", "2"]),
        ("1.5 'a' \"s\" true", ["1.5", "'a'", '"s"', "true"]),
        ("{ 1 2 }", ["{ 1 2 }"]),
        ("ZERO LIMIT", ["0", "2147483647"]),
        ("PRIMES", ["{ 2 3 5 7 }"]),
        ("1 2 swap", ["2", "1"]),
        ("1 dup", ["1", "1"]),
        ("1 2 drop", ["1"]),
        ("1 2 over", ["1", "2", "1"]),
        ("1 2 3 rot", ["2", "3", "1"]),
        ("1 five 2", ["2"]),
        ("1 2 [ 3 ] 4", ["4"]),
        ("1 if 2 end 3", ["3"]),
        ("1 2 let a b in end 3", ["3"]),
    ],
)
def test_known_values(line: str, known: list[str]) -> None:
    assert known_after(line) == known


def test_a_stack_word_reaching_below_what_is_known() -> None:
    assert known_after("five 1 swap") == ["?", "?"]
    assert known_after("five dup") == ["?", "?"]
    assert known_after("five 1 2 rot") == ["?", "?", "?"]
    assert known_after("drop 1") == ["1"]


def test_a_local_is_not_known() -> None:
    assert known_after("1 x", ("x",)) == []
    assert known_after("1 x 2", ("x",)) == ["2"]


@pytest.mark.parametrize(
    ("line", "known"),
    [
        ("1 2 +", ["3"]),
        ("1 2 + 3 *", ["9"]),
        ("7 2 / 7 2 %", ["3", "1"]),
        ("1.5 2.0 *", ["3.0"]),
        ("1 2 <", ["true"]),
        ("200u8 100u8 +.wrap", ["44u8"]),
        ("200u8 100u8 +.checked", ["none"]),
        ("7 0 /.checked", ["none"]),
        ("1.0 0.0 /.ieee", ["inf"]),
        ("300 to-u8.sat", ["255u8"]),
        ("{ 10 20 30 } 1u32 array-get", ["20"]),
        ('"héllo" str-len', ["6u32"]),
        ("LIMIT 1 -", ["2147483646"]),
        ("unit", ["unit"]),
    ],
)
def test_folding(line: str, known: list[str]) -> None:
    assert known_after(line) == known


def test_what_is_not_folded() -> None:
    assert known_after("five 1 +") == ["?"]
    assert known_after("1 five +") == ["?"]
    assert known_after("+") == ["?"]
    assert known_after('"a" print') == []
    assert known_after("1 strbuf-new") == []
    assert known_after("1 [ 1 + ] call") == []


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "200u8 100u8 +",
            "this always traps: integer overflow in `+` (u8 200u8, u8 100u8)",
        ),
        ("1 0 /", "this always traps: division by zero in `/` (i32 1, i32 0)"),
        (
            "1.0 0.0 /",
            "this always traps: division by zero in `/` (f32 1.0, f32 0.0)",
        ),
        (
            "1 2 + 2147483647 +",
            "this always traps: integer overflow in `+` (i32 3, i32 2147483647)",
        ),
        (
            "LIMIT 1 +",
            "this always traps: integer overflow in `+` (i32 2147483647, i32 1)",
        ),
        (
            "LIMIT dup +",
            (
                "this always traps: integer overflow in `+` "
                "(i32 2147483647, i32 2147483647)"
            ),
        ),
        (
            "{ 1 2 } 5u32 array-get",
            "this always traps: index out of range in `array-get` (index 5, length 2)",
        ),
        ("300 to-u8", "this always traps: value out of range in `to-u8` (i32 300)"),
        (
            "1u8 8u32 shl",
            "this always traps: shift amount out of range in `shl` (u8 1u8, u32 8u32)",
        ),
        (
            "-1.0 sqrt",
            "this always traps: invalid float operation in `sqrt` (f32 -1.0)",
        ),
    ],
)
def test_a_certain_trap_is_an_error(line: str, message: str) -> None:
    assert line_fails(line) == message


def test_even_where_it_might_not_run() -> None:
    assert get_failure_message("word t ( bool -> ) if 1 0 / drop end end") == (
        "this always traps: division by zero in `/` (i32 1, i32 0)"
    )


def test_the_error_points_at_the_operation() -> None:
    with pytest.raises(IndeErrors) as err:
        check(parse("word t ( -> i32 )\n  200 100\n  *\n  2147483647 *\nend"))
    location = err.value.errors[0].location
    assert location is not None
    assert (location.line, location.column) == (4, 14)


@pytest.mark.parametrize(
    ("body", "name"),
    [
        ("0 /", "/"),
        ("0 /.wrap", "/.wrap"),
        ("0 /.sat", "/.sat"),
        ("0 %", "%"),
        ("0 %.wrap", "%.wrap"),
        ("0 %.sat", "%.sat"),
        ("ZERO /", "/"),
        ("5 0 swap drop /", "/"),
    ],
)
def test_a_zero_divisor_alone_is_enough(body: str, name: str) -> None:
    assert get_failure_message(f"word t ( i32 -> i32 ) {body} end") == (
        f"`{name}` always traps here: its divisor is zero"
    )


def test_the_forms_that_define_a_result_are_not_errors() -> None:
    accepts("word t ( i32 -> Option[i32] ) 0 /.checked end")
    accepts("word t ( f32 -> f32 ) 0.0 / end")
    accepts("word t ( f32 -> f32 ) 0.0 /.ieee end")
    accepts("word t ( i32 -> i32 ) 1 / end")
    accepts("word t ( i32 i32 -> i32 ) / end")


def test_a_local_zero_is_not_known() -> None:
    accepts("word t ( i32 -> i32 ) 0 let z in z / end end")
