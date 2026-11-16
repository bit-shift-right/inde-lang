import io
import platform

import pytest

from inde import logger
from inde.errors import IndeError, IndeErrors
from inde.lexer import Location
from inde.parser import parse


class FakeTerminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_error_keeps_message_and_location_apart() -> None:
    error = IndeError("something is wrong", Location(3, 7, 40))
    assert error.message == "something is wrong"
    assert error.location == Location(3, 7, 40)
    assert str(error) == "3:7: something is wrong"
    assert str(IndeError("no position")) == "no position"


def test_parse_reports_every_error_in_source_order() -> None:
    source = (
        "word broken1 b end\nword ok ( -> ) end\n7 8 9\nword broken2 ( -> x ) end\n"
    )
    with pytest.raises(IndeErrors) as err:
        parse(source)
    errors = err.value.errors
    assert [error.location.line for error in errors] == [1, 3, 4]  # type: ignore[union-attr]
    assert "`(` to start the signature" in errors[0].message
    assert "outside a declaration" in errors[1].message
    assert "`x` is not a type" in errors[2].message


def test_lexical_error_is_reported_alone() -> None:
    with pytest.raises(IndeErrors) as err:
        parse('word b c end\nword a ( -> ) "unterminated')
    assert len(err.value.errors) == 1
    assert "unterminated string" in err.value.errors[0].message


def test_format_names_the_source_file() -> None:
    error = IndeError("bad thing", Location(3, 7, 40))
    plain = io.StringIO()  # not a terminal
    assert (
        logger.format_error("examples/demo.inde", error, plain)
        == "examples/demo.inde:3:7: error: bad thing"
    )
    assert (
        logger.format_error("demo.inde", IndeError("no position"), plain)
        == "demo.inde: error: no position"
    )


def test_no_color_codes_when_output_is_not_a_terminal() -> None:
    assert logger.decorate_text("x", logger.Color.RED, stream=io.StringIO()) == "x"


def test_color_codes_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.delenv("NO_COLOR", raising=False)
    terminal = FakeTerminal()
    assert logger.decorate_text("x", logger.Color.RED, stream=terminal) == (
        "\x1b[31;49mx\x1b[0m"
    )
    monkeypatch.setenv("NO_COLOR", "1")
    assert logger.decorate_text("x", logger.Color.RED, stream=terminal) == "x"
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert logger.decorate_text("x", logger.Color.RED, stream=terminal) == "x"


def test_report_errors_writes_one_line_per_error() -> None:
    stream = io.StringIO()
    errors = [
        IndeError("first", Location(1, 1, 0)),
        IndeError("second", Location(2, 5, 9)),
    ]
    logger.report_errors("demo.inde", errors, stream)
    assert stream.getvalue() == (
        "demo.inde:1:1: error: first\ndemo.inde:2:5: error: second\n"
    )
