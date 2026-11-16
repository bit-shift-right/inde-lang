import os
import platform
import sys
from enum import Enum
from typing import TextIO

from inde.errors import IndeError


class Color(Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    DEFAULT = 39


def supports_color(stream: TextIO) -> bool:
    system = platform.system()
    if system == "Windows" or system == "":
        return False  # Don't support text decoration on these platforms
    if "NO_COLOR" in os.environ:
        return False
    return stream.isatty()


def decorate_text(
    text: str,
    foreground_color: Color = Color.DEFAULT,
    background_color: Color = Color.DEFAULT,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    strikethrough: bool = False,
    stream: TextIO | None = None,
) -> str:
    if not supports_color(stream or sys.stderr):
        return text
    start_code = "\x1b["
    reset_code = "\x1b[0m"

    start_code += f"{foreground_color.value}"
    start_code += f";{background_color.value + 10}"
    if bold:
        start_code += ";1"
    if italic:
        start_code += ";3"
    if underline:
        start_code += ";4"
    if strikethrough:
        start_code += ";9"
    start_code += "m"

    return start_code + text + reset_code


def format_error(
    source_name: str, error: IndeError, stream: TextIO | None = None
) -> str:
    header = source_name
    if error.location is not None:
        header += f":{error.location.line}:{error.location.column}"
    label = decorate_text("error:", Color.RED, stream=stream)
    return f"{header}: {label} {error.message}"


def report_errors(
    source_name: str, errors: list[IndeError], stream: TextIO | None = None
) -> None:
    stream = stream or sys.stderr
    for error in errors:
        print(format_error(source_name, error, stream), file=stream)
