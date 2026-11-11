import platform
import sys
from enum import Enum


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


class Logger:
    def __init__(self, program_name: str) -> None:
        self.program_name = program_name

    def decorate_text(
        self,
        text: str,
        foreground_color: Color = Color.DEFAULT,
        background_color: Color = Color.DEFAULT,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        strikethrough: bool = False,
    ) -> str:
        system = platform.system()
        if system == "Windows" or system == "":
            return text  # Don't support text decoration on these platforms
        start_code = "\x1b["
        reset_code = "\x1b[0m"

        start_code += f'{foreground_color.value}'
        start_code += f';{background_color.value + 10}'
        if bold:
            start_code += ';1'
        if italic:
            start_code += ';3'
        if underline:
            start_code += ';4'
        if strikethrough:
            start_code += ';9'
        start_code += "m"

        return start_code + text + reset_code

    def print_error(self, message: str, location_string: str = "") -> None:
        print(
            f'{location_string}{self.program_name}: '
            f'{self.decorate_text("error:", Color.RED)} '
            f'{message}',
            file=sys.stderr,
        )
