import sys
from pathlib import Path

from inde import checker, logger, parser
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Interpreter
from inde.ir import format_module
from inde.lowering import lower
from inde.repl import run_repl

TRAP_STATUS = 2


def read_source(source_name: str) -> str | None:
    try:
        return Path(source_name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"{source_name}: cannot read file: {error}", file=sys.stderr)
        return None


def run_file(source_name: str) -> int:
    source = read_source(source_name)
    if source is None:
        return 1

    try:
        parsed_program = parser.parse(source)
        checked_program = checker.check(parsed_program, require_main=True)
    except IndeErrors as failure:
        logger.report_errors(source_name, failure.errors)
        return 1

    interpreter = Interpreter(checked_program.words, checked_program.constants)
    try:
        return interpreter.run_main()
    except IndeTrap as trap:
        sys.stdout.flush()
        logger.report_trap(source_name, trap)
        return TRAP_STATUS


def show_ir(source_name: str, debug: bool) -> int:
    source = read_source(source_name)
    if source is None:
        return 1
    try:
        module = lower(parser.parse(source))
    except IndeErrors as failure:
        logger.report_errors(source_name, failure.errors)
        return 1
    print(format_module(module, debug=debug), end="")
    return 0


USAGE = "usage: python -m inde [--ir | --ir-debug] [file]"


def main() -> int:
    arguments = sys.argv[1:]
    if arguments and arguments[0] in ("--ir", "--ir-debug"):
        if len(arguments) != 2:
            print(USAGE, file=sys.stderr)
            return 1
        return show_ir(arguments[1], debug=arguments[0] == "--ir-debug")
    if len(arguments) > 1 or (arguments and arguments[0].startswith("--")):
        print(USAGE, file=sys.stderr)
        return 1
    if arguments:
        return run_file(arguments[0])
    run_repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
