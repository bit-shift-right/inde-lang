import sys
from pathlib import Path

from inde import checker, logger, parser
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Interpreter
from inde.repl import run_repl

TRAP_STATUS = 2


def run_file(source_name: str) -> int:
    try:
        source = Path(source_name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"{source_name}: cannot read file: {error}", file=sys.stderr)
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


def main() -> int:
    if len(sys.argv) > 1:
        return run_file(sys.argv[1])
    run_repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
