import sys
from pathlib import Path

from inde import checker, logger
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Interpreter
from inde.ir import Module, format_module
from inde.ir_parser import IRSyntaxError, parse_module
from inde.lowering import lower
from inde.machine import Machine
from inde.modules import load_program
from inde.repl import run_repl
from inde.verifier import IRError, verify

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
        parsed_program = load_program(Path(source_name), source)
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


def compile_file(source_name: str) -> Module | int:
    source = read_source(source_name)
    if source is None:
        return 1
    written = source_name.endswith(".ir")
    try:
        if written:
            module = parse_module(source)
        else:
            module = lower(load_program(Path(source_name), source))
    except IndeErrors as failure:
        logger.report_errors(source_name, failure.errors)
        return 1
    except IRSyntaxError as failure:
        print(
            f"{source_name}:{failure.line}: error: {failure.message}", file=sys.stderr
        )
        return 1
    try:
        verify(module)
    except IRError as failure:
        problem = (
            "the IR is invalid"
            if written
            else ("internal error: the compiled IR is invalid")
        )
        print(f"{source_name}: {problem}", file=sys.stderr)
        for line in failure.problems:
            print(f"  {line}", file=sys.stderr)
        return 1 if written else 3
    return module


def show_ir(source_name: str, debug: bool) -> int:
    module = compile_file(source_name)
    if isinstance(module, int):
        return module
    print(format_module(module, debug=debug), end="")
    return 0


def run_ir(source_name: str) -> int:
    module = compile_file(source_name)
    if isinstance(module, int):
        return module
    try:
        return Machine(module).run_main()
    except IndeTrap as trap:
        sys.stdout.flush()
        logger.report_trap(source_name, trap)
        return TRAP_STATUS


USAGE = "usage: python -m inde [--ir | --ir-debug | --run-ir] [file.inde | file.ir]"


def main() -> int:
    arguments = sys.argv[1:]
    if arguments and arguments[0] in ("--ir", "--ir-debug", "--run-ir"):
        if len(arguments) != 2:
            print(USAGE, file=sys.stderr)
            return 1
        if arguments[0] == "--run-ir":
            return run_ir(arguments[1])
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
