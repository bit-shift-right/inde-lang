import os
import sys
from collections.abc import Callable, Sequence
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
from inde.system import OUTPUT_FAILED, System
from inde.verifier import IRError, verify

TRAP_STATUS = 2


def read_source(source_name: str) -> str | None:
    try:
        return Path(source_name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"{source_name}: cannot read file: {error}", file=sys.stderr)
        return None


def flush_output() -> bool:
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return False
    return True


def finish(source_name: str, run: Callable[[], int]) -> int:
    try:
        status = run()
    except IndeTrap as trap:
        flush_output()
        try:
            logger.report_trap(source_name, trap)
        except (OSError, ValueError):
            pass
        return TRAP_STATUS
    if not flush_output():
        try:
            print(OUTPUT_FAILED, file=sys.stderr)
        except (OSError, ValueError):
            pass
        return TRAP_STATUS
    return status


def run_file(source_name: str, arguments: Sequence[str] = ()) -> int:
    source = read_source(source_name)
    if source is None:
        return 1

    try:
        parsed_program = load_program(Path(source_name), source)
        checked_program = checker.check(parsed_program, require_main=True)
    except IndeErrors as failure:
        logger.report_errors(source_name, failure.errors)
        return 1

    interpreter = Interpreter(
        checked_program.words, checked_program.constants, system=System(args=arguments)
    )
    return finish(source_name, interpreter.run_main)


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


def run_ir(source_name: str, arguments: Sequence[str] = ()) -> int:
    module = compile_file(source_name)
    if isinstance(module, int):
        return module
    return finish(source_name, Machine(module, system=System(args=arguments)).run_main)


USAGE = (
    "usage: python -m inde [file.inde [arguments...]]\n"
    "       python -m inde --run-ir file.inde|file.ir [arguments...]\n"
    "       python -m inde --ir|--ir-debug file.inde|file.ir"
)


def main() -> int:
    arguments = sys.argv[1:]
    if arguments and arguments[0] in ("--ir", "--ir-debug", "--run-ir"):
        flag, rest = arguments[0], arguments[1:]
        if not rest or (flag != "--run-ir" and len(rest) != 1):
            print(USAGE, file=sys.stderr)
            return 1
        if flag == "--run-ir":
            return run_ir(rest[0], rest[1:])
        return show_ir(rest[0], debug=flag == "--ir-debug")
    if arguments and arguments[0].startswith("--"):
        print(USAGE, file=sys.stderr)
        return 1
    if arguments:
        return run_file(arguments[0], arguments[1:])
    run_repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
