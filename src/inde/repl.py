from __future__ import annotations

import sys
from typing import TextIO

from inde import logger
from inde.errors import IndeErrors, IndeTrap
from inde.interpreter import Session
from inde.system import System

SOURCE_NAME = "repl"
PROMPT = ">> "
CONTINUE_PROMPT = ".. "


def run_repl(
    input_stream: TextIO | None = None,
    output: TextIO | None = None,
    error_output: TextIO | None = None,
    session: Session | None = None,
) -> None:
    input_stream = input_stream or sys.stdin
    output = output or sys.stdout
    error_output = error_output or sys.stderr
    if session is None:
        system = System(stdout=output, stderr=error_output, stdin=input_stream)
        session = Session(output, system=system)
    interactive = input_stream.isatty()

    pending = ""
    while True:
        if interactive:
            prompt = CONTINUE_PROMPT if pending else PROMPT
            print(prompt, end="", file=output, flush=True)
        line = input_stream.readline()
        at_end = line == ""
        blank = line.strip() == ""
        if blank and not pending:
            if at_end:
                break
            continue
        pending += line
        if not blank and session.is_incomplete(pending):
            continue
        try:
            session.run_line(pending)
        except IndeErrors as failure:
            logger.report_errors(SOURCE_NAME, failure.errors, error_output)
        except IndeTrap as trap:
            logger.report_trap(SOURCE_NAME, trap, error_output)
        else:
            print(session.show_stack(), file=output)
        pending = ""
        if at_end:
            break
    if interactive:
        print(file=output)
