import sys
from pathlib import Path

from inde import checker, logger, parser
from inde.errors import IndeErrors

DEFAULT_SOURCE = "./examples/checker_demo.inde"


def main() -> int:
    source_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE

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

    for word in checked_program.words.values():
        print(f"{word.name} {word.effect}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
