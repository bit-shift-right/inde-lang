import sys
from pathlib import Path

from inde import ast, logger, parser
from inde.errors import IndeErrors

DEFAULT_SOURCE = "./examples/parser_demo_test.inde"


def main() -> int:
    source_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE

    try:
        source = Path(source_name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"{source_name}: cannot read file: {error}", file=sys.stderr)
        return 1

    try:
        parsed_program = parser.parse(source)
    except IndeErrors as failure:
        logger.report_errors(source_name, failure.errors)
        return 1

    print(ast.dump(parsed_program))
    return 0


if __name__ == "__main__":
    sys.exit(main())
