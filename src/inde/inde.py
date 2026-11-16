from pathlib import Path

from inde import ast, parser

PROGRAM_NAME = Path(__file__).stem


def main() -> None:
    with open("./examples/parser_demo_test.inde", "r+") as source_file:
        source = source_file.read()

    parsed_program = parser.parse(source, PROGRAM_NAME)
    print(ast.dump(parsed_program))


if __name__ == "__main__":
    main()
