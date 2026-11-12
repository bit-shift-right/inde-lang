from pathlib import Path

from inde import lexer

PROGRAM_NAME = Path(__file__).stem


def main() -> None:
    with open("./examples/lexer_grammar_test.inde", "r+") as source_file:
        source = source_file.read()

    source_tokens = lexer.tokenize(source, PROGRAM_NAME)
    lexer.dump_tokens(source_tokens)


if __name__ == "__main__":
    main()
