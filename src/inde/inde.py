from inde import lexer


def main() -> None:
    with open("./examples/lexer_grammar_test.inde", "r+") as source_file:
        source = source_file.read()

    source_tokens = lexer.tokenize(source, "inde")
    lexer.dump_tokens(source_tokens)


if __name__ == "__main__":
    main()
