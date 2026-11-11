import argparse
from importlib.metadata import version

from inde import lexer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="inde",
        description="A modern, stack-oriented, concatenative programming language.",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s version {version('inde-lang')}",
    )

    parser.parse_args()

    lexer.tokenize("    abc", "inde")


if __name__ == "__main__":
    main()
