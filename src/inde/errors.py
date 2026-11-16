from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inde.lexer import Location


class IndeError(Exception):
    def __init__(self, message: str, location: Location | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.location = location

    def __str__(self) -> str:
        if self.location is None:
            return self.message
        return f"{self.location.line}:{self.location.column}: {self.message}"


class IndeErrors(Exception):
    def __init__(self, errors: list[IndeError]) -> None:
        super().__init__(f"{len(errors)} error(s)")
        self.errors = errors

    def __str__(self) -> str:
        return "\n".join(str(error) for error in self.errors)


class IndeTrap(Exception):
    def __init__(self, message: str, location: Location | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.location = location

    def __str__(self) -> str:
        if self.location is None:
            return self.message
        return f"{self.location.line}:{self.location.column}: {self.message}"
