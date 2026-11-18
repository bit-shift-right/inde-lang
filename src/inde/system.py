from __future__ import annotations

import io
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import BinaryIO, TextIO, TypeVar

from inde.errors import IndeTrap
from inde.lexer import Location, quote
from inde.numeric import format_float
from inde.type_model import (
    CHAR,
    FILE,
    IO_ERROR,
    IO_ERROR_KIND,
    STR,
    U8,
    UNIT,
    Type,
    array_of,
    option_of,
    result_of,
)
from inde.values import (
    Data,
    IoFailure,
    OpenFile,
    Value,
    array,
    none,
    some,
    string,
)

UNIT_VALUE = Value(UNIT, Data("unit"))
OUTPUT_FAILED = "output failed"
MAX_LENGTH = 2**32 - 1
CHUNK = 1 << 16
TOO_LARGE = "too large to read at once"
INVALID_INPUT = "standard input is not valid UTF-8"

ERROR_KINDS: tuple[tuple[type[OSError], str, str], ...] = (
    (FileNotFoundError, "not-found", "file not found"),
    (PermissionError, "permission-denied", "permission denied"),
    (FileExistsError, "already-exists", "already exists"),
    (IsADirectoryError, "is-a-directory", "is a directory"),
    (NotADirectoryError, "not-a-directory", "not a directory"),
)

Done = TypeVar("Done")


def failure(kind: str, message: str, path: str | None = None) -> Value:
    return Value(IO_ERROR, IoFailure(kind, path, message))


def from_os_error(error: OSError, path: str | None) -> Value:
    for kind_type, kind, message in ERROR_KINDS:
        if isinstance(error, kind_type):
            return failure(kind, message, path)
    reason = error.strerror or str(error) or "input or output failed"
    return failure("other-error", reason[:1].lower() + reason[1:], path)


def ok(value_type: Type, value: Value) -> Value:
    return Value(result_of(value_type, IO_ERROR), Data("ok", (value,)))


def err(value_type: Type, error: Value) -> Value:
    return Value(result_of(value_type, IO_ERROR), Data("err", (error,)))


def readable(text: str) -> str:
    return text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


def display(value: Value) -> str:
    if value.type == UNIT:
        return "unit"
    data = value.data
    if isinstance(data, bool):
        return "true" if data else "false"
    if isinstance(data, float):
        return format_float(data, value.type)
    if isinstance(data, (int, str)):
        return str(data)
    if isinstance(data, IoFailure):
        return data.text()
    raise AssertionError(f"`{value.type}` has no display text")


def too_large(raw: bytes) -> bool:
    return len(raw) > MAX_LENGTH


def quoted_form(value: Value) -> str:
    data = value.data
    if value.type == STR and isinstance(data, str):
        return quote(data, '"')
    if value.type == CHAR and isinstance(data, str):
        return quote(data, "'")
    return display(value)


def text_of(value: Value) -> str:
    assert isinstance(value.data, str)
    return value.data


def bytes_of(value: Value) -> bytes:
    assert isinstance(value.data, tuple)
    return bytes(
        element.data for element in value.data if isinstance(element.data, int)
    )


def bytes_value(raw: bytes) -> Value:
    return array(U8, tuple(Value(U8, byte) for byte in raw))


def strip_line(raw: bytes) -> bytes:
    if raw.endswith(b"\n"):
        raw = raw[:-1]
        if raw.endswith(b"\r"):
            raw = raw[:-1]
    return raw


class System:
    def __init__(
        self,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        stdin: BinaryIO | TextIO | None = None,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.stdout = sys.stdout if stdout is None else stdout
        self.stderr = sys.stderr if stderr is None else stderr
        self.stdin = sys.stdin.buffer if stdin is None else stdin
        self.args = tuple(readable(argument) for argument in args)
        self.env = os.environ if env is None else env

    def emit(self, text: str, error: bool, location: Location | None) -> None:
        try:
            if error:
                self.stdout.flush()
                self.stderr.write(text)
                self.stderr.flush()
            else:
                self.stdout.write(text)
        except (OSError, ValueError):
            raise IndeTrap(OUTPUT_FAILED, location) from None

    def report(self, text: str, stream: TextIO | None = None) -> None:
        try:
            self.stdout.flush()
        except (OSError, ValueError):
            pass
        try:
            print(text, file=self.stderr if stream is None else stream)
        except (OSError, ValueError):
            pass

    def run(
        self, name: str, operands: list[Value], location: Location | None
    ) -> list[Value]:
        if name in ("print", "write", "eprint", "ewrite"):
            ending = "\n" if name.endswith("print") else ""
            self.emit(display(operands[0]) + ending, name.startswith("e"), location)
            return []
        if name == "show":
            return [string(display(operands[0]))]
        if name == "show-quoted":
            return [string(quoted_form(operands[0]))]
        if name == "io-error-kind":
            data = operands[0].data
            assert isinstance(data, IoFailure)
            return [Value(IO_ERROR_KIND, Data(data.kind))]
        if name == "args":
            return [array(STR, tuple(string(argument) for argument in self.args))]
        if name == "env-get":
            return [self.env_get(text_of(operands[0]))]
        if name in ("read-line", "read-all"):
            return [self.read_input(name == "read-line", location)]
        if name.startswith("file-"):
            return self.file_operation(name, operands)
        return [self.whole_file(name, operands)]

    def env_get(self, key: str) -> Value:
        if not key or "=" in key or "\0" in key:
            return none(STR)
        value = self.env.get(key)
        return none(STR) if value is None else some(string(readable(value)))

    def read_input(self, line: bool, location: Location | None) -> Value:
        try:
            self.stdout.flush()
        except (OSError, ValueError):
            raise IndeTrap(OUTPUT_FAILED, location) from None
        value_type = option_of(STR) if line else STR
        try:
            raw = self.stdin.readline() if line else self.stdin.read()
        except OSError as error:
            return err(value_type, from_os_error(error, None))
        except UnicodeDecodeError:
            return err(value_type, failure("invalid-text", INVALID_INPUT))
        if isinstance(raw, str):
            raw = raw.encode("utf-8", "surrogateescape")
        if line and not raw:
            return ok(value_type, none(STR))
        return decoded(raw, value_type, None, line)

    def whole_file(self, name: str, operands: list[Value]) -> Value:
        path = text_of(operands[0])
        if name in ("read-file", "read-file-bytes"):
            value_type = STR if name == "read-file" else BYTES
            try:
                raw = guard(path, lambda: self.load(path))
            except Failed as failed:
                return err(value_type, failed.error)
            if name == "read-file-bytes":
                return sized(raw, path)
            return decoded(raw, STR, path, False)
        content = operands[1]
        if name == "write-file-bytes":
            raw = bytes_of(content)
        else:
            raw = text_of(content).encode()
        mode = "ab" if name == "append-file" else "wb"
        try:
            guard(path, lambda: self.store(path, mode, raw))
        except Failed as failed:
            return err(UNIT, failed.error)
        return ok(UNIT, UNIT_VALUE)

    def load(self, path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()

    def store(self, path: str, mode: str, raw: bytes) -> None:
        with open(path, mode) as handle:
            handle.write(raw)

    def file_operation(self, name: str, operands: list[Value]) -> list[Value]:
        if name in ("file-open", "file-create", "file-append"):
            path = text_of(operands[0])
            mode = {"file-open": "rb", "file-create": "wb", "file-append": "ab"}[name]
            try:
                handle = guard(path, lambda: opened(path, mode))
            except Failed as failed:
                return [err(FILE, failed.error)]
            return [ok(FILE, Value(FILE, OpenFile(handle, path, mode != "rb")))]
        handle_value = operands[0]
        opened_file = handle_value.data
        assert isinstance(opened_file, OpenFile)
        if name == "file-close":
            try:
                guard(opened_file.path, opened_file.handle.close)
            except Failed as failed:
                return [err(UNIT, failed.error)]
            return [ok(UNIT, UNIT_VALUE)]
        return [handle_value, self.file_step(name, opened_file, operands)]

    def file_step(
        self, name: str, opened_file: OpenFile, operands: list[Value]
    ) -> Value:
        path = opened_file.path
        handle = opened_file.handle
        writes = name in ("file-write", "file-write-bytes")
        value_type = FILE_RESULTS.get(name, UNIT)
        if writes != opened_file.writing:
            purpose = "writing" if writes else "reading"
            return err(
                value_type,
                failure("other-error", f"the file is not open for {purpose}", path),
            )
        try:
            if writes:
                content = operands[1]
                raw = (
                    bytes_of(content)
                    if name == "file-write-bytes"
                    else (text_of(content).encode())
                )
                guard(path, lambda: handle.write(raw))
                return ok(UNIT, UNIT_VALUE)
            if name == "file-read-bytes":
                count = operands[1].data
                assert isinstance(count, int)
                return sized(guard(path, lambda: read_up_to(handle, count)), path)
            if name == "file-read-all":
                return decoded(guard(path, handle.read), STR, path, False)
            line = guard(path, handle.readline)
        except Failed as failed:
            return err(value_type, failed.error)
        if not line:
            return ok(value_type, none(STR))
        return decoded(line, value_type, path, True)


class Failed(Exception):
    def __init__(self, error: Value) -> None:
        super().__init__("input or output failed")
        self.error = error


def guard(path: str, action: Callable[[], Done]) -> Done:
    try:
        return action()
    except OSError as error:
        raise Failed(from_os_error(error, path)) from None
    except ValueError:
        raise Failed(failure("other-error", "invalid path", path)) from None


def opened(path: str, mode: str) -> BinaryIO:
    raw = io.FileIO(path, mode[0])
    if mode == "rb":
        return io.BufferedReader(raw)
    return io.BufferedWriter(raw)


def read_up_to(handle: BinaryIO, count: int) -> bytes:
    parts: list[bytes] = []
    remaining = count
    while remaining > 0:
        part = handle.read(min(remaining, CHUNK))
        if not part:
            break
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def sized(raw: bytes, path: str) -> Value:
    if too_large(raw):
        return err(BYTES, failure("other-error", TOO_LARGE, path))
    return ok(BYTES, bytes_value(raw))


def decoded(raw: bytes, value_type: Type, path: str | None, line: bool) -> Value:
    if too_large(raw):
        return err(value_type, failure("other-error", TOO_LARGE, path))
    try:
        text = string((strip_line(raw) if line else raw).decode("utf-8"))
    except UnicodeDecodeError:
        message = "not valid UTF-8 text" if path else INVALID_INPUT
        return err(value_type, failure("invalid-text", message, path))
    return ok(value_type, some(text) if line else text)


BYTES = array_of(U8)
FILE_RESULTS = {
    "file-read-line": option_of(STR),
    "file-read-all": STR,
    "file-read-bytes": BYTES,
}
