from __future__ import annotations

from inde.ast import (
    Body,
    Constraint,
    ConstraintReference,
    Declaration,
    Item,
    Let,
    QuotationType,
    StackEffect,
    StackType,
    TypeDeclaration,
    TypeExpression,
    TypeReference,
    TypeVariable,
    WordDeclaration,
    WordReference,
)
from inde.lexer import Location

ROW = "a"
QUOTATION_NAMES = ("Q", "P", "W", "Z", "Y", "X")


def field_words(declaration: TypeDeclaration) -> list[Declaration]:
    if len(declaration.variants) != 1 or not declaration.variants[0].field_names:
        return []
    return Generator(declaration).words()


def variables_of(expression: TypeExpression) -> list[str]:
    if isinstance(expression, TypeVariable):
        return [expression.name]
    if isinstance(expression, TypeReference):
        found: list[str] = []
        for argument in expression.args:
            for name in variables_of(argument):
                if name not in found:
                    found.append(name)
        return found
    return []


class Generator:
    def __init__(self, declaration: TypeDeclaration) -> None:
        self.declaration = declaration
        self.variant = declaration.variants[0]
        self.constructor = self.variant.name
        self.public = declaration.public and not declaration.opaque
        self.location: Location | None = self.variant.location
        self.record = TypeReference(
            name=declaration.name,
            args=tuple(TypeVariable(name=p) for p in declaration.params),
            location=self.location,
        )
        self.locals = tuple(f"#{i}" for i in range(len(self.variant.fields)))
        self.quotation = next(
            name for name in QUOTATION_NAMES if name not in declaration.params
        )

    def words(self) -> list[Declaration]:
        found: list[Declaration] = []
        for index, name in enumerate(self.variant.field_names):
            base = f"{self.constructor}-{name}"
            found.extend(
                [
                    self.fetch(f"{base}@", index, name),
                    self.store(f"{base}!", index, name),
                    self.change(f"change-{base}", index),
                    self.exchange(f"exchange-{base}", index),
                ]
            )
        return found

    def word(self, name: str) -> WordReference:
        return WordReference(name=name, location=self.location)

    def effect(
        self,
        inputs: tuple[TypeExpression, ...],
        outputs: tuple[TypeExpression, ...],
        row: str | None = None,
    ) -> StackEffect:
        return StackEffect(
            inputs=StackType(row=row, types=inputs, location=self.location),
            outputs=StackType(row=row, types=outputs, location=self.location),
            location=self.location,
        )

    def unrestricted(self, index: int) -> tuple[Constraint, ...]:
        return tuple(
            Constraint(
                var=name,
                bounds=(ConstraintReference(name="Unrestricted"),),
                location=self.location,
            )
            for name in variables_of(self.variant.fields[index])
        )

    def rebuilt(self, index: int, value: str) -> tuple[Item, ...]:
        names = list(self.locals)
        names[index] = value
        return (*(self.word(name) for name in names), self.word(self.constructor))

    def unpacked(self, names: tuple[str, ...], body: Body) -> tuple[Item, ...]:
        return (
            self.word(f"{self.constructor}-unpack"),
            Let(names=names, body=body, location=self.location),
        )

    def declare(
        self,
        name: str,
        signature: StackEffect,
        where: tuple[Constraint, ...],
        body: Body,
        copies_field: str | None = None,
    ) -> WordDeclaration:
        return WordDeclaration(
            name=name,
            signature=signature,
            where=where,
            body=body,
            public=self.public,
            copies_field=copies_field,
            location=self.location,
        )

    def fetch(self, name: str, index: int, field: str) -> WordDeclaration:
        field_type = self.variant.fields[index]
        value = self.locals[index]
        body = self.unpacked(
            self.locals, (*self.rebuilt(index, value), self.word(value))
        )
        signature = self.effect((self.record,), (self.record, field_type))
        return self.declare(name, signature, self.unrestricted(index), body, field)

    def store(self, name: str, index: int, field: str) -> WordDeclaration:
        field_type = self.variant.fields[index]
        body = (
            self.word("swap"),
            *self.unpacked(("#new", *self.locals), self.rebuilt(index, "#new")),
        )
        signature = self.effect((self.record, field_type), (self.record,))
        return self.declare(name, signature, self.unrestricted(index), body, field)

    def change(self, name: str, index: int) -> WordDeclaration:
        field_type = self.variant.fields[index]
        quotation = TypeVariable(name=self.quotation, location=self.location)
        bound = QuotationType(
            effect=self.effect((field_type,), (field_type,), ROW),
            location=self.location,
        )
        where = (
            Constraint(var=self.quotation, bounds=(bound,), location=self.location),
        )
        inner = Let(names=("#new",), body=self.rebuilt(index, "#new"))
        body = (
            self.word("swap"),
            *self.unpacked(
                ("#action", *self.locals),
                (
                    self.word(self.locals[index]),
                    self.word("#action"),
                    self.word("call"),
                    inner,
                ),
            ),
        )
        signature = self.effect((self.record, quotation), (self.record,), ROW)
        return self.declare(name, signature, where, body)

    def exchange(self, name: str, index: int) -> WordDeclaration:
        field_type = self.variant.fields[index]
        body = (
            self.word("swap"),
            *self.unpacked(
                ("#new", *self.locals),
                (*self.rebuilt(index, "#new"), self.word(self.locals[index])),
            ),
        )
        signature = self.effect((self.record, field_type), (self.record, field_type))
        return self.declare(name, signature, (), body)
