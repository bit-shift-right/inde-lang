from inde.inference import Inference
from inde.type_model import (
    F32,
    I32,
    LINEAR,
    STR,
    UNRESTRICTED,
    Effect,
    Stack,
    Type,
    quotation_of,
)

T = Type("T")
Q = Type("Q")


def quotation(
    inputs: tuple[Type, ...],
    outputs: tuple[Type, ...],
    rows: tuple[str, str] | None = None,
    qualifier: Type = UNRESTRICTED,
) -> Type:
    return quotation_of(Effect(inputs, outputs, rows=rows), qualifier)


def test_quotation_types_display() -> None:
    assert str(quotation((I32,), (I32,))) == "[ i32 -> i32 ]"
    assert str(quotation((), ())) == "[ -> ]"
    assert str(quotation((), (STR,), qualifier=LINEAR)) == "linear [ -> str ]"
    assert str(quotation((I32,), (), ("..a", "..b"))) == "[ ..a i32 -> ..b ]"
    assert str(quotation((), (), ("?..1", "?..2"))) == "[ ..? -> ..? ]"
    assert str(quotation((), (), qualifier=Q)) == "Q"
    effect = Effect((quotation((), (), ("..a", "..b"), Q),), (), rows=("..a", "..b"))
    assert str(effect) == "( ..a Q -> ..b ) where Q: [ ..a -> ..b ]"


def test_a_bound_row_is_flattened() -> None:
    inference = Inference()
    row = inference.fresh_row()
    below = inference.fresh_row()
    assert inference.bind_row(row, Stack(below, (I32,)))
    assert inference.flatten(Stack(row, (STR,))) == Stack(below, (I32, STR))
    assert inference.flatten(Stack("..a", (STR,))) == Stack("..a", (STR,))


def test_flattening_follows_every_binding() -> None:
    inference = Inference()
    first, second, third = (inference.fresh_row() for _ in range(3))
    unknown = inference.fresh("a test type", None)
    assert inference.bind_row(first, Stack(second, (unknown,)))
    assert inference.bind_row(second, Stack(third, (STR,)))
    assert inference.unify(unknown, I32)
    assert inference.flatten(Stack(first)) == Stack(third, (STR, I32))


def test_stacks_of_the_same_types_unify() -> None:
    inference = Inference()
    assert inference.unify_stacks(Stack("..a", (I32, STR)), Stack("..a", (I32, STR)))
    assert not inference.unify_stacks(Stack("..a", (I32,)), Stack("..a", (F32,)))
    assert inference.row_bindings == {}


def test_types_are_compared_from_the_top() -> None:
    inference = Inference()
    unknown = inference.fresh("a test type", None)
    row = inference.fresh_row()
    assert inference.unify_stacks(Stack(row, (unknown,)), Stack("..a", (STR, I32)))
    assert inference.resolve(unknown) == I32
    assert inference.flatten(Stack(row)) == Stack("..a", (STR,))


def test_an_unknown_row_takes_what_is_left() -> None:
    inference = Inference()
    row = inference.fresh_row()
    assert inference.unify_stacks(Stack("..a", (I32, STR)), Stack(row))
    assert inference.flatten(Stack(row, (F32,))) == Stack("..a", (I32, STR, F32))


def test_two_unknown_rows_become_one() -> None:
    inference = Inference()
    first, second = inference.fresh_row(), inference.fresh_row()
    assert inference.unify_stacks(Stack(first, (I32,)), Stack(second, (I32,)))
    assert inference.flatten(Stack(first)).row == inference.flatten(Stack(second)).row
    assert inference.unify_stacks(Stack(second), Stack("..a", (STR,)))
    assert inference.flatten(Stack(first)) == Stack("..a", (STR,))


def test_a_row_variable_is_only_equal_to_itself() -> None:
    inference = Inference()
    assert inference.unify_stacks(Stack("..a"), Stack("..a"))
    assert not inference.unify_stacks(Stack("..a"), Stack("..b"))
    assert not inference.unify_stacks(Stack("..a", (I32,)), Stack("..a"))
    assert not inference.unify_stacks(Stack(".."), Stack("..", (I32,)))
    assert not inference.bind_row("..a", Stack("..b"))


def test_a_row_cannot_contain_itself() -> None:
    inference = Inference()
    row = inference.fresh_row()
    assert not inference.unify_stacks(Stack(row), Stack(row, (I32,)))
    assert inference.unify_stacks(Stack(row), Stack(row))
    assert inference.row_bindings == {}


def test_quotations_of_the_same_effect_unify() -> None:
    inference = Inference()
    unknown = inference.fresh("a test type", None)
    assert inference.unify(quotation((I32,), (unknown,)), quotation((I32,), (STR,)))
    assert inference.resolve(unknown) == STR
    assert inference.resolve(quotation((), (unknown,))) == quotation((), (STR,))


def test_quotations_of_different_effects_do_not() -> None:
    inference = Inference()
    assert not inference.unify(quotation((I32,), ()), quotation((F32,), ()))
    assert not inference.unify(quotation((I32,), ()), quotation((I32, I32), ()))
    assert not inference.unify(quotation((), ()), I32)


def test_the_qualifiers_must_match() -> None:
    inference = Inference()
    assert not inference.unify(
        quotation((), (), qualifier=LINEAR), quotation((), (), qualifier=UNRESTRICTED)
    )
    qualifier = inference.fresh("a test qualifier", None)
    assert inference.unify(
        quotation((), (), qualifier=qualifier), quotation((), (), qualifier=LINEAR)
    )
    assert inference.resolve(qualifier) == LINEAR


def test_a_quotation_that_names_its_rows() -> None:
    inference = Inference()
    before, after = inference.fresh_row(), inference.fresh_row()
    assert inference.unify(
        quotation((), (), (before, after)), quotation((I32,), (I32,))
    )
    flat_before = inference.flatten(Stack(before))
    flat_after = inference.flatten(Stack(after))
    assert flat_before.types == (I32,)
    assert flat_after.types == (I32,)
    assert flat_before.row == flat_after.row


def test_each_comparison_picks_a_fresh_row() -> None:
    inference = Inference()
    any_depth = quotation((I32,), (I32,))
    assert inference.unify(any_depth, quotation((I32,), (I32,), ("..a", "..a")))
    assert inference.unify(any_depth, quotation((STR, I32), (STR, I32), ("..b", "..b")))
    assert not inference.unify(any_depth, quotation((I32,), (I32,), ("..a", "..b")))
    assert not inference.unify(any_depth, quotation((STR, I32), (STR, I32)))


def test_instantiating_renames_rows() -> None:
    inference = Inference()
    effect = Effect((quotation((), (), ("..a", "..b"), Q),), (), rows=("..a", "..b"))
    first = inference.instantiate("call", effect, None)
    second = inference.instantiate("call", effect, None)
    assert first.rows is not None and second.rows is not None
    assert first.rows[0] != first.rows[1]
    assert set(first.rows).isdisjoint(second.rows)
    inner = first.inputs[0].effect
    assert inner is not None and inner.rows == first.rows


def test_a_quotation_over_one_row_works_at_any_depth() -> None:
    inference = Inference()
    row, other = inference.fresh_row(), inference.fresh_row()
    same = Effect((I32,), (I32,), rows=(row, row))
    assert inference.generalize(same) == Effect((I32,), (I32,))
    different = Effect((I32,), (), rows=(row, other))
    assert inference.generalize(different) == different
    named = Effect((), (), rows=("..a", "..a"))
    assert inference.generalize(named) == named


def test_generalizing_looks_through_bindings() -> None:
    inference = Inference()
    row, below = inference.fresh_row(), inference.fresh_row()
    assert inference.bind_row(row, Stack(below, (I32,)))
    effect = Effect((), (I32, I32), rows=(row, below))
    assert inference.generalize(effect) == Effect((I32,), (I32, I32))
