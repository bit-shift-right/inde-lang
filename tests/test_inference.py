from inde.inference import Inference
from inde.type_model import BOOL, F32, I32, STR, Effect, Type, array_of, option_of

T = Type("T")
U = Type("U")


def test_types_display() -> None:
    assert str(Type("?1")) == "_"
    assert str(option_of(Type("?2"))) == "Option[_]"
    assert str(Effect((T, T), (T,), (("T", "Num"),))) == "( T T -> T ) where T: Num"
    assert str(
        Effect((T, U), (), (("T", "Eq"), ("T", "Unrestricted"), ("U", "Ord")))
    ) == ("( T U -> ) where T: Eq & Unrestricted, U: Ord")
    assert Effect((option_of(T), U, T), (U,)).variables() == ("T", "U")


def test_equal_types_unify() -> None:
    inference = Inference()
    assert inference.unify(I32, I32)
    assert inference.unify(option_of(I32), option_of(I32))
    assert inference.unify(T, T)
    assert inference.bindings == {}


def test_an_unknown_takes_the_other_type() -> None:
    inference = Inference()
    unknown = inference.fresh("a test type", None)
    assert inference.unify(unknown, I32)
    assert inference.resolve(unknown) == I32
    assert inference.unify(unknown, I32)
    assert not inference.unify(unknown, F32)


def test_an_unknown_on_either_side() -> None:
    inference = Inference()
    first = inference.fresh("a", None)
    second = inference.fresh("b", None)
    assert inference.unify(STR, first)
    assert inference.unify(first, second)
    assert inference.resolve(second) == STR


def test_two_unknowns_become_one() -> None:
    inference = Inference()
    first = inference.fresh("a", None)
    second = inference.fresh("b", None)
    assert inference.unify(first, second)
    assert inference.has_unknowns(first)
    assert inference.unify(second, BOOL)
    assert inference.resolve(first) == BOOL
    assert not inference.has_unknowns(first)


def test_different_types_do_not_unify() -> None:
    inference = Inference()
    assert not inference.unify(I32, F32)
    assert not inference.unify(T, U)
    assert not inference.unify(T, I32)


def test_resolve_looks_inside_types() -> None:
    inference = Inference()
    unknown = inference.fresh("a", None)
    wrapped = Type("Result", (option_of(unknown), STR))
    assert inference.has_unknowns(wrapped)
    assert inference.unify(unknown, I32)
    assert inference.resolve(wrapped) == Type("Result", (option_of(I32), STR))
    assert not inference.has_unknowns(wrapped)


def test_instantiate_gives_each_call_its_own_unknowns() -> None:
    inference = Inference()
    declared = Effect((T, T), (option_of(T),), (("T", "Num"),))
    first = inference.instantiate("+.checked", declared, None)
    second = inference.instantiate("+.checked", declared, None)
    assert first.inputs[0] == first.inputs[1]
    assert first.inputs[0] != second.inputs[0]
    assert first.outputs == (option_of(first.inputs[0]),)
    assert first.where == ((first.inputs[0].name, "Num"),)
    assert inference.unify(first.inputs[0], I32)
    assert inference.resolve(first.outputs[0]) == option_of(I32)
    assert inference.has_unknowns(second.outputs[0])


def test_unify_all() -> None:
    inference = Inference()
    unknown = inference.fresh("a", None)
    assert inference.unify_all([I32, unknown], [I32, STR])
    assert inference.resolve(unknown) == STR
    assert not inference.unify_all([I32], [I32, I32])
    assert not inference.unify_all([I32, STR], [I32, BOOL])
    assert inference.unify_all([], [])


def test_unify_looks_inside_types() -> None:
    inference = Inference()
    unknown = inference.fresh("a", None)
    assert inference.unify(option_of(unknown), option_of(I32))
    assert inference.resolve(unknown) == I32


def test_unify_several_arguments() -> None:
    inference = Inference()
    value = inference.fresh("a", None)
    error = inference.fresh("b", None)
    assert inference.unify(
        Type("Result", (value, STR)), Type("Result", (array_of(I32), error))
    )
    assert inference.resolve(value) == array_of(I32)
    assert inference.resolve(error) == STR


def test_unify_nested() -> None:
    inference = Inference()
    unknown = inference.fresh("a", None)
    assert inference.unify(
        option_of(array_of(option_of(unknown))), option_of(array_of(option_of(BOOL)))
    )
    assert inference.resolve(unknown) == BOOL


def test_one_unknown_in_two_places() -> None:
    inference = Inference()
    unknown = inference.fresh("a", None)
    pair = Type("Pair", (unknown, unknown))
    assert inference.unify(pair, Type("Pair", (I32, I32)))
    assert not Inference().unify(
        Type("Pair", (Type("?9"), Type("?9"))), Type("Pair", (I32, STR))
    )


def test_shapes_that_differ() -> None:
    inference = Inference()
    unknown = inference.fresh("a", None)
    assert not inference.unify(option_of(unknown), array_of(I32))
    assert not inference.unify(option_of(I32), option_of(F32))
    assert not inference.unify(option_of(I32), I32)
    assert inference.unify(option_of(T), option_of(T))
    assert not inference.unify(option_of(T), option_of(I32))


def test_occurs_check() -> None:
    inference = Inference()
    unknown = inference.fresh("a", None)
    assert not inference.unify(unknown, option_of(unknown))
    assert not inference.unify(array_of(option_of(unknown)), array_of(unknown))
    assert inference.bindings == {}
    other = inference.fresh("b", None)
    assert inference.unify(other, option_of(unknown))
    assert not inference.unify(unknown, array_of(other))
