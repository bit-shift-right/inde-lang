import sys
from dataclasses import replace
from pathlib import Path

import pytest

from inde import __main__ as command
from inde import ir
from inde.ast import Program
from inde.ir import (
    Call,
    CallQuotation,
    Const,
    FuncRef,
    If,
    LiteralMatch,
    LocalOp,
    Make,
    Match,
    Primitive,
    Return,
    Runtime,
    RuntimeDeclaration,
    StackOp,
    Unpack,
    While,
)
from inde.lowering import lower
from inde.parser import parse
from inde.type_model import (
    BOOL,
    CHAR,
    F32,
    I32,
    LINEAR,
    STR,
    STRBUF,
    U8,
    U32,
    UNIT,
    UNRESTRICTED,
    Effect,
    Type,
    array_of,
    option_of,
    quotation_of,
)
from inde.verifier import IRError, verify

SHAPE = Type("Shape")
SHAPES = "type Shape = circle( f32 ) | rect( f32 f32 )\n"


def compiled(source: str) -> ir.Module:
    return lower(parse(source))


def problems(module: ir.Module) -> list[str]:
    with pytest.raises(IRError) as err:
        verify(module)
    return err.value.problems


def with_main(
    body: list[ir.Instruction],
    declarations: str = "",
    slots: list[tuple[str, Type]] | None = None,
    outputs: tuple[Type, ...] = (),
) -> ir.Module:
    module = compiled(declarations + "\nword main ( -> ) end")
    main = module.function("$main")
    main.body = tuple(body)
    main.slots = slots or []
    main.outputs = outputs
    return module


def main_fails(
    body: list[ir.Instruction], slots: list[tuple[str, Type]] | None = None
) -> str:
    found = problems(with_main(body, slots=slots))
    assert len(found) == 1, found
    return found[0].removeprefix("in $main: ")


def const(value_type: Type, value: ir.Value) -> Const:
    return Const(type=value_type, value=value)


def drop(value_type: Type) -> StackOp:
    return StackOp(op="drop", type=value_type)


def imported(
    module: ir.Module,
    call: Runtime,
    inputs: tuple[Type, ...],
    outputs: tuple[Type, ...],
) -> ir.Module:
    module.runtime.append(RuntimeDeclaration(call.full_name, inputs, outputs))
    return module


VALID_PROGRAMS = [
    SHAPES
    + "word area ( Shape -> f32 ) match case circle dup * case rect * end end\n"
    + "word main ( -> ) 2.0 circle area print end",
    "word main ( -> Result[unit, str] ) unit ok end",
    (
        "type List[T] = empty | cons( T List[T] )\n"
        "word main ( -> ) empty 1 swap cons empty 1 swap cons == print end"
    ),
    (
        "word twice ( T -> T T ) where T: Unrestricted dup end\n"
        'word main ( -> ) "a" twice str-concat print 2 twice + print end'
    ),
    (
        "word dip ( ..a T Q -> ..b T ) where Q: [ ..a -> ..b ]\n"
        "  swap [ ] curry compose call\n"
        "end\n"
        "word main ( -> ) 1 2 3 [ + ] dip + print end"
    ),
    (
        "word half ( i32 -> Option[i32] )\n"
        "  dup 2 % 0 == if 2 / some else drop none end\n"
        "end\n"
        "word quarter ( i32 -> Option[i32] ) let n in n half ? half end end\n"
        "word main ( -> ) 8 quarter drop end"
    ),
    (
        "word count ( i32 i32 -> i32 )\n"
        "  let total n in n 0 == if total else total n + n 1 - count end end\n"
        "end\n"
        "word main ( -> ) 0 10 count print end"
    ),
    (
        "word main ( -> )\n"
        "  0 { 1 2 3 } arraybuf-from-array [ + ] arraybuf-consume print\n"
        "  strbuf-new 'a' strbuf-push [ strbuf-freeze ] curry call print\n"
        "  { { 1 } { } } { { 1 } { } } != print\n"
        "  'x' match case 'x' 1 else 0 end print\n"
        "  true match case true 1 case false 0 end print\n"
        "  0 while dup 3 < do 1 + end print\n"
        "end"
    ),
]


@pytest.mark.parametrize("source", VALID_PROGRAMS)
def test_compiled_programs_are_valid(source: str) -> None:
    verify(compiled(source))


def test_operand_types() -> None:
    body = [const(F32, 1.0), const(F32, 2.0), Primitive(type=I32, op="add"), drop(I32)]
    assert main_fails(body) == "`i32.add`: needs (i32 i32), found (f32 f32)"


def test_a_function_must_leave_its_results() -> None:
    assert main_fails([const(I32, 1)]) == (
        "the function ends with (i32), but its results are ()"
    )
    module = compiled("word one ( -> i32 ) 1 end\nword main ( -> ) end")
    module.function("$one").body = ()
    assert problems(module) == [
        "in $one: the function ends with (), but its results are (i32)"
    ]


def test_an_instruction_needs_its_operands() -> None:
    assert main_fails([drop(I32)]) == (
        "`drop.i32`: needs (i32), but the stack holds only ()"
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            [Runtime(name="strbuf_new"), StackOp(op="dup", type=STRBUF)],
            "`dup.strbuf`: dup needs an unrestricted type, but strbuf is linear",
        ),
        (
            [Runtime(name="strbuf_new"), drop(STRBUF)],
            "`drop.strbuf`: drop needs an unrestricted type, but strbuf is linear",
        ),
    ],
)
def test_linear_values_are_never_copied_or_discarded(
    body: list[ir.Instruction], message: str
) -> None:
    module = with_main(body)
    imported(module, Runtime(name="strbuf_new"), (), (STRBUF,))
    assert problems(module) == [f"in $main: {message}"]


def test_a_linear_value_cannot_be_left_behind() -> None:
    module = imported(
        with_main([Runtime(name="strbuf_new")]),
        Runtime(name="strbuf_new"),
        (),
        (STRBUF,),
    )
    assert problems(module) == [
        "in $main: the function ends with (strbuf), but its results are ()"
    ]


def test_stack_words() -> None:
    body = [
        const(I32, 1),
        const(BOOL, True),
        StackOp(op="over", type=BOOL),
        drop(I32),
        drop(BOOL),
        drop(I32),
    ]
    assert main_fails(body) == "`over.bool`: copies the second value, a i32"
    assert main_fails([StackOp(op="swap", type=I32)]) == (
        "`swap.i32`: dup, over and drop name a type, and swap and rot do not"
    )


def test_code_after_return_can_never_run() -> None:
    assert main_fails([Return(), const(I32, 1)]) == "`i32.const 1` can never run"


@pytest.mark.parametrize(
    ("value_type", "value", "message"),
    [
        (U8, 300, "300 is not a u8"),
        (I32, True, "True is not a i32"),
        (F32, 0.1, "0.1 is not exactly an f32"),
        (CHAR, "\ud800", "'\\ud800' is not a Unicode scalar value"),
        (STR, "x", "there are no str constants"),
    ],
)
def test_constants(value_type: Type, value: ir.Value, message: str) -> None:
    found = main_fails([const(value_type, value), drop(value_type)])
    assert found.endswith(f": {message}")


def test_static_data() -> None:
    module = compiled('word main ( -> ) { 1 2 } drop "a" drop end')
    module.data[0] = replace(module.data[0], value=(1, "two"))
    assert problems(module) == ["data $a0: 'two' is not a i32"]
    module = compiled('word main ( -> ) "a" drop end')
    module.function("$main").body = (ir.DataRef(kind="array", name="$s0"),)
    assert problems(module) == ["in $main: `data.array $s0`: $s0 is str data"]


@pytest.mark.parametrize(
    ("instruction", "message"),
    [
        (Primitive(type=U8, op="neg"), "there is no u8.neg"),
        (Primitive(type=F32, op="add", modifier="wrap"), "f32.add has no .wrap form"),
        (Primitive(type=I32, op="shl", modifier="sat"), "i32.shl has no .sat form"),
        (Primitive(type=F32, op="shl"), "there is no f32.shl"),
        (Primitive(type=I32, op="to_char"), "there is no conversion from i32 to char"),
        (
            Primitive(type=U32, op="to_char", modifier="wrap"),
            "u32.to_char has no .wrap form",
        ),
        (Primitive(type=I32, op="to_bits"), "there is no i32.to_bits"),
        (Primitive(type=STR, op="eq"), "there is no str.eq"),
        (
            Primitive(type=STRBUF, op="select"),
            "select needs an unrestricted type, but strbuf is linear",
        ),
    ],
)
def test_primitive_forms(instruction: Primitive, message: str) -> None:
    assert main_fails([instruction]) == f"`{instruction.text()}`: {message}"


def test_a_checked_result_needs_its_option_type() -> None:
    body = [
        const(I32, 1),
        const(I32, 2),
        Primitive(type=I32, op="add", modifier="checked"),
        drop(option_of(I32)),
    ]
    assert main_fails(body) == (
        "`i32.add.checked`: %Option<i32> is not in the type table"
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            [LocalOp(op="get", slot="$x")],
            "`local.get $x`: the slot is empty",
        ),
        (
            [
                const(I32, 1),
                LocalOp(op="set", slot="$x"),
                const(I32, 2),
                LocalOp(op="set", slot="$x"),
            ],
            "`local.set $x`: the slot is already full",
        ),
        (
            [const(I32, 1), LocalOp(op="set", slot="$x")],
            "$x must be empty at the end of the function",
        ),
        (
            [const(I32, 1), LocalOp(op="set", slot="$y")],
            "`local.set $y`: there is no slot $y",
        ),
    ],
)
def test_slots(body: list[ir.Instruction], message: str) -> None:
    assert main_fails(body, slots=[("$x", I32)]) == message


def test_linear_slots_can_only_be_set_and_taken() -> None:
    module = with_main(
        [
            Runtime(name="strbuf_new"),
            LocalOp(op="set", slot="$b"),
            LocalOp(op="get", slot="$b"),
        ],
        slots=[("$b", STRBUF)],
    )
    imported(module, Runtime(name="strbuf_new"), (), (STRBUF,))
    assert problems(module) == [
        (
            "in $main: `local.get $b`: local.get needs an unrestricted type, "
            "but strbuf "
            "is linear"
        )
    ]


def test_a_slot_must_be_empty_at_a_return() -> None:
    body = [const(I32, 1), LocalOp(op="set", slot="$x"), Return()]
    assert (
        main_fails(body, slots=[("$x", I32)])
        == "`return`: $x must be empty at a return"
    )


def test_a_block_cannot_reach_below_its_values() -> None:
    body = [
        const(I32, 1),
        const(BOOL, True),
        If(inputs=(), outputs=(), then_body=(drop(I32), const(I32, 2))),
        drop(I32),
    ]
    assert main_fails(body) == "`drop.i32`: needs (i32), but its block has only ()"


def test_branches_must_agree() -> None:
    body = [
        const(BOOL, True),
        If(
            inputs=(),
            outputs=(I32,),
            then_body=(const(I32, 1),),
            else_body=(const(F32, 1.0),),
        ),
        drop(I32),
    ]
    assert main_fails(body) == (
        "`if [-> i32]`: a branch leaves (f32), but the block leaves (i32)"
    )
    body = [
        const(BOOL, True),
        If(inputs=(), outputs=(I32,), then_body=(const(I32, 1),)),
        drop(I32),
    ]
    assert (
        main_fails(body)
        == "`if [-> i32]`: without else, an if must leave what it takes"
    )


def test_branches_must_leave_the_slots_alike() -> None:
    body = [
        const(I32, 1),
        LocalOp(op="set", slot="$x"),
        const(BOOL, True),
        If(
            inputs=(),
            outputs=(),
            then_body=(LocalOp(op="drop", slot="$x"),),
            else_body=(),
        ),
    ]
    assert main_fails(body, slots=[("$x", I32)]) == (
        "`if [->]`: its branches leave the slots in different states"
    )


def test_a_branch_that_returns_imposes_nothing() -> None:
    body = [
        const(I32, 1),
        LocalOp(op="set", slot="$x"),
        const(BOOL, True),
        If(
            inputs=(),
            outputs=(),
            then_body=(LocalOp(op="drop", slot="$x"), Return()),
            else_body=(),
        ),
        LocalOp(op="drop", slot="$x"),
    ]
    verify(with_main(body, slots=[("$x", I32)]))


def test_while() -> None:
    found = main_fails(
        [
            const(I32, 0),
            While(
                types=(I32,),
                condition=(
                    StackOp(op="dup", type=I32),
                    const(I32, 3),
                    Primitive(type=I32, op="lt"),
                ),
                body=(drop(I32),),
            ),
            drop(I32),
        ]
    )
    assert found == "`while [i32]`: the body must leave (i32), but it leaves ()"
    found = main_fails(
        [
            const(I32, 0),
            While(types=(I32,), condition=(), body=()),
            drop(I32),
        ]
    )
    assert found == (
        "`while [i32]`: the condition must leave (i32 bool), but it leaves (i32)"
    )


def test_constructor_matches() -> None:
    module = compiled(SHAPES + "word main ( -> ) 1.0 circle drop end")
    match = Match(
        type=SHAPE,
        inputs=(),
        outputs=(),
        arms=(("circle", (drop(F32),)),),
    )
    module.function("$main").body = (
        const(F32, 1.0),
        Make(type=SHAPE, variant="circle"),
        match,
    )
    assert problems(module) == [
        (
            "in $main: `match %Shape [->]`: its arms are circle, but every variant "
            "must appear exactly once: circle, rect"
        )
    ]
    module.function("$main").body = (
        const(F32, 1.0),
        Make(type=SHAPE, variant="square"),
    )
    assert problems(module) == [
        "in $main: `make %Shape.square`: %Shape has no variant square"
    ]
    module.function("$main").body = (
        const(F32, 1.0),
        Make(type=SHAPE, variant="circle"),
        Unpack(type=SHAPE, variant="circle"),
        drop(F32),
    )
    assert problems(module) == [
        "in $main: `unpack %Shape.circle`: only a type with one variant can be unpacked"
    ]


@pytest.mark.parametrize(
    ("matched", "arms", "else_body", "message"),
    [
        (I32, ((1, ()), (1, ())), (), "a value has two arms"),
        (I32, ((1, ()),), None, "it needs an else arm for the values without one"),
        (
            BOOL,
            ((True, ()), (False, ())),
            (),
            "both booleans have an arm, so else can never run",
        ),
        (F32, ((1.0, ()),), (), "a f32 cannot be matched against literals"),
    ],
)
def test_literal_matches(
    matched: Type,
    arms: tuple[tuple[ir.Value, ir.Body], ...],
    else_body: ir.Body | None,
    message: str,
) -> None:
    value: ir.Value = True if matched == BOOL else 1.0 if matched == F32 else 1
    match = LiteralMatch(
        type=matched, inputs=(), outputs=(), arms=arms, else_body=else_body
    )
    found = main_fails([const(matched, value), match])
    assert found == f"`{match.text()}`: {message}"


def test_calls() -> None:
    assert main_fails([Call(function="$nowhere")]) == (
        "`call $nowhere`: there is no function $nowhere"
    )
    module = compiled("word f ( i32 -> i32 ) 1 + end\nword main ( -> ) 1 f drop end")
    module.function("$main").body = (const(F32, 1.0), Call(function="$f"), drop(I32))
    assert problems(module) == ["in $main: `call $f`: needs (i32), found (f32)"]


def test_tail_calls() -> None:
    module = compiled(
        "word f ( i32 -> i32 ) 1 + end\n"
        "word g ( str i32 -> str i32 ) f end\n"
        "word main ( -> ) end"
    )
    verify(module)
    module.function("$g").outputs = (I32,)
    assert problems(module) == [
        (
            "in $g: `return_call $f (at 2:31)`: a tail call leaves (str i32), but the "
            "results are (i32)"
        )
    ]


def test_quotation_calls() -> None:
    effect = quotation_of(Effect((I32,), (I32,)), UNRESTRICTED)
    module = compiled("word main ( -> ) 1 [ 1 + ] call drop end")
    verify(module)
    main = module.function("$main")
    main.body = (
        const(I32, 1),
        FuncRef(function="$main.q0"),
        CallQuotation(inputs=(I32,), outputs=(I32, I32)),
        drop(I32),
    )
    assert problems(module) == [
        (
            "in $main: `call.quot [i32 -> i32 i32]`: needs a quotation of this "
            "effect, "
            f"found {ir.ir_type(effect)}"
        )
    ]


@pytest.mark.parametrize(
    ("call", "inputs", "outputs", "message"),
    [
        (
            Runtime(name="str_len"),
            (STR,),
            (I32,),
            (
                "it is imported as (str) -> (i32), but the runtime interface gives "
                "(str) -> (u32)"
            ),
        ),
        (
            Runtime(name="str_reverse"),
            (STR,),
            (STR,),
            "rt.str_reverse is not in the runtime interface",
        ),
        (
            Runtime(name="print", arguments=(SHAPE,)),
            (SHAPE,),
            (),
            "%Shape cannot be displayed",
        ),
        (
            Runtime(name="array_len", arguments=(STRBUF,)),
            (array_of(STRBUF),),
            (U32,),
            (
                "array<strbuf> holds linear elements, but the elements of an array "
                "must "
                "be unrestricted"
            ),
        ),
        (
            Runtime(name="arraybuf_get", arguments=(STRBUF,)),
            (Type("arraybuf", (STRBUF,)), U32),
            (Type("arraybuf", (STRBUF,)), STRBUF),
            "rt.arraybuf_get<strbuf> needs an unrestricted type, but strbuf is linear",
        ),
    ],
)
def test_runtime_calls(
    call: Runtime, inputs: tuple[Type, ...], outputs: tuple[Type, ...], message: str
) -> None:
    module = with_main([call], declarations=SHAPES)
    imported(module, call, inputs, outputs)
    found = problems(module)
    assert any(problem.endswith(message) for problem in found), found


def test_a_runtime_call_must_be_imported() -> None:
    assert main_fails([Runtime(name="strbuf_new")]) == (
        "`rt.strbuf_new`: this runtime call is not imported"
    )


@pytest.mark.parametrize(
    ("value", "quotation", "result"),
    [
        (I32, ([I32, I32], [I32]), ([I32], [I32])),
        (I32, ([], [STR]), ([], [I32, STR])),
        (STRBUF, ([STRBUF], [STR]), None),
    ],
)
def test_curry_signatures(
    value: Type,
    quotation: tuple[list[Type], list[Type]],
    result: tuple[list[Type], list[Type]] | None,
) -> None:
    given = quotation_of(Effect(tuple(quotation[0]), tuple(quotation[1])), UNRESTRICTED)
    call = Runtime(name="curry", arguments=(value, given))
    if result is None:
        expected = quotation_of(Effect((), (STR,)), LINEAR)
    else:
        expected = quotation_of(
            Effect(tuple(result[0]), tuple(result[1])), UNRESTRICTED
        )
    module = with_main([])
    imported(module, call, (value, given), (expected,))
    verify(module)


def test_curry_needs_the_value_its_quotation_takes() -> None:
    given = quotation_of(Effect((STR,), ()), UNRESTRICTED)
    call = Runtime(name="curry", arguments=(I32, given))
    module = with_main([const(I32, 1), FuncRef(function="$main"), call])
    imported(module, call, (I32, given), (given,))
    assert any(
        "gives the quotation a i32, but it takes (str)" in problem
        for problem in problems(module)
    )


def test_compose_signatures() -> None:
    first = quotation_of(Effect((I32,), (I32,)), UNRESTRICTED)
    second = quotation_of(Effect((), (I32,)), UNRESTRICTED)
    composed = quotation_of(Effect((I32,), (I32, I32)), UNRESTRICTED)
    call = Runtime(name="compose", arguments=(first, second))
    verify(imported(with_main([]), call, (first, second), (composed,)))
    wider = quotation_of(Effect((STR, I32), (I32,)), UNRESTRICTED)
    call = Runtime(name="compose", arguments=(first, wider))
    expected = quotation_of(Effect((STR, I32), (I32,)), UNRESTRICTED)
    verify(imported(with_main([]), call, (first, wider), (expected,)))


def test_consume_signature() -> None:
    call = Runtime(name="arraybuf_consume", arguments=(I32, (STR,)))
    buffer = Type("arraybuf", (I32,))
    quotation = quotation_of(Effect((STR, I32), (STR,)), UNRESTRICTED)
    verify(imported(with_main([]), call, (STR, buffer, quotation), (STR,)))
    module = imported(with_main([]), call, (buffer, quotation), ())
    module.function("$main").body = (call,)
    assert any("runtime interface gives" in problem for problem in problems(module))


def test_the_type_table() -> None:
    module = compiled(SHAPES + "word main ( -> ) 1.0 circle drop end")
    module.types.clear()
    assert problems(module)[0] == (
        "in $main: `make %Shape.circle (at 2:22)`: %Shape is not in the type table"
    )


def test_a_qualifier_is_checked_against_the_fields() -> None:
    holder = Type("Holder")
    module = compiled(
        "type Holder = holder( strbuf )\n"
        "word main ( -> ) strbuf-new holder holder-unpack strbuf-freeze drop end"
    )
    verify(module)
    module.types[0] = replace(module.types[0], linear=False)
    assert problems(module)[0] == (
        "type %Holder: it holds a linear value, so it must be linear"
    )
    assert holder == module.types[0].type


def test_core_types_have_their_shape() -> None:
    module = compiled("word main ( -> ) 1 some drop end")
    module.types[0] = replace(module.types[0], variants=(("some", (I32,)),))
    assert problems(module)[0] == (
        "type %Option<i32>: it does not have the variants of the core Option"
    )


def test_a_core_type_is_linear_only_when_it_must_be() -> None:
    module = compiled("word main ( -> ) 1 some drop end")
    module.types[0] = replace(module.types[0], linear=True)
    assert problems(module)[0] == (
        "type %Option<i32>: the core Option is linear only when it holds a linear value"
    )
    module = compiled(
        "linear type Token = token\nword main ( -> ) token match case token end end"
    )
    verify(module)


def test_the_entry_point() -> None:
    module = compiled("word main ( -> ) end")
    module.entry = ("$start", "unit")
    assert problems(module) == ["entry: there is no function $start"]
    module.entry = ("$main", "result")
    assert problems(module) == [
        (
            "entry: $main must leave a %Result<unit, E> with a displayable E for the "
            "result form"
        )
    ]


def test_the_runtime_version() -> None:
    module = compiled("word main ( -> ) end")
    module.runtime_version = 4
    assert problems(module) == [
        "the module needs runtime 4, but this implementation provides runtime 3"
    ]


def test_names_are_unique() -> None:
    module = compiled("word main ( -> ) end")
    module.functions.append(module.functions[0])
    assert "function $main is defined twice" in problems(module)


def test_every_function_is_checked() -> None:
    module = compiled("word f ( -> ) end\nword g ( -> ) end\nword main ( -> ) end")
    module.function("$f").body = (const(I32, 1),)
    module.function("$g").body = (const(I32, 1),)
    assert problems(module) == [
        "in $f: the function ends with (i32), but its results are ()",
        "in $g: the function ends with (i32), but its results are ()",
    ]


def test_the_command_line_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    program = tmp_path / "p.inde"
    program.write_text("word main ( -> ) 1 drop end", encoding="utf-8")

    def broken(parsed: Program) -> ir.Module:
        module = lower(parsed)
        module.function("$main").body = ()
        module.function("$main").outputs = (UNIT,)
        return module

    monkeypatch.setattr(command, "lower", broken)
    monkeypatch.setattr(sys, "argv", ["inde", "--ir", str(program)])
    assert command.main() == 3
    err = capsys.readouterr().err
    assert "internal error: the compiled IR is invalid" in err
    assert "in $main: the function ends with (), but its results are (unit)" in err
