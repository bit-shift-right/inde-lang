import io
import sys
from pathlib import Path

import pytest

from inde import __main__ as command
from inde.errors import IndeErrors
from inde.ir import Module, format_module
from inde.lowering import lower
from inde.parser import parse


def compile_source(source: str) -> Module:
    return lower(parse(source))


def module_text(source: str) -> str:
    return format_module(compile_source(source))


def function_text(source: str, name: str) -> str:
    lines = module_text(source).splitlines()
    start = lines.index(
        next(line for line in lines if line.startswith(f"  func {name} "))
    )
    end = lines.index("  end", start)
    return "\n".join(line[2:] for line in lines[start : end + 1])


def body_of(source: str, name: str = "$main") -> list[str]:
    lines = function_text(source, name).splitlines()[1:-1]
    return [line.strip() for line in lines if not line.strip().startswith("local ")]


def main_body(body: str, declarations: str = "") -> list[str]:
    return body_of(f"{declarations}\nword main ( -> ) {body} end")


def fails(source: str) -> str:
    with pytest.raises(IndeErrors) as err:
        compile_source(source)
    return err.value.errors[0].message


SHAPES = """\
type Shape = circle( f32 ) | rect( f32 f32 )

word area ( Shape -> f32 )
  match
    case circle dup * 3.14159 *
    case rect *
  end
end

word main ( -> )
  "area: " print
  2.0 circle area print
end
"""


def test_a_whole_module() -> None:
    assert module_text(SHAPES) == (
        "module\n"
        "  runtime 1\n"
        "\n"
        "  type %Shape = variant { circle(f32) rect(f32 f32) }\n"
        "\n"
        '  data $s0 = utf8 "area: "\n'
        "\n"
        "  import rt.print<str> (str) -> ()\n"
        "  import rt.print<f32> (f32) -> ()\n"
        "\n"
        "  func $area (%Shape) -> (f32)\n"
        "    match %Shape [-> f32]\n"
        "      case circle\n"
        "        dup.f32\n"
        "        f32.mul\n"
        "        f32.const 3.14159\n"
        "        f32.mul\n"
        "      case rect\n"
        "        f32.mul\n"
        "    end\n"
        "  end\n"
        "\n"
        "  func $main () -> ()\n"
        "    data.str $s0\n"
        "    rt.print<str>\n"
        "    f32.const 2.0\n"
        "    make %Shape.circle\n"
        "    call $area\n"
        "    rt.print<f32>\n"
        "  end\n"
        "\n"
        "  entry $main unit\n"
        "end\n"
    )


def test_debug_information() -> None:
    text = format_module(compile_source(SHAPES), debug=True)
    assert "    call $area  ; 12:14 area\n" in text
    assert "        f32.mul  ; 6:15 *\n" in text


def test_constants() -> None:
    assert main_body(
        "5 drop 2u8 drop -1i64 drop 0.1 drop -0.0f64 drop 'a' drop true drop "
        "unit drop 18446744073709551615u64 drop"
    ) == [
        "i32.const 5",
        "drop.i32",
        "u8.const 2",
        "drop.u8",
        "i64.const -1",
        "drop.i64",
        "f32.const 0.1",
        "drop.f32",
        "f64.const -0.0",
        "drop.f64",
        "char.const U+0061",
        "drop.char",
        "bool.const true",
        "drop.bool",
        "unit.const",
        "drop.unit",
        "u64.const 18446744073709551615",
        "drop.u64",
    ]


def test_named_constants_are_their_values() -> None:
    declarations = 'const LIMIT = 10u8\nconst GREETING = "hi"\n'
    assert main_body("LIMIT drop GREETING drop", declarations) == [
        "u8.const 10",
        "drop.u8",
        "data.str $s0",
        "drop.str",
    ]


def test_static_data() -> None:
    text = module_text(
        "word main ( -> )\n"
        '  "a" "a" str-concat "b\\n" str-concat drop\n'
        "  { 1 2 } drop { { 'x' } { } } drop { } { 3 } == drop\n"
        '  { "p" "q" } drop { 1 2 } drop\n'
        "end"
    )
    assert (
        '  data $s0 = utf8 "a"\n'
        '  data $s1 = utf8 "b\\n"\n'
        "  data $a0 = array<i32> [1 2]\n"
        "  data $a1 = array<array<char>> [[U+0078] []]\n"
        "  data $a2 = array<i32> []\n"
        "  data $a3 = array<i32> [3]\n"
        '  data $a4 = array<str> ["p" "q"]\n'
    ) in text
    assert text.count("[1 2]") == 1


def test_stack_words_name_the_types_they_copy_or_discard() -> None:
    assert main_body('1 "s" over swap rot dup drop drop drop drop') == [
        "i32.const 1",
        "data.str $s0",
        "over.i32",
        "swap",
        "rot",
        "dup.i32",
        "drop.i32",
        "drop.i32",
        "drop.str",
        "drop.i32",
    ]


@pytest.mark.parametrize(
    ("body", "instruction"),
    [
        ("1 2 +", "i32.add"),
        ("1u8 2u8 -.wrap", "u8.sub.wrap"),
        ("1i64 2i64 *.sat", "i64.mul.sat"),
        ("7 2 /.checked", "i32.div.checked"),
        ("7.5 2.0 %", "f32.rem"),
        ("1.0f64 0.0f64 /.ieee", "f64.div.ieee"),
        ("-3 abs.checked", "i32.abs.checked"),
        ("2.0 sqrt", "f32.sqrt"),
        ("1.0 2.0 3.0 fma.checked", "f32.fma.checked"),
        ("1.5 round", "f32.round"),
        ("1.0 -2.0 copysign", "f32.copysign"),
        ("5u32 3u32 bit-and", "u32.bit_and"),
        ("5 bit-not", "i32.bit_not"),
        ("1 3u32 shl.wrap", "i32.shl.wrap"),
        ("1u64 3u32 rotr", "u64.rotr"),
        ("12u16 count-ones", "u16.count_ones"),
        ("1 2 <", "i32.lt"),
        ("1.0 2.0 >=", "f32.ge"),
        ("'a' 'b' !=", "char.ne"),
        ("true false <", "bool.lt"),
        ("true false xor", "bool.xor"),
        ("true not", "bool.not"),
        ("true 1 2 select", "select.i32"),
        ("'a' char-utf8-len", "char.utf8_len"),
        ("300 to-u8.sat", "i32.to_u8.sat"),
        ("1 to-f64", "i32.to_f64"),
        ("1.5f64 to-f32.ieee", "f64.to_f32.ieee"),
        ("97u32 to-char.checked", "u32.to_char.checked"),
        ("'a' to-u32", "char.to_u32"),
        ("1 to-i32", "i32.to_i32"),
        ("1.0 to-bits", "f32.to_bits"),
        ("1u64 from-bits", "u64.from_bits_f64"),
    ],
)
def test_primitive_instructions(body: str, instruction: str) -> None:
    lowered = main_body(f"{body} drop")
    assert lowered[-2] == instruction


@pytest.mark.parametrize(
    ("body", "call", "declaration"),
    [
        ('"a" str-len', "rt.str_len", "(str) -> (u32)"),
        ('"a" "b" str-concat', "rt.str_concat", "(str str) -> (str)"),
        ('"a" 0u32 1u32 str-slice', "rt.str_slice", "(str u32 u32) -> (%Option<str>)"),
        ('"a" "b" <', "rt.str_lt", "(str str) -> (bool)"),
        ("'a' str-from-char", "rt.str_from_char", "(char) -> (str)"),
        ("{ 1 } array-len", "rt.array_len<i32>", "(array<i32>) -> (u32)"),
        (
            "{ 1 } 0u32 array-get.checked",
            "rt.array_get.checked<i32>",
            "(array<i32> u32) -> (%Option<i32>)",
        ),
        (
            "arraybuf-new 1u8 arraybuf-push arraybuf-freeze",
            "rt.arraybuf_new<u8>",
            "() -> (arraybuf<u8>)",
        ),
        (
            "strbuf-new 'a' strbuf-push strbuf-freeze",
            "rt.strbuf_push",
            "(strbuf char) -> (strbuf)",
        ),
    ],
)
def test_runtime_calls(body: str, call: str, declaration: str) -> None:
    source = f"word main ( -> ) {body} drop end"
    assert call in body_of(source)
    assert f"  import {call} {declaration}\n" in module_text(source)


def test_print_is_a_runtime_call_per_type() -> None:
    assert main_body('1 print "a" print 1.5f64 print unit print') == [
        "i32.const 1",
        "rt.print<i32>",
        "data.str $s0",
        "rt.print<str>",
        "f64.const 1.5",
        "rt.print<f64>",
        "unit.const",
        "rt.print<unit>",
    ]


def test_unit_equality_needs_no_comparison() -> None:
    assert main_body("unit unit == drop") == [
        "unit.const",
        "unit.const",
        "drop.unit",
        "drop.unit",
        "bool.const true",
        "drop.bool",
    ]


def test_derived_equality_is_compiled() -> None:
    source = (
        "type Point = point( i32 str )\n"
        "word main ( -> )\n"
        '  1 "a" point 1 "a" point != drop\n'
        "  { 1 } { 1 } == drop\n"
        "end"
    )
    assert body_of(source) == [
        "i32.const 1",
        "data.str $s0",
        "make %Point.point",
        "i32.const 1",
        "data.str $s0",
        "make %Point.point",
        "call $==<%Point>",
        "bool.not",
        "drop.bool",
        "data.array $a0",
        "data.array $a0",
        "call $==<array<i32>>",
        "drop.bool",
    ]
    assert function_text(source, "$==<%Point>") == (
        "func $==<%Point> (%Point %Point) -> (bool)\n"
        "  local $b %Point\n"
        "  local $a.point.0 i32\n"
        "  local $a.point.1 str\n"
        "  local $b.point.0 i32\n"
        "  local $b.point.1 str\n"
        "  local.set $b\n"
        "  match %Point [-> bool]\n"
        "    case point\n"
        "      local.set $a.point.1\n"
        "      local.set $a.point.0\n"
        "      local.take $b\n"
        "      match %Point [-> bool]\n"
        "        case point\n"
        "          local.set $b.point.1\n"
        "          local.set $b.point.0\n"
        "          local.take $a.point.0\n"
        "          local.take $b.point.0\n"
        "          i32.eq\n"
        "          local.take $a.point.1\n"
        "          local.take $b.point.1\n"
        "          rt.str_eq\n"
        "          bool.and\n"
        "      end\n"
        "  end\n"
        "end"
    )
    assert "func $==<array<i32>> (array<i32> array<i32>) -> (bool)" in module_text(
        source
    )


def test_unequal_variants_drop_their_fields() -> None:
    source = "word main ( -> ) none 5 some == drop end"
    lowered = body_of(source, "$==<%Option<i32>>")
    assert lowered[:10] == [
        "local.set $b",
        "match %Option<i32> [-> bool]",
        "case none",
        "local.take $b",
        "match %Option<i32> [-> bool]",
        "case none",
        "bool.const true",
        "case some",
        "drop.i32",
        "bool.const false",
    ]

    assert lowered[12:18] == [
        "local.set $a.some.0",
        "local.take $b",
        "match %Option<i32> [-> bool]",
        "case none",
        "local.drop $a.some.0",
        "bool.const false",
    ]


def test_constructors_and_unpacking() -> None:
    declarations = "type Pair2 = pair2( i32 str )\n"
    assert main_body(
        '1 "a" pair2 pair2-unpack drop drop 5 some drop', declarations
    ) == [
        "i32.const 1",
        "data.str $s0",
        "make %Pair2.pair2",
        "unpack %Pair2.pair2",
        "drop.str",
        "drop.i32",
        "i32.const 5",
        "make %Option<i32>.some",
        "drop.%Option<i32>",
    ]


def test_types_are_settled_by_their_use() -> None:

    assert main_body("none 1u8 some == drop { } { 1u16 } == drop") == [
        "make %Option<u8>.none",
        "u8.const 1",
        "make %Option<u8>.some",
        "call $==<%Option<u8>>",
        "drop.bool",
        "data.array $a0",
        "data.array $a1",
        "call $==<array<u16>>",
        "drop.bool",
    ]


def test_the_type_table() -> None:
    text = module_text(
        "type Box[T] = box( T )\n"
        "type Holder = holder( strbuf )\n"
        "word main ( -> )\n"
        '  1 box drop "s" box drop strbuf-new holder holder-unpack strbuf-freeze drop\n'
        "end"
    )
    assert "  type %Box<i32> = variant { box(i32) }\n" in text
    assert "  type %Box<str> = variant { box(str) }\n" in text
    assert "  linear type %Holder = variant { holder(strbuf) }\n" in text


def test_nested_and_recursive_types() -> None:
    text = module_text(
        "type List[T] = empty | cons( T List[T] )\n"
        "word main ( -> ) empty 1 swap cons 5 some pair drop end"
    )
    assert "  type %List<i32> = variant { empty cons(i32 %List<i32>) }\n" in text
    assert (
        "  type %Pair<%List<i32>, %Option<i32>> = "
        "variant { pair(%List<i32> %Option<i32>) }\n"
    ) in text


GENERICS = """\
word twice ( T -> T T ) where T: Unrestricted dup end
word swap-pair ( Pair[A, B] -> Pair[B, A] ) pair-unpack swap pair end
word main ( -> )
  "a" twice drop drop
  5 twice drop drop
  6 twice drop drop
  1 "x" pair swap-pair drop
end
"""


def test_one_function_per_instance() -> None:
    text = module_text(GENERICS)
    assert "func $twice<str> (str) -> (str str)" in text
    assert "func $twice<i32> (i32) -> (i32 i32)" in text
    assert text.count("func $twice<i32>") == 1
    assert "func $swap-pair<i32, str> (%Pair<i32, str>) -> (%Pair<str, i32>)" in text
    assert body_of(GENERICS, "$twice<str>") == ["dup.str"]


def test_unused_generic_words_are_not_compiled() -> None:
    text = module_text("word id ( T -> T ) end\nword main ( -> ) end")
    assert "$id" not in text


def test_unused_words_are_compiled() -> None:
    text = module_text("word spare ( i32 -> i32 ) 1 + end\nword main ( -> ) end")
    assert "func $spare (i32) -> (i32)" in text


def test_polymorphic_recursion_cannot_be_compiled() -> None:
    message = fails(
        "word grow ( T -> ) where T: Unrestricted dup drop some grow end\n"
        "word main ( -> ) 1 grow end"
    )
    assert message == (
        "this program cannot be compiled: `grow` is used at ever larger types, "
        "so it would need endless copies (polymorphic recursion)"
    )


ROWS = """\
word dip ( ..a T Q -> ..b T ) where Q: [ ..a -> ..b ]
  swap [ ] curry compose call
end
word apply-first ( ..a array[i32] Q -> ..a Option[i32] )
  where Q: [ ..a array[i32] -> ..a array[i32] ]
  call 0u32 array-get.checked ? some
end
word main ( -> )
  1 2 [ 1 + ] dip drop drop
  "s" 1 2 3 [ + ] dip drop drop drop
  { 7 } [ ] apply-first drop
  { 7 } [ dup drop ] apply-first drop
end
"""


def test_a_row_takes_only_the_values_its_quotation_needs() -> None:
    text = module_text(ROWS)
    assert (
        "func $dip<i32, [i32 -> i32], (i32), (i32)> (i32 i32 [i32 -> i32]) -> (i32 i32)"
    ) in text
    assert (
        "func $dip<i32, [i32 i32 -> i32], (i32 i32), (i32)> "
        "(i32 i32 i32 [i32 i32 -> i32]) -> (i32 i32)"
    ) in text


def test_a_quotation_can_be_narrower_than_its_parameter() -> None:
    text = module_text(ROWS)
    assert "func $apply-first<[->], ()> (array<i32> [->]) -> (%Option<i32>)" in text
    assert (
        "func $apply-first<[array<i32> -> array<i32>], ()> "
        "(array<i32> [array<i32> -> array<i32>]) -> (%Option<i32>)"
    ) in text


def test_quotation_literals_become_functions() -> None:
    source = (
        "word adder ( i32 -> [ i32 -> i32 ] ) [ + ] curry end\n"
        "word main ( -> ) 5 3 adder call drop [ [ 1 ] call ] call drop end"
    )
    assert body_of(source, "$adder") == [
        "func.ref $adder.q0",
        "rt.curry<i32, [i32 i32 -> i32]>",
    ]
    assert function_text(source, "$adder.q0") == (
        "func $adder.q0 (i32 i32) -> (i32)\n  i32.add\nend"
    )
    assert body_of(source) == [
        "i32.const 5",
        "i32.const 3",
        "call $adder",
        "call.quot [i32 -> i32]",
        "drop.i32",
        "func.ref $main.q0",
        "call.quot [-> i32]",
        "drop.i32",
    ]
    assert body_of(source, "$main.q0") == [
        "func.ref $main.q1",
        "return_call.quot [-> i32]",
    ]
    assert (
        "import rt.curry<i32, [i32 i32 -> i32]> "
        "(i32 [i32 i32 -> i32]) -> ([i32 -> i32])"
    ) in module_text(source)


def test_compose_and_linear_quotations() -> None:
    source = (
        "word main ( -> )\n"
        "  [ 1 + ] [ 2 * ] compose drop\n"
        "  strbuf-new [ strbuf-freeze ] curry call drop\n"
        "end"
    )
    text = module_text(source)
    assert (
        "import rt.compose<[i32 -> i32], [i32 -> i32]> "
        "([i32 -> i32] [i32 -> i32]) -> ([i32 -> i32])"
    ) in text
    assert (
        "import rt.curry<strbuf, [strbuf -> str]> "
        "(strbuf [strbuf -> str]) -> (linear [-> str])"
    ) in text
    assert "call.quot [-> str]" in body_of(source)


def test_consume_names_its_row() -> None:
    source = (
        "word main ( -> )\n"
        "  0 { 1 2 } arraybuf-from-array [ + ] arraybuf-consume print\n"
        "end"
    )
    assert "rt.arraybuf_consume<i32, (i32)>" in body_of(source)
    assert (
        "import rt.arraybuf_consume<i32, (i32)> "
        "(i32 arraybuf<i32> [i32 i32 -> i32]) -> (i32)"
    ) in module_text(source)


def test_blocks_name_only_the_values_they_touch() -> None:
    source = (
        "word pick ( str i32 i32 bool -> str i32 )\n"
        "  if drop else swap drop end\n"
        "end\n"
        "word main ( -> ) end"
    )
    assert body_of(source, "$pick") == [
        "if [i32 i32 -> i32]",
        "drop.i32",
        "else",
        "swap",
        "drop.i32",
        "end",
    ]


def test_an_if_without_else() -> None:
    source = "word bump ( i32 bool -> i32 ) if 1 + end end\nword main ( -> ) end"
    assert body_of(source, "$bump") == [
        "if [i32 -> i32]",
        "i32.const 1",
        "i32.add",
        "end",
    ]


def test_while() -> None:
    source = (
        "word sum-to ( i32 -> i32 )\n"
        "  0 swap while dup 0 > do swap over + swap 1 - end drop\n"
        "end\n"
        "word main ( -> ) end"
    )
    assert body_of(source, "$sum-to") == [
        "i32.const 0",
        "swap",
        "while [i32 i32]",
        "dup.i32",
        "i32.const 0",
        "i32.gt",
        "do",
        "swap",
        "over.i32",
        "i32.add",
        "swap",
        "i32.const 1",
        "i32.sub",
        "end",
        "drop.i32",
    ]


def test_literal_matches() -> None:
    source = (
        "const ONE = 1u8\n"
        "word f ( u8 -> i32 ) match case ONE 10 case 2u8 20 else 0 end end\n"
        'word g ( str -> bool ) match case "yes" true else false end end\n'
        "word h ( bool -> i32 ) match case true 1 case false 0 end end\n"
        "word main ( -> ) end"
    )
    assert body_of(source, "$f") == [
        "match.u8 [-> i32]",
        "case 1",
        "i32.const 10",
        "case 2",
        "i32.const 20",
        "else",
        "i32.const 0",
        "end",
    ]
    assert body_of(source, "$g")[:2] == ["match.str [-> bool]", 'case "yes"']
    assert body_of(source, "$h") == [
        "match.bool [-> i32]",
        "case true",
        "i32.const 1",
        "case false",
        "i32.const 0",
        "end",
    ]


def test_a_match_pushes_fields_in_its_arms() -> None:
    source = (
        "word size ( str Option[i32] -> str i32 )\n"
        "  match case some 1 + case none 0 end\n"
        "end\n"
        "word main ( -> ) end"
    )
    assert body_of(source, "$size") == [
        "match %Option<i32> [-> i32]",
        "case some",
        "i32.const 1",
        "i32.add",
        "case none",
        "i32.const 0",
        "end",
    ]


def test_locals_become_slots() -> None:
    source = (
        "word f ( i32 i32 strbuf -> str i32 )\n"
        "  let a b buffer in buffer strbuf-freeze a b + end\n"
        "end\n"
        "word main ( -> ) end"
    )
    assert function_text(source, "$f") == (
        "func $f (i32 i32 strbuf) -> (str i32)\n"
        "  local $a i32\n"
        "  local $b i32\n"
        "  local $buffer strbuf\n"
        "  local.set $buffer\n"
        "  local.set $b\n"
        "  local.set $a\n"
        "  local.take $buffer\n"
        "  rt.strbuf_freeze\n"
        "  local.get $a\n"
        "  local.get $b\n"
        "  i32.add\n"
        "  local.drop $b\n"
        "  local.drop $a\n"
        "end"
    )


def test_unused_locals_are_dropped_at_end() -> None:
    source = "word f ( i32 -> ) let x in end end\nword main ( -> ) end"
    assert body_of(source, "$f") == ["local.set $x", "local.drop $x"]


def test_each_let_has_its_own_slots() -> None:
    source = (
        "word f ( i32 str -> i32 )\n"
        "  let x in end let x in x end\n"
        "end\n"
        "word main ( -> ) end"
    )
    text = function_text(source, "$f")
    assert "  local $x str\n  local $x.1 i32\n" in text
    assert body_of(source, "$f") == [
        "local.set $x",
        "local.drop $x",
        "local.set $x.1",
        "local.get $x.1",
        "local.drop $x.1",
    ]


def test_tail_calls() -> None:
    source = (
        "word down ( i32 -> i32 )\n"
        "  dup 0 > if 1 - down else 1 + end\n"
        "end\n"
        "word main ( -> ) 3 down drop end"
    )
    assert body_of(source, "$down") == [
        "dup.i32",
        "i32.const 0",
        "i32.gt",
        "if [i32 -> i32]",
        "i32.const 1",
        "i32.sub",
        "return_call $down",
        "else",
        "i32.const 1",
        "i32.add",
        "end",
    ]
    assert "call $down" in body_of(source)


def test_a_tail_call_empties_the_slots_first() -> None:
    source = (
        "word count ( i32 i32 -> i32 )\n"
        "  let total n in n 0 == if total else total n + n 1 - count end end\n"
        "end\n"
        "word main ( -> ) end"
    )
    lowered = body_of(source, "$count")
    tail = lowered.index("return_call $count")
    assert lowered[tail - 2 : tail] == ["local.drop $n", "local.drop $total"]
    assert lowered[-2:] == ["local.drop $n", "local.drop $total"]


def test_a_tail_call_may_leave_values_below_its_arguments() -> None:
    source = (
        "word g ( i32 -> i32 ) 1 + end\n"
        "word f ( str i32 -> str i32 ) g end\n"
        "word main ( -> ) end"
    )
    assert body_of(source, "$f") == ["return_call $g"]


def test_nothing_in_a_loop_is_in_tail_position() -> None:
    source = (
        "word g ( i32 -> i32 ) 1 + end\n"
        "word f ( i32 -> i32 ) while dup 10 < do g end end\n"
        "word main ( -> ) end"
    )
    assert "call $g" in body_of(source, "$f")


def test_question_mark_on_a_result() -> None:
    source = (
        "word half ( i32 -> Result[i32, str] )\n"
        '  dup 2 % 0 == if 2 / ok else drop "odd" err end\n'
        "end\n"
        "word quarter ( i32 -> Result[i32, str] )\n"
        '  "note" swap half ? swap drop half\n'
        "end\n"
        "word main ( -> ) end"
    )
    assert function_text(source, "$quarter") == (
        "func $quarter (i32) -> (%Result<i32, str>)\n"
        "  local $? str\n"
        "  data.str $s1\n"
        "  swap\n"
        "  call $half\n"
        "  match %Result<i32, str> [str -> str i32]\n"
        "    case ok\n"
        "    case err\n"
        "      local.set $?\n"
        "      drop.str\n"
        "      local.take $?\n"
        "      make %Result<i32, str>.err\n"
        "      return\n"
        "  end\n"
        "  swap\n"
        "  drop.str\n"
        "  return_call $half\n"
        "end"
    )


def test_question_mark_on_an_option_empties_the_slots() -> None:
    source = (
        "word f ( i32 -> Option[i32] )\n"
        "  let n in n 2 *.checked ? n + some end\n"
        "end\n"
        "word main ( -> ) end"
    )
    assert body_of(source, "$f")[:11] == [
        "local.set $n",
        "local.get $n",
        "i32.const 2",
        "i32.mul.checked",
        "match %Option<i32> [-> i32]",
        "case none",
        "local.drop $n",
        "make %Option<i32>.none",
        "return",
        "case some",
        "end",
    ]


def test_question_mark_keeps_the_callers_row() -> None:
    source = (
        "word w ( ..a Q -> ..a Option[i32] ) where Q: [ ..a -> ..a ]\n"
        "  call 5 some ? some\n"
        "end\n"
        "word main ( -> )\n"
        '  strbuf-new [ "x" strbuf-append ] w drop strbuf-freeze drop\n'
        "end"
    )
    name = "$w<[strbuf -> strbuf], (strbuf)>"
    assert function_text(source, name).splitlines()[0] == (
        f"func {name} (strbuf [strbuf -> strbuf]) -> (strbuf %Option<i32>)"
    )
    assert body_of(source, name)[:7] == [
        "call.quot [strbuf -> strbuf]",
        "i32.const 5",
        "make %Option<i32>.some",
        "match %Option<i32> [-> i32]",
        "case none",
        "make %Option<i32>.none",
        "return",
    ]


def test_the_entry_point() -> None:
    assert module_text("word main ( -> ) end").endswith("  entry $main unit\nend\n")
    text = module_text("word main ( -> Result[unit, str] ) unit ok end")
    assert text.endswith("  entry $main result\nend\n")


def test_a_program_with_errors_is_not_compiled() -> None:
    assert fails("word main ( -> ) 1 end") == (
        "`main` declares that it leaves nothing, but its body leaves `i32`"
    )
    assert fails("word f ( -> ) end") == "the program has no `main`"


def run_command(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "argv", ["inde", *arguments])
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    status = command.main()
    return status, out.getvalue(), err.getvalue()


def test_the_command_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    program = tmp_path / "shapes.inde"
    program.write_text(SHAPES, encoding="utf-8")
    status, out, _ = run_command(["--ir", str(program)], monkeypatch)
    assert status == 0
    assert out == module_text(SHAPES)
    status, out, _ = run_command(["--ir-debug", str(program)], monkeypatch)
    assert "; 12:14 area" in out
    status, _, err = run_command(["--ir"], monkeypatch)
    assert status == 1
    assert err.startswith("usage:")
    broken = tmp_path / "broken.inde"
    broken.write_text("word main ( -> ) 1 end", encoding="utf-8")
    status, _, err = run_command(["--ir", str(broken)], monkeypatch)
    assert status == 1
    assert "leaves nothing" in err
