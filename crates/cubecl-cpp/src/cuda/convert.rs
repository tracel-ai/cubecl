//! Cuda conversion functions
#![allow(unused)]

use core::fmt;

use crate::{
    Dialect,
    shared::{Component, Elem, FP8Kind, FmtLeft, Instruction, Item, UnaryInstruction, Value},
};

/// special cast function for recursive conversion in the case of minifloat to minifloat conversion
///
/// Needs to jump through a lot of hoops to deal with CUDA nonsense.
/// The overview of available conversions is as follows:
///
/// | From                     | To             | Extra args                 |
/// | ------------------------ | -------------- | -------------------------- |
/// | f16/bf16/f32/f64         | e4m3/e5m2      | Interpretation, saturation |
/// | f16/bf16/f32/f64         | e3m2/e2m3/e2m1 | Interpretation, rounding   |
/// | bf16/f32/f64             | e8m0           | saturation, rounding       |
/// | e4m3/e5m2/e3m2/e2m3/e2m1 | f16            | Interpretation,            |
/// | e8m0                     | bf16           |                            |
///
/// When the input and output don't match these options, we need to do a two-step conversion.
/// When the input is a minifloat we always need to cast out to `f16`/`bf16`, and then convert to
/// the actual out type if it differs. Trying to cast ints also requires an extra conversion, and
/// so does `f16` to `e8m0` (though it's not recommended to do that anyways, you should be using
/// `e5m2` for that since you don't have 8 bits of exponent in f16).
///
/// See also:
/// <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html>
/// <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html>
/// <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html>
pub(crate) fn special_cast<D: Dialect>(
    f: &mut std::fmt::Formatter,
    input: &Value<D>,
    out: &Value<D>,
) -> fmt::Result {
    let mut current_in = *input;

    if matches!(
        input.elem().unpacked(),
        Elem::FP4(_) | Elem::FP6(_) | Elem::FP8(_)
    ) {
        let half_elem = match input.elem().unpacked() {
            Elem::FP8(FP8Kind::UE8M0) => Elem::BF16,
            _ => Elem::F16,
        };
        // The decode intrinsics only exist in scalar and x2 forms, so a single unpacked source
        // value has no intrinsic that fills a wider destination. Decode it on its own and let
        // the trailing assign broadcast the result. A packed source already carries one value
        // per destination lane of the x2 intrinsic, so it keeps the destination's width.
        let broadcast = input.item().vectorization() == 1
            && input.elem().packing_factor() == 1
            && out.item().vectorization() > 1;
        let item = if broadcast {
            Item::Scalar(half_elem)
        } else {
            out.item().with_elem(half_elem)
        };
        let out_var = if item == out.item() {
            *out
        } else {
            Value::tmp(item)
        };
        if *item.elem() == Elem::F16 {
            cast_minifloat_to_half(f, current_in, out_var)?;
        } else {
            cast_scale_to_bfloat(f, current_in, out_var)?;
        }
        current_in = out_var;
    }

    let in_vec = match current_in.item() {
        Item::Scalar(_) => 1,
        Item::Vector(_, vectorization) | Item::NativeVector(_, vectorization) => vectorization,
        _ => panic!("Invalid input item for special cast"),
    };

    // Broadcast scalars to packing factor. A minifloat source has already been decoded to half
    // above, so this widens whatever `current_in` holds now, not the type it started as.
    if out.item().packing_factor() > 1 && in_vec == 1 {
        let tmp = Value::tmp(Item::new(current_in.elem(), out.item().packing_factor()));
        let assign = Instruction::Assign(UnaryInstruction {
            input: current_in,
            out: tmp,
        });
        writeln!(f, "{assign}")?;
        current_in = tmp;
    }

    let in_vec = match current_in.item() {
        Item::Scalar(_) => 1,
        Item::Vector(_, vectorization) | Item::NativeVector(_, vectorization) => vectorization,
        _ => panic!("Invalid input item for special cast"),
    };

    if matches!(
        current_in.elem(),
        Elem::U8
            | Elem::U16
            | Elem::U32
            | Elem::U64
            | Elem::I8
            | Elem::I16
            | Elem::I32
            | Elem::I64
            | Elem::Bool
    ) {
        // Precision is irrelevant for int, so use bf16 for the range
        let tmp = Value::tmp(Item::new(Elem::BF16, in_vec));
        let assign = Instruction::Assign(UnaryInstruction {
            input: current_in,
            out: tmp,
        });
        writeln!(f, "{assign}")?;
        current_in = tmp;
    }

    if matches!(out.elem().unpacked(), Elem::FP4(_) | Elem::FP6(_)) {
        return cast_to_fp4_fp6(f, current_in, *out);
    }

    if matches!(out.elem().unpacked(), Elem::FP8(FP8Kind::UE8M0)) {
        // Scale can't be converted from half...
        if matches!(current_in.elem(), Elem::F16) {
            let item = current_in.item().with_elem(Elem::BF16);
            let tmp = Value::tmp(item);
            let assign = Instruction::Assign(UnaryInstruction {
                input: current_in,
                out: tmp,
            });
            writeln!(f, "{assign}")?;
            current_in = tmp;
        }
        return cast_to_scale(f, current_in, *out);
    }

    if matches!(out.elem().unpacked(), Elem::FP8(_)) {
        return cast_to_fp8(f, current_in, *out);
    }

    if current_in.item() != out.item() {
        let assign = Instruction::Assign(UnaryInstruction {
            input: current_in,
            out: *out,
        });
        writeln!(f, "{assign}")?;
    }

    Ok(())
}

/// Convert any float to fp4/fp6, with round to nearest
fn cast_to_fp4_fp6<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Value<D>,
    out: Value<D>,
) -> fmt::Result {
    let out_opt = out.optimized();
    let packing = out_opt.item().packing_factor();
    let packed = packing == 2;
    let pack_suffix = if packed { "2" } else { "" };

    let (out_ty, interpretation) = match out_opt.elem() {
        Elem::FP4(kind) => ("fp4", format!("{kind:?}")),
        Elem::FP4x2(kind) => ("fp4x2", format!("{kind:?}")),
        Elem::FP6(kind) => ("fp6", format!("{kind:?}")),
        Elem::FP6x2(kind) => ("fp6x2", format!("{kind:?}")),
        _ => unreachable!("Must be fp4 or fp6"),
    };

    let in_ty = match input.elem().unpacked() {
        Elem::F64 => format!("double{pack_suffix}"),
        Elem::TF32 | Elem::F32 => format!("float{pack_suffix}"),
        Elem::F16 => format!("halfraw{pack_suffix}"),
        Elem::BF16 => format!("bfloat16raw{pack_suffix}"),
        _ => unreachable!(),
    };

    let input = input.optimized();

    handle_unroll(f, out, |f, i| {
        let in_value = float_to_packed(input, i, packing);

        write!(
            f,
            "__nv_cvt_{in_ty}_to_{out_ty}({in_value}, __NV_{interpretation}, cudaRoundNearest)",
        )
    })
}

/// Convert any float except f16 to e8m0
fn cast_to_scale<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Value<D>,
    out: Value<D>,
) -> fmt::Result {
    let out_opt = out.optimized();
    let packing = out_opt.item().packing_factor();
    let packed = packing > 1;
    let pack_suffix = if packed { "2" } else { "" };

    let out_ty = match out_opt.elem() {
        Elem::FP8(_) => "e8m0",
        Elem::FP8x2(_) => "e8m0x2",
        _ => unreachable!("Must be scale factor"),
    };

    let in_ty = match input.elem() {
        Elem::F64 => format!("double{pack_suffix}"),
        Elem::TF32 | Elem::F32 => format!("float{pack_suffix}"),
        Elem::BF16 => format!("bfloat16{pack_suffix}raw"),
        _ => unreachable!(),
    };

    let input = input.optimized();

    handle_unroll(f, out, |f, i| {
        let in_value = float_to_packed(input, i, packing);

        write!(
            f,
            "__nv_cvt_{in_ty}_to_{out_ty}({in_value}, __NV_NOSAT, cudaRoundPosInf)",
        )
    })
}

/// Convert any float to fp8 (except e8m0)
fn cast_to_fp8<D: Dialect>(f: &mut fmt::Formatter, input: Value<D>, out: Value<D>) -> fmt::Result {
    let out_opt = out.optimized();
    let packing = out_opt.item().packing_factor();
    let packed = packing > 1;
    let pack_suffix = if packed { "2" } else { "" };

    let (out_ty, interpretation) = match out_opt.elem() {
        Elem::FP8(kind) => ("fp8", format!("{kind:?}")),
        Elem::FP8x2(kind) => ("fp8x2", format!("{kind:?}")),
        _ => unreachable!("Must be fp8"),
    };

    let in_ty = match input.elem() {
        Elem::F64 => format!("double{pack_suffix}"),
        Elem::TF32 | Elem::F32 => format!("float{pack_suffix}"),
        Elem::BF16 => format!("bfloat16raw{pack_suffix}"),
        Elem::F16 => format!("halfraw{pack_suffix}"),
        _ => unreachable!(),
    };

    let input = input.optimized();

    handle_unroll(f, out, |f, i| {
        let in_value = float_to_packed(input, i, packing);

        write!(
            f,
            "__nv_cvt_{in_ty}_to_{out_ty}({in_value}, __NV_NOSAT, __NV_{interpretation})",
        )
    })
}

/// Pack types that normally wouldn't be optimized into a `vec2` for conversion
fn float_to_packed<D: Dialect>(input: Value<D>, i: usize, packing: usize) -> String {
    match input.elem() {
        Elem::TF32 | Elem::F32 => {
            let i = i * packing;
            if packing > 1 {
                format!("float2 {{ {}, {} }}", input.index(i), input.index(i + 1))
            } else {
                format!("{}", input.index(i))
            }
        }
        Elem::F64 => {
            let i = i * packing;
            if packing > 1 {
                format!("double2 {{ {}, {} }}", input.index(i), input.index(i + 1))
            } else {
                format!("{}", input.index(i))
            }
        }
        Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => format!("{}", input.index(i)),
        _ => unreachable!(),
    }
}

/// Convert any FP8/6/4 except e8m0 to half
fn cast_minifloat_to_half<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Value<D>,
    out: Value<D>,
) -> fmt::Result {
    let in_opt = input.optimized();
    let out_opt = out.optimized().item();

    let (in_ty, interpretation) = match in_opt.elem() {
        Elem::FP4(kind) => ("fp4", format!("{kind:?}")),
        Elem::FP4x2(kind) => ("fp4x2", format!("{kind:?}")),
        Elem::FP6(kind) => ("fp6", format!("{kind:?}")),
        Elem::FP6x2(kind) => ("fp6x2", format!("{kind:?}")),
        Elem::FP8(kind) => ("fp8", format!("{kind:?}")),
        Elem::FP8x2(kind) => ("fp8x2", format!("{kind:?}")),
        _ => unreachable!("can only cast minifloat"),
    };

    let out_ty = match out_opt.elem() {
        Elem::F16 => "halfraw",
        Elem::F16x2 => "halfraw2",
        _ => unreachable!("out type must be half"),
    };

    handle_unroll(f, out, |f, i| {
        let input = in_opt.index(i);
        write!(
            f,
            "{}(__nv_cvt_{in_ty}_to_{out_ty}({input}, __NV_{interpretation}))",
            out_opt.elem()
        )
    })
}

/// Convert an e8m0 scaling factor to bf16
fn cast_scale_to_bfloat<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Value<D>,
    out: Value<D>,
) -> fmt::Result {
    let in_opt = input.optimized();
    let out_opt = out.optimized().item();

    let in_ty = match in_opt.elem() {
        Elem::FP8(_) => "e8m0",
        Elem::FP8x2(_) => "e8m0x2",
        _ => unreachable!("must be scaling factor in e8m0 format"),
    };

    let out_ty = match out_opt.elem() {
        Elem::BF16 => "bf16raw",
        Elem::BF16x2 => "bf162raw",
        _ => unreachable!("out type must be half"),
    };

    handle_unroll(f, out, |f, i| {
        let input = in_opt.index(i);
        write!(
            f,
            "{}(__nv_cvt_{in_ty}_to_{out_ty}({input}))",
            out_opt.elem()
        )
    })
}

fn handle_unroll<D: Dialect>(
    f: &mut fmt::Formatter,
    out: Value<D>,
    mut op: impl FnMut(&mut fmt::Formatter, usize) -> fmt::Result,
) -> fmt::Result {
    let out_opt = out.item().optimized();
    let vec = match out_opt {
        Item::Scalar(_) => 1,
        Item::Vector(_, vectorization) | Item::NativeVector(_, vectorization) => vectorization,
        _ => panic!("Invalid input item for special cast"),
    };
    let out_var = if out.item() != out_opt {
        Value::tmp(out_opt)
    } else {
        out
    };
    write!(f, "{} = ", out_var.fmt_left())?;
    if vec > 1 {
        writeln!(f, "{out_opt} {{")?;
    }
    for i in 0..vec {
        op(f, i)?;
        if i + 1 < vec {
            f.write_str(",\n")?;
        }
    }
    if vec > 1 {
        write!(f, "\n}}")?;
    }
    f.write_str(";\n")?;

    if out.item() != out_opt {
        writeln!(
            f,
            "{} = reinterpret_cast<{}&>({out_var});",
            out.fmt_left(),
            out.item()
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cuda::{CudaDialect, mma::CudaWmmaCompiler},
        shared::{FP4Kind, FP6Kind},
    };
    use std::fmt::Display;

    type Cuda = CudaDialect<CudaWmmaCompiler>;

    /// Every `__nv_cvt_*` symbol declared by the CUDA minifloat headers, as of 13.3.1.
    ///
    /// The point of pinning these is that the decode direction only ever comes in matched
    /// scalar/scalar and x2/2-wide pairs. `special_cast` composes the name from the source and
    /// destination types independently, so a source and destination of mismatched width compose a
    /// name that does not exist. This list is what makes that detectable without a GPU, which
    /// matters because fp4, fp6 and e8m0 conversion needs Blackwell to run at all.
    const DECLARED_INTRINSICS: &[&str] = &[
        "__nv_cvt_bfloat162raw_to_e8m0x2",
        "__nv_cvt_bfloat16raw2_to_fp4x2",
        "__nv_cvt_bfloat16raw2_to_fp6x2",
        "__nv_cvt_bfloat16raw2_to_fp8x2",
        "__nv_cvt_bfloat16raw_to_e8m0",
        "__nv_cvt_bfloat16raw_to_fp4",
        "__nv_cvt_bfloat16raw_to_fp6",
        "__nv_cvt_bfloat16raw_to_fp8",
        "__nv_cvt_double2_to_e8m0x2",
        "__nv_cvt_double2_to_fp4x2",
        "__nv_cvt_double2_to_fp6x2",
        "__nv_cvt_double2_to_fp8x2",
        "__nv_cvt_double_to_e8m0",
        "__nv_cvt_double_to_fp4",
        "__nv_cvt_double_to_fp6",
        "__nv_cvt_double_to_fp8",
        "__nv_cvt_e8m0_to_bf16raw",
        "__nv_cvt_e8m0x2_to_bf162raw",
        "__nv_cvt_float2_to_e8m0x2",
        "__nv_cvt_float2_to_fp4x2",
        "__nv_cvt_float2_to_fp6x2",
        "__nv_cvt_float2_to_fp8x2",
        "__nv_cvt_float_to_e8m0",
        "__nv_cvt_float_to_fp4",
        "__nv_cvt_float_to_fp6",
        "__nv_cvt_float_to_fp8",
        "__nv_cvt_fp4_to_halfraw",
        "__nv_cvt_fp4x2_to_halfraw2",
        "__nv_cvt_fp6_to_halfraw",
        "__nv_cvt_fp6x2_to_halfraw2",
        "__nv_cvt_fp8_to_halfraw",
        "__nv_cvt_fp8x2_to_halfraw2",
        "__nv_cvt_halfraw2_to_fp4x2",
        "__nv_cvt_halfraw2_to_fp6x2",
        "__nv_cvt_halfraw2_to_fp8x2",
        "__nv_cvt_halfraw_to_fp4",
        "__nv_cvt_halfraw_to_fp6",
        "__nv_cvt_halfraw_to_fp8",
    ];

    struct SpecialCast {
        input: Value<Cuda>,
        out: Value<Cuda>,
    }

    impl Display for SpecialCast {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            special_cast(f, &self.input, &self.out)
        }
    }

    fn emit(input: Item<Cuda>, out: Item<Cuda>) -> String {
        SpecialCast {
            input: Value::Value { id: 0, item: input },
            out: Value::Value { id: 1, item: out },
        }
        .to_string()
    }

    fn called_intrinsics(source: &str) -> Vec<&str> {
        let mut found = Vec::new();
        let mut rest = source;
        while let Some(start) = rest.find("__nv_cvt_") {
            rest = &rest[start..];
            let end = rest
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            found.push(&rest[..end]);
            rest = &rest[end..];
        }
        found
    }

    /// Both the unpacked kinds and the x2 kinds, since a packed source carries two logical values
    /// per storage element and reaches a different branch.
    const MINIFLOATS: &[Elem<Cuda>] = &[
        Elem::FP4(FP4Kind::E2M1),
        Elem::FP4x2(FP4Kind::E2M1),
        Elem::FP6(FP6Kind::E2M3),
        Elem::FP6(FP6Kind::E3M2),
        Elem::FP6x2(FP6Kind::E2M3),
        Elem::FP8(FP8Kind::E4M3),
        Elem::FP8(FP8Kind::E5M2),
        Elem::FP8(FP8Kind::UE8M0),
        Elem::FP8x2(FP8Kind::E4M3),
    ];

    const FLOATS: &[Elem<Cuda>] = &[Elem::F16, Elem::BF16, Elem::F32, Elem::F64];

    /// Number of values an item holds, counting both its lanes and the values packed into each.
    fn logical_width(item: Item<Cuda>) -> usize {
        item.vectorization() * item.elem().packing_factor()
    }

    /// Every conversion the compiler can ask for, as a description and the CUDA it emits.
    ///
    /// `cast_expand_elem` only rejects a value count mismatch when the source has more than one
    /// lane, so a single lane reaches codegen against any destination width even when it carries
    /// a packed pair. Anything wider has to match one for one.
    fn conversion_matrix() -> Vec<(String, String)> {
        let mut cases = Vec::new();

        for &minifloat in MINIFLOATS {
            for &float in FLOATS {
                for mini_width in [1, 2, 4] {
                    for float_width in [1, 2, 4] {
                        let mini = Item::new(minifloat, mini_width);
                        let float = Item::new(float, float_width);
                        let matched = logical_width(mini) == logical_width(float);

                        // Narrowing a packed source into fewer values than it holds also reaches
                        // codegen and also emits a symbol that does not exist, but that is a
                        // separate defect from the broadcast this file fixes.
                        let widening = logical_width(float) >= logical_width(mini);
                        if matched || (mini.vectorization() == 1 && widening) {
                            cases
                                .push((format!("decode {mini:?} to {float:?}"), emit(mini, float)));
                        }
                        // The broadcast encode direction is tracked separately as #1459, so only
                        // the matched widths are asserted here.
                        if matched {
                            cases
                                .push((format!("encode {float:?} to {mini:?}"), emit(float, mini)));
                        }
                    }
                }
            }
        }

        // A minifloat destination decodes through the same half intermediate before re-encoding,
        // so the broadcast has to survive it too. Only a single unpacked source is covered: a
        // packed or multi-lane source decoded into a narrower destination composes a name that
        // does not exist, but that is the same separate defect as the narrowing above.
        for &from_elem in MINIFLOATS {
            if from_elem.packing_factor() != 1 {
                continue;
            }
            for &to_elem in MINIFLOATS {
                if from_elem == to_elem {
                    continue;
                }
                for to_width in [2, 4] {
                    let from = Item::Scalar(from_elem);
                    let to = Item::new(to_elem, to_width);
                    cases.push((format!("convert {from:?} to {to:?}"), emit(from, to)));
                }
            }
        }

        cases
    }

    /// No conversion may name an intrinsic the CUDA headers do not declare, at any combination of
    /// source and destination width. A scalar source broadcast into a vector is the case where a
    /// width mismatch composes `__nv_cvt_fp8_to_halfraw2`.
    #[test]
    fn every_emitted_intrinsic_is_declared() {
        let mut undeclared = Vec::new();

        for (case, source) in conversion_matrix() {
            for name in called_intrinsics(&source) {
                if !DECLARED_INTRINSICS.contains(&name) {
                    undeclared.push(format!("{case}: `{name}`\n{source}"));
                }
            }
        }

        assert!(
            undeclared.is_empty(),
            "{} conversions named an intrinsic no CUDA header declares:\n{}",
            undeclared.len(),
            undeclared.join("\n")
        );
    }

    /// A vector temporary read past its own width compiles to nothing, so the widths of the
    /// intermediates have to agree with the width the surrounding unroll indexes them at.
    #[test]
    fn no_temporary_is_read_past_its_width() {
        let mut out_of_bounds = Vec::new();

        for (case, source) in conversion_matrix() {
            // Declarations look like `__half_2 _tmp_0 = ...`, reads like `_tmp_0.i_2`.
            let mut widths = std::collections::HashMap::new();
            for line in source.lines() {
                let mut words = line.split_whitespace();
                if let (Some(ty), Some(name)) = (words.next(), words.next())
                    && name.starts_with("_tmp_")
                    && let Some((_, width)) = ty.rsplit_once('_')
                    && let Ok(width) = width.parse::<usize>()
                {
                    widths.insert(name.to_string(), width);
                }
            }

            for (name, width) in &widths {
                let read = format!("{name}.i_");
                let mut rest = source.as_str();
                while let Some(start) = rest.find(&read) {
                    rest = &rest[start + read.len()..];
                    let end = rest
                        .find(|c: char| !c.is_ascii_digit())
                        .unwrap_or(rest.len());
                    if let Ok(index) = rest[..end].parse::<usize>()
                        && index >= *width
                    {
                        out_of_bounds.push(format!(
                            "{case}: {name} has width {width}, read at i_{index}\n{source}"
                        ));
                    }
                }
            }
        }

        assert!(
            out_of_bounds.is_empty(),
            "{} conversions read a temporary past its width:\n{}",
            out_of_bounds.len(),
            out_of_bounds.join("\n")
        );
    }
}
