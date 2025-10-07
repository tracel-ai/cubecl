use core::fmt;

use crate::{
    Dialect,
    shared::{Component, Elem, FP8Kind, FmtLeft, Instruction, Item, UnaryInstruction, Variable},
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
    input: &Variable<D>,
    out: &Variable<D>,
) -> fmt::Result {
    let mut current_in = *input;

    if matches!(
        input.elem().unpacked(),
        Elem::FP4(_) | Elem::FP6(_) | Elem::FP8(_)
    ) {
        let mut item = out.item();
        item.elem = match input.elem().unpacked() {
            Elem::FP8(FP8Kind::UE8M0) => Elem::BF16,
            _ => Elem::F16,
        };
        let out_var = if item == out.item() {
            *out
        } else {
            Variable::tmp(item)
        };
        if item.elem == Elem::F16 {
            cast_minifloat_to_half(f, current_in, out_var)?;
        } else {
            cast_scale_to_bfloat(f, current_in, out_var)?;
        }
        current_in = out_var;
    }

    // Broadcast scalars to packing factor
    if out.item().packing_factor() > 1 && input.item().vectorization == 1 {
        let tmp = Variable::tmp(Item {
            elem: input.item().elem,
            vectorization: out.item().packing_factor(),
            native: input.item().native,
        });
        let assign = Instruction::Assign(UnaryInstruction {
            input: current_in,
            out: tmp,
        });
        writeln!(f, "{assign}")?;
        current_in = tmp;
    }

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
        let tmp = Variable::tmp(Item {
            elem: Elem::BF16,
            vectorization: current_in.item().vectorization,
            native: current_in.item().native,
        });
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
            let mut item = current_in.item();
            item.elem = Elem::BF16;
            let tmp = Variable::tmp(item);
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
    input: Variable<D>,
    out: Variable<D>,
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
    input: Variable<D>,
    out: Variable<D>,
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
fn cast_to_fp8<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Variable<D>,
    out: Variable<D>,
) -> fmt::Result {
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
fn float_to_packed<D: Dialect>(input: Variable<D>, i: usize, packing: usize) -> String {
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
    input: Variable<D>,
    out: Variable<D>,
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
    input: Variable<D>,
    out: Variable<D>,
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
    out: Variable<D>,
    mut op: impl FnMut(&mut fmt::Formatter, usize) -> fmt::Result,
) -> fmt::Result {
    let out_opt = out.item().optimized();
    let vec = out_opt.vectorization;
    let out_var = if out.item() != out_opt {
        Variable::tmp(out_opt)
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
