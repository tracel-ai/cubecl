use super::{Component, Dialect, Elem, FmtLeft, Value};
use std::fmt::Display;

pub trait Unary<D: Dialect> {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let out_item = out.item();

        if out_item.vectorization() == 1 {
            write!(f, "{} = ", out.fmt_left())?;
            Self::format_scalar(f, *input, *out_item.elem())?;
            f.write_str(";\n")
        } else {
            Self::unroll_vec(f, input, out, *out_item.elem(), out_item.vectorization())
        }
    }

    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out_elem: Elem<D>,
    ) -> std::fmt::Result;

    fn unroll_vec(
        f: &mut std::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
        out_elem: Elem<D>,
        index: usize,
    ) -> std::fmt::Result {
        let mut write_op = |index, out_elem, input: &Value<D>, out: &Value<D>| {
            let out_item = out.item();
            let out = out.fmt_left();
            writeln!(f, "{out} = {out_item}{{")?;

            for i in 0..index {
                let inputi = input.index(i);

                Self::format_scalar(f, inputi, out_elem)?;
                f.write_str(",")?;
            }

            f.write_str("};\n")
        };

        if Self::can_optimize() {
            let optimized = Value::optimized_args([*input, *out]);
            let [input, out_optimized] = optimized.args;

            let item_out_original = out.item();
            let item_out_optimized = out_optimized.item();

            let (index, out_elem) = match optimized.optimization_factor {
                Some(factor) => (index / factor, out_optimized.elem()),
                None => (index, out_elem),
            };

            if item_out_original != item_out_optimized {
                let out_tmp = Value::tmp(item_out_optimized);

                write_op(index, out_elem, &input, &out_tmp)?;
                let qualifier = out.const_qualifier();
                let addr_space = D::address_space_for_value(out);
                let out_fmt = out.fmt_left();
                writeln!(
                    f,
                    "{out_fmt} = reinterpret_cast<{addr_space}{item_out_original}{qualifier}&>({out_tmp});\n"
                )
            } else {
                write_op(index, out_elem, &input, &out_optimized)
            }
        } else {
            write_op(index, out_elem, input, out)
        }
    }

    fn can_optimize() -> bool {
        true
    }
}

pub trait FunctionFmt<D: Dialect> {
    fn base_function_name() -> &'static str;
    fn function_name(elem: Elem<D>) -> String {
        if Self::half_support() {
            let prefix = match elem {
                Elem::F16 | Elem::BF16 => D::compile_instruction_half_function_name_prefix(),
                Elem::F16x2 | Elem::BF16x2 => D::compile_instruction_half2_function_name_prefix(),
                _ => "",
            };
            format!(
                "{prefix}{}",
                D::compile_fast_math_function_name(Self::base_function_name())
            )
        } else {
            D::compile_fast_math_function_name(Self::base_function_name()).into()
        }
    }
    fn format_unary<Input: Display>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        if Self::half_support() {
            // Dialects without a half prefix (e.g. Metal) reuse the float-taking
            // function and lack `bfloat` intrinsics, so bfloat must round-trip through
            // float. Dialects with a prefix (e.g. CUDA's `h`) have native intrinsics.
            let no_half_prefix = D::compile_instruction_half_function_name_prefix().is_empty();
            match elem {
                Elem::BF16 | Elem::BF16x2 if no_half_prefix => {
                    write!(f, "{}({}(float({input})))", elem, Self::function_name(elem))
                }
                _ => write!(f, "{}({input})", Self::function_name(elem)),
            }
        } else {
            match elem {
                Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                    write!(f, "{}({}(float({input})))", elem, Self::function_name(elem))
                }
                // Small ints cause issues around auto-promotion on AMD, so use explicit cast on out
                Elem::U16 | Elem::U8 | Elem::I16 | Elem::I8 => {
                    write!(f, "{elem}({}({input}))", Self::function_name(elem))
                }
                _ => write!(f, "{}({input})", Self::function_name(elem)),
            }
        }
    }

    fn half_support() -> bool;
}

macro_rules! function {
    ($name:ident, $func:expr) => {
        function!($name, $func, true);
    };
    ($name:ident, $func:expr, $half_support:expr) => {
        pub struct $name;

        impl<D: Dialect> FunctionFmt<D> for $name {
            fn base_function_name() -> &'static str {
                $func
            }
            fn half_support() -> bool {
                $half_support
            }
        }

        impl<D: Dialect> Unary<D> for $name {
            fn format_scalar<Input: Display>(
                f: &mut std::fmt::Formatter<'_>,
                input: Input,
                elem: Elem<D>,
            ) -> std::fmt::Result {
                Self::format_unary(f, input, elem)
            }

            fn can_optimize() -> bool {
                $half_support
            }
        }
    };
}

function!(Log, "log");
function!(FastLog, "__logf", false);
function!(Sin, "sin");
function!(Cos, "cos");
function!(Tan, "tan", false);
function!(Sinh, "sinh", false);
function!(Cosh, "cosh", false);
function!(ArcCos, "acos", false);
function!(ArcSin, "asin", false);
function!(ArcTan, "atan", false);
function!(ArcSinh, "asinh", false);
function!(ArcCosh, "acosh", false);
function!(ArcTanh, "atanh", false);
function!(FastSin, "__sinf", false);
function!(FastCos, "__cosf", false);
function!(Sqrt, "sqrt");
function!(InverseSqrt, "rsqrt");
function!(FastSqrt, "__fsqrt_rn", false);
function!(FastInverseSqrt, "__frsqrt_rn", false);
function!(Exp, "exp");
function!(FastExp, "__expf", false);
function!(Ceil, "ceil");
function!(Trunc, "trunc");
function!(Floor, "floor");
function!(Round, "rint");
function!(FastRecip, "__frcp_rn", false);
function!(FastTanh, "__tanhf", false);

function!(Erf, "erf", false);
function!(Abs, "abs", false);

pub struct Neg;

impl<D: Dialect> Unary<D> for Neg {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        _out_elem: Elem<D>,
    ) -> std::fmt::Result {
        writeln!(f, "-{}", input)
    }
}

pub struct Log1p;

impl<D: Dialect> Unary<D> for Log1p {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        _out_elem: Elem<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_log1p_scalar(f, input)
    }

    fn can_optimize() -> bool {
        false
    }
}

pub struct Expm1;

impl<D: Dialect> Unary<D> for Expm1 {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        _out_elem: Elem<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_expm1_scalar(f, input)
    }

    fn can_optimize() -> bool {
        false
    }
}

pub struct Tanh;

impl<D: Dialect> Unary<D> for Tanh {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        _out_elem: Elem<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_tanh_scalar(f, input)
    }

    fn can_optimize() -> bool {
        false
    }
}

pub struct Degrees;

impl<D: Dialect> Unary<D> for Degrees {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        write!(f, "{input}*{elem}(57.29577951308232f)")
    }

    fn can_optimize() -> bool {
        false
    }
}

pub struct Radians;

impl<D: Dialect> Unary<D> for Radians {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        write!(f, "{input}*{elem}(0.017453292519943295f)")
    }

    fn can_optimize() -> bool {
        false
    }
}

pub fn zero_extend<D: Dialect>(input: impl Component<D>) -> String {
    match input.elem() {
        Elem::I8 => format!("{}({}({input}))", Elem::<D>::U32, Elem::<D>::U8),
        Elem::I16 => format!("{}({}({input}))", Elem::<D>::U32, Elem::<D>::U16),
        Elem::U8 => format!("{}({input})", Elem::<D>::U32),
        Elem::U16 => format!("{}({input})", Elem::<D>::U32),
        _ => unreachable!("zero extend only supports integer < 32 bits"),
    }
}

pub struct CountBits;

impl<D: Dialect> Unary<D> for CountBits {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_popcount_scalar(f, input, elem)
    }
}

pub struct ReverseBits;

impl<D: Dialect> Unary<D> for ReverseBits {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_reverse_bits_scalar(f, input, elem)
    }
}

pub struct LeadingZeros;

impl<D: Dialect> Unary<D> for LeadingZeros {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_leading_zeros_scalar(f, input, elem)
    }
}

pub struct TrailingZeros;

impl<D: Dialect> Unary<D> for TrailingZeros {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_trailing_zeros_scalar(f, input, elem)
    }
}

pub struct FindFirstSet;

impl<D: Dialect> Unary<D> for FindFirstSet {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out_elem: Elem<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_find_first_set(f, input, out_elem)
    }
}

pub struct BitwiseNot;

impl<D: Dialect> Unary<D> for BitwiseNot {
    fn format_scalar<Input>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out_elem: Elem<D>,
    ) -> std::fmt::Result
    where
        Input: Component<D>,
    {
        // Bitwise negation may widen, so use explicit cast
        write!(f, "{out_elem}(~{input})")
    }
}

pub struct Not;

impl<D: Dialect> Unary<D> for Not {
    fn format_scalar<Input>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        _out_elem: Elem<D>,
    ) -> std::fmt::Result
    where
        Input: Component<D>,
    {
        write!(f, "!{input}")
    }
}

pub struct Assign;

impl<D: Dialect> Unary<D> for Assign {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let item = out.item();
        let item = item.value_ty();

        if item.vectorization() == 1 || input.item().value_ty() == item {
            write!(f, "{} = ", out.fmt_left())?;
            Self::format_scalar(f, *input, *item.elem())?;
            f.write_str(";\n")
        } else {
            Self::unroll_vec(f, input, out, *item.elem(), item.vectorization())
        }
    }

    fn format_scalar<Input>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result
    where
        Input: Component<D>,
    {
        // Cast only when necessary.
        if elem != input.elem() {
            match elem {
                Elem::TF32 => write!(f, "nvcuda::wmma::__float_to_tf32({input})"),
                // A direct construction between the two half types is
                // ambiguous on HIP — `__hip_bfloat16` exposes a dozen implicit
                // conversion operators, so `__half(bf16)` (and the reverse)
                // fails to compile. Route half<->half through `float`, which
                // is a single unambiguous conversion on every dialect.
                elem if is_half(elem) && is_half(input.elem()) => {
                    write!(f, "{elem}(float({input}))")
                }
                elem => write!(f, "{elem}({input})"),
            }
        } else {
            write!(f, "{input}")
        }
    }

    fn can_optimize() -> bool {
        false
    }
}

/// The two scalar half-precision float types, whose direct
/// interconstruction is ambiguous on HIP.
fn is_half<D: Dialect>(elem: Elem<D>) -> bool {
    matches!(elem, Elem::F16 | Elem::BF16)
}

fn elem_function_name<D: Dialect>(base_name: &'static str, elem: Elem<D>) -> String {
    // Math functions prefix (no leading underscores)
    let prefix = match elem {
        Elem::F16 | Elem::BF16 => D::compile_instruction_half_function_name_prefix(),
        Elem::F16x2 | Elem::BF16x2 => D::compile_instruction_half2_function_name_prefix(),
        _ => "",
    };
    if prefix.is_empty() {
        base_name.to_string()
    } else if prefix == "h" || prefix == "h2" {
        format!("__{prefix}{base_name}")
    } else {
        panic!("Unknown prefix '{prefix}'");
    }
}

// `isnan` / `isinf` are defined for cuda/hip/metal with same prefixes for half/bf16 on cuda/hip
pub struct IsNan;

impl<D: Dialect> Unary<D> for IsNan {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        _elem: Elem<D>,
    ) -> std::fmt::Result {
        // Format unary function name based on *input* elem dtype
        let elem = input.elem();
        write!(f, "{}({input})", elem_function_name("isnan", elem))
    }

    fn can_optimize() -> bool {
        true
    }
}

pub struct IsInf;

impl<D: Dialect> Unary<D> for IsInf {
    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        _elem: Elem<D>,
    ) -> std::fmt::Result {
        // Format unary function name based on *input* elem dtype
        let elem = input.elem();
        write!(f, "{}({input})", elem_function_name("isinf", elem))
    }

    fn can_optimize() -> bool {
        true
    }
}
