use super::{Component, Dialect, Elem, FmtLeft, Variable};
use std::fmt::Display;

pub trait Unary<D: Dialect> {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out_item = out.item();

        if out_item.vectorization == 1 {
            write!(f, "{} = ", out.fmt_left())?;
            Self::format_scalar(f, *input, out_item.elem)?;
            f.write_str(";\n")
        } else {
            Self::unroll_vec(f, input, out, out_item.elem, out_item.vectorization)
        }
    }

    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out_elem: Elem<D>,
    ) -> std::fmt::Result;

    fn unroll_vec(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<D>,
        out: &Variable<D>,
        out_elem: Elem<D>,
        index: usize,
    ) -> std::fmt::Result {
        let mut write_op = |index, out_elem, input: &Variable<D>, out: &Variable<D>| {
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
            let optimized = Variable::optimized_args([*input, *out]);
            let [input, out_optimized] = optimized.args;

            let item_out_original = out.item();
            let item_out_optimized = out_optimized.item();

            let (index, out_elem) = match optimized.optimization_factor {
                Some(factor) => (index / factor, out_optimized.elem()),
                None => (index, out_elem),
            };

            if item_out_original != item_out_optimized {
                let out_tmp = Variable::tmp(item_out_optimized);

                write_op(index, out_elem, &input, &out_tmp)?;
                let qualifier = out.const_qualifier();
                let addr_space = D::address_space_for_variable(out);
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
            format!("{prefix}{}", Self::base_function_name())
        } else {
            Self::base_function_name().into()
        }
    }
    fn format_unary<Input: Display>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        if Self::half_support() {
            write!(f, "{}({input})", Self::function_name(elem))
        } else {
            match elem {
                Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                    write!(f, "{}({}(float({input})))", elem, Self::function_name(elem))
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
function!(Cos, "cos");
function!(Sin, "sin");
function!(Sqrt, "sqrt");
function!(Exp, "exp");
function!(Ceil, "ceil");
function!(Floor, "floor");
function!(Round, "rint");

function!(Erf, "erf", false);
function!(Abs, "abs", false);

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
        _out_elem: Elem<D>,
    ) -> std::fmt::Result
    where
        Input: Component<D>,
    {
        write!(f, "~{input}")
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
        input: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let item = out.item();

        if item.vectorization == 1 || input.item() == item {
            write!(f, "{} = ", out.fmt_left())?;
            Self::format_scalar(f, *input, item.elem)?;
            f.write_str(";\n")
        } else {
            Self::unroll_vec(f, input, out, item.elem, item.vectorization)
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
                elem => write!(f, "{elem}({input})"),
            }
        } else {
            write!(f, "{input}")
        }
    }
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
