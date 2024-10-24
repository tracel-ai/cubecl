use super::{Component, Dialect, Elem, FmtLeft, Variable};
use std::fmt::Display;

pub trait Unary<D: Dialect> {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let item = out.item();

        if item.vectorization == 1 {
            write!(f, "{} = ", out.fmt_left())?;
            Self::format_scalar(f, *input, item.elem)?;
            f.write_str(";\n")
        } else {
            Self::unroll_vec(f, input, out, item.elem, item.vectorization)
        }
    }

    fn format_scalar<Input: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result;

    fn unroll_vec(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<D>,
        out: &Variable<D>,
        elem: Elem<D>,
        index: usize,
    ) -> std::fmt::Result {
        let mut write_op = |index, elem, input: &Variable<D>, out: &Variable<D>| {
            let out_item = out.item();
            let out = out.fmt_left();
            writeln!(f, "{out} = {out_item}{{")?;

            for i in 0..index {
                let inputi = input.index(i);

                Self::format_scalar(f, inputi, elem)?;
                f.write_str(",")?;
            }

            f.write_str("};\n")
        };

        if Self::can_optimize() {
            let optimized = Variable::optimized_args([*input, *out]);
            let [input, out_optimized] = optimized.args;

            let item_out_original = out.item();
            let item_out_optimized = out_optimized.item();

            let (index, elem) = match optimized.optimization_factor {
                Some(factor) => (index / factor, out_optimized.elem()),
                None => (index, elem),
            };

            if item_out_original != item_out_optimized {
                let out_tmp = Variable::tmp(item_out_optimized);

                write_op(index, elem, &input, &out_tmp)?;

                writeln!(
                    f,
                    "{out} = reinterpret_cast<{item_out_original}&>({out_tmp});\n"
                )
            } else {
                write_op(index, elem, &input, &out_optimized)
            }
        } else {
            write_op(index, elem, input, out)
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
            match elem {
                Elem::F16 | Elem::BF16 => return format!("h{}", Self::base_function_name()),
                Elem::F162 | Elem::BF162 => return format!("h2{}", Self::base_function_name()),
                _ => (),
            };
        }

        Self::base_function_name().into()
    }
    fn format_unary<Input: Display>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        if Self::half_support() {
            return write!(f, "{}({input})", Self::function_name(elem));
        }

        match elem {
            Elem::F16 | Elem::F162 | Elem::BF16 | Elem::BF162 => {
                write!(f, "{}({}(float({input})))", elem, Self::function_name(elem))
            }
            _ => write!(f, "{}({input})", Self::function_name(elem)),
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
function!(Log1p, "log1p");
function!(Cos, "cos");
function!(Sin, "sin");
function!(Sqrt, "sqrt");
function!(Exp, "exp");
function!(Ceil, "ceil");
function!(Floor, "floor");
function!(Round, "rint");

function!(Tanh, "tanh", false);
function!(Erf, "erf", false);
function!(Abs, "abs", false);

pub struct Not;

impl<D: Dialect> Unary<D> for Not {
    fn format_scalar<Input>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        _elem: Elem<D>,
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
            write!(f, "{elem}({input})")
        } else {
            write!(f, "{input}")
        }
    }
}
