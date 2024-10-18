use super::{Component, Elem, FmtLeft, Variable};
use std::fmt::Display;

pub trait Unary {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
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

    fn format_scalar<Input: Component>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem,
    ) -> std::fmt::Result;

    fn unroll_vec(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
        index: usize,
    ) -> std::fmt::Result {
        let optimized = Variable::optimized_args([*input, *out]);
        let [input, out] = optimized.args;
        let (index, elem) = match optimized.optimization_factor {
            Some(factor) => (index / factor, out.elem()),
            None => (index, elem),
        };

        let out_item = out.item();
        let out = out.fmt_left();
        writeln!(f, "{out} = {out_item}{{")?;

        for i in 0..index {
            let inputi = input.index(i);

            Self::format_scalar(f, inputi, elem)?;
            f.write_str(",")?;
        }

        f.write_str("};\n")
    }
}

pub trait FunctionFmt {
    fn base_function_name() -> &'static str;
    fn function_name(elem: Elem) -> String {
        match elem {
            Elem::F16 | Elem::BF16 => format!("h{}", Self::base_function_name()),
            Elem::F162 | Elem::BF162 => format!("h2{}", Self::base_function_name()),
            _ => Self::base_function_name().into(),
        }
    }
    fn format_unary<Input: Display>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        elem: Elem,
    ) -> std::fmt::Result {
        write!(f, "{}({input})", Self::function_name(elem))
    }
}

macro_rules! function {
    ($name:ident, $func:expr) => {
        pub struct $name;

        impl FunctionFmt for $name {
            fn base_function_name() -> &'static str {
                $func
            }
        }

        impl Unary for $name {
            fn format_scalar<Input: Display>(
                f: &mut std::fmt::Formatter<'_>,
                input: Input,
                elem: Elem,
            ) -> std::fmt::Result {
                Self::format_unary(f, input, elem)
            }
        }
    };
}

function!(Abs, "abs");
function!(Log, "log");
function!(Log1p, "log1p");
function!(Cos, "cos");
function!(Sin, "sin");
function!(Tanh, "tanh");
function!(Sqrt, "sqrt");
function!(Exp, "exp");
function!(Erf, "erf");
function!(Ceil, "ceil");
function!(Floor, "floor");
function!(Round, "rint");

pub struct Not;

impl Unary for Not {
    fn format_scalar<Input>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        _elem: Elem,
    ) -> std::fmt::Result
    where
        Input: Component,
    {
        write!(f, "!{input}")
    }
}

pub struct Assign;

impl Unary for Assign {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
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
        elem: Elem,
    ) -> std::fmt::Result
    where
        Input: Component,
    {
        // Cast only when necessary.
        if elem != input.elem() {
            write!(f, "{elem}({input})")
        } else {
            write!(f, "{input}")
        }
    }
}
