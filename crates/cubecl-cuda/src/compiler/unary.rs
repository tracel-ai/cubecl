use super::{Component, Elem, Variable};
use std::fmt::Display;

pub trait Unary {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let item = out.item();

        Self::unroll_vec(f, input, out, item.elem, item.vectorization)
    }

    fn format_scalar<Input, Out>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out: Out,
        elem: Elem,
    ) -> std::fmt::Result
    where
        Input: Component,
        Out: Component;

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

        for i in 0..index {
            let inputi = input.index(i);
            let outi = out.index(i);

            Self::format_scalar(f, inputi, outi, elem)?;
        }

        Ok(())
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
    fn format_unary<Input: Display, Output: Display>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out: Output,
        elem: Elem,
    ) -> std::fmt::Result {
        writeln!(f, "{out} = {}({input});", Self::function_name(elem))
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
            fn format_scalar<Input: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                input: Input,
                out: Out,
                elem: Elem,
            ) -> std::fmt::Result {
                Self::format_unary(f, input, out, elem)
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
    fn format_scalar<Input, Out>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out: Out,
        _elem: Elem,
    ) -> std::fmt::Result
    where
        Input: Component,
        Out: Component,
    {
        writeln!(f, "{out} = !{input};")
    }
}

pub struct Assign;

impl Unary for Assign {
    fn format_scalar<Input, Out>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out: Out,
        elem: Elem,
    ) -> std::fmt::Result
    where
        Input: Component,
        Out: Component,
    {
        // Cast only when necessary.
        if elem != input.elem() {
            writeln!(f, "{out} = {elem}({input});")
        } else {
            writeln!(f, "{out} = {input};")
        }
    }
}
