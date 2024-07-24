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
        let (is_optimized, index, elem) = match optimized.optimization_factor {
            Some(factor) => (true, index / factor, out.elem()),
            None => (false, index, elem),
        };

        for i in 0..index {
            let inputi = input.index(i, is_optimized);
            let outi = out.index(i, is_optimized);

            Self::format_scalar(f, inputi, outi, elem)?;
        }

        Ok(())
    }
}

macro_rules! function {
    ($name:ident, $func:expr) => {
        pub struct $name;

        impl Unary for $name {
            fn format_scalar<Input: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                input: Input,
                out: Out,
                elem: Elem,
            ) -> std::fmt::Result {
                match elem {
                    Elem::F16 => f.write_fmt(format_args!("{out} = h{}({input});\n", $func)),
                    Elem::F162 => f.write_fmt(format_args!("{out} = h2{}({input});\n", $func)),
                    Elem::BF16 => f.write_fmt(format_args!("{out} = h{}({input});\n", $func)),
                    Elem::BF162 => f.write_fmt(format_args!("{out} = h2{}({input});\n", $func)),
                    Elem::F32 => f.write_fmt(format_args!("{out} = __{}f({input});\n", $func)),
                    _ => f.write_fmt(format_args!("{out} = {}({input});\n", $func)),
                }
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
function!(Erf, "erff");
function!(Ceil, "ceil");
function!(Floor, "floor");

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
        f.write_fmt(format_args!("{out} = !{input};\n"))
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
            f.write_fmt(format_args!("{out} = {elem}({input});\n"))
        } else {
            f.write_fmt(format_args!("{out} = {input};\n"))
        }
    }
}
