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
        if index == 1 {
            return Self::format_scalar(f, *input, *out, elem);
        }

        // let input_optimized = input.optimized();
        // let output_optimized = out.optimized();
        // let optimized = input_optimized.elem() == output_optimized.elem();

        // let (input, out) = if optimized {
        //     (input_optimized, output_optimized)
        // } else {
        //     (*input, *out)
        // };

        for i in 0..index {
            let inputi = input.index(i, false);
            let outi = out.index(i, false);

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
                _elem: Elem,
            ) -> std::fmt::Result {
                f.write_fmt(format_args!("{out} = {}({input});\n", $func))
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
