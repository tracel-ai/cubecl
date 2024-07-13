use super::{Component, Elem, InstructionSettings, Item, Variable};
use std::fmt::Display;

pub trait Unary {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let item = out.item();
        let settings = Self::settings(*item.elem());

        match item {
            Item::Vec4(elem) => {
                if settings.native_vec4 {
                    Self::format_native_vec4(f, input, out, elem)
                } else {
                    Self::unroll_vec4(f, input, out, elem)
                }
            }
            Item::Vec3(elem) => {
                if settings.native_vec3 {
                    Self::format_native_vec3(f, input, out, elem)
                } else {
                    Self::unroll_vec3(f, input, out, elem)
                }
            }
            Item::Vec2(elem) => {
                if settings.native_vec2 {
                    Self::format_native_vec2(f, input, out, elem)
                } else {
                    Self::unroll_vec2(f, input, out, elem)
                }
            }
            Item::Scalar(elem) => Self::format_scalar(f, *input, *out, elem),
        }
    }

    fn settings(_elem: Elem) -> InstructionSettings {
        InstructionSettings::default()
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

    fn format_native_vec4(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::format_scalar(f, *input, *out, elem)
    }

    fn format_native_vec3(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::format_scalar(f, *input, *out, elem)
    }

    fn format_native_vec2(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::format_scalar(f, *input, *out, elem)
    }

    fn unroll_vec2(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::unroll_vec(f, input, out, elem, 2)
    }

    fn unroll_vec3(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::unroll_vec(f, input, out, elem, 3)
    }

    fn unroll_vec4(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::unroll_vec(f, input, out, elem, 4)
    }

    fn unroll_vec(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
        index: usize,
    ) -> std::fmt::Result {
        for i in 0..index {
            let inputi = input.index(i);
            let outi = out.index(i);

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
