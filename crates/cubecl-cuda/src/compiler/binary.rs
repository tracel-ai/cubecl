use super::{Component, Elem, Variable};
use std::fmt::Display;

pub trait Binary {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let item = out.item().de_optimized();
        Self::unroll_vec(f, lhs, rhs, out, item.elem, item.vectorization.into())
    }

    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Out,
        elem: Elem,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
        Out: Component;

    fn unroll_vec(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
        index: usize,
    ) -> core::fmt::Result {
        if index == 1 {
            return Self::format_scalar(f, *lhs, *rhs, *out, elem);
        }

        let optimized = Variable::optimized_args([*lhs, *rhs, *out]);
        let [lhs, rhs, out] = optimized.args;
        let (is_optimized, index) = match optimized.optimization_factor {
            Some(factor) => (true, index / factor),
            None => (false, index),
        };

        for i in 0..index {
            let lhsi = lhs.index(i, is_optimized);
            let rhsi = rhs.index(i, is_optimized);
            let outi = out.index(i, is_optimized);

            Self::format_scalar(f, lhsi, rhsi, outi, elem)?;
        }

        Ok(())
    }
}

macro_rules! operator {
    ($name:ident, $op:expr) => {
        operator!(
            $name,
            $op,
            InstructionSettings {
                native_vec4: false,
                native_vec3: false,
                native_vec2: false,
            }
        );
    };
    ($name:ident, $op:expr, $vectorization:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn format_scalar<Lhs: Display, Rhs: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                out: Out,
                _elem: Elem,
            ) -> std::fmt::Result {
                f.write_fmt(format_args!("{out} = {lhs} {} {rhs};\n", $op))
            }
        }
    };
}

macro_rules! function {
    ($name:ident, $op:expr) => {
        function!(
            $name,
            $op,
            InstructionSettings {
                native_vec4: false,
                native_vec3: false,
                native_vec2: true,
            }
        );
    };
    ($name:ident, $op:expr, $vectorization:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn format_scalar<Lhs: Display, Rhs: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                out: Out,
                _elem: Elem,
            ) -> std::fmt::Result {
                f.write_fmt(format_args!("{out} = {}({lhs}, {rhs});\n", $op))
            }
        }
    };
}

operator!(Add, "+");
operator!(Sub, "-");
operator!(Div, "/");
operator!(Mul, "*");
operator!(Modulo, "%");
operator!(Equal, "==");
operator!(NotEqual, "!=");
operator!(Lower, "<");
operator!(LowerEqual, "<=");
operator!(Greater, ">");
operator!(GreaterEqual, ">=");
operator!(ShiftLeft, "<<");
operator!(ShiftRight, ">>");
operator!(BitwiseAnd, "&");
operator!(BitwiseXor, "^");
operator!(Or, "||");
operator!(And, "&&");

function!(Powf, "powf");
function!(Max, "max");
function!(Min, "min");

pub struct IndexAssign;
pub struct Index;

impl Binary for IndexAssign {
    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Out,
        elem: Elem,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
        Out: Component,
    {
        let elem_rhs = rhs.elem();
        // Cast only when necessary.
        if elem != elem_rhs {
            // if let Elem::Bool = elem_rhs {
            //     match rhs.item() {
            //         Item::Vec4(_) => {
            //             f.write_fmt(format_args!("{out}[{lhs}] = make_uint4({elem}({rhs}.x), {elem}({rhs}.y), {elem}({rhs}.z), {elem}({rhs}.w));\n"))
            //         },
            //         Item::Vec3(_) => todo!(),
            //         Item::Vec2(_) => todo!(),
            //         Item::Scalar(_) => todo!(),
            //     }
            // } else {
            // }
            f.write_fmt(format_args!("{out}[{lhs}] = {elem}({rhs});\n"))
        } else {
            f.write_fmt(format_args!("{out}[{lhs}] = {rhs};\n"))
        }
    }

    fn unroll_vec(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
        index: usize,
    ) -> std::fmt::Result {
        if index == 1 {
            return Self::format_scalar(f, *lhs, *rhs, *out, elem);
        }

        for i in 0..index {
            let lhsi = lhs.index(i, false);
            let rhsi = rhs.index(i, false);
            Self::format_scalar(f, lhsi, rhsi, *out, elem)?;
        }

        Ok(())
    }

    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        if let Variable::Local {
            id: _,
            item: _,
            depth: _,
        } = out
        {
            return IndexAssignVector::format(f, lhs, rhs, out);
        };

        let elem = out.elem();
        let item = lhs.item();

        Self::unroll_vec(f, lhs, rhs, out, elem, item.vectorization.into())
    }
}

impl Binary for Index {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        if let Variable::Local {
            id: _,
            item: _,
            depth: _,
        } = lhs
        {
            return IndexVector::format(f, lhs, rhs, out);
        }

        Self::format_scalar(f, *lhs, *rhs, *out, out.elem())
    }

    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Out,
        _elem: Elem,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
        Out: Component,
    {
        f.write_fmt(format_args!("{out} = {lhs}[{rhs}];\n"))
    }
}

/// The goal is to support indexing of vectorized types.
///
/// # Examples
///
/// ```c
/// float4 rhs;
/// float item = var[0]; // We want that.
/// float item = var.x; // So we compile to that.
/// ```
struct IndexVector;

/// The goal is to support indexing of vectorized types.
///
/// # Examples
///
/// ```c
/// float4 var;
///
/// var[0] = 1.0; // We want that.
/// var.x = 1.0;  // So we compile to that.
/// ```
struct IndexAssignVector;

impl IndexVector {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let index = match rhs {
            Variable::ConstantScalar(value, _elem) => value.as_usize(),
            _ => {
                let elem = out.elem();
                return f.write_fmt(format_args!("{out} = *(({elem}*)&{lhs} + {rhs});\n"));
            }
        };

        let out = out.index(index, false);
        let lhs = lhs.index(index, false);

        f.write_fmt(format_args!("{out} = {lhs};\n"))
    }
}

impl IndexAssignVector {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let index = match lhs {
            Variable::ConstantScalar(value, _) => value.as_usize(),
            _ => {
                let elem = out.elem();
                return f.write_fmt(format_args!("*(({elem}*)&{out} + {lhs}) = {rhs};\n"));
            }
        };

        let out = out.index(index, false);
        let rhs = rhs.index(index, false);

        f.write_fmt(format_args!("{out} = {rhs};\n"))
    }
}
