use super::{Component, Elem, Variable};
use std::fmt::{Display, Formatter};

pub trait Binary {
    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let item = out.item().de_optimized();
        Self::unroll_vec(f, lhs, rhs, out, item.elem, item.vectorization)
    }

    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut Formatter<'_>,
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
        f: &mut Formatter<'_>,
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
        f: &mut Formatter<'_>,
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
        let item_out = out.item();
        let item_rhs = rhs.item();

        let format_vec = |f: &mut Formatter<'_>, cast: bool| {
            let is_vec_native = item_out.is_vec_native();
            f.write_str("{\n")?;
            let var = "broadcasted";
            f.write_fmt(format_args!("{item_out} {var};\n"))?;
            for i in 0..item_out.vectorization {
                if is_vec_native {
                    let char = match i {
                        0 => 'x',
                        1 => 'y',
                        2 => 'z',
                        3 => 'w',
                        _ => panic!("Invalid"),
                    };
                    if cast {
                        f.write_fmt(format_args!("{var}.{char} = ({}){};\n", elem, rhs.index(i)))?;
                    } else {
                        f.write_fmt(format_args!("{var}.{char} = {};\n", rhs.index(i)))?;
                    }
                } else {
                    if cast {
                        f.write_fmt(format_args!("{var}.i_{i} = ({}){};\n", elem, rhs.index(i)))?;
                    } else {
                        f.write_fmt(format_args!("{var}.i_{i} = {};\n", rhs.index(i)))?;
                    }
                }
            }
            f.write_fmt(format_args!("{out}[{lhs}] = {var};\n"))?;
            f.write_str("}")?;

            Ok(())
        };

        if item_out.vectorization != item_rhs.vectorization {
            format_vec(f, item_out != item_rhs)
        } else if elem != item_rhs.elem {
            if item_out.vectorization > 1 {
                format_vec(f, true)?;
            } else {
                f.write_fmt(format_args!("{out}[{lhs}] = ({elem}){rhs};\n"))?;
            }
            Ok(())
        } else {
            f.write_fmt(format_args!("{out}[{lhs}] = {rhs};\n"))
        }
    }

    fn unroll_vec(
        f: &mut Formatter<'_>,
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
            let lhsi = lhs.index(i, lhs.item().is_optimized());
            let rhsi = rhs.index(i, rhs.item().is_optimized());
            Self::format_scalar(f, lhsi, rhsi, *out, elem)?;
        }

        Ok(())
    }

    fn format(
        f: &mut Formatter<'_>,
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

        Self::unroll_vec(f, lhs, rhs, out, elem, item.vectorization)
    }
}

impl Binary for Index {
    fn format(
        f: &mut Formatter<'_>,
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
        f: &mut Formatter<'_>,
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
        let item_out = out.item();
        let item_lhs = lhs.item();

        let format_vec = |f: &mut Formatter<'_>| {
            let is_vec_native = item_out.is_vec_native();
            f.write_str("{\n")?;
            let var = "broadcasted";
            f.write_fmt(format_args!("{item_out} {var};\n"))?;
            for i in 0..item_out.vectorization {
                if is_vec_native {
                    let char = match i {
                        0 => 'x',
                        1 => 'y',
                        2 => 'z',
                        3 => 'w',
                        _ => panic!("Invalid"),
                    };
                    f.write_fmt(format_args!("{var}.{char} = {elem}({lhs}[{rhs}].i_{i});\n",))?;
                } else {
                    f.write_fmt(format_args!("{var}.i_{i} = {elem}({lhs}[{rhs}].i_{i});\n",))?;
                }
            }
            f.write_fmt(format_args!("{out} = {var};\n"))?;
            f.write_str("}")?;

            Ok(())
        };

        if elem != item_lhs.elem {
            if item_out.vectorization > 1 {
                format_vec(f)?;
            } else {
                f.write_fmt(format_args!("{out} = ({elem}){lhs}[{rhs}];\n"))?;
            }
            Ok(())
        } else {
            f.write_fmt(format_args!("{out} = {lhs}[{rhs}];\n"))
        }
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
        f: &mut Formatter<'_>,
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
        f: &mut Formatter<'_>,
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
