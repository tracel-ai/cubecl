use crate::compiler::FmtLeft;

use super::{Component, Elem, Item, Variable};
use std::fmt::{Display, Formatter};

pub trait Binary {
    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let out_item = out.item();
        if out.item().vectorization == 1 {
            let out = out.fmt_left();
            write!(f, "{out} = ")?;
            Self::format_scalar(f, *lhs, *rhs, out_item)?;
            f.write_str(";\n")
        } else {
            Self::unroll_vec(f, lhs, rhs, out)
        }
    }

    fn format_scalar<Lhs, Rhs>(
        f: &mut Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component;

    fn unroll_vec(
        f: &mut Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> core::fmt::Result {
        let item_out = out.item();

        let optimized = Variable::optimized_args([*lhs, *rhs, *out]);
        let [lhs, rhs, out] = optimized.args;
        let index = match optimized.optimization_factor {
            Some(factor) => item_out.vectorization / factor,
            None => item_out.vectorization,
        };

        let out = out.fmt_left();
        writeln!(f, "{out} = {{")?;
        for i in 0..index {
            let lhsi = lhs.index(i);
            let rhsi = rhs.index(i);

            Self::format_scalar(f, lhsi, rhsi, item_out)?;
            f.write_str(", ")?;
        }

        f.write_str("};\n")
    }
}

macro_rules! operator {
    ($name:ident, $op:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn format_scalar<Lhs: Display, Rhs: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                _item: Item,
            ) -> std::fmt::Result {
                write!(f, "{lhs} {} {rhs}", $op)
            }
        }
    };
}

macro_rules! function {
    ($name:ident, $op:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn format_scalar<Lhs: Display, Rhs: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                _item: Item,
            ) -> std::fmt::Result {
                write!(f, "{}({lhs}, {rhs})", $op)
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
operator!(BitwiseOr, "|");
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
    fn format_scalar<Lhs, Rhs>(
        f: &mut Formatter<'_>,
        _lhs: Lhs,
        rhs: Rhs,
        item_out: Item,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
    {
        let item_rhs = rhs.item();

        let format_vec = |f: &mut Formatter<'_>, cast: bool| {
            f.write_str("{\n")?;
            for i in 0..item_out.vectorization {
                if cast {
                    writeln!(f, "{}({}),", item_out.elem, rhs.index(i))?;
                } else {
                    writeln!(f, "{},", rhs.index(i))?;
                }
            }
            f.write_str("}")?;

            Ok(())
        };

        if item_out.vectorization != item_rhs.vectorization {
            format_vec(f, item_out != item_rhs)
        } else if item_out.elem != item_rhs.elem {
            if item_out.vectorization > 1 {
                format_vec(f, true)?;
            } else {
                write!(f, "{}({rhs})", item_out.elem)?;
            }
            Ok(())
        } else {
            write!(f, "{rhs}")
        }
    }

    fn unroll_vec(
        f: &mut Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let item_lhs = lhs.item();
        let out_item = out.item();
        let out = out.fmt_left();

        for i in 0..item_lhs.vectorization {
            let lhsi = lhs.index(i);
            let rhsi = rhs.index(i);
            write!(f, "{out}[{lhs}] = ")?;
            Self::format_scalar(f, lhsi, rhsi, out_item)?;
            f.write_str(";\n")?;
        }

        Ok(())
    }

    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        if matches!(out, Variable::Local { .. } | Variable::ConstLocal { .. }) {
            return IndexAssignVector::format(f, lhs, rhs, out);
        };

        let out_item = out.item();

        if lhs.item().vectorization == 1 {
            write!(f, "{}[{lhs}] = ", out.fmt_left())?;
            Self::format_scalar(f, *lhs, *rhs, out_item)?;
            f.write_str(";\n")
        } else {
            Self::unroll_vec(f, lhs, rhs, out)
        }
    }
}

impl Binary for Index {
    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        if matches!(lhs, Variable::Local { .. } | Variable::ConstLocal { .. }) {
            return IndexVector::format(f, lhs, rhs, out);
        }

        let item_out = out.item();
        if let Elem::Atomic(inner) = item_out.elem {
            write!(f, "{inner}* {out} = &{lhs}[{rhs}];")
        } else {
            let out = out.fmt_left();
            write!(f, "{out} = ")?;
            Self::format_scalar(f, *lhs, *rhs, item_out)?;
            f.write_str(";\n")
        }
    }

    fn format_scalar<Lhs, Rhs>(
        f: &mut Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item_out: Item,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
    {
        let item_lhs = lhs.item();

        let format_vec = |f: &mut Formatter<'_>| {
            f.write_str("{\n")?;
            for i in 0..item_out.vectorization {
                write!(f, "{}({lhs}[{rhs}].i_{i}),", item_out.elem)?;
            }
            f.write_str("}")?;

            Ok(())
        };

        if item_out.elem != item_lhs.elem {
            if item_out.vectorization > 1 {
                format_vec(f)
            } else {
                write!(f, "{}({lhs}[{rhs}])", item_out.elem)
            }
        } else {
            write!(f, "{lhs}[{rhs}];")
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
                let out = out.fmt_left();
                return writeln!(f, "{out} = *(({elem}*)&{lhs} + {rhs});");
            }
        };

        let out = out.index(index);
        let lhs = lhs.index(index);

        let out = out.fmt_left();
        writeln!(f, "{out} = {lhs};")
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
                return writeln!(f, "*(({elem}*)&{out} + {lhs}) = {rhs};");
            }
        };

        let out = out.index(index);
        let rhs = rhs.index(index);

        writeln!(f, "{out} = {rhs};")
    }
}
