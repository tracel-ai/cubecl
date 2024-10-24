use crate::shared::FmtLeft;

use super::{Component, Dialect, Elem, Item, Variable};
use std::{
    fmt::{Display, Formatter},
    marker::PhantomData,
};

pub trait Binary<D: Dialect> {
    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
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
        item: Item<D>,
    ) -> std::fmt::Result
    where
        Lhs: Component<D>,
        Rhs: Component<D>;

    fn unroll_vec(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let optimized = Variable::optimized_args([*lhs, *rhs, *out]);
        let [lhs, rhs, out_optimized] = optimized.args;

        let item_out_original = out.item();
        let item_out_optimized = out_optimized.item();

        let index = match optimized.optimization_factor {
            Some(factor) => item_out_optimized.vectorization / factor,
            None => item_out_optimized.vectorization,
        };

        let mut write_op =
            |lhs: &Variable<D>, rhs: &Variable<D>, out: &Variable<D>, item_out: Item<D>| {
                let out = out.fmt_left();
                writeln!(f, "{out} = {item_out}{{")?;
                for i in 0..index {
                    let lhsi = lhs.index(i);
                    let rhsi = rhs.index(i);

                    Self::format_scalar(f, lhsi, rhsi, item_out)?;
                    f.write_str(", ")?;
                }

                f.write_str("};\n")
            };

        if item_out_original == item_out_optimized {
            write_op(&lhs, &rhs, out, item_out_optimized)
        } else {
            let out_tmp = Variable::tmp(item_out_optimized);

            write_op(&lhs, &rhs, &out_tmp, item_out_optimized)?;

            let out = out.fmt_left();

            writeln!(
                f,
                "{out} = reinterpret_cast<{item_out_original}&>({out_tmp});\n"
            )?;

            Ok(())
        }
    }
}

macro_rules! operator {
    ($name:ident, $op:expr) => {
        pub struct $name;

        impl<D: Dialect> Binary<D> for $name {
            fn format_scalar<Lhs: Display, Rhs: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                _item: Item<D>,
            ) -> std::fmt::Result {
                write!(f, "{lhs} {} {rhs}", $op)
            }
        }
    };
}

macro_rules! function {
    ($name:ident, $op:expr) => {
        pub struct $name;

        impl<D: Dialect> Binary<D> for $name {
            fn format_scalar<Lhs: Display, Rhs: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                _item: Item<D>,
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

impl<D: Dialect> Binary<D> for IndexAssign {
    fn format_scalar<Lhs, Rhs>(
        f: &mut Formatter<'_>,
        _lhs: Lhs,
        rhs: Rhs,
        item_out: Item<D>,
    ) -> std::fmt::Result
    where
        Lhs: Component<D>,
        Rhs: Component<D>,
    {
        let item_rhs = rhs.item();

        let format_vec = |f: &mut Formatter<'_>, cast: bool| {
            writeln!(f, "{item_out}{{")?;
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
        } else if rhs.is_const() && item_rhs.vectorization > 1 {
            // Reinterpret cast in case rhs is optimized
            write!(f, "reinterpret_cast<{item_out} const&>({rhs})")
        } else {
            write!(f, "{rhs}")
        }
    }

    fn unroll_vec(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
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
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
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

impl<D: Dialect> Binary<D> for Index {
    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
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
        item_out: Item<D>,
    ) -> std::fmt::Result
    where
        Lhs: Component<D>,
        Rhs: Component<D>,
    {
        let item_lhs = lhs.item();

        let format_vec = |f: &mut Formatter<'_>| {
            writeln!(f, "{item_out}{{")?;
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
            write!(f, "{lhs}[{rhs}]")
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
struct IndexVector<D: Dialect> {
    dialect: PhantomData<D>,
}

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
struct IndexAssignVector<D: Dialect> {
    dialect: PhantomData<D>,
}

impl<D: Dialect> IndexVector<D> {
    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
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

impl<D: Dialect> IndexAssignVector<D> {
    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
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
