use super::{Component, Elem, Variable};
use std::fmt::{Display, Formatter};

pub trait Binary {
    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        Self::unroll_vec(f, lhs, rhs, out)
    }

    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Out,
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
    ) -> core::fmt::Result {
        let item_out = out.item();

        if item_out.vectorization == 1 {
            return Self::format_scalar(f, *lhs, *rhs, *out);
        }

        let optimized = Variable::optimized_args([*lhs, *rhs, *out]);
        let [lhs, rhs, out] = optimized.args;
        let index = match optimized.optimization_factor {
            Some(factor) => item_out.vectorization / factor,
            None => item_out.vectorization,
        };

        for i in 0..index {
            let lhsi = lhs.index(i);
            let rhsi = rhs.index(i);
            let outi = out.index(i);

            Self::format_scalar(f, lhsi, rhsi, outi)?;
        }

        Ok(())
    }
}

macro_rules! operator {
    ($name:ident, $op:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn format_scalar<Lhs: Display, Rhs: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                out: Out,
            ) -> std::fmt::Result {
                write!(f, "{out} = {lhs} {} {rhs};\n", $op)
            }
        }
    };
}

macro_rules! function {
    ($name:ident, $op:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn format_scalar<Lhs: Display, Rhs: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                out: Out,
            ) -> std::fmt::Result {
                write!(f, "{out} = {}({lhs}, {rhs});\n", $op)
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
    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Out,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
        Out: Component,
    {
        let item_out = out.item();
        let item_rhs = rhs.item();

        let format_vec = |f: &mut Formatter<'_>, cast: bool| {
            f.write_str("{\n")?;
            let var = "broadcasted";
            writeln!(f, "{item_out} {var};")?;
            for i in 0..item_out.vectorization {
                if cast {
                    writeln!(f, "{var}.i_{i} = {}({});", item_out.elem, rhs.index(i))?;
                } else {
                    writeln!(f, "{var}.i_{i} = {};", rhs.index(i))?;
                }
            }
            writeln!(f, "{out}[{lhs}] = {var};")?;
            f.write_str("}")?;

            Ok(())
        };

        if item_out.vectorization != item_rhs.vectorization {
            format_vec(f, item_out != item_rhs)
        } else if item_out.elem != item_rhs.elem {
            if item_out.vectorization > 1 {
                format_vec(f, true)?;
            } else {
                writeln!(f, "{out}[{lhs}] = {}({rhs});", item_out.elem)?;
            }
            Ok(())
        } else {
            writeln!(f, "{out}[{lhs}] = {rhs};")
        }
    }

    fn unroll_vec(
        f: &mut Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let item_lhs = lhs.item();

        if item_lhs.vectorization == 1 {
            return Self::format_scalar(f, *lhs, *rhs, *out);
        }

        for i in 0..item_lhs.vectorization {
            let lhsi = lhs.index(i);
            let rhsi = rhs.index(i);
            Self::format_scalar(f, lhsi, rhsi, *out)?;
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

        Self::unroll_vec(f, lhs, rhs, out)
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

        Self::format_scalar(f, *lhs, *rhs, *out)
    }

    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Out,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
        Out: Component,
    {
        let item_out = out.item();
        let item_lhs = lhs.item();

        let format_vec = |f: &mut Formatter<'_>| {
            f.write_str("{\n")?;
            let var = "broadcasted";
            writeln!(f, "{item_out} {var};")?;
            for i in 0..item_out.vectorization {
                writeln!(f, "{var}.i_{i} = {}({lhs}[{rhs}].i_{i});", item_out.elem)?;
            }
            writeln!(f, "{out} = {var};")?;
            f.write_str("}")?;

            Ok(())
        };

        if item_out.elem != item_lhs.elem {
            if item_out.vectorization > 1 {
                format_vec(f)?;
            } else {
                writeln!(f, "{out} = {}({lhs}[{rhs}]);", item_out.elem)?;
            }
            Ok(())
        } else if let Elem::Atomic(inner) = item_out.elem {
            writeln!(f, "{inner}* {out} = &{lhs}[{rhs}];")
        } else {
            writeln!(f, "{out} = {lhs}[{rhs}];")
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
                return writeln!(f, "{out} = *(({elem}*)&{lhs} + {rhs});");
            }
        };

        let out = out.index(index);
        let lhs = lhs.index(index);

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
