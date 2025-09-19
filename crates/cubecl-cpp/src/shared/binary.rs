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
            Some(factor) => item_out_original.vectorization / factor,
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
            let addr_space = D::address_space_for_variable(out);
            let out = out.fmt_left();

            writeln!(
                f,
                "{out} = reinterpret_cast<{addr_space}{item_out_original}&>({out_tmp});\n"
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
                out_item: Item<D>,
            ) -> std::fmt::Result {
                let out_elem = out_item.elem();
                match out_elem {
                    // prevent auto-promotion rules to kick-in in order to stay in the same type
                    // this is because of fusion and vectorization that can do elemwise operations on vectorized type,
                    // the resulting elements need to be of the same type.
                    Elem::<D>::I16 | Elem::<D>::U16 | Elem::<D>::I8 | Elem::<D>::U8 => {
                        write!(f, "{out_elem}({lhs} {} {rhs})", $op)
                    }
                    _ => write!(f, "{lhs} {} {rhs}", $op),
                }
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

pub struct HiMul;

impl<D: Dialect> Binary<D> for HiMul {
    // Powf doesn't support half and no half equivalent exists
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Item<D>,
    ) -> std::fmt::Result {
        let out_elem = out.elem;
        match out_elem {
            Elem::I32 => write!(f, "__mulhi({lhs}, {rhs})"),
            Elem::U32 => write!(f, "__umulhi({lhs}, {rhs})"),
            Elem::I64 => write!(f, "__mul64hi({lhs}, {rhs})"),
            Elem::U64 => write!(f, "__umul64hi({lhs}, {rhs})"),
            _ => unimplemented!("HiMul only supports 32 and 64 bit ints"),
        }
    }

    // Powf doesn't support half and no half equivalent exists
    fn unroll_vec(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let item_out = out.item();
        let index = out.item().vectorization;

        let out = out.fmt_left();
        writeln!(f, "{out} = {item_out}{{")?;
        for i in 0..index {
            let lhsi = lhs.index(i);
            let rhsi = rhs.index(i);

            Self::format_scalar(f, lhsi, rhsi, item_out)?;
            f.write_str(", ")?;
        }

        f.write_str("};\n")
    }
}

pub struct SaturatingAdd;

impl<D: Dialect> Binary<D> for SaturatingAdd {
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Item<D>,
    ) -> std::fmt::Result {
        D::compile_saturating_add(f, lhs, rhs, out)
    }
}

pub struct SaturatingSub;

impl<D: Dialect> Binary<D> for SaturatingSub {
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Item<D>,
    ) -> std::fmt::Result {
        D::compile_saturating_sub(f, lhs, rhs, out)
    }
}

pub struct Powf;

impl<D: Dialect> Binary<D> for Powf {
    // Powf doesn't support half and no half equivalent exists
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item<D>,
    ) -> std::fmt::Result {
        let elem = item.elem;
        let lhs = lhs.to_string();
        let rhs = rhs.to_string();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                let lhs = format!("float({lhs})");
                let rhs = format!("float({rhs})");
                write!(f, "{elem}(")?;
                D::compile_instruction_powf(f, &lhs, &rhs, Elem::F32)?;
                write!(f, ")")
            }
            _ => D::compile_instruction_powf(f, &lhs, &rhs, elem),
        }
    }

    // Powf doesn't support half and no half equivalent exists
    fn unroll_vec(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let item_out = out.item();
        let index = out.item().vectorization;

        let out = out.fmt_left();
        writeln!(f, "{out} = {item_out}{{")?;
        for i in 0..index {
            let lhsi = lhs.index(i);
            let rhsi = rhs.index(i);

            Self::format_scalar(f, lhsi, rhsi, item_out)?;
            f.write_str(", ")?;
        }

        f.write_str("};\n")
    }
}

pub struct Powi;

impl<D: Dialect> Binary<D> for Powi {
    // Powi doesn't support half and no half equivalent exists
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item<D>,
    ) -> std::fmt::Result {
        let elem = item.elem;
        let lhs = lhs.to_string();
        let rhs = rhs.to_string();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                let lhs = format!("float({lhs})");

                write!(f, "{elem}(")?;
                D::compile_instruction_powf(f, &lhs, &rhs, Elem::F32)?;
                write!(f, ")")
            }
            _ => D::compile_instruction_powf(f, &lhs, &rhs, elem),
        }
    }

    // Powi doesn't support half and no half equivalent exists
    fn unroll_vec(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let item_out = out.item();
        let index = out.item().vectorization;

        let out = out.fmt_left();
        writeln!(f, "{out} = {item_out}{{")?;
        for i in 0..index {
            let lhsi = lhs.index(i);
            let rhsi = rhs.index(i);

            Self::format_scalar(f, lhsi, rhsi, item_out)?;
            f.write_str(", ")?;
        }

        f.write_str("};\n")
    }
}

pub struct Max;

impl<D: Dialect> Binary<D> for Max {
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_max_function_name(f, item)?;
        write!(f, "({lhs}, {rhs})")
    }
}

pub struct Min;

impl<D: Dialect> Binary<D> for Min {
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item<D>,
    ) -> std::fmt::Result {
        D::compile_instruction_min_function_name(f, item)?;
        write!(f, "({lhs}, {rhs})")
    }
}

pub struct IndexAssign;
pub struct Index;

impl IndexAssign {
    pub fn format<D: Dialect>(
        f: &mut Formatter<'_>,
        index: &Variable<D>,
        value: &Variable<D>,
        out_list: &Variable<D>,
        line_size: u32,
    ) -> std::fmt::Result {
        if matches!(
            out_list,
            Variable::LocalMut { .. } | Variable::LocalConst { .. }
        ) {
            return IndexAssignVector::format(f, index, value, out_list);
        };

        if line_size > 0 {
            let mut item = out_list.item();
            item.vectorization = line_size as usize;
            let addr_space = D::address_space_for_variable(out_list);
            let qualifier = out_list.const_qualifier();
            let tmp = Variable::tmp_declared(item);

            writeln!(
                f,
                "{qualifier} {addr_space}{item} *{tmp} = reinterpret_cast<{qualifier} {item}*>({out_list});"
            )?;

            return IndexAssign::format(f, index, value, &tmp, 0);
        }

        let out_item = out_list.item();

        if index.item().vectorization == 1 {
            write!(f, "{}[{index}] = ", out_list.fmt_left())?;
            Self::format_scalar(f, *index, *value, out_item)?;
            f.write_str(";\n")
        } else {
            Self::unroll_vec(f, index, value, out_list)
        }
    }
    fn format_scalar<D: Dialect, Lhs, Rhs>(
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
            write!(f, "reinterpret_cast<")?;
            D::compile_local_memory_qualifier(f)?;
            write!(f, " {item_out} const&>({rhs})")
        } else {
            write!(f, "{rhs}")
        }
    }

    fn unroll_vec<D: Dialect>(
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
}

impl Index {
    pub(crate) fn format<D: Dialect>(
        f: &mut Formatter<'_>,
        list: &Variable<D>,
        index: &Variable<D>,
        out: &Variable<D>,
        line_size: u32,
    ) -> std::fmt::Result {
        if matches!(
            list,
            Variable::LocalMut { .. } | Variable::LocalConst { .. } | Variable::ConstantScalar(..)
        ) {
            return IndexVector::format(f, list, index, out);
        }

        if line_size > 0 {
            let mut item = list.item();
            item.vectorization = line_size as usize;
            let addr_space = D::address_space_for_variable(list);
            let qualifier = list.const_qualifier();
            let tmp = Variable::tmp_declared(item);

            writeln!(
                f,
                "{qualifier} {addr_space}{item} *{tmp} = reinterpret_cast<{qualifier} {item}*>({list});"
            )?;

            return Index::format(f, &tmp, index, out, 0);
        }

        let item_out = out.item();
        if let Elem::Atomic(inner) = item_out.elem {
            let addr_space = D::address_space_for_variable(list);
            writeln!(f, "{addr_space}{inner}* {out} = &{list}[{index}];")
        } else {
            let out = out.fmt_left();
            write!(f, "{out} = ")?;
            Self::format_scalar(f, *list, *index, item_out)?;
            f.write_str(";\n")
        }
    }

    fn format_scalar<D: Dialect, Lhs, Rhs>(
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
    _dialect: PhantomData<D>,
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
    _dialect: PhantomData<D>,
}

impl<D: Dialect> IndexVector<D> {
    fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        match rhs {
            Variable::ConstantScalar(value, _elem) => {
                let index = value.as_usize();
                let out = out.index(index);
                let lhs = lhs.index(index);
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs};")
            }
            _ => {
                let elem = out.elem();
                let qualifier = out.const_qualifier();
                let addr_space = D::address_space_for_variable(out);
                let out = out.fmt_left();
                writeln!(
                    f,
                    "{out} = reinterpret_cast<{addr_space}{elem}{qualifier}*>(&{lhs})[{rhs}];"
                )
            }
        }
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
                let addr_space = D::address_space_for_variable(out);
                return writeln!(f, "*(({addr_space}{elem}*)&{out} + {lhs}) = {rhs};");
            }
        };

        let out = out.index(index);
        let rhs = rhs.index(index);

        writeln!(f, "{out} = {rhs};")
    }
}
