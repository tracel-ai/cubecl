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
        let out_item = *out.item().value_ty();
        if let Item::Vector(..) = out_item {
            Self::unroll_vec(f, lhs, rhs, out)
        } else {
            let out = out.fmt_left();
            write!(f, "{out} = ")?;
            Self::format_scalar(f, *lhs, *rhs, out_item)?;
            f.write_str(";\n")
        }
    }

    fn format_scalar<Lhs: Component<D>, Rhs: Component<D>>(
        f: &mut Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item<D>,
    ) -> std::fmt::Result;

    fn unroll_vec(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let mut write_op = |index: usize,
                            lhs: &Variable<D>,
                            rhs: &Variable<D>,
                            out: &Variable<D>,
                            item_out: Item<D>| {
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

        if Self::can_optimize() {
            let optimized = Variable::optimized_args([*lhs, *rhs, *out]);
            let [lhs, rhs, out_optimized] = optimized.args;

            let item_out_original = *out.item().value_ty();
            let item_out_optimized = *out_optimized.item().value_ty();

            let index = match item_out_optimized {
                Item::Vector(_, vectorization) => vectorization,
                _ => 1,
            };

            if item_out_original == item_out_optimized {
                write_op(index, &lhs, &rhs, out, item_out_optimized)
            } else {
                let out_tmp = Variable::tmp(item_out_optimized);
                write_op(index, &lhs, &rhs, &out_tmp, item_out_optimized)?;
                let addr_space = D::address_space_for_variable(out);
                let out = out.fmt_left();

                writeln!(
                    f,
                    "{out} = reinterpret_cast<{addr_space}{item_out_original}&>({out_tmp});\n"
                )?;

                Ok(())
            }
        } else {
            let index = match out.item() {
                Item::Vector(_, vectorization) => vectorization,
                _ => 1,
            };

            write_op(index, lhs, rhs, out, out.item())
        }
    }

    fn can_optimize() -> bool {
        true
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

pub struct Remainder;

impl<D: Dialect> Binary<D> for Remainder {
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out_item: Item<D>,
    ) -> std::fmt::Result {
        let out_elem = out_item.elem();
        match out_elem {
            Elem::<D>::I16 | Elem::<D>::U16 | Elem::<D>::I8 | Elem::<D>::U8 => {
                write!(f, "{out_elem}({lhs} % {rhs})")
            }
            Elem::<D>::F16 | Elem::<D>::BF16 => {
                let f32 = Elem::<D>::F32;
                write!(f, "{out_elem}(fmodf({f32}({lhs}), {f32}({rhs}))))")
            }
            Elem::<D>::F32 => {
                write!(f, "fmodf({lhs}, {rhs})")
            }
            Elem::<D>::F64 => {
                write!(f, "fmod({lhs}, {rhs})")
            }
            _ => write!(f, "{lhs} % {rhs}"),
        }
    }

    fn can_optimize() -> bool {
        false
    }
}

pub struct ModFloor;

impl<D: Dialect> Binary<D> for ModFloor {
    fn format_scalar<Lhs: Component<D>, Rhs: Component<D>>(
        f: &mut Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item<D>,
    ) -> std::fmt::Result {
        let is_uint = matches!(item.elem(), Elem::U8 | Elem::U16 | Elem::U32 | Elem::U64);
        if is_uint {
            // Remainder is cheaper and unsigned ints don't have a difference
            return Remainder::format_scalar(f, lhs, rhs, item);
        }

        let floor = {
            let prefix = match item.elem() {
                Elem::F16 | Elem::BF16 => D::compile_instruction_half_function_name_prefix(),
                Elem::F16x2 | Elem::BF16x2 => D::compile_instruction_half2_function_name_prefix(),
                _ => "",
            };
            format!("{prefix}floor")
        };

        let is_int = matches!(item.elem(), Elem::I8 | Elem::I16 | Elem::I32 | Elem::I64);
        let out_elem = item.elem();
        if is_int {
            write!(
                f,
                "{lhs} - {rhs} * ({out_elem}){floor}((float){lhs} / (float){rhs})"
            )
        } else {
            write!(f, "{lhs} - {rhs} * {floor}({lhs} / {rhs})")
        }
    }
}

pub struct FastDiv;

impl<D: Dialect> Binary<D> for FastDiv {
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        _out_item: Item<D>,
    ) -> std::fmt::Result {
        // f32 only
        write!(f, "__fdividef({lhs}, {rhs})")
    }
}

pub struct HiMul;

impl<D: Dialect> Binary<D> for HiMul {
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Item<D>,
    ) -> std::fmt::Result {
        let out_elem = out.elem();
        match out_elem {
            Elem::I32 => write!(f, "__mulhi({lhs}, {rhs})"),
            Elem::U32 => write!(f, "__umulhi({lhs}, {rhs})"),
            Elem::I64 => write!(f, "__mul64hi({lhs}, {rhs})"),
            Elem::U64 => write!(f, "__umul64hi({lhs}, {rhs})"),
            _ => writeln!(f, "#error HiMul only supports 32 and 64 bit ints"),
        }
    }

    fn can_optimize() -> bool {
        false
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
        let elem = *item.elem();
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

    fn can_optimize() -> bool {
        false
    }
}

pub struct FastPowf;

impl<D: Dialect> Binary<D> for FastPowf {
    // Only executed for f32
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        _item: Item<D>,
    ) -> std::fmt::Result {
        write!(f, "__powf({lhs}, {rhs})")
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
        let elem = *item.elem();
        let lhs = lhs.to_string();
        let rhs = rhs.to_string();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                let lhs = format!("float({lhs})");

                write!(f, "{elem}(")?;
                D::compile_instruction_powf(f, &lhs, &rhs, Elem::F32)?;
                write!(f, ")")
            }
            Elem::F64 => {
                // RHS needs to be a double.
                let rhs = format!("double({rhs})");

                D::compile_instruction_powf(f, &lhs, &rhs, elem)
            }
            _ => D::compile_instruction_powf(f, &lhs, &rhs, elem),
        }
    }
}
pub struct ArcTan2;

impl<D: Dialect> Binary<D> for ArcTan2 {
    // ArcTan2 doesn't support half and no half equivalent exists
    fn format_scalar<Lhs: Display, Rhs: Display>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item<D>,
    ) -> std::fmt::Result {
        let elem = item.elem();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                write!(f, "{elem}(atan2(float({lhs}), float({rhs})))")
            }
            _ => {
                write!(f, "atan2({lhs}, {rhs})")
            }
        }
    }

    fn can_optimize() -> bool {
        false
    }
}

pub struct Hypot;

impl<D: Dialect> Binary<D> for Hypot {
    // Hypot doesn't support half and no half equivalent exists
    fn format_scalar<Lhs, Rhs>(
        f: &mut Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item<D>,
    ) -> std::fmt::Result
    where
        Lhs: Component<D>,
        Rhs: Component<D>,
    {
        let elem = *item.elem();
        let lhs = lhs.to_string();
        let rhs = rhs.to_string();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                let lhs = format!("float({lhs})");
                let rhs = format!("float({rhs})");
                write!(f, "{elem}(")?;
                D::compile_instruction_hypot(f, &lhs, &rhs, Elem::F32)?;
                write!(f, ")")
            }
            _ => D::compile_instruction_hypot(f, &lhs, &rhs, elem),
        }
    }

    fn can_optimize() -> bool {
        false
    }
}

pub struct Rhypot;

impl<D: Dialect> Binary<D> for Rhypot {
    // Rhypot doesn't support half and no half equivalent exists
    fn format_scalar<Lhs, Rhs>(
        f: &mut Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        item: Item<D>,
    ) -> std::fmt::Result
    where
        Lhs: Component<D>,
        Rhs: Component<D>,
    {
        let elem = *item.elem();
        let lhs = lhs.to_string();
        let rhs = rhs.to_string();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                let lhs = format!("float({lhs})");
                let rhs = format!("float({rhs})");
                write!(f, "{elem}(")?;
                D::compile_instruction_rhypot(f, &lhs, &rhs, Elem::F32)?;
                write!(f, ")")
            }
            _ => D::compile_instruction_rhypot(f, &lhs, &rhs, elem),
        }
    }

    fn can_optimize() -> bool {
        false
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

pub struct Index;

impl Index {
    pub(crate) fn format<D: Dialect>(
        f: &mut Formatter<'_>,
        list: &Variable<D>,
        index: &Variable<D>,
        out: &Variable<D>,
        vector_size: u32,
    ) -> std::fmt::Result {
        if list.item().vectorization() != out.item().vectorization() {
            let item = Item::new(list.elem(), vector_size as usize);
            let item_ptr = out.item();
            let tmp = Variable::tmp_declared(item);

            writeln!(
                f,
                "{item_ptr} {tmp} = reinterpret_cast<{item_ptr}>({list});"
            )?;

            writeln!(f, "{item_ptr} {out} = &{tmp}[{index}];")
        } else {
            let item_out = out.item();
            if matches!(item_out.elem(), Elem::Barrier(_)) {
                let addr_space = D::address_space_for_variable(list);
                writeln!(
                    f,
                    "{addr_space}{}& {out} = {list}[{index}];",
                    item_out.elem()
                )
            } else {
                writeln!(f, "{item_out} {out} = &{list}[{index}];")
            }
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
pub struct ExtractComponent<D: Dialect> {
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
pub struct InsertComponent<D: Dialect> {
    _dialect: PhantomData<D>,
}

impl<D: Dialect> ExtractComponent<D> {
    pub fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        match rhs {
            Variable::Constant(value, _elem) => {
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
                let lhs = lhs.ensure_lvalue(f)?;
                let out = out.fmt_left();
                writeln!(
                    f,
                    "{out} = reinterpret_cast<{addr_space}{elem}{qualifier}*>(&{lhs})[{rhs}];"
                )
            }
        }
    }
}

impl<D: Dialect> InsertComponent<D> {
    pub fn format(
        f: &mut Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let index = match lhs {
            Variable::Constant(value, _) => value.as_usize(),
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
