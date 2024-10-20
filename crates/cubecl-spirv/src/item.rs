use std::mem::transmute;

use cubecl_core::ir as core;
use rspirv::spirv::{Capability, CooperativeMatrixUse, Decoration, Scope, StorageClass, Word};

use crate::{compiler::SpirvCompiler, target::SpirvTarget, variable::ConstVal};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Item {
    Scalar(Elem),
    // Vector of scalars. Must be 2, 3, or 4, or 8/16 for OpenCL only
    Vector(Elem, u32),
    Array(Box<Item>, u32),
    RuntimeArray(Box<Item>),
    Struct(Vec<Item>),
    Pointer(StorageClass, Box<Item>),
    CoopMatrix {
        ty: Elem,
        rows: u32,
        columns: u32,
        ident: CooperativeMatrixUse,
    },
}

impl Item {
    pub fn id<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>) -> Word {
        let id = match self {
            Item::Scalar(elem) => elem.id(b),
            Item::Vector(elem, vec) => {
                let elem = elem.id(b);
                b.type_vector(elem, *vec)
            }
            Item::Array(item, len) => {
                let size = item.size();
                let item = item.id(b);
                let len = b.const_u32(*len);
                let ty = b.type_array(item, len);
                if !b.state.array_types.contains_key(&ty) {
                    b.decorate(ty, Decoration::ArrayStride, vec![size.into()]);
                    b.state.array_types.insert(ty, ty);
                }
                ty
            }
            Item::RuntimeArray(item) => {
                let size = item.size();
                let item = item.id(b);
                if let Some(existing) = b.state.array_types.get(&item) {
                    *existing
                } else {
                    let ty = b.type_runtime_array(item);
                    b.decorate(ty, Decoration::ArrayStride, vec![size.into()]);
                    b.state.array_types.insert(item, ty);
                    ty
                }
            }
            Item::Struct(vec) => {
                let items: Vec<_> = vec.iter().map(|item| item.id(b)).collect();
                b.type_struct(items)
            }
            Item::Pointer(storage_class, item) => {
                let item = item.id(b);
                b.type_pointer(None, *storage_class, item)
            }
            Item::CoopMatrix {
                ty,
                rows,
                columns,
                ident,
            } => {
                let ty = ty.id(b);
                let scope = b.const_u32(Scope::Subgroup as u32);
                let usage = b.const_u32(*ident as u32);
                b.type_cooperative_matrix_khr(ty, scope, *rows, *columns, usage)
            }
        };
        if b.debug && !b.state.debug_types.contains(&id) {
            b.debug_name(id, format!("{self}"));
            b.state.debug_types.insert(id);
        }
        id
    }

    pub fn size(&self) -> u32 {
        match self {
            Item::Scalar(elem) => elem.size(),
            Item::Vector(elem, factor) => elem.size() * *factor,
            Item::Array(item, len) => item.size() * *len,
            Item::RuntimeArray(item) => item.size(),
            Item::Struct(vec) => vec.iter().map(|it| it.size()).sum(),
            Item::Pointer(_, item) => item.size(),
            Item::CoopMatrix { ty, .. } => ty.size(),
        }
    }

    pub fn elem(&self) -> Elem {
        match self {
            Item::Scalar(elem) => *elem,
            Item::Vector(elem, _) => *elem,
            Item::Array(item, _) => item.elem(),
            Item::RuntimeArray(item) => item.elem(),
            Item::Struct(_) => Elem::Void,
            Item::Pointer(_, item) => item.elem(),
            Item::CoopMatrix { ty, .. } => *ty,
        }
    }

    pub fn constant<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>, value: ConstVal) -> Word {
        let scalar = self.elem().constant(b, value);
        b.get_or_insert_const(value, self.clone(), |b| {
            let ty = self.id(b);
            match self {
                Item::Scalar(_) => scalar,
                Item::Vector(_, vec) => b.constant_composite(ty, (0..*vec).map(|_| scalar)),
                Item::Array(item, len) => {
                    let elem = item.constant(b, value);
                    b.constant_composite(ty, (0..*len).map(|_| elem))
                }
                Item::RuntimeArray(_) => unimplemented!("Can't create constant runtime array"),
                Item::Struct(elems) => {
                    let items = elems
                        .iter()
                        .map(|item| item.constant(b, value))
                        .collect::<Vec<_>>();
                    b.constant_composite(ty, items)
                }
                Item::Pointer(_, _) => unimplemented!("Can't create constant pointer"),
                Item::CoopMatrix { .. } => unimplemented!("Can't create constant cmma matrix"),
            }
        })
    }

    pub fn const_u32<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>, value: u32) -> Word {
        b.static_cast(ConstVal::Bit32(value), &Elem::Int(32, false), self)
    }

    pub fn cast_to<T: SpirvTarget>(
        &self,
        b: &mut SpirvCompiler<T>,
        out_id: Option<Word>,
        obj: Word,
        other: &Item,
    ) -> Word {
        let ty = other.id(b);

        let matching_vec = match (self, other) {
            (Item::Scalar(_), Item::Scalar(_)) => true,
            (Item::Vector(_, factor_from), Item::Vector(_, factor_to)) => factor_from == factor_to,
            _ => false,
        };
        let matching_elem = self.elem() == other.elem();

        let broadcast = |b: &mut SpirvCompiler<T>, obj: Word, out_id: Option<Word>| -> Word {
            match (self, other) {
                (Item::Scalar(elem), Item::Vector(_, factor)) => {
                    let item = Item::Vector(*elem, *factor);
                    let ty = item.id(b);
                    b.composite_construct(ty, out_id, (0..*factor).map(|_| obj).collect::<Vec<_>>())
                        .unwrap()
                }
                (from, to) => panic!("Invalid cast from {from:?} to {to:?}"),
            }
        };

        let swap_sign = |b: &mut SpirvCompiler<T>,
                         obj: Word,
                         out_id: Option<Word>,
                         width: u32,
                         target_sign: bool| match (width, target_sign) {
            (64, false) => {
                let zero = ConstVal::Bit64(0);
                let zero = b.static_cast(zero, &Elem::Int(64, true), other);
                let id = out_id.unwrap_or_else(|| b.id());
                T::s_max(b, ty, obj, zero, id);
                id
            }
            (_, false) => {
                let zero = ConstVal::Bit32(0);
                let zero = b.static_cast(zero, &Elem::Int(64, true), other);
                let id = out_id.unwrap_or_else(|| b.id());
                T::s_max(b, ty, obj, zero, id);
                id
            }
            (64, true) => {
                let max = ConstVal::Bit64(i64::MAX as u64);
                let max = b.static_cast(max, &Elem::Int(64, true), other);
                b.bitwise_and(ty, out_id, obj, max).unwrap()
            }
            (_, true) => {
                let max = ConstVal::Bit32(i32::MAX as u32);
                let max = b.static_cast(max, &Elem::Int(32, true), other);
                b.bitwise_and(ty, out_id, obj, max).unwrap()
            }
        };

        let convert_i_width =
            |b: &mut SpirvCompiler<T>, obj: Word, out_id: Option<Word>, signed: bool| {
                if signed {
                    b.s_convert(ty, out_id, obj).unwrap()
                } else {
                    b.u_convert(ty, out_id, obj).unwrap()
                }
            };

        let convert_int = |b: &mut SpirvCompiler<T>,
                           obj: Word,
                           out_id: Option<Word>,
                           (width_self, signed_self),
                           (width_other, signed_other)| {
            let sign_differs = signed_self != signed_other;
            let width_differs = width_self != width_other;
            match (sign_differs, width_differs) {
                (true, true) => {
                    let sign_swap = swap_sign(b, obj, None, width_self, signed_other);
                    convert_i_width(b, sign_swap, out_id, signed_other)
                }
                (true, false) => swap_sign(b, obj, out_id, width_self, signed_other),
                (false, true) => convert_i_width(b, obj, out_id, signed_other),
                (false, false) => b.copy_object(ty, out_id, obj).unwrap(),
            }
        };

        let cast_elem = |b: &mut SpirvCompiler<T>, obj: Word, out_id: Option<Word>| -> Word {
            match (self.elem(), other.elem()) {
                (Elem::Bool, Elem::Int(_, _)) => {
                    let one = other.constant(b, 1u32.into());
                    let zero = other.constant(b, 0u32.into());
                    b.select(ty, out_id, obj, one, zero).unwrap()
                }
                (Elem::Bool, Elem::Float(_)) => {
                    let one = other.constant(b, 1f32.into());
                    let zero = other.constant(b, 0f32.into());
                    b.select(ty, out_id, obj, one, zero).unwrap()
                }
                (Elem::Int(_, _), Elem::Bool) => {
                    let one = self.constant(b, 1u32.into());
                    b.i_equal(ty, out_id, obj, one).unwrap()
                }
                (Elem::Int(width_self, signed_self), Elem::Int(width_other, signed_other)) => {
                    convert_int(
                        b,
                        obj,
                        out_id,
                        (width_self, signed_self),
                        (width_other, signed_other),
                    )
                }
                (Elem::Int(_, false), Elem::Float(_)) => b.convert_u_to_f(ty, out_id, obj).unwrap(),
                (Elem::Int(_, true), Elem::Float(_)) => b.convert_s_to_f(ty, out_id, obj).unwrap(),
                (Elem::Float(_), Elem::Bool) => {
                    let one = self.constant(b, 1f32.into());
                    b.i_equal(ty, out_id, obj, one).unwrap()
                }
                (Elem::Float(_), Elem::Int(_, false)) => b.convert_f_to_u(ty, out_id, obj).unwrap(),
                (Elem::Float(_), Elem::Int(_, true)) => b.convert_f_to_s(ty, out_id, obj).unwrap(),
                (Elem::Float(_), Elem::Float(_)) => b.f_convert(ty, out_id, obj).unwrap(),
                (from, to) => panic!("Invalid cast from {from:?} to {to:?}"),
            }
        };

        match (matching_vec, matching_elem) {
            (true, true) if out_id.is_some() => b.copy_object(ty, out_id, obj).unwrap(),
            (true, true) => obj,
            (true, false) => cast_elem(b, obj, out_id),
            (false, true) => broadcast(b, obj, out_id),
            (false, false) => {
                let broadcast = broadcast(b, obj, None);
                cast_elem(b, broadcast, out_id)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Elem {
    Void,
    Bool,
    Int(u32, bool),
    Float(u32),
}

impl Elem {
    pub fn id<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>) -> Word {
        let id = match self {
            Elem::Void => b.type_void(),
            Elem::Bool => b.type_bool(),
            Elem::Int(width, _) => b.type_int(*width, 0),
            Elem::Float(width) => {
                if *width == 16 {
                    b.capabilities.insert(Capability::Float16);
                }
                b.type_float(*width)
            }
        };
        if b.debug && !b.state.debug_types.contains(&id) {
            b.debug_name(id, format!("{self}"));
            b.state.debug_types.insert(id);
        }
        id
    }

    pub fn size(&self) -> u32 {
        match self {
            Elem::Void => 0,
            Elem::Bool => 1,
            Elem::Int(size, _) => *size / 8,
            Elem::Float(size) => *size / 8,
        }
    }

    pub fn constant<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>, value: ConstVal) -> Word {
        b.get_or_insert_const(value, Item::Scalar(*self), |b| {
            let ty = self.id(b);
            match self {
                Elem::Void => unreachable!(),
                Elem::Bool if value.as_u64() == 1 => b.constant_true(ty),
                Elem::Bool => b.constant_false(ty),
                Elem::Int(64, _) => b.constant_bit64(ty, value.as_u64()),
                Elem::Int(_, _) => b.constant_bit32(ty, value.as_u32()),
                Elem::Float(64) => b.constant_bit64(ty, value.as_u64()),
                Elem::Float(_) => b.constant_bit32(ty, value.as_u32()),
            }
        })
    }
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_item(&mut self, item: core::Item) -> Item {
        let size = item.elem.size() as u32 * 8;
        let elem = match item.elem {
            core::Elem::Float(_) => Elem::Float(size),
            core::Elem::Int(_) => Elem::Int(size, true),
            core::Elem::AtomicInt(_) => Elem::Int(size, true),
            core::Elem::UInt => Elem::Int(size, false),
            core::Elem::AtomicUInt => Elem::Int(size, false),
            core::Elem::Bool => Elem::Bool,
        };
        let vectorization = item.vectorization.map(|it| it.get()).unwrap_or(1);
        if vectorization == 1 {
            Item::Scalar(elem)
        } else {
            Item::Vector(elem, vectorization as u32)
        }
    }

    pub fn static_core(&mut self, val: core::Variable, item: &Item) -> Word {
        let val = val.as_const().unwrap();
        unsafe {
            let value = match (val, item.elem()) {
                (core::ConstantScalarValue::Int(val, _), Elem::Bool) => {
                    ConstVal::Bit32((val == 1) as u32)
                }
                (core::ConstantScalarValue::Int(val, _), Elem::Int(64, false)) => {
                    ConstVal::Bit64(val as u64)
                }
                (core::ConstantScalarValue::Int(val, _), Elem::Int(64, true)) => {
                    ConstVal::Bit64(transmute::<i64, u64>(val))
                }
                (core::ConstantScalarValue::Int(val, _), Elem::Int(_, false)) => {
                    ConstVal::Bit32(val as i32 as u32)
                }
                (core::ConstantScalarValue::Int(val, _), Elem::Int(_, true)) => {
                    ConstVal::Bit32(transmute::<i32, u32>(val as i32))
                }
                (core::ConstantScalarValue::Int(val, _), Elem::Float(64)) => {
                    ConstVal::Bit64((val as f64).to_bits())
                }
                (core::ConstantScalarValue::Int(val, _), Elem::Float(32)) => {
                    ConstVal::Bit32((val as f32).to_bits())
                }
                (core::ConstantScalarValue::Float(val, _), Elem::Bool) => {
                    ConstVal::Bit32((val == 1.0) as u32)
                }
                (core::ConstantScalarValue::Float(val, _), Elem::Int(64, false)) => {
                    ConstVal::Bit64(val as u64)
                }
                (core::ConstantScalarValue::Float(val, _), Elem::Int(64, true)) => {
                    ConstVal::Bit64(transmute::<i64, u64>(val as i64))
                }
                (core::ConstantScalarValue::Float(val, _), Elem::Int(_, false)) => {
                    ConstVal::Bit32(val as u32)
                }
                (core::ConstantScalarValue::Float(val, _), Elem::Int(_, true)) => {
                    ConstVal::Bit32(transmute::<i32, u32>(val as i32))
                }
                (core::ConstantScalarValue::Float(val, _), Elem::Float(64)) => {
                    ConstVal::Bit64(val.to_bits())
                }
                (core::ConstantScalarValue::Float(val, _), Elem::Float(_)) => {
                    ConstVal::Bit32((val as f32).to_bits())
                }
                (core::ConstantScalarValue::UInt(val), Elem::Bool) => {
                    ConstVal::Bit32((val == 1) as u32)
                }
                (core::ConstantScalarValue::UInt(val), Elem::Int(64, false)) => {
                    ConstVal::Bit64(val)
                }
                (core::ConstantScalarValue::UInt(val), Elem::Int(64, true)) => {
                    ConstVal::Bit64(val as i64 as u64) //clip sign bit
                }
                (core::ConstantScalarValue::UInt(val), Elem::Int(_, false)) => {
                    ConstVal::Bit32(val as u32)
                }
                (core::ConstantScalarValue::UInt(val), Elem::Int(_, true)) => {
                    ConstVal::Bit32(val as i32 as u32) //clip sign bit
                }
                (core::ConstantScalarValue::UInt(val), Elem::Float(64)) => {
                    ConstVal::Bit64((val as f64).to_bits())
                }
                (core::ConstantScalarValue::UInt(val), Elem::Float(_)) => {
                    ConstVal::Bit32((val as f32).to_bits())
                }
                (core::ConstantScalarValue::Bool(val), Elem::Bool) => ConstVal::Bit32(val as u32),
                (core::ConstantScalarValue::Bool(val), Elem::Int(64, _)) => {
                    ConstVal::Bit64(val as u64)
                }
                (core::ConstantScalarValue::Bool(val), Elem::Int(_, _)) => {
                    ConstVal::Bit32(val as u32)
                }
                (core::ConstantScalarValue::Bool(val), Elem::Float(64)) => {
                    ConstVal::Bit64((val as u32 as f64).to_bits())
                }
                (core::ConstantScalarValue::Bool(val), Elem::Float(_)) => {
                    ConstVal::Bit32((val as u32 as f32).to_bits())
                }
                _ => unreachable!(),
            };
            item.constant(self, value)
        }
    }

    pub fn static_cast(&mut self, val: ConstVal, from: &Elem, item: &Item) -> Word {
        let elem_cast = match (from, item.elem()) {
            (Elem::Bool, Elem::Int(64, _)) => ConstVal::Bit64(val.as_u64()),
            (Elem::Bool, Elem::Int(_, _)) => ConstVal::Bit32(val.as_u32()),
            (Elem::Bool, Elem::Float(64)) => ConstVal::Bit64((val.as_u32() as f64).to_bits()),
            (Elem::Bool, Elem::Float(_)) => ConstVal::Bit32((val.as_u32() as f32).to_bits()),
            (Elem::Int(_, _), Elem::Bool) => ConstVal::Bit32((val.as_u64() == 1) as u32),
            (Elem::Int(_, _), Elem::Int(64, _)) => ConstVal::Bit64(val.as_u64()),
            (Elem::Int(_, _), Elem::Int(_, _)) => ConstVal::Bit32(val.as_u32()),
            (Elem::Int(_, false), Elem::Float(64)) => {
                ConstVal::Bit64((val.as_u64() as f64).to_bits())
            }
            (Elem::Int(_, true), Elem::Float(64)) => unsafe {
                ConstVal::Bit64((transmute::<u64, i64>(val.as_u64()) as f64).to_bits())
            },
            (Elem::Int(_, false), Elem::Float(_)) => {
                ConstVal::Bit32((val.as_u64() as f32).to_bits())
            }
            (Elem::Int(64, true), Elem::Float(_)) => unsafe {
                ConstVal::Bit32((transmute::<u64, i64>(val.as_u64()) as f32).to_bits())
            },
            (Elem::Int(_, true), Elem::Float(_)) => unsafe {
                ConstVal::Bit32((transmute::<u32, i32>(val.as_u32()) as f32).to_bits())
            },
            (Elem::Float(64), Elem::Bool) => {
                ConstVal::Bit32((f64::from_bits(val.as_u64()) == 1.0) as u32)
            }
            (Elem::Float(_), Elem::Bool) => {
                ConstVal::Bit32((f32::from_bits(val.as_u32()) == 1.0) as u32)
            }
            (Elem::Float(64), Elem::Int(64, false)) => {
                ConstVal::Bit64(f64::from_bits(val.as_u64()) as u64)
            }
            (Elem::Float(_), Elem::Int(64, false)) => {
                ConstVal::Bit64(f32::from_bits(val.as_u32()) as u64)
            }
            (Elem::Float(64), Elem::Int(64, true)) => ConstVal::Bit64(unsafe {
                transmute::<i64, u64>(f64::from_bits(val.as_u64()) as i64)
            }),
            (Elem::Float(_), Elem::Int(64, true)) => ConstVal::Bit64(unsafe {
                transmute::<i64, u64>(f32::from_bits(val.as_u32()) as i64)
            }),
            (Elem::Float(64), Elem::Int(_, false)) => {
                ConstVal::Bit32(f64::from_bits(val.as_u64()) as u32)
            }
            (Elem::Float(_), Elem::Int(_, false)) => {
                ConstVal::Bit32(f32::from_bits(val.as_u32()) as u32)
            }
            (Elem::Float(64), Elem::Int(_, true)) => ConstVal::Bit32(unsafe {
                transmute::<i32, u32>(f64::from_bits(val.as_u64()) as i32)
            }),
            (Elem::Float(_), Elem::Int(_, true)) => ConstVal::Bit32(unsafe {
                transmute::<i32, u32>(f32::from_bits(val.as_u32()) as i32)
            }),
            (Elem::Float(64), Elem::Float(64)) => val,
            (Elem::Float(64), Elem::Float(_)) => {
                ConstVal::Bit32((f64::from_bits(val.as_u64()) as f32).to_bits())
            }
            (Elem::Float(_), Elem::Float(64)) => {
                ConstVal::Bit64((f32::from_bits(val.as_u32()) as f64).to_bits())
            }
            _ => val,
        };
        item.constant(self, elem_cast)
    }
}

impl std::fmt::Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Scalar(elem) => write!(f, "{elem}"),
            Item::Vector(elem, factor) => write!(f, "vec{factor}<{elem}>"),
            Item::Array(item, len) => write!(f, "array<{item}, {len}>"),
            Item::RuntimeArray(item) => write!(f, "array<{item}>"),
            Item::Struct(members) => {
                write!(f, "struct<")?;
                for item in members {
                    write!(f, "{item}")?;
                }
                f.write_str(">")
            }
            Item::Pointer(class, item) => write!(f, "ptr<{class:?}, {item}>"),
            Item::CoopMatrix { ty, ident, .. } => write!(f, "matrix<{ty}, {ident:?}>"),
        }
    }
}

impl std::fmt::Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Elem::Void => write!(f, "void"),
            Elem::Bool => write!(f, "bool"),
            Elem::Int(width, false) => write!(f, "u{width}"),
            Elem::Int(width, true) => write!(f, "i{width}"),
            Elem::Float(width) => write!(f, "f{width}"),
        }
    }
}

pub trait HasId {
    fn id<T: SpirvTarget>(b: &mut SpirvCompiler<T>) -> Word;
}

impl HasId for u32 {
    fn id<T: SpirvTarget>(b: &mut SpirvCompiler<T>) -> Word {
        Elem::Int(32, false).id(b)
    }
}
