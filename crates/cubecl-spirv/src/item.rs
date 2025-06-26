use cubecl_core::ir::{self as core, FloatKind, IntKind, UIntKind};
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
        if b.debug_symbols && !b.state.debug_types.contains(&id) {
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

    pub fn same_vectorization(&self, elem: Elem) -> Item {
        match self {
            Item::Scalar(_) => Item::Scalar(elem),
            Item::Vector(_, factor) => Item::Vector(elem, *factor),
            _ => unreachable!(),
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

    /// Broadcast a scalar to a vector if needed, ex: f32 -> vec2<f32>, vec2<f32> -> vec2<f32>
    pub fn broadcast<T: SpirvTarget>(
        &self,
        b: &mut SpirvCompiler<T>,
        obj: Word,
        out_id: Option<Word>,
        other: &Item,
    ) -> Word {
        match (self, other) {
            (Item::Scalar(elem), Item::Vector(_, factor)) => {
                let item = Item::Vector(*elem, *factor);
                let ty = item.id(b);
                b.composite_construct(ty, out_id, (0..*factor).map(|_| obj).collect::<Vec<_>>())
                    .unwrap()
            }
            _ => obj,
        }
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
            let width_differs = width_self != width_other;
            let sign_extend = signed_self && signed_other;
            match width_differs {
                true => convert_i_width(b, obj, out_id, sign_extend),
                false => b.copy_object(ty, out_id, obj).unwrap(),
            }
        };

        let cast_elem = |b: &mut SpirvCompiler<T>, obj: Word, out_id: Option<Word>| -> Word {
            match (self.elem(), other.elem()) {
                (Elem::Bool, Elem::Int(_, _)) => {
                    let one = other.const_u32(b, 1);
                    let zero = other.const_u32(b, 0);
                    b.select(ty, out_id, obj, one, zero).unwrap()
                }
                (Elem::Bool, Elem::Float(_)) | (Elem::Bool, Elem::Relaxed) => {
                    let one = other.const_u32(b, 1);
                    let zero = other.const_u32(b, 0);
                    b.select(ty, out_id, obj, one, zero).unwrap()
                }
                (Elem::Int(_, _), Elem::Bool) => {
                    let zero = self.const_u32(b, 0);
                    b.i_not_equal(ty, out_id, obj, zero).unwrap()
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
                (Elem::Int(_, false), Elem::Float(_)) | (Elem::Int(_, false), Elem::Relaxed) => {
                    b.convert_u_to_f(ty, out_id, obj).unwrap()
                }
                (Elem::Int(_, true), Elem::Float(_)) | (Elem::Int(_, true), Elem::Relaxed) => {
                    b.convert_s_to_f(ty, out_id, obj).unwrap()
                }
                (Elem::Float(_), Elem::Bool) | (Elem::Relaxed, Elem::Bool) => {
                    let zero = self.const_u32(b, 0);
                    b.f_unord_not_equal(ty, out_id, obj, zero).unwrap()
                }
                (Elem::Float(_), Elem::Int(_, false)) | (Elem::Relaxed, Elem::Int(_, false)) => {
                    b.convert_f_to_u(ty, out_id, obj).unwrap()
                }
                (Elem::Float(_), Elem::Int(_, true)) | (Elem::Relaxed, Elem::Int(_, true)) => {
                    b.convert_f_to_s(ty, out_id, obj).unwrap()
                }
                (Elem::Float(_), Elem::Float(_))
                | (Elem::Float(_), Elem::Relaxed)
                | (Elem::Relaxed, Elem::Float(_)) => b.f_convert(ty, out_id, obj).unwrap(),
                (Elem::Bool, Elem::Bool) => b.copy_object(ty, out_id, obj).unwrap(),
                (Elem::Relaxed, Elem::Relaxed) => b.copy_object(ty, out_id, obj).unwrap(),
                (from, to) => panic!("Invalid cast from {from:?} to {to:?}"),
            }
        };

        match (matching_vec, matching_elem) {
            (true, true) if out_id.is_some() => b.copy_object(ty, out_id, obj).unwrap(),
            (true, true) => obj,
            (true, false) => cast_elem(b, obj, out_id),
            (false, true) => self.broadcast(b, obj, out_id, other),
            (false, false) => {
                let broadcast = self.broadcast(b, obj, None, other);
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
    Relaxed,
}

impl Elem {
    pub fn id<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>) -> Word {
        let id = match self {
            Elem::Void => b.type_void(),
            Elem::Bool => b.type_bool(),
            Elem::Int(width, _) => b.type_int(*width, 0),
            Elem::Float(width) => b.type_float(*width),
            Elem::Relaxed => b.type_float(32),
        };
        if b.debug_symbols && !b.state.debug_types.contains(&id) {
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
            Elem::Relaxed => 4,
        }
    }

    pub fn constant<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>, value: ConstVal) -> Word {
        b.get_or_insert_const(value, Item::Scalar(*self), |b| {
            let ty = self.id(b);
            match self {
                Elem::Void => unreachable!(),
                Elem::Bool if value.as_u64() != 0 => b.constant_true(ty),
                Elem::Bool => b.constant_false(ty),
                _ => match value {
                    ConstVal::Bit32(val) => b.constant_bit32(ty, val),
                    ConstVal::Bit64(val) => b.constant_bit64(ty, val),
                },
            }
        })
    }
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_item(&mut self, item: core::Item) -> Item {
        let elem = self.compile_elem(item.elem);
        let vectorization = item.vectorization.map(|it| it.get()).unwrap_or(1);
        if vectorization == 1 {
            Item::Scalar(elem)
        } else {
            Item::Vector(elem, vectorization as u32)
        }
    }

    pub fn compile_elem(&mut self, elem: core::Elem) -> Elem {
        match elem {
            core::Elem::Float(
                core::FloatKind::E2M1
                | core::FloatKind::E2M3
                | core::FloatKind::E3M2
                | core::FloatKind::E4M3
                | core::FloatKind::E5M2
                | core::FloatKind::UE8M0,
            ) => panic!("Minifloat not supported in SPIR-V"),
            core::Elem::Float(core::FloatKind::BF16) => panic!("BFloat16 not supported in SPIR-V"),
            core::Elem::Float(FloatKind::F16) => {
                self.capabilities.insert(Capability::Float16);
                Elem::Float(16)
            }
            core::Elem::Float(FloatKind::TF32) => panic!("TF32 not supported in SPIR-V"),
            core::Elem::Float(FloatKind::Flex32) => Elem::Relaxed,
            core::Elem::Float(FloatKind::F32) => Elem::Float(32),
            core::Elem::Float(FloatKind::F64) => {
                self.capabilities.insert(Capability::Float64);
                Elem::Float(64)
            }
            core::Elem::AtomicFloat(FloatKind::F16) => {
                self.capabilities.insert(Capability::Float16);
                Elem::Float(16)
            }
            core::Elem::AtomicFloat(FloatKind::F32) => Elem::Float(32),
            core::Elem::AtomicFloat(FloatKind::Flex32) => Elem::Relaxed,
            core::Elem::AtomicFloat(FloatKind::F64) => {
                self.capabilities.insert(Capability::Float64);
                Elem::Float(64)
            }
            core::Elem::AtomicFloat(
                core::FloatKind::E2M1
                | core::FloatKind::E2M3
                | core::FloatKind::E3M2
                | core::FloatKind::E4M3
                | core::FloatKind::E5M2
                | core::FloatKind::UE8M0,
            ) => panic!("Minifloat not supported in SPIR-V"),
            core::Elem::AtomicFloat(core::FloatKind::BF16) => {
                panic!("BFloat16 not supported in SPIR-V")
            }
            core::Elem::AtomicFloat(core::FloatKind::TF32) => {
                panic!("TF32 not supported in SPIR-V")
            }
            core::Elem::Int(IntKind::I8) => {
                self.capabilities.insert(Capability::Int8);
                Elem::Int(8, true)
            }
            core::Elem::Int(IntKind::I16) => {
                self.capabilities.insert(Capability::Int16);
                Elem::Int(16, true)
            }
            core::Elem::Int(IntKind::I32) => Elem::Int(32, true),
            core::Elem::Int(IntKind::I64) => {
                self.capabilities.insert(Capability::Int64);
                Elem::Int(64, true)
            }
            core::Elem::AtomicInt(IntKind::I8) => {
                self.capabilities.insert(Capability::Int8);
                Elem::Int(8, true)
            }
            core::Elem::AtomicInt(IntKind::I16) => {
                self.capabilities.insert(Capability::Int16);
                Elem::Int(16, true)
            }
            core::Elem::AtomicInt(IntKind::I32) => Elem::Int(32, true),
            core::Elem::AtomicInt(IntKind::I64) => {
                self.capabilities.insert(Capability::Int64Atomics);
                Elem::Int(64, true)
            }
            core::Elem::UInt(UIntKind::U64) => {
                self.capabilities.insert(Capability::Int64);
                Elem::Int(64, false)
            }
            core::Elem::UInt(UIntKind::U32) => Elem::Int(32, false),
            core::Elem::UInt(UIntKind::U16) => {
                self.capabilities.insert(Capability::Int16);
                Elem::Int(16, false)
            }
            core::Elem::UInt(UIntKind::U8) => {
                self.capabilities.insert(Capability::Int8);
                Elem::Int(8, false)
            }
            core::Elem::AtomicUInt(UIntKind::U8) => {
                self.capabilities.insert(Capability::Int8);
                Elem::Int(8, false)
            }
            core::Elem::AtomicUInt(UIntKind::U16) => {
                self.capabilities.insert(Capability::Int16);
                Elem::Int(16, false)
            }
            core::Elem::AtomicUInt(UIntKind::U32) => Elem::Int(32, false),
            core::Elem::AtomicUInt(UIntKind::U64) => {
                self.capabilities.insert(Capability::Int64);
                self.capabilities.insert(Capability::Int64Atomics);
                Elem::Int(64, false)
            }
            core::Elem::Bool => Elem::Bool,
        }
    }

    pub fn static_core(&mut self, val: core::Variable, item: &Item) -> Word {
        let val = val.as_const().unwrap();

        let value = match (val, item.elem()) {
            (core::ConstantScalarValue::Int(val, _), Elem::Bool) => ConstVal::from_bool(val != 0),
            (core::ConstantScalarValue::Int(val, _), Elem::Int(width, false)) => {
                ConstVal::from_uint(val as u64, width)
            }
            (core::ConstantScalarValue::Int(val, _), Elem::Int(width, true)) => {
                ConstVal::from_int(val, width)
            }
            (core::ConstantScalarValue::Int(val, _), Elem::Float(width)) => {
                ConstVal::from_float(val as f64, width)
            }
            (core::ConstantScalarValue::Int(val, _), Elem::Relaxed) => {
                ConstVal::from_float(val as f64, 32)
            }
            (core::ConstantScalarValue::Float(val, _), Elem::Bool) => {
                ConstVal::from_bool(val != 0.0)
            }
            (core::ConstantScalarValue::Float(val, _), Elem::Int(width, false)) => {
                ConstVal::from_uint(val as u64, width)
            }
            (core::ConstantScalarValue::Float(val, _), Elem::Int(width, true)) => {
                ConstVal::from_int(val as i64, width)
            }
            (core::ConstantScalarValue::Float(val, _), Elem::Float(width)) => {
                ConstVal::from_float(val, width)
            }
            (core::ConstantScalarValue::Float(val, _), Elem::Relaxed) => {
                ConstVal::from_float(val, 32)
            }
            (core::ConstantScalarValue::UInt(val, _), Elem::Bool) => ConstVal::from_bool(val != 0),
            (core::ConstantScalarValue::UInt(val, _), Elem::Int(width, false)) => {
                ConstVal::from_uint(val, width)
            }
            (core::ConstantScalarValue::UInt(val, _), Elem::Int(width, true)) => {
                ConstVal::from_int(val as i64, width)
            }
            (core::ConstantScalarValue::UInt(val, _), Elem::Float(width)) => {
                ConstVal::from_float(val as f64, width)
            }
            (core::ConstantScalarValue::UInt(val, _), Elem::Relaxed) => {
                ConstVal::from_float(val as f64, 32)
            }
            (core::ConstantScalarValue::Bool(val), Elem::Bool) => ConstVal::from_bool(val),
            (core::ConstantScalarValue::Bool(val), Elem::Int(width, _)) => {
                ConstVal::from_uint(val as u64, width)
            }
            (core::ConstantScalarValue::Bool(val), Elem::Float(width)) => {
                ConstVal::from_float(val as u32 as f64, width)
            }
            (core::ConstantScalarValue::Bool(val), Elem::Relaxed) => {
                ConstVal::from_float(val as u32 as f64, 32)
            }
            (_, Elem::Void) => unreachable!(),
        };
        item.constant(self, value)
    }

    pub fn static_cast(&mut self, val: ConstVal, from: &Elem, item: &Item) -> Word {
        let elem_cast = match (*from, item.elem()) {
            (Elem::Bool, Elem::Int(width, _)) => ConstVal::from_uint(val.as_u32() as u64, width),
            (Elem::Bool, Elem::Float(width)) => ConstVal::from_float(val.as_u32() as f64, width),
            (Elem::Bool, Elem::Relaxed) => ConstVal::from_float(val.as_u32() as f64, 32),
            (Elem::Int(_, _), Elem::Bool) => ConstVal::from_bool(val.as_u64() != 0),
            (Elem::Int(_, false), Elem::Int(width, _)) => ConstVal::from_uint(val.as_u64(), width),
            (Elem::Int(w_in, true), Elem::Int(width, _)) => {
                ConstVal::from_uint(val.as_int(w_in) as u64, width)
            }
            (Elem::Int(_, false), Elem::Float(width)) => {
                ConstVal::from_float(val.as_u64() as f64, width)
            }
            (Elem::Int(_, false), Elem::Relaxed) => ConstVal::from_float(val.as_u64() as f64, 32),
            (Elem::Int(in_w, true), Elem::Float(width)) => {
                ConstVal::from_float(val.as_int(in_w) as f64, width)
            }
            (Elem::Int(in_w, true), Elem::Relaxed) => {
                ConstVal::from_float(val.as_int(in_w) as f64, 32)
            }
            (Elem::Float(in_w), Elem::Bool) => ConstVal::from_bool(val.as_float(in_w) != 0.0),
            (Elem::Relaxed, Elem::Bool) => ConstVal::from_bool(val.as_float(32) != 0.0),
            (Elem::Float(in_w), Elem::Int(out_w, false)) => {
                ConstVal::from_uint(val.as_float(in_w) as u64, out_w)
            }
            (Elem::Relaxed, Elem::Int(out_w, false)) => {
                ConstVal::from_uint(val.as_float(32) as u64, out_w)
            }
            (Elem::Float(in_w), Elem::Int(out_w, true)) => {
                ConstVal::from_int(val.as_float(in_w) as i64, out_w)
            }
            (Elem::Relaxed, Elem::Int(out_w, true)) => {
                ConstVal::from_int(val.as_float(32) as i64, out_w)
            }
            (Elem::Float(in_w), Elem::Float(out_w)) => {
                ConstVal::from_float(val.as_float(in_w), out_w)
            }
            (Elem::Relaxed, Elem::Float(out_w)) => ConstVal::from_float(val.as_float(32), out_w),
            (Elem::Float(in_w), Elem::Relaxed) => ConstVal::from_float(val.as_float(in_w), 32),
            (Elem::Bool, Elem::Bool) => val,
            (Elem::Relaxed, Elem::Relaxed) => val,
            (_, Elem::Void) | (Elem::Void, _) => unreachable!(),
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
            Elem::Relaxed => write!(f, "flex32"),
        }
    }
}
