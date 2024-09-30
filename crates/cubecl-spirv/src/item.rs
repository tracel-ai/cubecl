use cubecl_core::ir as core;
use rspirv::spirv::{Capability, CooperativeMatrixUse, Decoration, StorageClass, Word};

use crate::{compiler::SpirvCompiler, target::SpirvTarget};

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
        scope: Word,
    },
}

impl Item {
    pub fn id<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>) -> Word {
        match self {
            Item::Scalar(elem) => elem.id(b),
            Item::Vector(elem, vec) => {
                let elem = elem.id(b);
                b.type_vector(elem, *vec)
            }
            Item::Array(item, len) => {
                let item = item.id(b);
                let len = b.const_u32(*len);
                b.type_array(item, len)
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
                scope,
            } => {
                let ty = ty.id(b);
                b.type_cooperative_matrix_khr(ty, *scope, *rows, *columns, *ident as u32)
            }
        }
    }

    fn size(&self) -> u32 {
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

    pub fn constant<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>, value: u64) -> Word {
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

    pub fn cast_to<T: SpirvTarget>(
        &self,
        b: &mut SpirvCompiler<T>,
        object: Word,
        other: &Item,
    ) -> Word {
        if self == other {
            return object;
        }

        let broadcast = match (self, other) {
            (Item::Vector(_, factor), Item::Vector(_, factor2)) if factor == factor2 => object,
            (Item::Scalar(_), Item::Scalar(_)) => object,
            (Item::Scalar(elem), Item::Vector(_, factor)) => {
                let item = Item::Vector(*elem, *factor);
                let ty = item.id(b);
                b.composite_construct(ty, None, (0..*factor).map(|_| object).collect::<Vec<_>>())
                    .unwrap()
            }
            (from, to) => panic!("Invalid cast from {from:?} to {to:?}"),
        };

        if self.elem() == other.elem() {
            return broadcast;
        }

        let ty = other.id(b);

        match (self.elem(), other.elem()) {
            (Elem::Bool, Elem::Int(_, _)) => {
                let one = other.constant(b, 1);
                let zero = other.constant(b, 0);
                b.select(ty, None, broadcast, one, zero).unwrap()
            }
            (Elem::Bool, Elem::Float(_)) => {
                let one = other.constant(b, 1f32.to_bits() as u64);
                let zero = other.constant(b, 0f32.to_bits() as u64);
                b.select(ty, None, broadcast, one, zero).unwrap()
            }
            (Elem::Int(_, _), Elem::Bool) => {
                let one = other.constant(b, 1);
                b.i_equal(ty, None, broadcast, one).unwrap()
            }
            (Elem::Int(_, false), Elem::Int(_, false)) => b.u_convert(ty, None, broadcast).unwrap(),
            (Elem::Int(_, true), Elem::Int(_, false)) => {
                b.sat_convert_s_to_u(ty, None, broadcast).unwrap()
            }
            (Elem::Int(_, true), Elem::Int(_, true)) => b.s_convert(ty, None, broadcast).unwrap(),
            (Elem::Int(_, false), Elem::Int(_, true)) => {
                b.sat_convert_u_to_s(ty, None, broadcast).unwrap()
            }
            (Elem::Int(_, false), Elem::Float(_)) => b.convert_u_to_f(ty, None, broadcast).unwrap(),
            (Elem::Int(_, true), Elem::Float(_)) => b.convert_s_to_f(ty, None, broadcast).unwrap(),
            (Elem::Float(_), Elem::Bool) => {
                let one = other.constant(b, 1f32.to_bits() as u64);
                b.i_equal(ty, None, broadcast, one).unwrap()
            }
            (Elem::Float(_), Elem::Int(_, false)) => b.convert_f_to_u(ty, None, broadcast).unwrap(),
            (Elem::Float(_), Elem::Int(_, true)) => b.convert_f_to_s(ty, None, broadcast).unwrap(),
            (Elem::Float(_), Elem::Float(_)) => b.f_convert(ty, None, broadcast).unwrap(),
            (from, to) => panic!("Invalid cast from {from:?} to {to:?}"),
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
        match self {
            Elem::Void => b.type_void(),
            Elem::Bool => b.type_bool(),
            Elem::Int(width, signed) => b.type_int(*width, if *signed { 1 } else { 0 }),
            Elem::Float(width) => {
                if *width == 16 {
                    b.capabilities.insert(Capability::Float16);
                }
                b.type_float(*width)
            }
        }
    }

    pub fn size(&self) -> u32 {
        match self {
            Elem::Void => 0,
            Elem::Bool => 1,
            Elem::Int(size, _) => *size / 8,
            Elem::Float(size) => *size / 8,
        }
    }

    pub fn constant<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>, value: u64) -> Word {
        b.get_or_insert_const(value, Item::Scalar(*self), |b| {
            let ty = self.id(b);
            match self {
                Elem::Void => unreachable!(),
                Elem::Bool => match value == 1 {
                    true => b.constant_true(ty),
                    false => b.constant_false(ty),
                },
                Elem::Int(width, _) => match *width {
                    64 => b.constant_bit64(ty, value),
                    _ => b.constant_bit32(ty, value as u32),
                },
                Elem::Float(width) => match *width {
                    64 => b.constant_bit64(ty, value),
                    _ => b.constant_bit32(ty, value as u32),
                },
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
}

pub trait HasId {
    fn id<T: SpirvTarget>(b: &mut SpirvCompiler<T>) -> Word;
}

impl HasId for u32 {
    fn id<T: SpirvTarget>(b: &mut SpirvCompiler<T>) -> Word {
        Elem::Int(32, false).id(b)
    }
}
