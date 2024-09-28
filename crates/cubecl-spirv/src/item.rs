use cubecl_core::ir as core;
use rspirv::spirv::{Capability, CooperativeMatrixUse, Decoration, StorageClass, Word};

use crate::{compiler::SpirvCompiler, target::SpirvTarget};

#[derive(Debug, Clone, PartialEq, Eq)]
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
                b.type_array(item, *len)
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Elem {
    Void,
    Bool,
    Int(u32),
    Float(u32),
}

impl Elem {
    pub fn id<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>) -> Word {
        match self {
            Elem::Void => b.type_void(),
            Elem::Bool => b.type_bool(),
            Elem::Int(width) => b.type_int(*width, 0),
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
            Elem::Int(size) => *size / 8,
            Elem::Float(size) => *size / 8,
        }
    }
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_item(&mut self, item: core::Item) -> Item {
        let size = item.elem.size() as u32 * 8;
        let elem = match item.elem {
            core::Elem::Float(_) => Elem::Float(size),
            core::Elem::Int(_) => Elem::Int(size),
            core::Elem::AtomicInt(_) => Elem::Int(size),
            core::Elem::UInt => Elem::Int(size),
            core::Elem::AtomicUInt => Elem::Int(size),
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
        Elem::Int(32).id(b)
    }
}
