use cubecl_core::ir::{self as core, AddressSpace, ClampMode, FloatKind, IntKind, UIntKind};
use rspirv::spirv::{
    Capability, CooperativeMatrixUse, FPEncoding, Scope, StorageClass, TensorClampMode, Word,
};
use serde::{Deserialize, Serialize};

use crate::{compiler::SpirvCompiler, target::SpirvTarget, variable::ConstVal};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Item {
    Scalar(Elem),
    // Vector of scalars. Must be 2, 3, or 4, unless long vectors extension is enabled
    Vector(Elem, u32),
    Pointer(StorageClass, Box<Item>),
    Array(Box<Item>, u32),
    DynamicArray(Box<Item>),
    CoopMatrix {
        ty: Elem,
        rows: u32,
        columns: u32,
        ident: CooperativeMatrixUse,
        scope: Scope,
    },
    TensorLayout {
        dims: usize,
        clamp_mode: TensorClampMode,
    },
    TensorView {
        dims: usize,
        has_dims: bool,
        permutation: Vec<u32>,
    },
}

impl Item {
    pub fn id<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>) -> Word {
        let id = match self {
            Item::Scalar(elem) => elem.id(b),
            Item::Vector(elem, vec) => {
                let elem = elem.id(b);
                if b.compilation_options.vulkan.supports_long_vectors {
                    let len = b.const_u32(*vec);
                    b.type_vector_id_ext(elem, len)
                } else {
                    b.type_vector(elem, *vec)
                }
            }
            Item::Pointer(storage_class, item) => {
                let item = item.id(b);
                b.type_pointer(None, *storage_class, item)
            }
            Item::Array(item, size) => {
                let item = item.id(b);
                let id = b.id();
                let size = b.const_u32(*size);
                b.type_array_id(Some(id), item, size)
            }
            Item::DynamicArray(item) => {
                let item = item.id(b);
                let id = b.id();
                b.type_runtime_array_id(Some(id), item)
            }
            Item::CoopMatrix {
                ty,
                rows,
                columns,
                ident,
                scope,
            } => {
                let ty = ty.id(b);
                let scope = b.const_u32(*scope as u32);
                let usage = b.const_u32(*ident as u32);
                b.type_cooperative_matrix_khr(ty, scope, *rows, *columns, usage)
            }
            Item::TensorLayout { dims, clamp_mode } => {
                let dim = b.const_u32(*dims as u32);
                let clamp_mode = b.const_u32(*clamp_mode as u32);
                b.type_tensor_layout_nv(dim, clamp_mode)
            }
            Item::TensorView {
                dims,
                has_dims,
                permutation,
            } => {
                let bool = b.type_bool();
                let dim = b.const_u32(*dims as u32);
                let has_dims = if *has_dims {
                    b.constant_true(bool)
                } else {
                    b.constant_false(bool)
                };
                let permutation = permutation
                    .iter()
                    .map(|it| b.const_u32(*it))
                    .collect::<Vec<_>>();
                b.type_tensor_view_nv(dim, has_dims, permutation)
            }
        };
        if b.debug_symbols && !b.state.debug_types.contains(&id) {
            b.debug_name(id, format!("{self}"));
            b.state.debug_types.insert(id);
        }
        id
    }

    pub fn builtin_u32() -> Self {
        Item::Scalar(Elem::Int(32, false))
    }

    pub fn value_type(&self) -> Item {
        match self {
            Item::Pointer(_, item) => item.value_type(),
            Item::Array(item, _) => item.value_type(),
            Item::DynamicArray(item) => item.value_type(),
            other => other.clone(),
        }
    }

    pub fn size(&self) -> u32 {
        match self {
            Item::Scalar(elem) => elem.size(),
            Item::Vector(elem, factor) => elem.size() * *factor,
            Item::Pointer(_, item) => item.size(),
            Item::Array(item, size) => item.size() * *size,
            Item::DynamicArray(item) => item.size(),
            Item::CoopMatrix { ty, .. } => ty.size(),
            Item::TensorLayout { .. } => 1,
            Item::TensorView { .. } => 1,
        }
    }

    pub fn elem(&self) -> Elem {
        match self {
            Item::Scalar(elem) => *elem,
            Item::Vector(elem, _) => *elem,
            Item::Pointer(_, item) => item.elem(),
            Item::Array(item, _) => item.elem(),
            Item::DynamicArray(item) => item.elem(),
            Item::CoopMatrix { ty, .. } => *ty,
            Item::TensorLayout { .. } => Elem::Void,
            Item::TensorView { .. } => Elem::Void,
        }
    }

    pub fn same_vectorization(&self, elem: Elem) -> Item {
        match self {
            Item::Scalar(_) => Item::Scalar(elem),
            Item::Vector(_, factor) => Item::Vector(elem, *factor),
            _ => unreachable!(),
        }
    }

    pub fn vectorization(&self) -> u32 {
        match self {
            Item::Vector(_, factor) => *factor,
            _ => 1,
        }
    }

    pub fn constant<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>, value: ConstVal) -> Word {
        let scalar = self.elem().constant(b, value);
        let ty = self.id(b);
        match self {
            Item::Scalar(_) => scalar,
            Item::Vector(_, vec) => b.constant_composite(ty, (0..*vec).map(|_| scalar)),
            Item::Pointer(_, _) => unimplemented!("Can't create constant pointer"),
            Item::Array(_, _) => unimplemented!("Can't create constant pointer"),
            Item::DynamicArray(_) => unimplemented!("Can't create constant pointer"),
            Item::CoopMatrix { .. } => unimplemented!("Can't create constant cmma matrix"),
            Item::TensorLayout { .. } => unimplemented!("Can't create constant cmma matrix"),
            Item::TensorView { .. } => unimplemented!("Can't create constant cmma matrix"),
        }
    }

    pub fn const_u32<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>, value: u32) -> Word {
        b.static_cast(ConstVal::Bit32(value), &Elem::Int(32, false), self)
            .0
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
            (Item::Scalar(_), Item::Vector(..)) => false,
            (Item::Vector(_, factor_from), Item::Vector(_, factor_to)) => factor_from == factor_to,
            _ => true,
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
                (Elem::Bool, Elem::Float(_, _)) | (Elem::Bool, Elem::Relaxed) => {
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
                (Elem::Int(_, false), Elem::Float(_, _)) | (Elem::Int(_, false), Elem::Relaxed) => {
                    b.convert_u_to_f(ty, out_id, obj).unwrap()
                }
                (Elem::Int(_, true), Elem::Float(_, _)) | (Elem::Int(_, true), Elem::Relaxed) => {
                    b.convert_s_to_f(ty, out_id, obj).unwrap()
                }
                (Elem::Float(_, _), Elem::Bool) | (Elem::Relaxed, Elem::Bool) => {
                    let zero = self.const_u32(b, 0);
                    b.f_unord_not_equal(ty, out_id, obj, zero).unwrap()
                }
                (Elem::Float(_, _), Elem::Int(_, false)) | (Elem::Relaxed, Elem::Int(_, false)) => {
                    b.convert_f_to_u(ty, out_id, obj).unwrap()
                }
                (Elem::Float(_, _), Elem::Int(_, true)) | (Elem::Relaxed, Elem::Int(_, true)) => {
                    b.convert_f_to_s(ty, out_id, obj).unwrap()
                }
                (Elem::Float(32, _), Elem::Relaxed) | (Elem::Relaxed, Elem::Float(32, _)) => {
                    if out_id.is_some() {
                        b.copy_object(ty, out_id, obj).unwrap()
                    } else {
                        obj
                    }
                }
                (Elem::Float(_, _), Elem::Float(_, _))
                | (Elem::Float(_, _), Elem::Relaxed)
                | (Elem::Relaxed, Elem::Float(_, _)) => b.f_convert(ty, out_id, obj).unwrap(),
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

    pub fn is_array(&self) -> bool {
        matches!(self, Item::Array(..))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Elem {
    Void,
    Bool,
    Int(u32, bool),
    Float(u32, Option<FPEncoding>),
    Relaxed,
}

impl Elem {
    pub fn id<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>) -> Word {
        let id = match self {
            Elem::Void => b.type_void(),
            Elem::Bool => b.type_bool(),
            Elem::Int(width, _) => b.type_int(*width, 0),
            Elem::Float(width, encoding) => b.type_float(*width, *encoding),
            Elem::Relaxed => b.type_float(32, None),
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
            Elem::Float(size, _) => *size / 8,
            Elem::Relaxed => 4,
        }
    }

    pub fn constant<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>, value: ConstVal) -> Word {
        let ty = self.id(b);
        match self {
            Elem::Void => unreachable!(),
            Elem::Bool if value.as_u64() != 0 => b.constant_true(ty),
            Elem::Bool => b.constant_false(ty),
            _ => match value {
                ConstVal::Bit32(val) => b.dedup_constant_bit32(ty, val),
                ConstVal::Bit64(val) => b.dedup_constant_bit64(ty, val),
            },
        }
    }

    pub fn float_encoding(&self) -> Option<FPEncoding> {
        match self {
            Elem::Float(_, encoding) => *encoding,
            _ => None,
        }
    }

    pub fn width(&self) -> u32 {
        self.size() * 8
    }
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_type(&mut self, item: core::Type) -> Item {
        match item {
            core::Type::Scalar(storage) => Item::Scalar(self.compile_storage_type(storage)),
            core::Type::Vector(inner, size) => {
                Item::Vector(self.compile_storage_type(inner.storage_type()), size as u32)
            }
            core::Type::Atomic(inner) => self.compile_type(*inner),
            core::Type::Pointer(inner, class) => {
                let storage_class = compile_pointer_class(class);
                let item = self.compile_type(*inner);
                Item::Pointer(storage_class, Box::new(item))
            }
            core::Type::Semantic(semantic) => match semantic {
                core::SemanticType::BarrierToken
                | core::SemanticType::Pipeline
                | core::SemanticType::TensorMap => {
                    unimplemented!("Unsupported semantic type")
                }
                core::SemanticType::TensorLayout(dims, clamp_mode) => Item::TensorLayout {
                    dims,
                    clamp_mode: compile_clamp_mode(clamp_mode),
                },
                core::SemanticType::TensorView(dims, has_dims, permutation) => Item::TensorView {
                    dims,
                    has_dims,
                    permutation: permutation[..dims].to_vec(),
                },
            },
            core::Type::Array(inner, size, _) => {
                let item = self.compile_type(*inner);
                Item::Array(Box::new(item), size as u32)
            }
            core::Type::DynamicArray(inner, _) => {
                let item = self.compile_type(*inner);
                Item::DynamicArray(Box::new(item))
            }
        }
    }

    pub fn compile_storage_type(&mut self, ty: core::StorageType) -> Elem {
        match ty {
            core::StorageType::Scalar(ty) => self.compile_elem(ty),
            core::StorageType::Opaque(ty) => match ty {
                core::OpaqueType::Barrier(_) => {
                    unimplemented!("Barrier type not supported in SPIR-V")
                }
            },
            core::StorageType::Packed(_, _) => {
                unimplemented!("Packed types not yet supported in SPIR-V")
            }
        }
    }

    pub fn compile_elem(&mut self, elem: core::ElemType) -> Elem {
        match elem {
            core::ElemType::Float(
                core::FloatKind::E2M1
                | core::FloatKind::E2M3
                | core::FloatKind::E3M2
                | core::FloatKind::UE8M0,
            ) => panic!("Minifloat not supported in SPIR-V"),
            core::ElemType::Float(core::FloatKind::E4M3) => {
                self.capabilities.insert(Capability::Float8EXT);
                Elem::Float(8, Some(FPEncoding::Float8E4M3EXT))
            }
            core::ElemType::Float(core::FloatKind::E5M2) => {
                self.capabilities.insert(Capability::Float8EXT);
                Elem::Float(8, Some(FPEncoding::Float8E5M2EXT))
            }
            core::ElemType::Float(core::FloatKind::BF16) => {
                self.capabilities.insert(Capability::BFloat16TypeKHR);
                Elem::Float(16, Some(FPEncoding::BFloat16KHR))
            }
            core::ElemType::Float(FloatKind::F16) => {
                self.capabilities.insert(Capability::Float16);
                Elem::Float(16, None)
            }
            core::ElemType::Float(FloatKind::TF32) => panic!("TF32 not supported in SPIR-V"),
            core::ElemType::Float(FloatKind::Flex32) => Elem::Relaxed,
            core::ElemType::Float(FloatKind::F32) => Elem::Float(32, None),
            core::ElemType::Float(FloatKind::F64) => {
                self.capabilities.insert(Capability::Float64);
                Elem::Float(64, None)
            }
            core::ElemType::Int(IntKind::I8) => {
                self.capabilities.insert(Capability::Int8);
                Elem::Int(8, true)
            }
            core::ElemType::Int(IntKind::I16) => {
                self.capabilities.insert(Capability::Int16);
                Elem::Int(16, true)
            }
            core::ElemType::Int(IntKind::I32) => Elem::Int(32, true),
            core::ElemType::Int(IntKind::I64) => {
                self.capabilities.insert(Capability::Int64);
                Elem::Int(64, true)
            }
            core::ElemType::UInt(UIntKind::U64) => {
                self.capabilities.insert(Capability::Int64);
                Elem::Int(64, false)
            }
            core::ElemType::UInt(UIntKind::U32) => Elem::Int(32, false),
            core::ElemType::UInt(UIntKind::U16) => {
                self.capabilities.insert(Capability::Int16);
                Elem::Int(16, false)
            }
            core::ElemType::UInt(UIntKind::U8) => {
                self.capabilities.insert(Capability::Int8);
                Elem::Int(8, false)
            }
            core::ElemType::Bool => Elem::Bool,
        }
    }

    pub fn compile_function_param_type(&mut self, var: core::Variable) -> Word {
        match var.kind {
            core::VariableKind::GlobalBuffer(id) => {
                self.state.base_lookups.buffers[id as usize].struct_ptr_ty_id
            }
            core::VariableKind::TensorMap(_) => {
                unimplemented!("Tensor maps not supported")
            }
            core::VariableKind::LocalMut { .. }
            | core::VariableKind::LocalConst { .. }
            | core::VariableKind::Versioned { .. }
            | core::VariableKind::Constant(..)
            | core::VariableKind::GlobalScalar(_)
            | core::VariableKind::Builtin(..) => self.compile_type(var.ty).id(self),
            core::VariableKind::ConstantArray { .. } => {
                todo!("Constant arrays not yet supported for args")
            }
            core::VariableKind::Shared { id, .. } => self.state.base_lookups.shared[&id].ptr_ty_id,
            core::VariableKind::Matrix { mat, .. } => {
                let mat = self.compile_matrix(&mat);
                self.item(&mat).id(self)
            }
            core::VariableKind::Pipeline { .. } => {
                unimplemented!("Pipelines not supported")
            }
            core::VariableKind::BarrierToken { .. } => {
                unimplemented!("Barrier tokens not supported")
            }
            core::VariableKind::Aggregate { .. } => {
                unreachable!("Should be disaggregated at this point")
            }
        }
    }

    pub fn static_cast(&mut self, val: ConstVal, from: &Elem, item: &Item) -> (Word, ConstVal) {
        let elem_cast = match (*from, item.elem()) {
            (Elem::Bool, Elem::Int(width, _)) => ConstVal::from_uint(val.as_u32() as u64, width),
            (Elem::Bool, Elem::Float(width, encoding)) => {
                ConstVal::from_float(val.as_u32() as f64, width, encoding)
            }
            (Elem::Bool, Elem::Relaxed) => ConstVal::from_float(val.as_u32() as f64, 32, None),
            (Elem::Int(_, _), Elem::Bool) => ConstVal::from_bool(val.as_u64() != 0),
            (Elem::Int(_, false), Elem::Int(width, _)) => ConstVal::from_uint(val.as_u64(), width),
            (Elem::Int(w_in, true), Elem::Int(width, _)) => {
                ConstVal::from_uint(val.as_int(w_in) as u64, width)
            }
            (Elem::Int(_, false), Elem::Float(width, encoding)) => {
                ConstVal::from_float(val.as_u64() as f64, width, encoding)
            }
            (Elem::Int(_, false), Elem::Relaxed) => {
                ConstVal::from_float(val.as_u64() as f64, 32, None)
            }
            (Elem::Int(in_w, true), Elem::Float(width, encoding)) => {
                ConstVal::from_float(val.as_int(in_w) as f64, width, encoding)
            }
            (Elem::Int(in_w, true), Elem::Relaxed) => {
                ConstVal::from_float(val.as_int(in_w) as f64, 32, None)
            }
            (Elem::Float(in_w, encoding), Elem::Bool) => {
                ConstVal::from_bool(val.as_float(in_w, encoding) != 0.0)
            }
            (Elem::Relaxed, Elem::Bool) => ConstVal::from_bool(val.as_float(32, None) != 0.0),
            (Elem::Float(in_w, encoding), Elem::Int(out_w, false)) => {
                ConstVal::from_uint(val.as_float(in_w, encoding) as u64, out_w)
            }
            (Elem::Relaxed, Elem::Int(out_w, false)) => {
                ConstVal::from_uint(val.as_float(32, None) as u64, out_w)
            }
            (Elem::Float(in_w, encoding), Elem::Int(out_w, true)) => {
                ConstVal::from_int(val.as_float(in_w, encoding) as i64, out_w)
            }
            (Elem::Relaxed, Elem::Int(out_w, true)) => {
                ConstVal::from_int(val.as_float(32, None) as i64, out_w)
            }
            (Elem::Float(in_w, encoding), Elem::Float(out_w, encoding_out)) => {
                ConstVal::from_float(val.as_float(in_w, encoding), out_w, encoding_out)
            }
            (Elem::Relaxed, Elem::Float(out_w, encoding)) => {
                ConstVal::from_float(val.as_float(32, None), out_w, encoding)
            }
            (Elem::Float(in_w, encoding), Elem::Relaxed) => {
                ConstVal::from_float(val.as_float(in_w, encoding), 32, None)
            }
            (Elem::Bool, Elem::Bool) => val,
            (Elem::Relaxed, Elem::Relaxed) => val,
            (_, Elem::Void) | (Elem::Void, _) => unreachable!(),
        };
        let id = item.constant(self, elem_cast);
        (id, elem_cast)
    }
}

pub fn compile_pointer_class(class: AddressSpace) -> StorageClass {
    match class {
        AddressSpace::Global(_) => StorageClass::PhysicalStorageBuffer,
        AddressSpace::Shared => StorageClass::Workgroup,
        AddressSpace::Local => StorageClass::Function,
    }
}

impl std::fmt::Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Scalar(elem) => write!(f, "{elem}"),
            Item::Vector(elem, factor) => write!(f, "vec{factor}<{elem}>"),
            Item::Pointer(class, item) => write!(f, "ptr<{class:?}, {item}>"),
            Item::Array(item, size) => write!(f, "array<{item}, {size}>"),
            Item::DynamicArray(item) => write!(f, "array<{item}>"),
            Item::CoopMatrix { ty, ident, .. } => write!(f, "matrix<{ty}, {ident:?}>"),
            Item::TensorLayout { dims, clamp_mode } => {
                write!(f, "tensor_layout<{dims}, {clamp_mode:?}>")
            }
            Item::TensorView {
                dims,
                has_dims,
                permutation,
            } => {
                write!(
                    f,
                    "tensor_view<{:?}, has_dims: {has_dims}>",
                    &permutation[..*dims]
                )
            }
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
            Elem::Float(width, None) => write!(f, "f{width}"),
            Elem::Float(_, Some(FPEncoding::BFloat16KHR)) => write!(f, "bf16"),
            Elem::Float(_, Some(FPEncoding::Float8E4M3EXT)) => write!(f, "e4m3"),
            Elem::Float(_, Some(FPEncoding::Float8E5M2EXT)) => write!(f, "e5m2"),
            Elem::Relaxed => write!(f, "flex32"),
        }
    }
}

fn compile_clamp_mode(clamp_mode: ClampMode) -> TensorClampMode {
    match clamp_mode {
        ClampMode::Undefined => TensorClampMode::Undefined,
        ClampMode::Constant(_) => TensorClampMode::Constant,
        ClampMode::ClampToEdge => TensorClampMode::ClampToEdge,
        ClampMode::Repeat => TensorClampMode::Repeat,
        ClampMode::RepeatMirrored => TensorClampMode::RepeatMirrored,
    }
}
