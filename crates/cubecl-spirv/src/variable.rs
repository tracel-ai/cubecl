use std::mem::transmute;

use crate::{
    item::{Elem, HasId, Item},
    lookups::Array,
    SpirvCompiler, SpirvTarget,
};
use cubecl_core::{
    ir::{self as core, ConstantScalarValue, FloatKind, IntKind},
    ExecutionMode,
};
use rspirv::{
    dr::Builder,
    spirv::{BuiltIn, StorageClass, Word},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    SubgroupSize(Word),
    GlobalInputArray(Word, Item),
    GlobalOutputArray(Word, Item),
    GlobalScalar(Word, Elem),
    ConstantScalar(Word, ConstVal, Elem),
    Local {
        id: Word,
        item: Item,
    },
    LocalBinding {
        id: Word,
        item: Item,
    },
    Named {
        id: Word,
        item: Item,
        is_array: bool,
    },
    Slice {
        ptr: Box<Variable>,
        offset: Word,
        len: Word,
        const_len: Option<u32>,
        item: Item,
    },
    LocalScalar {
        id: Word,
        elem: Elem,
    },
    SharedMemory(Word, Item, u32),
    ConstantArray(Word, Item, u32),
    LocalArray(Word, Item, u32),
    Id(Word),
    LocalInvocationIndex(Word),
    LocalInvocationIdX(Word),
    LocalInvocationIdY(Word),
    LocalInvocationIdZ(Word),
    Rank(Word),
    WorkgroupId(Word),
    WorkgroupIdX(Word),
    WorkgroupIdY(Word),
    WorkgroupIdZ(Word),
    GlobalInvocationIndex(Word),
    GlobalInvocationIdX(Word),
    GlobalInvocationIdY(Word),
    GlobalInvocationIdZ(Word),
    WorkgroupSize(Word),
    WorkgroupSizeX(Word),
    WorkgroupSizeY(Word),
    WorkgroupSizeZ(Word),
    NumWorkgroups(Word),
    NumWorkgroupsX(Word),
    NumWorkgroupsY(Word),
    NumWorkgroupsZ(Word),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstVal {
    Bit32(u32),
    Bit64(u64),
}

impl ConstVal {
    pub fn as_u64(&self) -> u64 {
        match self {
            ConstVal::Bit32(val) => *val as u64,
            ConstVal::Bit64(val) => *val,
        }
    }

    pub fn as_u32(&self) -> u32 {
        match self {
            ConstVal::Bit32(val) => *val,
            ConstVal::Bit64(_) => panic!("Truncating 64 bit variable to 32 bit"),
        }
    }
}

impl From<ConstantScalarValue> for ConstVal {
    fn from(value: ConstantScalarValue) -> Self {
        unsafe {
            match value {
                ConstantScalarValue::Int(val, IntKind::I32) => {
                    ConstVal::Bit32(transmute::<i32, u32>(val as i32))
                }
                ConstantScalarValue::Int(val, IntKind::I64) => {
                    ConstVal::Bit64(transmute::<i64, u64>(val))
                }
                ConstantScalarValue::Float(val, FloatKind::F64) => ConstVal::Bit64(val.to_bits()),
                ConstantScalarValue::Float(val, FloatKind::F32) => {
                    ConstVal::Bit32((val as f32).to_bits())
                }
                ConstantScalarValue::Float(val, FloatKind::F16) => {
                    ConstVal::Bit32((val as f32).to_bits())
                }
                ConstantScalarValue::Float(_, FloatKind::BF16) => {
                    panic!("bf16 not supported in SPIR-V")
                }
                ConstantScalarValue::UInt(val) => ConstVal::Bit32(val as u32),
                ConstantScalarValue::Bool(val) => ConstVal::Bit32(val as u32),
            }
        }
    }
}

impl From<u32> for ConstVal {
    fn from(value: u32) -> Self {
        ConstVal::Bit32(value)
    }
}

impl From<f32> for ConstVal {
    fn from(value: f32) -> Self {
        ConstVal::Bit32(value.to_bits())
    }
}

impl Variable {
    pub fn id(&self) -> Word {
        match self {
            Variable::GlobalInputArray(id, _) => *id,
            Variable::GlobalOutputArray(id, _) => *id,
            Variable::GlobalScalar(id, _) => *id,
            Variable::ConstantScalar(id, _, _) => *id,
            Variable::Local { id, .. } => *id,
            Variable::LocalBinding { id, .. } => *id,
            Variable::Named { id, .. } => *id,
            Variable::Slice { ptr, .. } => ptr.id(),
            Variable::LocalScalar { id, .. } => *id,
            Variable::SharedMemory(id, _, _) => *id,
            Variable::ConstantArray(id, _, _) => *id,
            Variable::LocalArray(id, _, _) => *id,
            Variable::SubgroupSize(id) => *id,
            Variable::Id(id) => *id,
            Variable::LocalInvocationIndex(id) => *id,
            Variable::LocalInvocationIdX(id) => *id,
            Variable::LocalInvocationIdY(id) => *id,
            Variable::LocalInvocationIdZ(id) => *id,
            Variable::Rank(id) => *id,
            Variable::WorkgroupId(id) => *id,
            Variable::WorkgroupIdX(id) => *id,
            Variable::WorkgroupIdY(id) => *id,
            Variable::WorkgroupIdZ(id) => *id,
            Variable::GlobalInvocationIndex(id) => *id,
            Variable::GlobalInvocationIdX(id) => *id,
            Variable::GlobalInvocationIdY(id) => *id,
            Variable::GlobalInvocationIdZ(id) => *id,
            Variable::WorkgroupSize(id) => *id,
            Variable::WorkgroupSizeX(id) => *id,
            Variable::WorkgroupSizeY(id) => *id,
            Variable::WorkgroupSizeZ(id) => *id,
            Variable::NumWorkgroups(id) => *id,
            Variable::NumWorkgroupsX(id) => *id,
            Variable::NumWorkgroupsY(id) => *id,
            Variable::NumWorkgroupsZ(id) => *id,
        }
    }

    pub fn item(&self) -> Item {
        match self {
            Variable::GlobalInputArray(_, item) => item.clone(),
            Variable::GlobalOutputArray(_, item) => item.clone(),
            Variable::ConstantScalar(_, _, elem) => Item::Scalar(*elem),
            Variable::Local { item, .. } => item.clone(),
            Variable::LocalBinding { item, .. } => item.clone(),
            Variable::Named { item, .. } => item.clone(),
            Variable::Slice { item, .. } => item.clone(),
            Variable::LocalScalar { elem, .. } => Item::Scalar(*elem),
            Variable::SharedMemory(_, item, _) => item.clone(),
            Variable::ConstantArray(_, item, _) => item.clone(),
            Variable::LocalArray(_, item, _) => item.clone(),
            _ => Item::Scalar(Elem::Int(32, false)), // builtin
        }
    }

    pub fn elem(&self) -> Elem {
        self.item().elem()
    }

    pub fn has_len(&self) -> bool {
        matches!(
            self,
            Variable::GlobalInputArray(_, _)
                | Variable::GlobalOutputArray(_, _)
                | Variable::Named { .. }
                | Variable::Slice { .. }
                | Variable::SharedMemory(_, _, _)
                | Variable::ConstantArray(_, _, _)
                | Variable::LocalArray(_, _, _)
        )
    }

    pub fn as_const(&self) -> Option<ConstVal> {
        match self {
            Self::ConstantScalar(_, val, _) => Some(*val),
            _ => None,
        }
    }
}

pub enum IndexedVariable {
    Pointer(Word, Item),
    Composite(Word, u32, Item),
    Scalar(Variable),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Globals {
    Id,
    LocalInvocationId,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    LocalInvocationIndex,
    Rank,
    WorkgroupId,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    WorkgroupIndex,
    GlobalInvocationId,
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    GlobalInvocationIndex,
    WorkgroupSize,
    WorkgroupSizeX,
    WorkgroupSizeY,
    WorkgroupSizeZ,
    NumWorkgroups,
    NumWorkgroupsTotal,
    NumWorkgroupsX,
    NumWorkgroupsY,
    NumWorkgroupsZ,
    SubgroupSize,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_variable(&mut self, variable: core::Variable) -> Variable {
        match variable {
            core::Variable::ConstantScalar(value) => {
                let item = self.compile_item(core::Item::new(value.elem()));
                let elem_id = item.id(self);
                let map_val = value.into();

                if let Some(existing) = self.state.constants.get(&(map_val, item.clone())) {
                    Variable::ConstantScalar(*existing, map_val, item.elem())
                } else {
                    let id = match value {
                        core::ConstantScalarValue::Int(val, kind) => match kind {
                            core::IntKind::I32 => self.constant_bit32(elem_id, unsafe {
                                transmute::<i32, u32>(val as i32)
                            }),
                            core::IntKind::I64 => {
                                self.constant_bit64(elem_id, unsafe { transmute::<i64, u64>(val) })
                            }
                        },
                        core::ConstantScalarValue::Float(val, kind) => match kind {
                            core::FloatKind::F16 | core::FloatKind::F32 => {
                                self.constant_bit32(elem_id, (val as f32).to_bits())
                            }
                            core::FloatKind::BF16 => unimplemented!("BF16 not yet supported"),
                            core::FloatKind::F64 => self.constant_bit64(elem_id, val.to_bits()),
                        },
                        core::ConstantScalarValue::UInt(val) => {
                            self.constant_bit32(elem_id, val as u32)
                        }
                        core::ConstantScalarValue::Bool(val) => match val {
                            true => self.constant_true(elem_id),
                            false => self.constant_false(elem_id),
                        },
                    };
                    self.state.constants.insert((map_val, item.clone()), id);
                    Variable::ConstantScalar(id, map_val, item.elem())
                }
            }
            core::Variable::Slice { id, depth, .. } => self
                .state
                .slices
                .get(&(id, depth))
                .expect("Tried accessing non-existing slice")
                .into(),
            core::Variable::GlobalInputArray { id, item } => {
                let id = self.state.inputs[id as usize];
                Variable::GlobalInputArray(id, self.compile_item(item))
            }
            core::Variable::GlobalOutputArray { id, item } => {
                let id = self.state.outputs[id as usize];
                Variable::GlobalOutputArray(id, self.compile_item(item))
            }
            core::Variable::GlobalScalar { id, elem } => {
                let arr_id = self.state.named[&format!("scalars_{elem}")];
                let item = self.compile_item(core::Item::new(elem));
                let arr = Variable::GlobalInputArray(arr_id, item.clone());
                let const_id = self.const_u32(id as u32);
                let index =
                    Variable::ConstantScalar(const_id, (id as u32).into(), Elem::Int(32, false));
                let val = self.id();
                self.debug_name(val, format!("scalars_{elem}[{id}]"));
                self.read_indexed(val, &arr, &index);
                Variable::GlobalScalar(val, item.elem())
            }
            core::Variable::Local { id, item, depth } => {
                let item = self.compile_item(item);
                let var = self
                    .state
                    .variables
                    .get(&(id, depth))
                    .expect("Trying to use undeclared variable");
                Variable::Local { id: *var, item }
            }
            core::Variable::LocalBinding { id, item, depth } => {
                let item = self.compile_item(item);
                if let Some(binding) = self.state.bindings.get(&(id, depth)) {
                    Variable::LocalBinding { id: *binding, item }
                } else {
                    let binding = self.id();
                    self.debug_name(binding, format!("_{id}"));
                    self.state.bindings.insert((id, depth), binding);
                    Variable::LocalBinding { id: binding, item }
                }
            }
            core::Variable::UnitPos => Variable::LocalInvocationIndex(self.get_or_insert_global(
                Globals::LocalInvocationIndex,
                |b| {
                    let id = b.load_builtin(
                        BuiltIn::LocalInvocationIndex,
                        Item::Scalar(Elem::Int(32, false)),
                    );
                    b.debug_name(id, "UNIT_POS");
                    id
                },
            )),
            core::Variable::UnitPosX => Variable::LocalInvocationIdX(self.get_or_insert_global(
                Globals::LocalInvocationIdX,
                |b| {
                    let id = b.extract(Globals::LocalInvocationId, BuiltIn::LocalInvocationId, 0);
                    b.debug_name(id, "UNIT_POS_X");
                    id
                },
            )),
            core::Variable::UnitPosY => Variable::LocalInvocationIdY(self.get_or_insert_global(
                Globals::LocalInvocationIdY,
                |b| {
                    let id = b.extract(Globals::LocalInvocationId, BuiltIn::LocalInvocationId, 1);
                    b.debug_name(id, "UNIT_POS_Y");
                    id
                },
            )),
            core::Variable::UnitPosZ => Variable::LocalInvocationIdZ(self.get_or_insert_global(
                Globals::LocalInvocationIdZ,
                |b| {
                    let id = b.extract(Globals::LocalInvocationId, BuiltIn::LocalInvocationId, 2);
                    b.debug_name(id, "UNIT_POS_Z");
                    id
                },
            )),
            core::Variable::CubePosX => {
                Variable::WorkgroupIdX(self.get_or_insert_global(Globals::WorkgroupIdX, |b| {
                    let id = b.extract(Globals::WorkgroupId, BuiltIn::WorkgroupId, 0);
                    b.debug_name(id, "CUBE_POS_X");
                    id
                }))
            }
            core::Variable::CubePosY => {
                Variable::WorkgroupIdY(self.get_or_insert_global(Globals::WorkgroupIdY, |b| {
                    let id = b.extract(Globals::WorkgroupId, BuiltIn::WorkgroupId, 1);
                    b.debug_name(id, "CUBE_POS_Y");
                    id
                }))
            }
            core::Variable::CubePosZ => {
                Variable::WorkgroupIdZ(self.get_or_insert_global(Globals::WorkgroupIdZ, |b| {
                    let id = b.extract(Globals::WorkgroupId, BuiltIn::WorkgroupId, 2);
                    b.debug_name(id, "CUBE_POS_Z");
                    id
                }))
            }
            core::Variable::CubeDim => Variable::WorkgroupSize(self.state.cube_size),
            core::Variable::CubeDimX => Variable::WorkgroupSizeX(self.state.cube_dims[0]),
            core::Variable::CubeDimY => Variable::WorkgroupSizeY(self.state.cube_dims[1]),
            core::Variable::CubeDimZ => Variable::WorkgroupSizeZ(self.state.cube_dims[2]),
            core::Variable::CubeCount => Variable::WorkgroupSize(self.get_or_insert_global(
                Globals::NumWorkgroupsTotal,
                |b: &mut SpirvCompiler<T>| {
                    let int = b.type_int(32, 0);
                    let x = b.compile_variable(core::Variable::CubeCountX).id();
                    let y = b.compile_variable(core::Variable::CubeCountY).id();
                    let z = b.compile_variable(core::Variable::CubeCountZ).id();
                    let count = b.i_mul(int, None, x, y).unwrap();
                    let count = b.i_mul(int, None, count, z).unwrap();
                    b.debug_name(count, "CUBE_COUNT");
                    count
                },
            )),
            core::Variable::CubeCountX => {
                Variable::NumWorkgroupsX(self.get_or_insert_global(Globals::NumWorkgroupsX, |b| {
                    let id = b.extract(Globals::NumWorkgroups, BuiltIn::NumWorkgroups, 0);
                    b.debug_name(id, "CUBE_COUNT_X");
                    id
                }))
            }
            core::Variable::CubeCountY => {
                Variable::NumWorkgroupsY(self.get_or_insert_global(Globals::NumWorkgroupsY, |b| {
                    let id = b.extract(Globals::NumWorkgroups, BuiltIn::NumWorkgroups, 1);
                    b.debug_name(id, "CUBE_COUNT_Y");
                    id
                }))
            }
            core::Variable::CubeCountZ => {
                Variable::NumWorkgroupsZ(self.get_or_insert_global(Globals::NumWorkgroupsZ, |b| {
                    let id = b.extract(Globals::NumWorkgroups, BuiltIn::NumWorkgroups, 2);
                    b.debug_name(id, "CUBE_COUNT_Z");
                    id
                }))
            }
            core::Variable::SubcubeDim => {
                let id = self.get_or_insert_global(Globals::SubgroupSize, |b| {
                    let id =
                        b.load_builtin(BuiltIn::SubgroupSize, Item::Scalar(Elem::Int(32, false)));
                    b.debug_name(id, "SUBCUBE_DIM");
                    id
                });
                Variable::SubgroupSize(id)
            }
            core::Variable::CubePos => {
                let id = self.get_or_insert_global(Globals::WorkgroupIndex, |b| {
                    let x = b.compile_variable(core::Variable::CubePosX).id();
                    let y = b.compile_variable(core::Variable::CubePosY).id();
                    let z = b.compile_variable(core::Variable::CubePosZ).id();

                    let groups_x = b.compile_variable(core::Variable::CubeCountX).id();
                    let groups_y = b.compile_variable(core::Variable::CubeCountY).id();
                    let ty = u32::id(b);
                    let id = b.i_mul(ty, None, z, groups_y).unwrap();
                    let id = b.i_add(ty, None, id, y).unwrap();
                    let id = b.i_mul(ty, None, id, groups_x).unwrap();
                    let id = b.i_add(ty, None, id, x).unwrap();
                    b.debug_name(id, "CUBE_POS");
                    id
                });
                Variable::WorkgroupId(id)
            }
            core::Variable::AbsolutePos => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIndex, |b| {
                    let x = b.compile_variable(core::Variable::AbsolutePosX).id();
                    let y = b.compile_variable(core::Variable::AbsolutePosY).id();
                    let z = b.compile_variable(core::Variable::AbsolutePosZ).id();

                    let groups_x = b.compile_variable(core::Variable::CubeCountX).id();
                    let groups_y = b.compile_variable(core::Variable::CubeCountY).id();
                    let size_x = b.state.cube_dims[0];
                    let size_y = b.state.cube_dims[1];
                    let ty = u32::id(b);
                    let size_x = b.i_mul(ty, None, groups_x, size_x).unwrap();
                    let size_y = b.i_mul(ty, None, groups_y, size_y).unwrap();
                    let id = b.i_mul(ty, None, z, size_y).unwrap();
                    let id = b.i_add(ty, None, id, y).unwrap();
                    let id = b.i_mul(ty, None, id, size_x).unwrap();
                    let id = b.i_add(ty, None, id, x).unwrap();
                    b.debug_name(id, "ABSOLUTE_POS");
                    id
                });
                Variable::GlobalInvocationIndex(id)
            }
            core::Variable::AbsolutePosX => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIdX, |b| {
                    let id = b.extract(Globals::GlobalInvocationId, BuiltIn::GlobalInvocationId, 0);
                    b.debug_name(id, "ABSOLUTE_POS_X");
                    id
                });

                Variable::GlobalInvocationIdX(id)
            }
            core::Variable::AbsolutePosY => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIdY, |b| {
                    let id = b.extract(Globals::GlobalInvocationId, BuiltIn::GlobalInvocationId, 1);
                    b.debug_name(id, "ABSOLUTE_POS_Y");
                    id
                });

                Variable::GlobalInvocationIdY(id)
            }
            core::Variable::AbsolutePosZ => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIdZ, |b| {
                    let id = b.extract(Globals::GlobalInvocationId, BuiltIn::GlobalInvocationId, 2);
                    b.debug_name(id, "ABSOLUTE_POS_Z");
                    id
                });

                Variable::GlobalInvocationIdZ(id)
            }
            core::Variable::Rank => Variable::Rank(self.state.rank),
            core::Variable::ConstantArray { id, item, length } => {
                let item = self.compile_item(item);
                let id = self.state.const_arrays[id as usize].id;
                Variable::ConstantArray(id, item, length)
            }
            core::Variable::SharedMemory { id, item, length } => {
                let item = self.compile_item(item);
                let id = if let Some(arr) = self.state.shared_memories.get(&id) {
                    arr.id
                } else {
                    let arr_id = self.id();
                    let arr = Array {
                        id: arr_id,
                        item: item.clone(),
                        len: length,
                    };
                    self.state.shared_memories.insert(id, arr);
                    arr_id
                };
                Variable::SharedMemory(id, item, length)
            }
            core::Variable::LocalArray {
                id,
                item,
                depth,
                length,
            } => {
                let item = self.compile_item(item);
                let id = if let Some(arr) = self.state.local_arrays.get(&(id, depth)) {
                    arr.id
                } else {
                    let arr_ty = Item::Array(Box::new(item.clone()), length);
                    let ptr_ty = Item::Pointer(StorageClass::Function, Box::new(arr_ty)).id(self);
                    let arr_id = self.declare_function_variable(ptr_ty);
                    self.debug_name(arr_id, format!("array_{id}_{depth}"));
                    let arr = Array {
                        id: arr_id,
                        item: item.clone(),
                        len: length,
                    };
                    self.state.local_arrays.insert((id, depth), arr);
                    arr_id
                };
                Variable::LocalArray(id, item, length)
            }
            core::Variable::Matrix { .. } => todo!(),
        }
    }

    pub fn read(&mut self, variable: &Variable) -> Word {
        match variable {
            Variable::GlobalInputArray(id, _) => *id,
            Variable::GlobalOutputArray(id, _) => *id,
            Variable::SharedMemory(id, _, _) => *id,
            Variable::ConstantArray(id, _, _) => *id,
            Variable::LocalArray(id, _, _) => *id,
            Variable::Slice { ptr, .. } => self.read(ptr),
            Variable::Local { id, item } => {
                let ty = item.id(self);
                self.load(ty, None, *id, None, vec![]).unwrap()
            }
            Variable::Named { id, item, .. } => {
                let ty = item.id(self);
                self.load(ty, None, *id, None, vec![]).unwrap()
            }
            ssa => ssa.id(),
        }
    }

    pub fn read_to(&mut self, variable: &Variable, out_id: Word) -> Word {
        match variable {
            Variable::GlobalInputArray(_, _)
            | Variable::GlobalOutputArray(_, _)
            | Variable::SharedMemory(_, _, _)
            | Variable::ConstantArray(_, _, _)
            | Variable::LocalArray(_, _, _)
            | Variable::Slice { .. } => panic!("Can't read unindexed array"),
            Variable::Local { id, item } => {
                let ty = item.id(self);
                self.load(ty, Some(out_id), *id, None, vec![]).unwrap()
            }
            Variable::Named { id, item, .. } => {
                let ty = item.id(self);
                self.load(ty, Some(out_id), *id, None, vec![]).unwrap()
            }
            ssa => {
                let ty = ssa.item().id(self);
                self.copy_object(ty, Some(out_id), ssa.id()).unwrap()
            }
        }
    }

    pub fn read_as(&mut self, variable: &Variable, item: &Item) -> Word {
        let id = self.read(variable);
        variable.item().cast_to(self, id, item)
    }

    fn index(&mut self, variable: &Variable, index: &Variable, unchecked: bool) -> IndexedVariable {
        let access_chain = if unchecked {
            Builder::in_bounds_access_chain
        } else {
            Builder::access_chain
        };
        let index_id = self.read(index);
        match variable {
            Variable::GlobalInputArray(id, item)
            | Variable::GlobalOutputArray(id, item)
            | Variable::Named { id, item, .. } => {
                let ptr_ty =
                    Item::Pointer(StorageClass::StorageBuffer, Box::new(item.clone())).id(self);
                let zero = self.const_u32(0);
                let id = access_chain(self, ptr_ty, None, *id, vec![zero, index_id]).unwrap();

                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::Local {
                id,
                item: Item::Vector(elem, _),
            } => {
                let ptr_ty =
                    Item::Pointer(StorageClass::Function, Box::new(Item::Scalar(*elem))).id(self);
                let id = access_chain(self, ptr_ty, None, *id, vec![index_id]).unwrap();

                IndexedVariable::Pointer(id, Item::Scalar(*elem))
            }
            Variable::LocalBinding {
                id,
                item: Item::Vector(elem, vec),
            } => IndexedVariable::Composite(
                *id,
                index
                    .as_const()
                    .expect("Index into vector must be constant")
                    .as_u32(),
                Item::Vector(*elem, *vec),
            ),
            Variable::LocalBinding { .. } | Variable::Local { .. } => {
                let index = index
                    .as_const()
                    .expect("Index into vector must be constant")
                    .as_u32();
                if index > 0 {
                    panic!("Tried accessing {index}th element of scalar!");
                } else {
                    IndexedVariable::Scalar(variable.clone())
                }
            }
            Variable::Slice { ptr, offset, .. } => {
                let item = Item::Scalar(Elem::Int(32, false));
                let int = item.id(self);
                let index = self.i_add(int, None, *offset, index_id).unwrap();
                self.index(ptr, &Variable::LocalBinding { id: index, item }, unchecked)
            }
            Variable::SharedMemory(id, item, _) => {
                let ptr_ty =
                    Item::Pointer(StorageClass::Workgroup, Box::new(item.clone())).id(self);
                let id = access_chain(self, ptr_ty, None, *id, vec![index_id]).unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::ConstantArray(id, item, _) => {
                let ptr_ty =
                    Item::Pointer(StorageClass::UniformConstant, Box::new(item.clone())).id(self);
                let id = access_chain(self, ptr_ty, None, *id, vec![index_id]).unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::LocalArray(id, item, _) => {
                let ptr_ty = Item::Pointer(StorageClass::Function, Box::new(item.clone())).id(self);
                let id = access_chain(self, ptr_ty, None, *id, vec![index_id]).unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            var => unimplemented!("Can't index into {var:?}"),
        }
    }

    pub fn read_indexed(&mut self, out_id: Word, variable: &Variable, index: &Variable) -> Word {
        let checked = matches!(self.mode, ExecutionMode::Checked) && variable.has_len();
        let always_in_bounds = is_always_in_bounds(variable, index);
        let index_id = self.read(index);
        let indexed = self.index(variable, index, always_in_bounds);

        let read = |b: &mut Self| match indexed {
            IndexedVariable::Pointer(ptr, item) => {
                let ty = item.id(b);
                b.load(ty, Some(out_id), ptr, None, vec![]).unwrap()
            }
            IndexedVariable::Composite(var, index, item) => {
                let elem = item.elem();
                let ty = elem.id(b);
                b.composite_extract(ty, Some(out_id), var, vec![index])
                    .unwrap()
            }
            IndexedVariable::Scalar(var) => b.read_to(&var, out_id),
        };
        if checked && !always_in_bounds {
            self.compile_read_bound(variable, index_id, variable.item(), read)
        } else {
            read(self)
        }
    }

    pub fn write_id(&mut self, variable: &Variable) -> Word {
        match variable {
            Variable::LocalBinding { id, .. } => *id,
            Variable::Local { .. } => self.id(),
            Variable::LocalScalar { .. } => self.id(),
            Variable::ConstantScalar(_, _, _) => panic!("Can't write to constant scalar"),
            Variable::GlobalInputArray(_, _)
            | Variable::GlobalOutputArray(_, _)
            | Variable::Slice { .. }
            | Variable::Named { .. }
            | Variable::SharedMemory(_, _, _)
            | Variable::ConstantArray(_, _, _)
            | Variable::LocalArray(_, _, _) => panic!("Can't write to unindexed array"),
            global => panic!("Can't write to builtin {global:?}"),
        }
    }

    pub fn write(&mut self, variable: &Variable, value: Word) {
        match variable {
            Variable::Local { id, .. } => self.store(*id, value, None, vec![]).unwrap(),
            Variable::LocalScalar { id, .. } => self.store(*id, value, None, vec![]).unwrap(),
            Variable::Slice { ptr, .. } => self.write(ptr, value),
            _ => {}
        }
    }

    pub fn write_indexed(&mut self, out: &Variable, index: &Variable, value: Word) {
        let checked = matches!(self.mode, ExecutionMode::Checked) && out.has_len();
        let always_in_bounds = is_always_in_bounds(out, index);
        let index_id = self.read(index);
        let variable = self.index(out, index, always_in_bounds);

        let write = |b: &mut Self| match variable {
            IndexedVariable::Pointer(ptr, _) => b.store(ptr, value, None, vec![]).unwrap(),
            IndexedVariable::Composite(var, index, item) => {
                let out_id = b.write_id(out);
                let ty = item.id(b);
                b.composite_insert(ty, Some(out_id), value, var, vec![index])
                    .unwrap();
                b.write(out, out_id);
            }
            IndexedVariable::Scalar(var) => b.write(&var, value),
        };
        if checked && !always_in_bounds {
            self.compile_write_bound(out, index_id, write);
        } else {
            write(self)
        }
    }

    fn extract(&mut self, global: Globals, builtin: BuiltIn, idx: u32) -> Word {
        let composite_id = self.vec_global(global, builtin);
        let ty = Elem::Int(32, false).id(self);
        self.composite_extract(ty, None, composite_id, vec![idx])
            .unwrap()
    }

    fn vec_global(&mut self, global: Globals, builtin: BuiltIn) -> Word {
        let item = Item::Vector(Elem::Int(32, false), 3);

        self.get_or_insert_global(global, |b| b.load_builtin(builtin, item))
    }

    fn load_builtin(&mut self, builtin: BuiltIn, item: Item) -> Word {
        let item_id = item.id(self);
        let id = self.builtin(builtin, item);
        self.load(item_id, None, id, None, vec![]).unwrap()
    }
}

fn is_always_in_bounds(var: &Variable, index: &Variable) -> bool {
    let len = match var {
        Variable::SharedMemory(_, _, len)
        | Variable::ConstantArray(_, _, len)
        | Variable::LocalArray(_, _, len)
        | Variable::Slice {
            const_len: Some(len),
            ..
        } => *len,
        _ => return false,
    };

    let const_index = match index {
        Variable::ConstantScalar(_, value, _) => value.as_u32(),
        _ => return false,
    };

    const_index < len
}
