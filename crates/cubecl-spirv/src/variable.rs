use std::mem::transmute;

use crate::{
    item::{Elem, HasId, Item},
    SpirvCompiler, SpirvTarget,
};
use cubecl_core::{
    ir::{self as core},
    ExecutionMode,
};
use rspirv::spirv::{BuiltIn, StorageClass, Word};

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    SubgroupSize(Word),
    GlobalInputArray(Word, Item),
    GlobalOutputArray(Word, Item),
    ConstantScalar(Word, Elem),
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

impl Variable {
    pub fn id(&self) -> Word {
        match self {
            Variable::GlobalInputArray(id, _) => *id,
            Variable::GlobalOutputArray(id, _) => *id,
            Variable::ConstantScalar(id, _) => *id,
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
            Variable::ConstantScalar(_, elem) => Item::Scalar(*elem),
            Variable::Local { item, .. } => item.clone(),
            Variable::LocalBinding { item, .. } => item.clone(),
            Variable::Named { item, .. } => item.clone(),
            Variable::Slice { item, .. } => item.clone(),
            Variable::LocalScalar { elem, .. } => Item::Scalar(*elem),
            Variable::SharedMemory(_, item, _) => item.clone(),
            Variable::ConstantArray(_, item, _) => item.clone(),
            Variable::LocalArray(_, item, _) => item.clone(),
            _ => Item::Scalar(Elem::Int(32)), // builtin
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
}

pub enum IndexedVariable {
    Pointer(Word, Item),
    Composite(Word, u32, Item),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Globals {
    Id,
    LocalInvocationIndex,
    LocalInvocationId,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    Rank,
    WorkgroupId,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
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
                let map_val: u64 = match value {
                    core::ConstantScalarValue::Int(val, _) => unsafe { transmute::<i64, u64>(val) },
                    core::ConstantScalarValue::Float(val, _) => val.to_bits(),
                    core::ConstantScalarValue::UInt(val) => val,
                    core::ConstantScalarValue::Bool(val) => val as u64,
                };
                if let Some(existing) = self.state.constants.get(&(map_val, item.clone())) {
                    Variable::ConstantScalar(*existing, item.elem())
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
                    Variable::ConstantScalar(id, item.elem())
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
                    self.state.bindings.insert((id, depth), binding);
                    Variable::LocalBinding { id: binding, item }
                }
            }
            core::Variable::UnitPos => Variable::LocalInvocationIndex(
                self.get_or_insert_global(Globals::GlobalInvocationIndex, |b| {
                    b.load_builtin(BuiltIn::LocalInvocationIndex, Item::Scalar(Elem::Int(32)))
                }),
            ),
            core::Variable::UnitPosX => Variable::LocalInvocationIdX(
                self.get_or_insert_global(Globals::LocalInvocationIdX, |b| {
                    b.extract(Globals::LocalInvocationId, BuiltIn::LocalInvocationId, 0)
                }),
            ),
            core::Variable::UnitPosY => Variable::LocalInvocationIdX(
                self.get_or_insert_global(Globals::LocalInvocationIdX, |b| {
                    b.extract(Globals::LocalInvocationId, BuiltIn::LocalInvocationId, 1)
                }),
            ),
            core::Variable::UnitPosZ => Variable::LocalInvocationIdX(
                self.get_or_insert_global(Globals::LocalInvocationIdX, |b| {
                    b.extract(Globals::LocalInvocationId, BuiltIn::LocalInvocationId, 2)
                }),
            ),
            core::Variable::CubePosX => {
                Variable::WorkgroupIdX(self.get_or_insert_global(Globals::WorkgroupIdX, |b| {
                    b.extract(Globals::WorkgroupId, BuiltIn::WorkgroupId, 2)
                }))
            }
            core::Variable::CubePosY => {
                Variable::WorkgroupIdY(self.get_or_insert_global(Globals::WorkgroupIdY, |b| {
                    b.extract(Globals::WorkgroupId, BuiltIn::WorkgroupId, 1)
                }))
            }
            core::Variable::CubePosZ => {
                Variable::WorkgroupIdZ(self.get_or_insert_global(Globals::WorkgroupIdZ, |b| {
                    b.extract(Globals::WorkgroupId, BuiltIn::WorkgroupId, 2)
                }))
            }
            core::Variable::CubeDim => Variable::WorkgroupSize(self.state.cube_size),
            core::Variable::CubeDimX => Variable::WorkgroupSizeX(self.state.cube_dims[0]),
            core::Variable::CubeDimY => Variable::WorkgroupSizeY(self.state.cube_dims[1]),
            core::Variable::CubeDimZ => Variable::WorkgroupSizeZ(self.state.cube_dims[2]),
            core::Variable::CubeCount => todo!(),
            core::Variable::CubeCountX => {
                Variable::WorkgroupSizeZ(self.get_or_insert_global(Globals::NumWorkgroupsZ, |b| {
                    b.extract(Globals::NumWorkgroups, BuiltIn::NumWorkgroups, 0)
                }))
            }
            core::Variable::CubeCountY => {
                Variable::WorkgroupSizeY(self.get_or_insert_global(Globals::NumWorkgroupsZ, |b| {
                    b.extract(Globals::NumWorkgroups, BuiltIn::NumWorkgroups, 1)
                }))
            }
            core::Variable::CubeCountZ => {
                Variable::WorkgroupSizeZ(self.get_or_insert_global(Globals::NumWorkgroupsZ, |b| {
                    b.extract(Globals::NumWorkgroups, BuiltIn::NumWorkgroups, 2)
                }))
            }
            core::Variable::SubcubeDim => {
                let id = self.get_or_insert_global(Globals::SubgroupSize, |b| {
                    b.load_builtin(BuiltIn::SubgroupSize, Item::Scalar(Elem::Int(32)))
                });
                Variable::SubgroupSize(id)
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
                    let stride_y = b.i_mul(ty, None, groups_x, size_x).unwrap();
                    let size_y = b.i_mul(ty, None, groups_y, size_y).unwrap();
                    let stride_z = b.i_mul(ty, None, stride_y, size_y).unwrap();
                    let z = b.i_mul(ty, None, z, stride_z).unwrap();
                    let y = b.i_mul(ty, None, y, stride_y).unwrap();
                    let id = b.i_add(ty, None, y, z).unwrap();
                    b.i_add(ty, None, id, x).unwrap()
                });
                Variable::GlobalInvocationIndex(id)
            }
            core::Variable::AbsolutePosX => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIdX, |b| {
                    b.extract(Globals::GlobalInvocationId, BuiltIn::GlobalInvocationId, 0)
                });

                Variable::GlobalInvocationIdX(id)
            }
            core::Variable::AbsolutePosY => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIdY, |b| {
                    b.extract(Globals::GlobalInvocationId, BuiltIn::GlobalInvocationId, 1)
                });

                Variable::GlobalInvocationIdY(id)
            }
            core::Variable::AbsolutePosZ => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIdZ, |b| {
                    b.extract(Globals::GlobalInvocationId, BuiltIn::GlobalInvocationId, 2)
                });

                Variable::GlobalInvocationIdZ(id)
            }
            var => todo!("{var:?}"),
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

    fn index(&mut self, variable: &Variable, index: Word) -> IndexedVariable {
        match variable {
            Variable::GlobalInputArray(id, item)
            | Variable::GlobalOutputArray(id, item)
            | Variable::Named { id, item, .. } => {
                let ptr_ty =
                    Item::Pointer(StorageClass::StorageBuffer, Box::new(item.clone())).id(self);
                let zero = self.const_u32(0);
                let id = self
                    .access_chain(ptr_ty, None, *id, vec![zero, index])
                    .unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::Local {
                id,
                item: Item::Vector(elem, vec),
            } => IndexedVariable::Composite(*id, index, Item::Vector(*elem, *vec)),
            Variable::LocalBinding {
                id,
                item: Item::Vector(elem, vec),
            } => IndexedVariable::Composite(*id, index, Item::Vector(*elem, *vec)),
            Variable::Slice { ptr, offset, .. } => {
                let int = Elem::Int(32).id(self);
                let index = self.i_add(int, None, *offset, index).unwrap();
                self.index(ptr, index)
            }
            Variable::SharedMemory(id, item, _) => {
                let ptr_ty =
                    Item::Pointer(StorageClass::Workgroup, Box::new(item.clone())).id(self);
                let id = self.access_chain(ptr_ty, None, *id, vec![index]).unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::ConstantArray(id, item, _) => {
                let ptr_ty =
                    Item::Pointer(StorageClass::UniformConstant, Box::new(item.clone())).id(self);
                let id = self.access_chain(ptr_ty, None, *id, vec![index]).unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::LocalArray(id, item, _) => {
                let ptr_ty = Item::Pointer(StorageClass::Function, Box::new(item.clone())).id(self);
                let id = self.access_chain(ptr_ty, None, *id, vec![index]).unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }

            var => unimplemented!("Can't index into {var:?}"),
        }
    }

    pub fn read_indexed(&mut self, out_id: Word, variable: &Variable, index: Word) -> Word {
        let checked = matches!(self.mode, ExecutionMode::Checked);
        let read = |b: &mut Self| {
            let variable = b.index(variable, index);
            match variable {
                IndexedVariable::Pointer(ptr, item) => {
                    let ty = item.id(b);
                    b.load(ty, Some(out_id), ptr, None, vec![]).unwrap()
                }
                IndexedVariable::Composite(var, index, item) => {
                    let ty = item.id(b);
                    b.composite_extract(ty, Some(out_id), var, vec![index])
                        .unwrap()
                }
            }
        };
        if checked && variable.has_len() {
            self.compile_read_bound(variable, index, variable.item(), read)
        } else {
            read(self)
        }
    }

    pub fn write_id(&mut self, variable: &Variable) -> Word {
        match variable {
            Variable::LocalBinding { id, .. } => *id,
            Variable::Local { .. } => self.id(),
            Variable::LocalScalar { .. } => self.id(),
            Variable::ConstantScalar(_, _) => panic!("Can't write to constant scalar"),
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

    pub fn write_indexed(&mut self, out: &Variable, index: Word, value: Word) {
        let checked = matches!(self.mode, ExecutionMode::Checked);
        let write = |b: &mut Self| {
            let variable = b.index(out, index);
            match variable {
                IndexedVariable::Pointer(ptr, _) => b.store(ptr, value, None, vec![]).unwrap(),
                IndexedVariable::Composite(var, index, item) => {
                    let ty = item.id(b);
                    b.composite_insert(ty, None, value, var, vec![index])
                        .unwrap();
                }
            }
        };
        if checked && out.has_len() {
            self.compile_write_bound(out, index, write);
        } else {
            write(self)
        }
    }

    fn extract(&mut self, global: Globals, builtin: BuiltIn, idx: u32) -> Word {
        let composite_id = self.vec_global(global, builtin);
        let ty = Elem::Int(32).id(self);
        self.composite_extract(ty, None, composite_id, vec![idx])
            .unwrap()
    }

    fn vec_global(&mut self, global: Globals, builtin: BuiltIn) -> Word {
        let item = Item::Vector(Elem::Int(32), 3);

        self.get_or_insert_global(global, |b| b.load_builtin(builtin, item))
    }

    fn load_builtin(&mut self, builtin: BuiltIn, item: Item) -> Word {
        let item_id = item.id(self);
        let id = self.builtin(builtin, item);
        self.load(item_id, None, id, None, vec![]).unwrap()
    }
}
