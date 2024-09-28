use std::mem::transmute;

use crate::{
    item::{Elem, HasId, Item},
    SpirvCompiler, SpirvTarget,
};
use cubecl_core::ir::{self as core};
use rspirv::spirv::{BuiltIn, StorageClass, Word};

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    SubgroupSize(Word),
    GlobalInputArray(Word, Item),
    GlobalOutputArray(Word, Item),
    ConstantScalar(u32, Elem),
    Local {
        id: u32,
        item: Item,
    },
    LocalBinding {
        id: u32,
        item: Item,
    },
    Named {
        name: String,
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
        id: u16,
        elem: Elem,
        depth: u8,
    },
    SharedMemory(u16, Item, u32),
    ConstantArray(u16, Item, u32),
    LocalArray(u16, Item, u8, u32),
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
            Variable::GlobalInputArray(_, item) => todo!(),
            Variable::GlobalOutputArray(_, item) => todo!(),
            Variable::ConstantScalar(id, _) => *id,
            Variable::Local { id, .. } => *id,
            Variable::LocalBinding { id, .. } => *id,
            Variable::Named {
                name,
                item,
                is_array,
            } => todo!(),
            Variable::Slice { ptr, .. } => ptr.id(),
            Variable::LocalScalar { id, elem, depth } => todo!(),
            Variable::SharedMemory(_, item, _) => todo!(),
            Variable::ConstantArray(_, item, _) => todo!(),
            Variable::LocalArray(_, item, _, _) => todo!(),
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
            Variable::LocalArray(_, item, _, _) => item.clone(),
            _ => Item::Scalar(Elem::Int(32)), // builtin
        }
    }

    pub fn elem(&self) -> Elem {
        self.item().elem()
    }
}

pub enum IndexedVariable {
    Pointer(Word, Item),
    Composite(Word, u32, Item),
}

impl IndexedVariable {
    pub fn id(&self) -> Word {
        match self {
            IndexedVariable::Pointer(ptr, _) => *ptr,
            IndexedVariable::Composite(ptr, _, _) => *ptr,
        }
    }
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
                let elem = self.compile_item(core::Item::new(value.elem())).elem();
                let elem_id = elem.id(self);
                let map_val: u64 = match value {
                    core::ConstantScalarValue::Int(val, _) => unsafe { transmute::<i64, u64>(val) },
                    core::ConstantScalarValue::Float(val, _) => val.to_bits(),
                    core::ConstantScalarValue::UInt(val) => val,
                    core::ConstantScalarValue::Bool(val) => val as u64,
                };
                if let Some(existing) = self.state.constants.get(&(map_val, elem)) {
                    Variable::ConstantScalar(*existing, elem)
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
                    self.state.constants.insert((map_val, elem), id);
                    Variable::ConstantScalar(id, elem)
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
            Variable::Local { id, item } => {
                let ty = item.id(self);
                self.load(ty, None, *id, None, vec![]).unwrap()
            }
            Variable::Named {
                name,
                item,
                is_array,
            } => todo!(),
            Variable::Slice { ptr, item, .. } => self.read(ptr),
            Variable::LocalScalar { id, elem, depth } => todo!(),
            Variable::SharedMemory(_, item, _) => todo!(),
            Variable::ConstantArray(_, item, _) => todo!(),
            Variable::LocalArray(_, item, _, _) => todo!(),
            ssa => ssa.id(),
        }
    }

    pub fn read_indexed(&mut self, out_id: Word, variable: &IndexedVariable) -> Word {
        match variable {
            IndexedVariable::Pointer(ptr, item) => {
                let ty = item.id(self);
                self.load(ty, Some(out_id), *ptr, None, vec![]).unwrap()
            }
            IndexedVariable::Composite(var, index, item) => {
                let ty = item.id(self);
                self.composite_extract(ty, Some(out_id), *var, vec![*index])
                    .unwrap()
            }
        }
    }

    pub fn index(&mut self, variable: &Variable, index: Word) -> IndexedVariable {
        match variable {
            Variable::GlobalInputArray(id, item) | Variable::GlobalOutputArray(id, item) => {
                let ptr_ty =
                    Item::Pointer(StorageClass::StorageBuffer, Box::new(item.clone())).id(self);
                let zero = self.const_u32(0);
                let id = self
                    .access_chain(ptr_ty, None, *id, vec![zero, index])
                    .unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::Local { id, item } => todo!(),
            Variable::LocalBinding { id, item } => todo!(),
            Variable::Named {
                name,
                item,
                is_array,
            } => todo!(),
            Variable::Slice {
                ptr, offset, item, ..
            } => {
                let int = Elem::Int(32).id(self);
                let index = self.i_add(int, None, *offset, index).unwrap();
                self.index(ptr, index)
            }
            Variable::LocalScalar { id, elem, depth } => todo!(),
            Variable::SharedMemory(_, item, _) => todo!(),
            Variable::ConstantArray(_, item, _) => todo!(),
            Variable::LocalArray(_, item, _, _) => todo!(),
            var => unimplemented!("Can't index into {var:?}"),
        }
    }

    pub fn write_id(&mut self, variable: &Variable) -> Word {
        match variable {
            Variable::GlobalInputArray(_, item) => todo!(),
            Variable::GlobalOutputArray(_, item) => todo!(),
            Variable::ConstantScalar(_, elem) => todo!(),
            Variable::Local { .. } => self.id(),
            Variable::LocalBinding { id, .. } => *id,
            Variable::Named {
                name,
                item,
                is_array,
            } => todo!(),
            Variable::Slice { .. } => self.id(),
            Variable::LocalScalar { id, elem, depth } => todo!(),
            Variable::SharedMemory(_, item, _) => todo!(),
            Variable::ConstantArray(_, item, _) => todo!(),
            Variable::LocalArray(_, item, _, _) => todo!(),
            global => panic!("Can't write to builtin {global:?}"),
        }
    }

    pub fn write(&mut self, variable: &Variable, value: Word) {
        match variable {
            Variable::GlobalInputArray(_, item) => todo!(),
            Variable::GlobalOutputArray(_, item) => todo!(),
            Variable::ConstantScalar(_, elem) => todo!(),
            Variable::Local { id, .. } => self.store(*id, value, None, vec![]).unwrap(),
            Variable::Named {
                name,
                item,
                is_array,
            } => todo!(),
            Variable::Slice { ptr, .. } => self.write(ptr, value),
            Variable::LocalScalar { id, elem, depth } => todo!(),
            Variable::SharedMemory(_, item, _) => todo!(),
            Variable::ConstantArray(_, item, _) => todo!(),
            Variable::LocalArray(_, item, _, _) => todo!(),
            _ => {}
        }
    }

    pub fn write_indexed(&mut self, variable: &IndexedVariable, value: Word) {
        match variable {
            IndexedVariable::Pointer(ptr, _) => self.store(*ptr, value, None, vec![]).unwrap(),
            IndexedVariable::Composite(var, index, item) => {
                let ty = item.id(self);
                self.composite_insert(ty, None, value, *var, vec![*index])
                    .unwrap();
            }
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
