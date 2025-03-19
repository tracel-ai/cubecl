use cubecl_core::ir::{self, Builtin};
use rspirv::spirv::{BuiltIn, Word};

use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::{Globals, Variable},
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_builtin(&mut self, builtin: Builtin) -> Variable {
        match builtin {
            Builtin::UnitPos => Variable::LocalInvocationIndex(self.get_or_insert_global(
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
            Builtin::UnitPosX => Variable::LocalInvocationIdX(self.get_or_insert_global(
                Globals::LocalInvocationIdX,
                |b| {
                    let id = b.extract(Globals::LocalInvocationId, BuiltIn::LocalInvocationId, 0);
                    b.debug_name(id, "UNIT_POS_X");
                    id
                },
            )),
            Builtin::UnitPosY => Variable::LocalInvocationIdY(self.get_or_insert_global(
                Globals::LocalInvocationIdY,
                |b| {
                    let id = b.extract(Globals::LocalInvocationId, BuiltIn::LocalInvocationId, 1);
                    b.debug_name(id, "UNIT_POS_Y");
                    id
                },
            )),
            Builtin::UnitPosZ => Variable::LocalInvocationIdZ(self.get_or_insert_global(
                Globals::LocalInvocationIdZ,
                |b| {
                    let id = b.extract(Globals::LocalInvocationId, BuiltIn::LocalInvocationId, 2);
                    b.debug_name(id, "UNIT_POS_Z");
                    id
                },
            )),
            Builtin::CubePosX => {
                Variable::WorkgroupIdX(self.get_or_insert_global(Globals::WorkgroupIdX, |b| {
                    let id = b.extract(Globals::WorkgroupId, BuiltIn::WorkgroupId, 0);
                    b.debug_name(id, "CUBE_POS_X");
                    id
                }))
            }
            Builtin::CubePosY => {
                Variable::WorkgroupIdY(self.get_or_insert_global(Globals::WorkgroupIdY, |b| {
                    let id = b.extract(Globals::WorkgroupId, BuiltIn::WorkgroupId, 1);
                    b.debug_name(id, "CUBE_POS_Y");
                    id
                }))
            }
            Builtin::CubePosZ => {
                Variable::WorkgroupIdZ(self.get_or_insert_global(Globals::WorkgroupIdZ, |b| {
                    let id = b.extract(Globals::WorkgroupId, BuiltIn::WorkgroupId, 2);
                    b.debug_name(id, "CUBE_POS_Z");
                    id
                }))
            }
            Builtin::CubeDim => Variable::WorkgroupSize(self.state.cube_size),
            Builtin::CubeDimX => Variable::WorkgroupSizeX(self.state.cube_dims[0]),
            Builtin::CubeDimY => Variable::WorkgroupSizeY(self.state.cube_dims[1]),
            Builtin::CubeDimZ => Variable::WorkgroupSizeZ(self.state.cube_dims[2]),
            Builtin::CubeCount => Variable::WorkgroupSize(self.get_or_insert_global(
                Globals::NumWorkgroupsTotal,
                |b: &mut SpirvCompiler<T>| {
                    let int = b.type_int(32, 0);
                    let x = b.compile_variable(built_var(Builtin::CubeCountX)).id(b);
                    let y = b.compile_variable(built_var(Builtin::CubeCountY)).id(b);
                    let z = b.compile_variable(built_var(Builtin::CubeCountZ)).id(b);
                    let count = b.i_mul(int, None, x, y).unwrap();
                    let count = b.i_mul(int, None, count, z).unwrap();
                    b.debug_name(count, "CUBE_COUNT");
                    count
                },
            )),
            Builtin::CubeCountX => {
                Variable::NumWorkgroupsX(self.get_or_insert_global(Globals::NumWorkgroupsX, |b| {
                    let id = b.extract(Globals::NumWorkgroups, BuiltIn::NumWorkgroups, 0);
                    b.debug_name(id, "CUBE_COUNT_X");
                    id
                }))
            }
            Builtin::CubeCountY => {
                Variable::NumWorkgroupsY(self.get_or_insert_global(Globals::NumWorkgroupsY, |b| {
                    let id = b.extract(Globals::NumWorkgroups, BuiltIn::NumWorkgroups, 1);
                    b.debug_name(id, "CUBE_COUNT_Y");
                    id
                }))
            }
            Builtin::CubeCountZ => {
                Variable::NumWorkgroupsZ(self.get_or_insert_global(Globals::NumWorkgroupsZ, |b| {
                    let id = b.extract(Globals::NumWorkgroups, BuiltIn::NumWorkgroups, 2);
                    b.debug_name(id, "CUBE_COUNT_Z");
                    id
                }))
            }
            Builtin::PlaneDim => {
                let id = self.get_or_insert_global(Globals::SubgroupSize, |b| {
                    let id =
                        b.load_builtin(BuiltIn::SubgroupSize, Item::Scalar(Elem::Int(32, false)));
                    b.debug_name(id, "PLANE_DIM");
                    id
                });
                Variable::SubgroupSize(id)
            }
            Builtin::UnitPosPlane => {
                let id = self.get_or_insert_global(Globals::SubgroupInvocationId, |b| {
                    let id = b.load_builtin(
                        BuiltIn::SubgroupLocalInvocationId,
                        Item::Scalar(Elem::Int(32, false)),
                    );
                    b.debug_name(id, "UNIT_POS_PLANE");
                    id
                });
                Variable::SubgroupSize(id)
            }
            Builtin::CubePos => {
                let id = self.get_or_insert_global(Globals::WorkgroupIndex, |b| {
                    let x = b.compile_variable(built_var(Builtin::CubePosX)).id(b);
                    let y = b.compile_variable(built_var(Builtin::CubePosY)).id(b);
                    let z = b.compile_variable(built_var(Builtin::CubePosZ)).id(b);

                    let groups_x = b.compile_variable(built_var(Builtin::CubeCountX)).id(b);
                    let groups_y = b.compile_variable(built_var(Builtin::CubeCountY)).id(b);
                    let ty = Elem::Int(32, false).id(b);
                    let id = b.i_mul(ty, None, z, groups_y).unwrap();
                    let id = b.i_add(ty, None, id, y).unwrap();
                    let id = b.i_mul(ty, None, id, groups_x).unwrap();
                    let id = b.i_add(ty, None, id, x).unwrap();
                    b.debug_name(id, "CUBE_POS");
                    id
                });
                Variable::WorkgroupId(id)
            }
            Builtin::AbsolutePos => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIndex, |b| {
                    let x = b.compile_variable(built_var(Builtin::AbsolutePosX)).id(b);
                    let y = b.compile_variable(built_var(Builtin::AbsolutePosY)).id(b);
                    let z = b.compile_variable(built_var(Builtin::AbsolutePosZ)).id(b);

                    let groups_x = b.compile_variable(built_var(Builtin::CubeCountX)).id(b);
                    let groups_y = b.compile_variable(built_var(Builtin::CubeCountY)).id(b);
                    let size_x = b.state.cube_dims[0];
                    let size_y = b.state.cube_dims[1];
                    let ty = Elem::Int(32, false).id(b);
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
            Builtin::AbsolutePosX => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIdX, |b| {
                    let id = b.extract(Globals::GlobalInvocationId, BuiltIn::GlobalInvocationId, 0);
                    b.debug_name(id, "ABSOLUTE_POS_X");
                    id
                });

                Variable::GlobalInvocationIdX(id)
            }
            Builtin::AbsolutePosY => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIdY, |b| {
                    let id = b.extract(Globals::GlobalInvocationId, BuiltIn::GlobalInvocationId, 1);
                    b.debug_name(id, "ABSOLUTE_POS_Y");
                    id
                });

                Variable::GlobalInvocationIdY(id)
            }
            Builtin::AbsolutePosZ => {
                let id = self.get_or_insert_global(Globals::GlobalInvocationIdZ, |b| {
                    let id = b.extract(Globals::GlobalInvocationId, BuiltIn::GlobalInvocationId, 2);
                    b.debug_name(id, "ABSOLUTE_POS_Z");
                    id
                });

                Variable::GlobalInvocationIdZ(id)
            }
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

fn built_var(builtin: Builtin) -> ir::Variable {
    ir::Variable::builtin(builtin)
}
