use cubecl_core::ir::{self, Builtin, UIntKind};
use rspirv::spirv::{BuiltIn, Word};

use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::Variable,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_builtin(&mut self, builtin: Builtin) -> Variable {
        match builtin {
            Builtin::UnitPos => Variable::LocalInvocationIndex(self.insert_global(|b| {
                let id = b.load_builtin(
                    BuiltIn::LocalInvocationIndex,
                    Item::Scalar(Elem::Int(32, false)),
                );
                b.debug_name(id, "UNIT_POS");
                id
            })),
            Builtin::UnitPosX => Variable::LocalInvocationIdX(self.insert_global(|b| {
                let id = b.extract(BuiltIn::LocalInvocationId, 0);
                b.debug_name(id, "UNIT_POS_X");
                id
            })),
            Builtin::UnitPosY => Variable::LocalInvocationIdY(self.insert_global(|b| {
                let id = b.extract(BuiltIn::LocalInvocationId, 1);
                b.debug_name(id, "UNIT_POS_Y");
                id
            })),
            Builtin::UnitPosZ => Variable::LocalInvocationIdZ(self.insert_global(|b| {
                let id = b.extract(BuiltIn::LocalInvocationId, 2);
                b.debug_name(id, "UNIT_POS_Z");
                id
            })),
            Builtin::CubePosX => Variable::WorkgroupIdX(self.insert_global(|b| {
                let id = b.extract(BuiltIn::WorkgroupId, 0);
                b.debug_name(id, "CUBE_POS_X");
                id
            })),
            Builtin::CubePosY => Variable::WorkgroupIdY(self.insert_global(|b| {
                let id = b.extract(BuiltIn::WorkgroupId, 1);
                b.debug_name(id, "CUBE_POS_Y");
                id
            })),
            Builtin::CubePosZ => Variable::WorkgroupIdZ(self.insert_global(|b| {
                let id = b.extract(BuiltIn::WorkgroupId, 2);
                b.debug_name(id, "CUBE_POS_Z");
                id
            })),
            Builtin::CubePosCluster
            | Builtin::CubePosClusterX
            | Builtin::CubePosClusterY
            | Builtin::CubePosClusterZ => self.constant_var(0),
            Builtin::CubeDim => Variable::WorkgroupSize(self.state.cube_size),
            Builtin::CubeDimX => Variable::WorkgroupSizeX(self.state.cube_dims[0]),
            Builtin::CubeDimY => Variable::WorkgroupSizeY(self.state.cube_dims[1]),
            Builtin::CubeDimZ => Variable::WorkgroupSizeZ(self.state.cube_dims[2]),
            Builtin::CubeClusterDim
            | Builtin::CubeClusterDimX
            | Builtin::CubeClusterDimY
            | Builtin::CubeClusterDimZ => self.constant_var(1),
            Builtin::CubeCount => {
                Variable::WorkgroupSize(self.insert_global(|b: &mut SpirvCompiler<T>| {
                    let int = b.type_int(32, 0);
                    let x = b.compile_variable(built_var(Builtin::CubeCountX)).id(b);
                    let y = b.compile_variable(built_var(Builtin::CubeCountY)).id(b);
                    let z = b.compile_variable(built_var(Builtin::CubeCountZ)).id(b);
                    let count = b.i_mul(int, None, x, y).unwrap();
                    let count = b.i_mul(int, None, count, z).unwrap();
                    b.debug_name(count, "CUBE_COUNT");
                    count
                }))
            }
            Builtin::CubeCountX => Variable::NumWorkgroupsX(self.insert_global(|b| {
                let id = b.extract(BuiltIn::NumWorkgroups, 0);
                b.debug_name(id, "CUBE_COUNT_X");
                id
            })),
            Builtin::CubeCountY => Variable::NumWorkgroupsY(self.insert_global(|b| {
                let id = b.extract(BuiltIn::NumWorkgroups, 1);
                b.debug_name(id, "CUBE_COUNT_Y");
                id
            })),
            Builtin::CubeCountZ => Variable::NumWorkgroupsZ(self.insert_global(|b| {
                let id = b.extract(BuiltIn::NumWorkgroups, 2);
                b.debug_name(id, "CUBE_COUNT_Z");
                id
            })),
            Builtin::PlaneDim => {
                let id = self.insert_global(|b| {
                    let id =
                        b.load_builtin(BuiltIn::SubgroupSize, Item::Scalar(Elem::Int(32, false)));
                    b.debug_name(id, "PLANE_DIM");
                    id
                });
                Variable::SubgroupSize(id)
            }
            Builtin::UnitPosPlane => {
                let id = self.insert_global(|b| {
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
                let id = self.insert_global(|b| {
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
                let id = self.insert_global(|b| {
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
                let id = self.insert_global(|b| {
                    let id = b.extract(BuiltIn::GlobalInvocationId, 0);
                    b.debug_name(id, "ABSOLUTE_POS_X");
                    id
                });

                Variable::GlobalInvocationIdX(id)
            }
            Builtin::AbsolutePosY => {
                let id = self.insert_global(|b| {
                    let id = b.extract(BuiltIn::GlobalInvocationId, 1);
                    b.debug_name(id, "ABSOLUTE_POS_Y");
                    id
                });

                Variable::GlobalInvocationIdY(id)
            }
            Builtin::AbsolutePosZ => {
                let id = self.insert_global(|b| {
                    let id = b.extract(BuiltIn::GlobalInvocationId, 2);
                    b.debug_name(id, "ABSOLUTE_POS_Z");
                    id
                });

                Variable::GlobalInvocationIdZ(id)
            }
        }
    }

    fn constant_var(&mut self, value: u32) -> Variable {
        let var =
            ir::Variable::constant(ir::ConstantScalarValue::UInt(value as u64, UIntKind::U32));
        self.compile_variable(var)
    }

    fn extract(&mut self, builtin: BuiltIn, idx: u32) -> Word {
        let composite_id = self.vec_global(builtin);
        let ty = Elem::Int(32, false).id(self);
        self.composite_extract(ty, None, composite_id, vec![idx])
            .unwrap()
    }

    fn vec_global(&mut self, builtin: BuiltIn) -> Word {
        let item = Item::Vector(Elem::Int(32, false), 3);

        self.insert_global(|b| b.load_builtin(builtin, item))
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
