use cubecl_core::ir::Builtin;
use rspirv::spirv::{BuiltIn, Word};

use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    fn compile_builtin_u32(&mut self, builtin: Builtin) -> Word {
        self.compile_builtin(builtin, &Item::builtin_u32())
    }

    pub fn compile_builtin(&mut self, builtin: Builtin, ty: &Item) -> Word {
        match builtin {
            Builtin::UnitPos => self.insert_global(builtin, |b| {
                let id = b.load_builtin(BuiltIn::LocalInvocationIndex, ty);
                b.debug_name(id, "UNIT_POS");
                id
            }),
            Builtin::UnitPosX => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::LocalInvocationId, 0, ty);
                b.debug_name(id, "UNIT_POS_X");
                id
            }),
            Builtin::UnitPosY => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::LocalInvocationId, 1, ty);
                b.debug_name(id, "UNIT_POS_Y");
                id
            }),
            Builtin::UnitPosZ => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::LocalInvocationId, 2, ty);
                b.debug_name(id, "UNIT_POS_Z");
                id
            }),
            Builtin::CubePosX => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::WorkgroupId, 0, ty);
                b.debug_name(id, "CUBE_POS_X");
                id
            }),
            Builtin::CubePosY => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::WorkgroupId, 1, ty);
                b.debug_name(id, "CUBE_POS_Y");
                id
            }),
            Builtin::CubePosZ => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::WorkgroupId, 2, ty);
                b.debug_name(id, "CUBE_POS_Z");
                id
            }),
            Builtin::CubePosCluster
            | Builtin::CubePosClusterX
            | Builtin::CubePosClusterY
            | Builtin::CubePosClusterZ => ty.const_u32(self, 0),
            Builtin::CubeDim => self.state.cube_size,
            Builtin::CubeDimX => self.state.cube_dims[0],
            Builtin::CubeDimY => self.state.cube_dims[1],
            Builtin::CubeDimZ => self.state.cube_dims[2],
            Builtin::CubeClusterDim
            | Builtin::CubeClusterDimX
            | Builtin::CubeClusterDimY
            | Builtin::CubeClusterDimZ => ty.const_u32(self, 1),
            Builtin::CubeCount => self.insert_global(builtin, |b: &mut SpirvCompiler<T>| {
                let ty_id = ty.id(b);
                let x = b.compile_builtin_u32(Builtin::CubeCountX);
                let y = b.compile_builtin_u32(Builtin::CubeCountY);
                let z = b.compile_builtin_u32(Builtin::CubeCountZ);

                let x = Item::builtin_u32().cast_to(b, None, x, ty);
                let y = Item::builtin_u32().cast_to(b, None, y, ty);
                let z = Item::builtin_u32().cast_to(b, None, z, ty);

                let count = b.i_mul(ty_id, None, x, y).unwrap();
                let count = b.i_mul(ty_id, None, count, z).unwrap();
                b.debug_name(count, "CUBE_COUNT");
                count
            }),
            Builtin::CubeCountX => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::NumWorkgroups, 0, ty);
                b.debug_name(id, "CUBE_COUNT_X");
                id
            }),
            Builtin::CubeCountY => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::NumWorkgroups, 1, ty);
                b.debug_name(id, "CUBE_COUNT_Y");
                id
            }),
            Builtin::CubeCountZ => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::NumWorkgroups, 2, ty);
                b.debug_name(id, "CUBE_COUNT_Z");
                id
            }),
            Builtin::PlaneDim => self.insert_global(builtin, |b| {
                let id = b.load_builtin(BuiltIn::SubgroupSize, ty);
                b.debug_name(id, "PLANE_DIM");
                id
            }),
            Builtin::PlanePos => self.insert_global(builtin, |b| {
                let id = b.load_builtin(BuiltIn::SubgroupId, ty);
                b.debug_name(id, "PLANE_POS");
                id
            }),
            Builtin::UnitPosPlane => self.insert_global(builtin, |b| {
                let id = b.load_builtin(BuiltIn::SubgroupLocalInvocationId, ty);
                b.debug_name(id, "UNIT_POS_PLANE");
                id
            }),
            Builtin::CubePos => self.insert_global(builtin, |b| {
                let x = b.compile_builtin_u32(Builtin::CubePosX);
                let y = b.compile_builtin_u32(Builtin::CubePosY);
                let z = b.compile_builtin_u32(Builtin::CubePosZ);

                let x = Item::builtin_u32().cast_to(b, None, x, ty);
                let y = Item::builtin_u32().cast_to(b, None, y, ty);
                let z = Item::builtin_u32().cast_to(b, None, z, ty);

                let groups_x = b.compile_builtin_u32(Builtin::CubeCountX);
                let groups_y = b.compile_builtin_u32(Builtin::CubeCountY);

                let groups_x = Item::builtin_u32().cast_to(b, None, groups_x, ty);
                let groups_y = Item::builtin_u32().cast_to(b, None, groups_y, ty);

                let ty = ty.id(b);
                let id = b.i_mul(ty, None, z, groups_y).unwrap();
                let id = b.i_add(ty, None, id, y).unwrap();
                let id = b.i_mul(ty, None, id, groups_x).unwrap();
                let id = b.i_add(ty, None, id, x).unwrap();
                b.debug_name(id, "CUBE_POS");
                id
            }),
            Builtin::AbsolutePos => self.insert_global(builtin, |b| {
                let x = b.compile_builtin_u32(Builtin::AbsolutePosX);
                let y = b.compile_builtin_u32(Builtin::AbsolutePosY);
                let z = b.compile_builtin_u32(Builtin::AbsolutePosZ);

                let x = Item::builtin_u32().cast_to(b, None, x, ty);
                let y = Item::builtin_u32().cast_to(b, None, y, ty);
                let z = Item::builtin_u32().cast_to(b, None, z, ty);

                let groups_x = b.compile_builtin_u32(Builtin::CubeCountX);
                let groups_y = b.compile_builtin_u32(Builtin::CubeCountY);

                let groups_x = Item::builtin_u32().cast_to(b, None, groups_x, ty);
                let groups_y = Item::builtin_u32().cast_to(b, None, groups_y, ty);

                let size_x = ty.const_u32(b, b.cube_dim.x);
                let size_y = ty.const_u32(b, b.cube_dim.y);

                let ty = ty.id(b);
                let size_x = b.i_mul(ty, None, groups_x, size_x).unwrap();
                let size_y = b.i_mul(ty, None, groups_y, size_y).unwrap();
                let id = b.i_mul(ty, None, z, size_y).unwrap();
                let id = b.i_add(ty, None, id, y).unwrap();
                let id = b.i_mul(ty, None, id, size_x).unwrap();
                let id = b.i_add(ty, None, id, x).unwrap();
                b.debug_name(id, "ABSOLUTE_POS");
                id
            }),
            Builtin::AbsolutePosX => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::GlobalInvocationId, 0, ty);
                b.debug_name(id, "ABSOLUTE_POS_X");
                id
            }),
            Builtin::AbsolutePosY => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::GlobalInvocationId, 1, ty);
                b.debug_name(id, "ABSOLUTE_POS_Y");
                id
            }),
            Builtin::AbsolutePosZ => self.insert_global(builtin, |b| {
                let id = b.extract(BuiltIn::GlobalInvocationId, 2, ty);
                b.debug_name(id, "ABSOLUTE_POS_Z");
                id
            }),
        }
    }

    fn extract(&mut self, builtin: BuiltIn, idx: u32, ty: &Item) -> Word {
        let composite_id = self.vec_global(builtin);
        let ty = ty.id(self);
        self.composite_extract(ty, None, composite_id, vec![idx])
            .unwrap()
    }

    fn vec_global(&mut self, builtin: BuiltIn) -> Word {
        let item = Item::Vector(Elem::Int(32, false), 3);

        self.insert_builtin(builtin, |b| b.load_builtin(builtin, &item))
    }

    fn load_builtin(&mut self, builtin: BuiltIn, item: &Item) -> Word {
        let item_id = item.id(self);
        let id = self.builtin(builtin, item.clone());
        self.load(item_id, None, id, None, vec![]).unwrap()
    }
}
