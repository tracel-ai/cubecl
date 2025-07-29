use cubecl_core::{CubeDim, ir::Builtin};

const NB_PASSED_BUILTIN: usize = 9;

#[derive(Default, Debug, Clone)]
pub struct BuiltinArray {
    pub dims: [u32; NB_PASSED_BUILTIN],
    //[
    //  cube_dim_x
    //  cube_dim_y
    //  cube_dim_z
    //  cube_count_x
    //  cube_count_y
    //  cube_count_z
    //  unit_pos_x
    //  unit_pos_y
    //  unit_pos_z
    //]
}

impl BuiltinArray {
    pub(crate) const fn builtin_order() -> [Builtin; 9] {
        [
            Builtin::CubeDimX,
            Builtin::CubeDimY,
            Builtin::CubeDimZ,
            Builtin::CubeCountX,
            Builtin::CubeCountY,
            Builtin::CubeCountZ,
            Builtin::UnitPosX,
            Builtin::UnitPosY,
            Builtin::UnitPosZ,
        ]
    }
    pub(crate) fn set_cube_dim(&mut self, cube_dim: CubeDim) {
        self.dims[0] = cube_dim.x;
        self.dims[1] = cube_dim.y;
        self.dims[2] = cube_dim.z;
    }
    pub(crate) fn set_cube_count(&mut self, cube_count: [u32; 3]) {
        self.dims[3] = cube_count[0];
        self.dims[4] = cube_count[1];
        self.dims[5] = cube_count[2];
    }
    pub(crate) fn set_unit_pos(&mut self, unit_pos: [u32; 3]) {
        self.dims[6] = unit_pos[0];
        self.dims[7] = unit_pos[1];
        self.dims[8] = unit_pos[2];
    }
    pub(crate) const fn len() -> usize {
        NB_PASSED_BUILTIN
    }
}
