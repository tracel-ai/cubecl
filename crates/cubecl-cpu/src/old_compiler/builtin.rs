use cubecl_core::{CubeDim, ir::Builtin};

const NB_PASSED_BUILTIN: usize = 9;

///[
///  `cube_dim_x`
///  `cube_dim_y`
///  `cube_dim_z`
///  `cube_count_x`
///  `cube_count_y`
///  `cube_count_z`
///  `unit_pos_x`
///  `unit_pos_y`
///  `unit_pos_z`
///]
#[derive(Default, Debug, Clone)]
pub struct BuiltinArray(pub [u32; NB_PASSED_BUILTIN]);

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
    pub fn new(cube_dim: CubeDim, cube_count: [u32; 3]) -> BuiltinArray {
        Self([
            cube_dim.x,
            cube_dim.y,
            cube_dim.z,
            cube_count[0],
            cube_count[1],
            cube_count[2],
            0,
            0,
            0,
        ])
    }
    pub(crate) fn set_unit_pos(&mut self, unit_pos: [u32; 3]) {
        self.0[6] = unit_pos[0];
        self.0[7] = unit_pos[1];
        self.0[8] = unit_pos[2];
    }
    pub(crate) const fn len() -> usize {
        NB_PASSED_BUILTIN
    }
}
