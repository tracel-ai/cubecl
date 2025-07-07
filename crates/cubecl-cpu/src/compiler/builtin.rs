use cubecl_core::CubeDim;

#[derive(Default, Debug, Clone)]
pub struct Builtin {
    pub dims: [u32; 9],
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

impl Builtin {
    pub fn set_cube_dim(&mut self, cube_dim: CubeDim) {
        self.dims[0] = cube_dim.x;
        self.dims[1] = cube_dim.y;
        self.dims[2] = cube_dim.z;
    }
    pub fn get_cube_dim(&self) -> (u32, u32, u32) {
        (
            self.dims[0] as u32,
            self.dims[1] as u32,
            self.dims[2] as u32,
        )
    }
    pub fn set_cube_count(&mut self, cube_count: [u32; 3]) {
        self.dims[3] = cube_count[0];
        self.dims[4] = cube_count[1];
        self.dims[5] = cube_count[2];
    }
    pub fn set_unit_pos(&mut self, unit_pos: [u32; 3]) {
        self.dims[6] = unit_pos[0];
        self.dims[7] = unit_pos[1];
        self.dims[8] = unit_pos[2];
    }
    pub const fn len(&self) -> usize {
        self.dims.len()
    }
}
