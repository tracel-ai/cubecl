use cubecl_core::CubeDim;

#[derive(Default)]
pub struct Builtin {
    pub dims: [u64; 9],
    //[
    //  cube_dim_x
    //  cube_dim_y
    //  cube_dim_z
    //  cube_count_x
    //  cube_count_y
    //  cube_count_z
    //  cube_pos_x
    //  cube_pos_y
    //  cube_pos_z
    //]
}

impl Builtin {
    pub fn set_cube_dim(&mut self, cube_dim: CubeDim) {
        self.dims[0] = cube_dim.x as u64;
        self.dims[1] = cube_dim.y as u64;
        self.dims[2] = cube_dim.z as u64;
    }
    pub fn get_cube_dim(&self) -> (u32, u32, u32) {
        (
            self.dims[0] as u32,
            self.dims[1] as u32,
            self.dims[2] as u32,
        )
    }
    pub fn set_cube_count(&mut self, cube_count: (u32, u32, u32)) {
        self.dims[3] = cube_count.0 as u64;
        self.dims[4] = cube_count.1 as u64;
        self.dims[5] = cube_count.2 as u64;
    }
    pub fn set_cube_pos_x(&mut self, x: u32) {
        self.dims[6] = x as u64;
    }
    pub fn set_cube_pos_y(&mut self, y: u32) {
        self.dims[7] = y as u64;
    }
    pub fn set_cube_pos_z(&mut self, z: u32) {
        self.dims[8] = z as u64;
    }
}
