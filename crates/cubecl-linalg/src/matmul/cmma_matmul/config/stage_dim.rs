#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct StageDims {
    pub lhs: StageDim,
    pub rhs: StageDim,
    pub out: StageDim,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct StageDim {
    pub tile_size_x: u32,
    pub tile_size_y: u32,
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
}

impl StageDim {
    pub fn num_elements(&self) -> u32 {
        self.num_tiles_x * self.num_tiles_y * self.tile_num_elements()
    }

    pub fn tile_num_elements(&self) -> u32 {
        self.tile_size_x * self.tile_size_y
    }
}
