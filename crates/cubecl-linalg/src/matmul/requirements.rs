#[derive(Copy, Clone)]
pub struct MatmulProblem {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl MatmulProblem {
    pub fn new(m: u32, n: u32, k: u32) -> MatmulProblem {
        MatmulProblem { m, n, k }
    }
}

pub struct Requirements {
    pub num_planes: u32,
    pub num_cubes: u32,
}
