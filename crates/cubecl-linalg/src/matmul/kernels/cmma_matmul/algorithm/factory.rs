use cubecl_core::prelude::Numeric;

use super::base;

pub enum PlaneMma {
    PlaneMma16x16x16,
    PlaneMma32x8x16,
    PlaneMma8x32x16,
}

pub enum Accelerated {
    Accelerated16x16x16,
    Accelerated32x8x16,
    Accelerated8x32x16,
}

pub enum TileMatmul {
    PlaneMma(PlaneMma),
    Accelerated(Accelerated),
}

pub enum StageMatmul {
    RowAccumulate,
}

pub enum GlobalMatmul {
    Homogeneous,
}

pub enum BatchMatmul {
    OneToOne,
    OneToMany,
}

fn make_algorithm<EG: Numeric>(
    tile: TileMatmul,
    stage: StageMatmul,
    global: GlobalMatmul,
    batch: BatchMatmul,
) -> base::Algorithm<EG> {
}
