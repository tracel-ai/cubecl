pub trait CmmaStageSize: 'static + Send + Sync {
    const NUM_M: u32;
    const NUM_N: u32;
    const NUM_K: u32;
}

macro_rules! create_cmma_stage {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        pub struct $name;

        impl CmmaStageSize for $name {
            const NUM_M: u32 = $m;
            const NUM_N: u32 = $n;
            const NUM_K: u32 = $k;
        }
    };
}

// This list is not exhaustive. Add what you need.
create_cmma_stage!(S1x1x1, 1, 1, 1);
create_cmma_stage!(S1x1x2, 1, 1, 2);
create_cmma_stage!(S1x2x1, 1, 2, 1);
create_cmma_stage!(S2x1x1, 2, 1, 1);
create_cmma_stage!(S2x2x1, 2, 2, 1);
create_cmma_stage!(S2x2x2, 2, 2, 2);
create_cmma_stage!(S4x4x1, 4, 4, 1);
create_cmma_stage!(S4x4x2, 4, 4, 2);
create_cmma_stage!(S8x1x1, 8, 1, 1);
create_cmma_stage!(S8x8x1, 8, 8, 1);
