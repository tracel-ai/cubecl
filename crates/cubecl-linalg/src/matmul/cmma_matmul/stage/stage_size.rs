pub trait CmmaStageSize: 'static + Send + Sync {
    const M: u32;
    const N: u32;
    const K: u32;
}

macro_rules! create_cmma_stage {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        pub struct $name;

        impl CmmaStageSize for $name {
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;
        }
    };
}

create_cmma_stage!(S8x32x16, 8, 32, 16);
create_cmma_stage!(S16x16x16, 16, 16, 16);
create_cmma_stage!(S16x32x16, 16, 32, 16);
create_cmma_stage!(S16x16x32, 16, 16, 32);
create_cmma_stage!(S32x8x16, 32, 8, 16);
create_cmma_stage!(S32x16x16, 32, 16, 16);
create_cmma_stage!(S32x32x16, 32, 32, 16);
create_cmma_stage!(S32x32x32, 32, 32, 32);
create_cmma_stage!(S64x64x16, 64, 64, 16);
create_cmma_stage!(S64x64x32, 64, 64, 32);
create_cmma_stage!(S128x16x16, 128, 16, 16);
create_cmma_stage!(S128x128x16, 128, 128, 16);
