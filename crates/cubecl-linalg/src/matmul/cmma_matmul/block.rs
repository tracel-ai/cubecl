pub trait CmmaBlockSize: 'static + Send + Sync {
    const M: u32;
    const N: u32;
    const K: u32;
}

macro_rules! create_cmma_block {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        pub struct $name;

        impl CmmaBlockSize for $name {
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;
        }
    };
}

create_cmma_block!(B8x32x16, 8, 32, 16);
create_cmma_block!(B16x16x16, 16, 16, 16);
create_cmma_block!(B16x32x16, 16, 32, 16);
create_cmma_block!(B32x8x16, 32, 8, 16);
create_cmma_block!(B32x16x16, 32, 16, 16);
create_cmma_block!(B32x32x16, 32, 32, 16);
create_cmma_block!(B32x32x32, 32, 32, 32);
create_cmma_block!(B64x64x16, 64, 64, 16);
create_cmma_block!(B64x64x32, 64, 64, 32);
create_cmma_block!(B128x16x16, 128, 16, 16);
create_cmma_block!(B128x128x16, 128, 128, 16);
