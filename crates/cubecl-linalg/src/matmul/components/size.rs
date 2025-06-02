use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Debug, Clone, Copy)]
pub enum MatmulDim {
    M,
    N,
    K,
}

macro_rules! define_3d_size_base {
    ($name:ident, $ty:ty) => {
        #[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
        pub struct $name {
            pub m: $ty,
            pub n: $ty,
            pub k: $ty,
        }

        impl $name {
            pub fn new(m: u32, n: u32, k: u32) -> Self {
                $name {
                    m: <$ty>::try_from(m).unwrap(),
                    n: <$ty>::try_from(n).unwrap(),
                    k: <$ty>::try_from(k).unwrap(),
                }
            }

            pub fn get(&self, dim: MatmulDim) -> u32 {
                (match dim {
                    MatmulDim::M => self.m,
                    MatmulDim::N => self.n,
                    MatmulDim::K => self.k,
                }) as u32
            }

            pub fn m(&self) -> u32 {
                self.get(MatmulDim::M)
            }

            pub fn n(&self) -> u32 {
                self.get(MatmulDim::N)
            }

            pub fn k(&self) -> u32 {
                self.get(MatmulDim::K)
            }

            pub fn mn(&self) -> u32 {
                self.get(MatmulDim::M) * self.get(MatmulDim::N)
            }

            pub fn mk(&self) -> u32 {
                self.get(MatmulDim::M) * self.get(MatmulDim::K)
            }

            pub fn nk(&self) -> u32 {
                self.get(MatmulDim::N) * self.get(MatmulDim::K)
            }

            pub fn mnk(&self) -> u32 {
                self.get(MatmulDim::M) * self.get(MatmulDim::N) * self.get(MatmulDim::K)
            }
        }
    };
}

macro_rules! impl_from_tuple {
    ($name:ident, $ty_struct:ty, $ty_tuple:ty) => {
        impl From<($ty_tuple, $ty_tuple, $ty_tuple)> for $name {
            fn from(value: ($ty_tuple, $ty_tuple, $ty_tuple)) -> Self {
                Self {
                    m: value.0 as $ty_struct,
                    n: value.1 as $ty_struct,
                    k: value.2 as $ty_struct,
                }
            }
        }

        impl From<$name> for ($ty_tuple, $ty_tuple, $ty_tuple) {
            fn from(value: $name) -> Self {
                (
                    value.m as $ty_tuple,
                    value.n as $ty_tuple,
                    value.k as $ty_tuple,
                )
            }
        }
    };
}

// Number of elements in a tile
define_3d_size_base!(TileSize, u8);
impl_from_tuple!(TileSize, u8, u8);
impl_from_tuple!(TileSize, u8, u32);
impl_from_tuple!(TileSize, u8, i32);
impl_from_tuple!(TileSize, u8, u16);
impl_from_tuple!(TileSize, u8, usize);

// Number of tiles in a stage partition
define_3d_size_base!(PartitionSize, u8);
impl_from_tuple!(PartitionSize, u8, u8);
impl_from_tuple!(PartitionSize, u8, u32);
impl_from_tuple!(PartitionSize, u8, i32);
impl_from_tuple!(PartitionSize, u8, u16);
impl_from_tuple!(PartitionSize, u8, usize);

// Number of partitions in a stage
define_3d_size_base!(StageSize, u8);
impl_from_tuple!(StageSize, u8, u8);
impl_from_tuple!(StageSize, u8, u32);
impl_from_tuple!(StageSize, u8, i32);
impl_from_tuple!(StageSize, u8, u16);
impl_from_tuple!(StageSize, u8, usize);

// Shapes m,n,k of the problem
define_3d_size_base!(MatmulProblemSize, u32);
impl_from_tuple!(MatmulProblemSize, u32, u8);
impl_from_tuple!(MatmulProblemSize, u32, u32);
impl_from_tuple!(MatmulProblemSize, u32, i32);
impl_from_tuple!(MatmulProblemSize, u32, u16);
impl_from_tuple!(MatmulProblemSize, u32, usize);
