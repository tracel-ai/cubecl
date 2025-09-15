use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::{CubeType, prelude::*};

use crate::components::{InputPrecision, stage::FillReader};

#[derive(CubeType)]
/// Accumulator loader that zeros the accumulator
pub struct ZeroLoader<IP: InputPrecision> {
    #[cube(comptime)]
    _ty: PhantomData<IP>,
}

#[cube]
impl<IP: InputPrecision> ZeroLoader<IP> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ZeroLoader::<IP> { _ty: PhantomData }
    }

    /// Give a reader to the loaded data.
    pub fn reader(&self) -> FillReader<IP::Stage> {
        FillReader::new(IP::Stage::from_int(0))
    }
}
