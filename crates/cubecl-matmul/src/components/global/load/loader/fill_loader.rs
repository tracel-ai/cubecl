use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::{CubeType, prelude::*};

use crate::components::{InputPrecision, stage::FillStageReader};

#[derive(CubeType)]
/// Accumulator loader that zeros the accumulator
pub struct ZeroStageLoader<IP: InputPrecision> {
    #[cube(comptime)]
    _ty: PhantomData<IP>,
}

#[cube]
impl<IP: InputPrecision> ZeroStageLoader<IP> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ZeroStageLoader::<IP> { _ty: PhantomData }
    }

    /// Give a reader to the loaded data.
    pub fn reader(&self) -> FillStageReader<IP::Stage> {
        FillStageReader::new(IP::Stage::from_int(0))
    }
}
