use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::{CubeType, prelude::*};

use crate::components::{MatrixPrecision, stage::FilledStage};

#[derive(CubeType)]
/// Accumulator reader that zeros the accumulator
pub struct ZeroGlobalReader<IP: MatrixPrecision> {
    #[cube(comptime)]
    _ty: PhantomData<IP>,
}

#[cube]
impl<IP: MatrixPrecision> ZeroGlobalReader<IP> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ZeroGlobalReader::<IP> { _ty: PhantomData }
    }

    /// Give a reader to the loaded data.
    pub fn stage(&self) -> FilledStage<IP::Stage> {
        FilledStage::new(IP::Stage::from_int(0))
    }
}
