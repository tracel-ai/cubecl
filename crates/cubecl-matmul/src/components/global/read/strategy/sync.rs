use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::{
    MatmulPrecision,
    global::{GlobalConfig, read::SyncStrategy},
};

/// Simple synchronous barrier, using `cube_sync()`
pub struct Synchronous {}

#[cube]
impl SyncStrategy for Synchronous {
    type Barrier = ();

    fn create_barrier() -> Self::Barrier {}

    fn sync<MP: MatmulPrecision, G: GlobalConfig>(
        _barrier: &mut Self::Barrier,
        #[comptime] _config: G,
    ) {
        sync_cube();
    }
}
