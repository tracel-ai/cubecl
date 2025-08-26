use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::*;

#[cube]
/// Allows to copy a slice of data from global to shared memory asynchronously
pub trait CopyMechanism: CubeType + Sync + Send + 'static {
    /// Copy the `source` slice to `destination`, assuming `source`'s length <= `destination`'s
    fn memcpy_async<ES: Numeric>(
        this: &Self,
        source: &Slice<Line<ES>>,
        destination: &mut SliceMut<Line<ES>>,
    );
}

#[cube]
impl CopyMechanism for Barrier {
    fn memcpy_async<ES: Numeric>(
        this: &Self,
        source: &Slice<Line<ES>>,
        destination: &mut SliceMut<Line<ES>>,
    ) {
        this.memcpy_async(source, destination)
    }
}
