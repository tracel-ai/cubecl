use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::*;

#[cube]
/// Allows to copy a slice of data from global to shared memory asynchronously
pub trait CopyMechanism<ES: Numeric>: CubeType + Sync + Send + 'static {
    /// Copy the `source` slice to `destination`, assuming `source`'s length <= `destination`'s
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>);
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for Barrier<ES> {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>) {
        this.memcpy_async(source, destination)
    }
}
