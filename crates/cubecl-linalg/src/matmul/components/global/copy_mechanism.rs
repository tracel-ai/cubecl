use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::*;

#[cube]
pub trait CopyMechanism<ES: Numeric>: CubeType + Sync + Send + 'static {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>);
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for Barrier<ES> {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>) {
        this.memcpy_async(source, destination)
    }
}
