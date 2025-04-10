use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::pipeline::Pipeline;
use cubecl_core::prelude::*;

#[cube]
// TODO remove copy clone
// pub trait CopyMechanism<ES: Numeric>: CubeType + Sync + Send + 'static {
pub trait CopyMechanism<ES: Numeric>: CubeType + Clone + Copy + Sync + Send + 'static {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>);
    fn memcpy_async_tensor_to_shared_3d(
        this: &Self,
        source: &TensorMap<ES>,
        destination: &mut SliceMut<Line<ES>>,
        b: u32,
        x: u32,
        y: u32,
    );
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for Pipeline<ES> {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>) {
        this.memcpy_async(source, destination)
    }

    fn memcpy_async_tensor_to_shared_3d(
        _this: &Self,
        _source: &TensorMap<ES>,
        _destination: &mut SliceMut<Line<ES>>,
        _b: u32,
        _x: u32,
        _y: u32,
    ) {
        comptime!(unimplemented!(
            "Pipeline is not a supported synchronization mechanism for TMA"
        ));
        #[allow(unreachable_code)]
        ()
    }
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for Barrier<ES> {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>) {
        this.memcpy_async(source, destination)
    }

    fn memcpy_async_tensor_to_shared_3d(
        this: &Self,
        source: &TensorMap<ES>,
        destination: &mut SliceMut<Line<ES>>,
        b: u32,
        x: u32,
        y: u32,
    ) {
        this.memcpy_async_tensor_to_shared_3d(source, destination, b as i32, x as i32, y as i32);
    }
}
