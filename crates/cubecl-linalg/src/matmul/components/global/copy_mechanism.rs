use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::pipeline::Pipeline;
use cubecl_core::prelude::*;

#[cube]
pub trait CopyMechanism<ES: Numeric>: CubeType + Sync + Send + 'static {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>);
    fn tma_load_3d(
        this: &Self,
        source: &TensorMap<ES>,
        destination: &mut SliceMut<Line<ES>>,
        b: u32,
        x: u32,
        y: u32,
    );
    fn tma_load_im2col_4d(
        this: &Self,
        source: &TensorMap<ES>,
        destination: &mut SliceMut<Line<ES>>,
        coords: (i32, i32, i32, i32),
        offsets: (u16, u16),
    );
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for Pipeline<ES> {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>) {
        this.memcpy_async(source, destination)
    }

    fn tma_load_3d(
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

    fn tma_load_im2col_4d(
        _this: &Self,
        _source: &TensorMap<ES>,
        _destination: &mut SliceMut<Line<ES>>,
        _coords: (i32, i32, i32, i32),
        _offsets: (u16, u16),
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

    fn tma_load_3d(
        this: &Self,
        source: &TensorMap<ES>,
        destination: &mut SliceMut<Line<ES>>,
        b: u32,
        x: u32,
        y: u32,
    ) {
        this.tma_load_3d(source, destination, b as i32, x as i32, y as i32);
    }

    fn tma_load_im2col_4d(
        this: &Self,
        source: &TensorMap<ES>,
        destination: &mut SliceMut<Line<ES>>,
        coords: (i32, i32, i32, i32),
        offsets: (u16, u16),
    ) {
        let (n, h, w, c) = coords;
        let (h_offset, w_offset) = offsets;
        this.tma_load_im2col_4d(source, destination, n, h, w, c, h_offset, w_offset);
    }
}
