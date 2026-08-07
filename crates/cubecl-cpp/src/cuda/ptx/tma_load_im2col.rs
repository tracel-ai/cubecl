use cubecl_core::{
    self as cubecl, frontend::barrier::Barrier, ir::dialect::tma::TmaLoadIm2colOp, prelude::*,
};
use pliron::{derive::op_interface_impl, value::Value};

use crate::{
    cuda::ptx::{barrier_native_handle, generic_to_shared, tensor_map_address},
    shared::lowering::LowerOp,
    target::Cuda,
};

#[cube]
pub fn tma_load_im2col_3d(
    tensor_map: &TensorMap<u32, Im2col>,
    bar: &Barrier,
    smem: *const u32,
    pos: (i32, i32, i32),
    offset: u16,
) {
    let bar_handle = barrier_native_handle(bar);
    let smem = generic_to_shared::<u32>(smem);
    let descriptor_address = tensor_map_address(tensor_map);
    let (n, w, c) = pos;
    gpu_asm!(
        "cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes ",
        "[{smem}], [{tensor_map}, {{{c}, {w}, {n}}}], [{bar}], {{{offs_w}}};",
        smem = in(_) smem, tensor_map = in(_) descriptor_address,
        c = in(_) c, w = in(_) w, n = in(_) n, offs_w = in(_) offset,
        bar = in(_) bar_handle,
    );
}

#[cube]
pub fn tma_load_im2col_4d(
    tensor_map: &TensorMap<u32, Im2col>,
    bar: &Barrier,
    smem: *const u32,
    pos: (i32, i32, i32, i32),
    offset: (u16, u16),
) {
    let bar_handle = barrier_native_handle(bar);
    let smem = generic_to_shared::<u32>(smem);
    let descriptor_address = tensor_map_address(tensor_map);
    let (n, h, w, c) = pos;
    let (offs_h, offs_w) = offset;
    gpu_asm!(
        "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes ",
        "[{smem}], [{tensor_map}, {{{c}, {w}, {h}, {n}}}], [{bar}], {{{offs_w}, {offs_h}}};",
        smem = in(_) smem, tensor_map = in(_) descriptor_address,
        c = in(_) c, w = in(_) w, h = in(_) h, n = in(_) n,
        offs_w = in(_) offs_w, offs_h = in(_) offs_h,
        bar = in(_) bar_handle,
    );
}

#[cube]
pub fn tma_load_im2col_5d(
    tensor_map: &TensorMap<u32, Im2col>,
    bar: &Barrier,
    smem: *const u32,
    pos: (i32, i32, i32, i32, i32),
    offset: (u16, u16, u16),
) {
    let bar_handle = barrier_native_handle(bar);
    let smem = generic_to_shared::<u32>(smem);
    let descriptor_address = tensor_map_address(tensor_map);
    let (n, d, h, w, c) = pos;
    let (offs_d, offs_h, offs_w) = offset;
    gpu_asm!(
        "cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes ",
        "[{smem}], [{tensor_map}, {{{c}, {w}, {h}, {d}, {n}}}], [{bar}], {{{offs_w}, {offs_h}, {offs_d}}};",
        smem = in(_) smem, tensor_map = in(_) descriptor_address,
        c = in(_) c, w = in(_) w, h = in(_) h, d = in(_) d, n = in(_) n,
        offs_w = in(_) offs_w, offs_h = in(_) offs_h, offs_d = in(_) offs_d,
        bar = in(_) bar_handle,
    );
}

#[op_interface_impl]
impl LowerOp<Cuda> for TmaLoadIm2colOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx_mut();
        let tensor_map = self.tensor_map(ctx).into();
        let bar = self.barrier(ctx).into();
        let smem = self.destination(ctx).into();
        let pos = self.indices(ctx);
        let offsets = self.offsets(ctx);

        let pos_3 = (pos[0].into(), pos[1].into(), pos[2].into());

        match self.rank(ctx) {
            3 => {
                let offset = offsets[0].into();
                tma_load_im2col_3d::expand(scope, &tensor_map, &bar, &smem, pos_3, offset);
            }
            4 => {
                let pos = (pos_3.0, pos_3.1, pos_3.2, pos[3].into());
                let offset = (offsets[0].into(), offsets[1].into());
                tma_load_im2col_4d::expand(scope, &tensor_map, &bar, &smem, pos, offset);
            }
            5 => {
                let pos = (pos_3.0, pos_3.1, pos_3.2, pos[3].into(), pos[4].into());
                let offset = (offsets[0].into(), offsets[1].into(), offsets[2].into());
                tma_load_im2col_5d::expand(scope, &tensor_map, &bar, &smem, pos, offset);
            }
            _ => unreachable!("Should be 3D-5D"),
        }
        vec![]
    }
}
