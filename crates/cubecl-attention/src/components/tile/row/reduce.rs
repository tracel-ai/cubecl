use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};

#[cube]
pub trait RowOp<PL: PlaneLayout> {
    fn mask(is_active: bool) -> PL::E;

    fn neutral_element() -> PL::E;

    fn local_update(acc: PL::E, row: u32, col: u32, data: &PL, mask: PL::E) -> PL::E;

    fn plane_reduce(acc: PL::E) -> PL::E;
}

#[derive(CubeType)]
pub struct RowMax {}

#[derive(CubeType)]
pub struct RowSum {}

#[cube]
impl<PL: PlaneLayout> RowOp<PL> for RowMax {
    fn mask(is_active: bool) -> PL::E {
        PL::E::cast_from(!is_active) * PL::E::min_value()
    }

    fn neutral_element() -> PL::E {
        PL::E::min_value()
    }

    fn local_update(acc: PL::E, row: u32, col: u32, data: &PL, mask: PL::E) -> PL::E {
        Max::max(
            acc,
            data.get_at_coor(row, col, <Self as RowOp<PL>>::neutral_element()) + mask,
        )
    }

    fn plane_reduce(acc: PL::E) -> PL::E {
        plane_max::<PL::E>(acc)
    }
}

#[cube]
impl<PL: PlaneLayout> RowOp<PL> for RowSum {
    fn mask(is_active: bool) -> PL::E {
        PL::E::cast_from(is_active)
    }

    fn neutral_element() -> PL::E {
        PL::E::from_int(0)
    }

    fn local_update(acc: PL::E, row: u32, col: u32, data: &PL, mask: PL::E) -> PL::E {
        acc + data.get_at_coor(row, col, <Self as RowOp<PL>>::neutral_element()) * mask
    }

    fn plane_reduce(acc: PL::E) -> PL::E {
        plane_sum::<PL::E>(acc)
    }
}

#[cube]
pub fn row_op<PL: PlaneLayout, RO: RowOp<PL>>(vals: &mut Array<PL::E>, data: &PL) {
    let total_row_count = data.total_rows_count();

    #[unroll]
    for row in 0..total_row_count {
        let is_active = data.is_owned(row);

        let mask = RO::mask(is_active);

        let mut local = RO::neutral_element();

        #[unroll]
        for c in 0..data.num_cols() {
            let col = data.col_index(row, c);
            local = RO::local_update(local, row, col, &data, mask);
        }

        vals[row] = RO::plane_reduce(local);
    }
}
