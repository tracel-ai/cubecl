use cubecl_common::rand::Rng;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::{PlaneLayout, PlaneLayoutExpand, RowWise};

#[cube]
trait RowOp<E: Float> {
    fn mask(is_active: bool) -> E;

    fn neutral_element() -> E;

    fn local_update<PL: PlaneLayout<E>>(acc: E, row: u32, col: u32, data: &PL, mask: E) -> E;

    fn plane_reduce(acc: E) -> E;
}

#[derive(CubeType)]
struct RowMax {}

#[derive(CubeType)]
struct RowSum {}

#[cube]
impl<E: Float> RowOp<E> for RowMax {
    fn mask(is_active: bool) -> E {
        E::cast_from(!is_active) * E::min_value()
    }

    fn neutral_element() -> E {
        E::min_value()
    }

    fn local_update<PL: PlaneLayout<E>>(acc: E, row: u32, col: u32, data: &PL, mask: E) -> E {
        Max::max(acc, data.get_at_coor(row, col) + mask)
    }

    fn plane_reduce(acc: E) -> E {
        plane_max::<E>(acc)
    }
}

#[cube]
impl<E: Float> RowOp<E> for RowSum {
    fn mask(is_active: bool) -> E {
        E::cast_from(is_active)
    }

    fn neutral_element() -> E {
        E::from_int(0)
    }
    fn local_update<PL: PlaneLayout<E>>(acc: E, row: u32, col: u32, data: &PL, mask: E) -> E {
        acc + data.get_at_coor(row, col) * mask
    }

    fn plane_reduce(acc: E) -> E {
        plane_sum::<E>(acc)
    }
}

#[cube]
fn row_op<E: Float, PL: PlaneLayout<E>, RO: RowOp<E>>(vals: &mut Array<E>, data: &PL) {
    let total_row_count = data.total_rows_count();

    #[unroll]
    for row in 0..total_row_count {
        let is_active = data.is_owned(row);

        let mask = RO::mask(is_active);

        let mut local = RO::neutral_element();

        #[unroll]
        for c in 0..data.num_cols() {
            let col = data.col_index(row, c);
            local = RO::local_update::<PL>(local, row, col, &data, mask);
        }

        vals[row] = RO::plane_reduce(local);
    }
}

#[cube]
pub fn row_sum<E: Float, PL: PlaneLayout<E>>(placeholder: &mut RowWise<E>, data: &PL) {
    placeholder.fill(<RowSum as RowOp<E>>::neutral_element());
    row_op::<E, PL, RowSum>(&mut placeholder.vals, data)
}

#[cube]
pub fn row_max<E: Float, PL: PlaneLayout<E>>(
    placeholder: &mut RowWise<E>,
    base: &RowWise<E>,
    data: &PL,
) {
    placeholder.copy_from(base);
    row_op::<E, PL, RowMax>(&mut placeholder.vals, data);
}
