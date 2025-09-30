use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::{PlaneLayout, PlaneLayoutExpand, RowVal, RowWise};

#[cube]
trait RowOp<E: Float> {
    fn mask(is_active: bool) -> E;

    fn neutral_element(mask: E) -> E;

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

    fn neutral_element(mask: E) -> E {
        E::min_value() + mask
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

    fn neutral_element(mask: E) -> E {
        E::from_int(0) * mask
    }
    fn local_update<PL: PlaneLayout<E>>(acc: E, row: u32, col: u32, data: &PL, mask: E) -> E {
        acc + data.get_at_coor(row, col) * mask
    }

    fn plane_reduce(acc: E) -> E {
        plane_sum::<E>(acc)
    }
}

#[cube]
fn row_op<E: Float, PL: PlaneLayout<E>, RO: RowOp<E>>(data: &PL) -> RowWise<E> {
    let mut vals = Sequence::new();
    let total_row_count = data.total_rows_count();
    let owned_row_count = data.owned_rows_count();

    #[unroll]
    for row in 0..total_row_count {
        let is_active = data.is_owned(row);

        let mask = RO::mask(is_active);

        let mut local = RO::neutral_element(mask);

        #[unroll]
        for c in 0..data.num_cols() {
            let col = data.col_index(row, c);
            local = RO::local_update::<PL>(local, row, col, &data, mask);
        }

        let val = RO::plane_reduce(local);

        vals.push(RowVal::new(val));
    }

    RowWise::<E>::new(owned_row_count, vals)
}

#[cube]
pub fn row_sum<E: Float, PL: PlaneLayout<E>>(data: &PL) -> RowWise<E> {
    row_op::<E, PL, RowSum>(data)
}

#[cube]
pub fn row_max<E: Float, PL: PlaneLayout<E>>(base: RowWise<E>, data: &PL) -> RowWise<E> {
    base.max(&row_op::<E, PL, RowMax>(data))
}
