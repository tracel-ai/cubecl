//! The simdgroup matrix as a manual MMA: each lane holds its cells in registers, and an MMA
//! hands them to a `simdgroup_matrix` through `thread_elements()` and takes the product back
//! the same way.
//!
//! **The lane layout.** An 8x8 simdgroup matrix gives each of the 32 lanes two cells, adjacent
//! columns of one row. Lane `l` holds row `(l / 16) * 4 + (l / 2) % 4` and columns
//! `((l / 8) % 2) * 4 + (l % 2) * 2` and the one after it, for the left operand, the right one
//! (rows along `k`) and the accumulator alike. Apple does not document it; MLX's `BaseMMAFrag`
//! relies on it on every Apple GPU MLX runs on, and the manual MMA runtime tests check it on the
//! device they run on.

use cubecl_core::{
    cmma::{MatrixIdent, MatrixLayout, MatrixShape, MatrixType},
    ir::{
        ContiguousElements, CoordinatePart, ElementIndex, ElementPositions, MmaProperties,
        OperandFact, dialect::matrix::MmaManualOp, interfaces::TypedExt, prelude::*,
    },
};
use itertools::Itertools;
use pliron::{context::Context, r#type::TypedHandle};

use crate::{
    metal::{arch::MetalArchitecture, metal_op, supported_cmma_combinations_metal},
    shared::{CppValue, SupportedMmaCombinations, ty::TypeExtCPP},
};

/// The cells a lane holds of any operand: two adjacent columns of one row.
const CELLS_PER_LANE: usize = 2;

/// The manual MMA shapes and types: those of the simdgroup matrix, which one instruction runs
/// either way.
pub fn supported_mma_combinations_metal(arch: &MetalArchitecture) -> SupportedMmaCombinations {
    supported_cmma_combinations_metal(arch)
}

/// How a lane's registers map onto an operand's cells (see the module documentation).
pub fn mma_properties_metal() -> MmaProperties {
    MmaProperties {
        register_size_bits: 32,
        const_plane_lane_count: 32,
        register_layout_a: MatrixLayout::RowMajor,
        register_layout_b: MatrixLayout::RowMajor,
        register_layout_acc: MatrixLayout::RowMajor,
        contiguous_elements: ContiguousElements::new(contiguous_elements_metal),
        element_positions: OperandFact::new(|_, _, _| Some(element_positions_metal())),
        ..MmaProperties::default()
    }
}

/// A lane's cells sit side by side in a row, whatever the operand.
fn contiguous_elements_metal(
    _ctx: &Context,
    _ident: MatrixIdent,
    _matrix: TypedHandle<MatrixType>,
) -> usize {
    CELLS_PER_LANE
}

/// Where a lane's cells lie, whatever the operand (see the module documentation).
fn element_positions_metal() -> ElementPositions {
    let lane = CoordinatePart::of(ElementIndex::Lane);
    ElementPositions {
        row: vec![
            lane.in_runs_of(16).spaced_by(4),
            lane.in_runs_of(2).counted_over(4),
        ],
        col: vec![
            lane.in_runs_of(8).counted_over(2).spaced_by(4),
            lane.counted_over(2).spaced_by(CELLS_PER_LANE),
            CoordinatePart::of(ElementIndex::Nth),
        ],
    }
}

metal_op!(MmaManualOp, |op, ctx| {
    let MatrixShape { m, n, k } = op.shape(ctx).0;
    assert!(
        m == 8 && n == 8 && k == 8,
        "{m}x{n}x{k} manual MMA not supported. Only 8x8x8 simdgroup matrices are supported."
    );

    // `registers_a/b/c` are array values, only `registers_d` is a pointer (see `MmaManualOp`).
    let operands = [
        ("mma_a", op.registers_a(ctx)),
        ("mma_b", op.registers_b(ctx)),
        ("mma_c", op.registers_c(ctx)),
    ];
    let frag_d = op.registers_d(ctx);
    let elem_d = frag_d.scalar_ty(ctx).to_cpp(ctx);

    let declare = |name: &str, elem: &str| format!("simdgroup_matrix<{elem}, 8, 8> {name};\n");
    let mut out = String::from("{\n");
    for (name, registers) in operands {
        let elem = registers.scalar_ty(ctx).to_cpp(ctx);
        out.push_str(&declare(name, &elem));
        let cells = format!(
            "reinterpret_cast<thread const {elem}*>(&{})",
            registers.name(ctx)
        );
        out.push_str(
            &(0..CELLS_PER_LANE)
                .map(|cell| format!("{name}.thread_elements()[{cell}] = {cells}[{cell}];\n"))
                .join(""),
        );
    }
    out.push_str(&declare("mma_d", &elem_d));
    out.push_str("simdgroup_multiply_accumulate(mma_d, mma_a, mma_b, mma_c);\n");
    let cells_d = format!("reinterpret_cast<thread {elem_d}*>({})", frag_d.name(ctx));
    out.push_str(
        &(0..CELLS_PER_LANE)
            .map(|cell| format!("{cells_d}[{cell}] = mma_d.thread_elements()[{cell}];\n"))
            .join(""),
    );
    out.push_str("}\n");
    out
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lanes_of_a_plane_hold_every_cell_once() {
        const LANES: usize = 32;
        let positions = element_positions_metal();
        let mut held = [[0; 8]; 8];
        for lane in 0..LANES {
            for nth in 0..CELLS_PER_LANE {
                let (row, col) = positions.of_nth(lane, nth);
                held[row][col] += 1;
            }
        }
        assert_eq!(held, [[1; 8]; 8]);
    }
}
