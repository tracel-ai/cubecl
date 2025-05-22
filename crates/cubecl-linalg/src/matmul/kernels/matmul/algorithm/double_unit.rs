use super::{UnitMatmulSelection, base, unit_matmul_selection};
use cubecl_core::{ir::Elem, prelude::*};
use std::marker::PhantomData;

use crate::matmul::components::{
    MatmulLineSizes, MatmulProblem, MatrixLayout,
    batch::{self, CubeCountDispatch, CubeDispatch},
    global::{self, load::sync_buffer_cyclic},
    stage::{self, BufferReaderFamily, RowMajorTilingOrder},
    tile,
};

pub struct DoubleUnitAlgorithm<Dispatch = batch::TransposedDispatch> {
    pub _dispatch: PhantomData<Dispatch>,
}

impl<Dispatch> base::Algorithm for DoubleUnitAlgorithm<Dispatch>
where
    Dispatch: CubeDispatch + CubeCountDispatch,
{
    type TileMatmul = tile::register_matmul::RegisterMatmul;
    type StageMatmul = stage::unit_matmul::UnitMatmulFamily<Self::TileMatmul, BufferReaderFamily>;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;

    type BatchMatmul = batch::one_to_one::OneToOneMatmulFamily<Self::GlobalMatmul, Dispatch>;
    type MatmulSelection = UnitMatmulSelection;

    fn num_stages() -> (u32, u32) {
        (2, 2)
    }

    fn line_sizes(
        problem: &MatmulProblem,
        in_available: impl Iterator<Item = u8> + Clone,
        out_available: impl Iterator<Item = u8> + Clone,
        selection: &Self::MatmulSelection,
    ) -> MatmulLineSizes {
        let max_lhs = match problem.lhs_layout {
            MatrixLayout::RowMajor => selection.tile_shape.k,
            MatrixLayout::ColMajor => selection.tile_shape.m,
        };
        let max_rhs = match problem.rhs_layout {
            MatrixLayout::RowMajor => selection.tile_shape.n,
            MatrixLayout::ColMajor => selection.tile_shape.k,
        };
        let max_out = selection.tile_shape.n;

        MatmulLineSizes {
            lhs: MatmulLineSizes::maximize_lhs(
                problem,
                in_available
                    .clone()
                    .filter(|line_size| *line_size <= max_lhs as u8),
            ),
            rhs: MatmulLineSizes::maximize_rhs(
                problem,
                in_available.filter(|line_size| *line_size <= max_rhs as u8),
            ),
            out: MatmulLineSizes::maximize_out(
                problem,
                out_available.filter(|line_size| *line_size <= max_out as u8),
            ),
        }
    }

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim {
        let num_acc = selection.tile_count.m * selection.tile_count.n;
        let acc_per_unit = selection.acc_per_unit.0 * selection.acc_per_unit.1;
        let num_units_needed = num_acc.div_ceil(acc_per_unit);
        let num_planes = num_units_needed.div_ceil(selection.plane_dim);

        CubeDim::new(selection.plane_dim, num_planes, 1)
    }

    fn cube_count(selection: &Self::MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tile_count.m * selection.tile_shape.m;
        let n_stage = selection.tile_count.n * selection.tile_shape.n;
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }

    fn selection<R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _elem_stage: Elem,
        _elem_acc: Elem,
    ) -> Self::MatmulSelection {
        unit_matmul_selection(problem, plane_dim)
    }

    fn accumulator_shape(selection: &Self::MatmulSelection) -> (u32, u32) {
        selection.acc_per_unit
    }
}
