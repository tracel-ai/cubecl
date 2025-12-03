use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::components::global::memory::GlobalMemoryConfig;
use crate::components::global::multi_stage::EventLoadingMode;
use crate::components::global::read::ReaderMode;
use crate::components::global::{
    GlobalWriterConfig, LoadSpecializationConfig, PlaneRoleConfig, SpecializationTensorConfig,
    SpecializedLoadingSides,
};
use crate::components::stage::{StageConfig, StageMemoryConfig};
use crate::components::{AccG, error::MatmulSetupError};
use crate::components::{AvailableLineSizes, MatmulPrecision, MatmulProblem};
use crate::components::{LhsG, MatmulElems, MatmulLineSizes, MatmulSelection, RhsG};
use crate::components::{MatmulIdent, StageIdent, problem};
use cubecl_std::{
    CubeOption,
    tensor::{View, layout::Coords2d},
};
use std::fmt::Debug;
use std::hash::Hash;

/// A family of [matmuls](GlobalMatmul) working with any [precision](MatmulPrecision).
pub trait GlobalMatmulFamily: Send + Sync + 'static {
    /// The specific [GlobalMatmul] implementation associated with this family.
    type Matmul<MP: MatmulPrecision>: GlobalMatmul<MP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: GlobalConfig;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError>;

    /// Filters out line sizes that are incompatible with this matmul family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }
}

#[cube]
/// Provides matrix multiplication operations at the global level.
///
/// At the global level,
///  - Inputs are views over global memory, meaning access is given to
///    only parts of the global memory inputs at once.
///  - All planes within a Cube are used to solve the problem
///  - Dimensions M and N are fixed to an integer, but K is arbitrary large.
///    The matrix multiplication works only for size (M, _) · (_, N) = (M, N).
///    M and N should match the underlying Stage matmul's M and N.
///
/// # Assumptions
/// - Line sizes of the inputs evenly divide the dimension they are aligned with.
///
/// # Safety
///
/// It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
/// It is therefore important that Readers and Writers perform checks to avoid out-of-bounds
/// before reading data.
pub trait GlobalMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: GlobalConfig;

    /// Global reader for matrix A (Lhs)
    type LhsGlobalReader: CubeType;
    /// Global reader for matrix B (Rhs)
    type RhsGlobalReader: CubeType;
    /// Global reader for matrix C (Accumulator/Bias)
    type AccGlobalReader: CubeType;
    /// Writer to store the output stage into global memory
    type GlobalWriter: CubeType;

    /// The accumulator type for the tile matmul
    type Accumulators: CubeType;

    /// Performs the matrix multiplication over data loaded by the
    /// Lhs and Rhs readers, over the range given for K, and stores with
    /// using the output writer.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of Lhs and Rhs.
    fn execute(
        lhs_reader: Self::LhsGlobalReader,
        rhs_reader: Self::RhsGlobalReader,
        acc_reader: Self::AccGlobalReader,
        writer: Self::GlobalWriter,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );

    /// Initialize the global reader for Lhs, starting at row m and column k
    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader;

    /// Initialize the global reader for Rhs, starting at row k and column n
    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader;

    /// Initialize the global reader for Rhs, starting at row k and column n
    fn init_acc_global_reader(
        acc: CubeOption<View<Line<AccG<MP>>, Coords2d>>,
        #[comptime] config: Self::Config,
    ) -> Self::AccGlobalReader;

    /// Initialize the accumulator without data
    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators;

    /// Initialize the global writer at row m and column n
    fn init_global_writer(
        out: View<Line<AccG<MP>>, Coords2d, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedGlobalMatmulConfig<S: StageConfig> {
    pub stage_config: S,
    pub num_planes: u32,
    pub lhs_reader_config: GlobalReaderConfig,
    pub rhs_reader_config: GlobalReaderConfig,
    pub writer_config: GlobalWriterConfig,
    pub must_sync_plane_after_execution: bool,
}

impl<S: StageConfig> SharedGlobalMatmulConfig<S> {
    pub fn check_k_bounds(&self) -> bool {
        let from_lhs = self.lhs_reader_config.gmem_config.check_col_bounds;
        let from_rhs = self.rhs_reader_config.gmem_config.check_row_bounds;
        assert!(from_lhs == from_rhs);
        from_lhs
    }

    pub fn plane_dim(&self) -> u32 {
        self.stage_config.plane_dim()
    }

    pub fn plane_role_config(&self) -> PlaneRoleConfig {
        self.stage_config.plane_role_config()
    }

    pub fn specialized_loading_sides(&self) -> SpecializedLoadingSides {
        LoadSpecializationConfig {
            lhs: self.lhs_reader_config.specialization_tensor_config,
            rhs: self.rhs_reader_config.specialization_tensor_config,
        }
        .into()
    }
}

impl<S: StageConfig> GlobalConfig for SharedGlobalMatmulConfig<S> {
    type StageConfig = S;

    fn stage_config(&self) -> Self::StageConfig {
        self.stage_config
    }

    fn lhs_reader_config(&self) -> GlobalReaderConfig {
        self.lhs_reader_config
    }

    fn rhs_reader_config(&self) -> GlobalReaderConfig {
        self.rhs_reader_config
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_2d(self.plane_dim(), self.num_planes)
    }

    fn global_line_sizes(&self) -> MatmulLineSizes {
        MatmulLineSizes {
            lhs: self.lhs_reader_config.gmem_config.line_size as u8,
            rhs: self.rhs_reader_config.gmem_config.line_size as u8,
            out: self.writer_config.gmem_config.line_size as u8,
        }
    }

    fn writer_config(&self) -> GlobalWriterConfig {
        self.writer_config
    }

    fn must_sync_plane_after_execution(&self) -> bool {
        self.must_sync_plane_after_execution
    }
}

/// Configuration for the [global matmul](GlobalMatmul) level.
pub trait GlobalConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type StageConfig: StageConfig;

    /// Convert itself to the underlying stage matmul config
    fn stage_config(&self) -> Self::StageConfig;
    fn lhs_reader_config(&self) -> GlobalReaderConfig;
    fn rhs_reader_config(&self) -> GlobalReaderConfig;
    fn writer_config(&self) -> GlobalWriterConfig;
    fn cube_dim(&self) -> CubeDim;
    fn global_line_sizes(&self) -> MatmulLineSizes;
    fn must_sync_plane_after_execution(&self) -> bool;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalReaderConfig {
    pub gmem_config: GlobalMemoryConfig,
    pub smem_config: StageMemoryConfig,
    pub precompute_job: bool,
    pub plane_dim: u32,
    pub reader_mode: ReaderMode,
    pub event_loading_mode: EventLoadingMode,
    pub specialization_tensor_config: SpecializationTensorConfig,
    pub plane_role_config: PlaneRoleConfig,

    // ideally remove because doesn't apply to any kind of problem
    pub stage_ident: StageIdent,
}

impl GlobalReaderConfig {
    pub fn loading_planes_count(&self) -> u32 {
        self.smem_config.num_planes
    }

    pub fn loading_units_count(&self) -> u32 {
        self.plane_dim * self.loading_planes_count()
    }
}

/// Defines the non-contiguous stride alignment in terms of powers of two
pub fn stride_align_bits(problem: &MatmulProblem, dtypes: &MatmulElems, ident: MatmulIdent) -> u32 {
    let (strides, layout) = match ident {
        MatmulIdent::Lhs => (&problem.lhs_strides, problem.lhs_layout),
        MatmulIdent::Rhs => (&problem.rhs_strides, problem.rhs_layout),
        MatmulIdent::Out => return 31,
    };
    let exclude_dim = match layout {
        problem::MatrixLayout::RowMajor => strides.len() - 1,
        problem::MatrixLayout::ColMajor => strides.len() - 2,
    };
    strides
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != exclude_dim)
        .map(|(_, it)| (*it * dtypes.global(ident).size_bits()) / 8)
        .map(|it| it.trailing_zeros())
        .min()
        .unwrap_or(31)
}
