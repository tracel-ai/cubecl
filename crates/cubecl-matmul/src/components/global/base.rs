use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::components::error::MatmulSetupError;
use crate::components::global::RoleRuleConfig;
use crate::components::global::memory::GlobalMemoryConfig;
use crate::components::stage::StageMemoryConfig;
use crate::components::{
    AvailableLineSizes, MatmulPrecision, MatmulProblem, MatrixLayout, TilingScheme,
    global::{PlaneRoleConfig, SpecializedLoadingSides, multi_stage::EventLoadingMode},
    stage::StageConfig,
};
use crate::components::{LhsG, MatmulIdent, MatmulLineSizes, MatmulSelection, RhsG};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::{fmt::Debug, hash::Hash};

use super::{GlobalWriter, load::LoaderMode};

/// A family of [matmuls](GlobalMatmul) working with any [precision](MatmulPrecision).
pub trait GlobalMatmulFamily: Send + Sync + 'static {
    /// The specific [GlobalMatmul] implementation associated with this family.
    type Matmul<MP: MatmulPrecision>: GlobalMatmul<MP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: GlobalConfig;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
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
/// It is therefore important that Loaders and Writers perform checks to avoid out-of-bounds
/// before loading data.
pub trait GlobalMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: GlobalConfig;
    type LhsLoader: CubeType;
    type RhsLoader: CubeType;
    type AccumulatorLoader: CubeType;
    type Writer: GlobalWriter<MP::EO>;
    type Accumulator: CubeType;

    /// Performs the matrix multiplication over data loaded by the
    /// Lhs and Rhs loaders, over the range given for K, and stores with
    /// using the output writer.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of Lhs and Rhs.
    fn execute(
        lhs_loader: Self::LhsLoader,
        rhs_loader: Self::RhsLoader,
        writer: Self::Writer,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );

    /// Initialize the loader for Lhs, starting at row m and column k
    fn init_lhs_loader(
        lhs: VirtualTensor<LhsG<MP>>,
        m_offset: u32,
        k_offset: u32,
        nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader;

    /// Initialize the loader for Rhs, starting at row k and column n
    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        k_offset: u32,
        n_offset: u32,
        nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader;

    /// Initialize the accumulator without data
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Initialize the writer at row m and column n
    fn init_writer(
        out: VirtualTensor<MP::EO, ReadWrite>,
        m_offset: u32,
        n_offset: u32,
        nth_batch: u32,
        batch_offset: u32,
    ) -> Self::Writer;
}

/// Configuration for the [global matmul](GlobalMatmul) level.
pub trait GlobalConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    /// Underlying Stage matmul config
    type StageConfig: StageConfig;
    type StageMemoryConfig: StageMemoryConfig;

    /// Convert itself to the underlying stage matmul config
    fn stage_config(&self) -> Self::StageConfig;

    fn stage_memory_config(&self) -> Self::StageMemoryConfig;

    fn global_memory_config(&self, ident: MatmulIdent) -> GlobalMemoryConfig {
        GlobalMemoryConfig {
            elements_in_tile_row: self.tiling_scheme().elements_in_tile_row(ident),
            elements_in_tile_col: self.tiling_scheme().elements_in_tile_col(ident),
            elements_in_stage_row: self.tiling_scheme().elements_in_stage_row(ident),
            elements_in_stage_col: self.tiling_scheme().elements_in_stage_col(ident),
            global_line_size: self.global_line_size(ident),
            check_row_bounds: self.check_row_bounds(ident),
            check_col_bounds: self.check_col_bounds(ident),
            matrix_layout: self.matrix_layout(ident),
        }
    }

    /// Returns the line size for the global memory corresponding to the given ident
    fn global_line_size(&self, ident: MatmulIdent) -> u32;

    /// Returns the [TilingScheme]
    fn tiling_scheme(&self) -> TilingScheme {
        self.stage_config().tiling_scheme()
    }

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout(&self, ident: MatmulIdent) -> MatrixLayout;

    /// Returns the number of planes participating in loading `ident`
    fn num_loading_planes(&self, ident: MatmulIdent) -> u32;

    /// Indicates the specialization roles for the planes
    fn plane_role_config(&self) -> PlaneRoleConfig;

    /// Indicates plane roles are associated to loading which tensor input
    fn specialized_loading_sides(&self) -> SpecializedLoadingSides;

    /// How to identify the role of the plane depending on its index
    fn role_rule_config(&self) -> RoleRuleConfig {
        self.plane_role_config().rule
    }

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Whether to check if accessing a row would exceed bounds.
    fn check_row_bounds(&self, ident: MatmulIdent) -> bool;

    /// Whether to check if accessing a col would exceed bounds.
    fn check_col_bounds(&self, ident: MatmulIdent) -> bool;

    /// Whether to check if accessing a col for lhs or row for rhs would exceed bounds.
    fn check_k_bounds(&self) -> bool;

    /// Whether to put common computations for loading tasks once before loop
    fn precompute_job(&self) -> bool;

    /// The number of stages in stage memory
    fn num_stages(&self, ident: MatmulIdent) -> u32;

    /// Whether to check loader is balanced in comptime or runtime.
    ///
    /// Not supported by all loading strategies
    fn loader_mode(&self) -> LoaderMode;

    /// Whether event loading is constrained to be ordered
    fn event_loading_mode(&self, ident: MatmulIdent) -> EventLoadingMode;

    /// Whether the matmul is quantized
    fn quantized(&self) -> bool {
        self.stage_config().quantized()
    }

    /// The [CubeDim] arising from the [TilingScheme]
    fn cube_dim(&self) -> CubeDim;
}
