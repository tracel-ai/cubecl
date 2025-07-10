use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::components::error::MatmulSetupError;
use crate::components::global::RoleRuleConfig;
use crate::components::{
    AvailableLineSizes, Ident, InputIdent, MatmulPrecision, MatmulProblem, MatrixLayout,
    TilingScheme,
    global::{PlaneRoleConfig, SpecializedLoadingSides, multi_stage::EventLoadingMode},
    stage::{self, StageConfig},
};
use crate::components::{MatmulLineSizes, MatmulSelection};
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};
use std::{fmt::Debug, hash::Hash};

use super::{GlobalWriter, Quantization, load::LoaderMode};

/// A family of [matmuls](GlobalMatmul) working with any [precision](MatmulPrecision).
pub trait GlobalMatmulFamily: Send + Sync + 'static {
    type Matmul<MP: MatmulPrecision>: GlobalMatmul<MP, Config = Self::Config>;
    type Config: GlobalConfig;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError>;

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
///  - All planes within a Cube can collaborate to solve the problem
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
    /// LHS and RHS loaders, over the range given for K, and stores with
    /// using the output writer.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of LHS and RHS.
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
        lhs: VirtualTensor<MP::EI>,
        m_offset: u32,
        k_offset: u32,
        nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader;

    /// Initialize the loader for Rhs, starting at row k and column n
    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        k_offset: u32,
        n_offset: u32,
        nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader;

    /// Initialize the accumulator without data
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with zeros
    ///
    /// TODO: The global matmul is responsible to zero the accumulator not the batch.
    ///
    /// this is unused.
    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config);

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
    type StageConfig: stage::StageConfig;

    /// Convert itself to the underlying stage matmul config
    fn stage_config(&self) -> Self::StageConfig;

    /// Returns the line size for the global memory corresponding to the given ident
    fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32;

    fn tiling_scheme(&self) -> TilingScheme {
        self.stage_config().tiling_scheme()
    }

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout;

    fn num_loading_planes<I: Into<Ident>>(&self, ident: I) -> u32;
    fn plane_role_config(&self) -> PlaneRoleConfig;
    fn specialized_loading_sides(&self) -> SpecializedLoadingSides;
    fn role_rule_config(&self) -> RoleRuleConfig {
        self.plane_role_config().rule
    }

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Whether to check if accessing a row would exceed bounds.
    fn check_row_bounds<I: Into<Ident>>(&self, ident: I) -> bool;

    /// Whether to check if accessing a col would exceed bounds.
    fn check_col_bounds<I: Into<Ident>>(&self, ident: I) -> bool;

    /// Whether to check if accessing a col for lhs or row for rhs would exceed bounds.
    fn check_k_bounds(&self) -> bool;

    fn precompute_job(&self) -> bool;

    fn num_stages(&self, ident: InputIdent) -> u32;

    /// Whether to check loader is balanced in comptime or runtime.
    /// Not supported by all loading strategies
    fn loader_mode(&self) -> LoaderMode;

    fn event_loading_mode(&self, ident: InputIdent) -> EventLoadingMode;

    fn quantized(&self) -> bool {
        self.stage_config().quantized()
    }

    fn cube_dim(&self) -> CubeDim;
}
