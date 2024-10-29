use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::Matmul;

use super::{GmmConfig, Loader, Unloader};

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
/// It is therefore important that Loaders and Unloaders perform checks to avoid out-of-bounds
/// before loading data.
pub trait GlobalMatmul<
    EG: Numeric,
    ES: Numeric,
    Lhs: Loader<EG, ES, G>,
    Rhs: Loader<EG, ES, G>,
    Out: Unloader<EG, G>,
    G: GmmConfig,
>: 'static + Send + Sync + Matmul<EG, EG, Config = G>
{
    /// Performs the matrix multiplication over data loaded by the
    /// LHS and RHS loaders, over the range given for K, and stores with
    /// using the output unloader.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of LHS and RHS.
    fn execute(
        lhs_loader: Lhs,
        rhs_loader: Rhs,
        unloader: Out,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );
}
