use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::Matmul;

use super::TmmConfig;

#[cube]
/// Provides matrix multiplication operations at the tile level.
///
/// At this level, dimensions M, N and K are fixed to an integer, and the
/// matrix multiplication works only for size (M, K) · (K, N) = (M, N)
///
/// Assumptions:
///  - Slices given as inputs must always be valid. If the actual matrix multiplication
///    should be done on smaller sizes than M, N and K, padding with zeros must be done beforehand.
pub trait TileMatmul<I: Numeric, O: Numeric, T: TmmConfig>:
    'static + Send + Sync + Matmul<I, O, Config = T>
{
    /// Number of rows of LHS
    const M: u32;
    /// Number of columns of RHS
    const N: u32;
    /// Common dimension of LHS and RHS
    const K: u32;

    /// Contains LHS data
    type Lhs: CubeType;
    /// Contains RHS data
    type Rhs: CubeType;
    /// Contains output data
    type Out: CubeType;

    /// Executes the matrix multiplication of LHS and RHS, storing it to the output
    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out, #[comptime] config: T);

    /// Create the container for LHS data
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call fill_lhs prior to execute.
    fn init_lhs(#[comptime] config: T) -> Self::Lhs;

    /// Create the container for RHS data
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call fill_rhs prior to execute.
    fn init_rhs(#[comptime] config: T) -> Self::Rhs;

    /// Fill the container of LHS with data
    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs, #[comptime] config: T);

    /// Fill the container of RHS with data
    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs, #[comptime] config: T);

    /// Create the container to receive the execution output
    fn init_output(#[comptime] config: T) -> Self::Out;

    /// Write the content of the output container to the given slice
    fn read_output<C: Numeric>(
        out: &Self::Out,
        slice: &mut SliceMut<'_, Line<C>>,
        #[comptime] config: T,
    );
}
