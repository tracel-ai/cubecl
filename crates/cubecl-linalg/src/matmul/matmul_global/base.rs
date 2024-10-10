use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::TensorMatmul;

use super::{Loader, Unloader};

#[cube]
/// Execute a matmul over a block, accumulating for arbitrary k-dim, using one Cube.
pub trait GlobalMatmul<E: Numeric, Lhs: Loader<E>, Rhs: Loader<E>, Out: Unloader<E>>:
    'static + Send + Sync + TensorMatmul<E>
{
    fn execute(lhs_loader: Lhs, rhs_loader: Rhs, out_writer: Out, k_range: (u32, u32));
}
