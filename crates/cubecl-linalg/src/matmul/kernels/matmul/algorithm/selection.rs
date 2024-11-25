use cubecl_core::{
    client::ComputeClient,
    prelude::{Numeric, TensorHandleRef},
    Runtime,
};

use crate::matmul::{components::MatmulProblem, kernels::matmul::base::matmul_cube_preparation};

use super::{cmma::Cmma, plane_mma::PlaneMma};

pub struct CmmaSelector;

impl CmmaSelector {
    pub fn select_kernel<R: Runtime, EG: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        lhs: TensorHandleRef<'_, R>,
        rhs: TensorHandleRef<'_, R>,
        out: TensorHandleRef<'_, R>,
        problem: MatmulProblem,
    ) {
        // TODO if problem.m < problem.n...
        matmul_cube_preparation::<R, EG, Cmma<EG>>(client, lhs, rhs, out, problem);
    }
}

pub struct PlaneMmaSelector;

impl PlaneMmaSelector {
    pub fn select_kernel<R: Runtime, EG: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        lhs: TensorHandleRef<'_, R>,
        rhs: TensorHandleRef<'_, R>,
        out: TensorHandleRef<'_, R>,
        problem: MatmulProblem,
    ) {
        // TODO if problem.m < problem.n...
        matmul_cube_preparation::<R, EG, PlaneMma<EG>>(client, lhs, rhs, out, problem);
    }
}
