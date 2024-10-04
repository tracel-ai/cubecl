use super::Optimizer;

pub trait OptimizationPass {
    #[allow(unused)]
    fn apply_pre_ssa(&mut self, opt: &mut Optimizer) {}
    #[allow(unused)]
    fn apply_post_ssa(&mut self, opt: &mut Optimizer) {}
}
