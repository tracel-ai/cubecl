use super::Optimizer;

pub trait OptimizationPass {
    fn apply(&mut self, opt: &mut Optimizer);
}
