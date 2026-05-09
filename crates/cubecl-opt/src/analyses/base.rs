use core::{any::Any, cell::RefCell};

use alloc::rc::Rc;
use type_map::TypeMap;

use crate::{Function, GlobalState};

use super::{
    dominance::{Dominators, PostDominators},
    liveness::Liveness,
    post_order::PostOrder,
    uniformity::Uniformity,
};

/// An analysis used by optimization passes. Unlike optimization passes, analyses can have state
/// and persist until they're invalidated.
pub trait Analysis {
    /// Perform the analysis for the current optimizer state and return the persistent analysis state
    fn init(opt: &mut Function, state: &GlobalState) -> Self;
}

#[derive(Default, Clone, Debug)]
pub struct AnalysisCache {
    cache: Rc<RefCell<TypeMap>>,
}

impl AnalysisCache {
    pub fn get<A: Analysis + Any>(&self, func: &mut Function, state: &GlobalState) -> Rc<A> {
        let analysis = self.cache.borrow().get::<Rc<A>>().cloned();
        if let Some(analysis) = analysis {
            analysis
        } else {
            let analysis = Rc::new(A::init(func, state));
            self.cache.borrow_mut().insert(analysis.clone());
            analysis
        }
    }

    pub fn try_get<A: Any>(&self) -> Option<Rc<A>> {
        self.cache.borrow().get().cloned()
    }

    pub fn invalidate<A: Analysis + Any>(&self) {
        self.cache.borrow_mut().remove::<Rc<A>>();
    }
}

impl Function {
    /// Fetch an analysis if cached, or run it if not.
    pub fn analysis<A: Analysis + Any>(&mut self, state: &GlobalState) -> Rc<A> {
        let analyses = self.analysis_cache.clone();
        analyses.get(self, state)
    }

    /// Invalidate an analysis by removing it from the cache. The analysis is rerun when requested
    /// again.
    pub fn invalidate_analysis<A: Analysis + Any>(&self) {
        self.analysis_cache.invalidate::<A>();
    }

    /// Invalidate all analyses that rely on the structure of the control flow graph.
    pub fn invalidate_structure(&self) {
        self.invalidate_analysis::<PostOrder>();
        self.invalidate_analysis::<Dominators>();
        self.invalidate_analysis::<PostDominators>();
        self.invalidate_analysis::<Liveness>();
        self.invalidate_analysis::<Uniformity>();
    }
}
