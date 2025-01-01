use std::{any::Any, cell::RefCell, rc::Rc};

use type_map::TypeMap;

use crate::Optimizer;

pub trait Analysis {
    fn init(opt: &mut Optimizer) -> Self;
}

#[derive(Default, Clone, Debug)]
pub struct Analyses {
    cache: Rc<RefCell<TypeMap>>,
}

impl Analyses {
    pub fn get<A: Analysis + Any>(&self, opt: &mut Optimizer) -> Rc<A> {
        let analysis = self.cache.borrow().get::<Rc<A>>().cloned();
        if let Some(analysis) = analysis {
            analysis
        } else {
            let analysis = Rc::new(A::init(opt));
            self.cache.borrow_mut().insert(analysis.clone());
            analysis
        }
    }

    pub fn invalidate<A: Analysis + Any>(&self) {
        self.cache.borrow_mut().remove::<Rc<A>>();
    }
}

impl Optimizer {
    pub fn analysis<A: Analysis + Any>(&mut self) -> Rc<A> {
        let analyses = self.analyses.clone();
        analyses.get(self)
    }

    pub fn invalidate_analysis<A: Analysis + Any>(&self) {
        self.analyses.invalidate::<A>();
    }
}
