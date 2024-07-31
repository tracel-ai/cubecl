use crate::ir::{macros::cpa, Branch, Elem, Item, Scope, Variable, Vectorization};
use serde::{Deserialize, Serialize};

/// Perform a check bound on the index (lhs) of value (rhs)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct EarlyReturn {
    pub global: Variable,
    pub position: Variable,
}

impl EarlyReturn {
    #[allow(missing_docs)]
    pub fn expand(self, scope: &mut Scope) {
        let variable = self.global;
        let index = self.position;

        let array_len = scope.create_local(Item::new(Elem::UInt));
        let outside_bound = scope.create_local(Item::new(Elem::Bool));

        cpa!(scope, array_len = len(variable));
        cpa!(scope, outside_bound = index >= array_len);

        cpa!(scope, if(outside_bound).then(|scope| {
            scope.register(Branch::Return);
        }));
    }

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            global: self.global.vectorize(vectorization),
            position: self.position.vectorize(vectorization),
        }
    }
}
