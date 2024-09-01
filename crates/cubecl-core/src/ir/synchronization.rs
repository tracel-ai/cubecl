use serde::{Deserialize, Serialize};

use crate::new_ir::{Expr, Expression};

/// All synchronization types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum Synchronization {
    // Synchronizize units in a cube.
    SyncUnits,
    SyncStorage,
}

impl Expr for Synchronization {
    type Output = ();

    fn expression_untyped(&self) -> crate::new_ir::Expression {
        Expression::Sync(*self)
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}
