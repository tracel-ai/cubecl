mod base;

pub use base::*;

/// Re-entrant locking primitives, available in std environments.
#[cfg(feature = "std")]
pub mod reentrant;
