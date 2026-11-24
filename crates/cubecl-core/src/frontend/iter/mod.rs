//! Composable iterators for `#[cube]` functions, mirroring [`core::iter`].
//!
//! Iterator chains can be written in a kernel the same way as in plain Rust:
//!
//! ```ignore
//! #[cube(launch)]
//! fn add_scaled<F: Float>(a: &[F], b: &[F], output: &mut [F]) {
//!     let scale = F::new(2.0);
//!     output
//!         .iter_mut()
//!         .zip(a.iter().zip(b.iter()).map(|(x, y)| *x + *y * scale))
//!         .for_each(|(out, value)| *out = value);
//! }
//! ```
//!
//! `#[cube]` emits the body of a function twice: unchanged, where the chain is type checked against
//! [`core::iter`], and rewritten for expansion, where `x.map(f)` becomes
//! `x.__expand_map_method(scope, f)`. This module provides the second half, so a chain must also be
//! valid [`core::iter`] code and a method keeps the name, generics and results of its
//! [`core::iter`] counterpart.
//!
//! A chain is evaluated while the kernel is built and produces one loop with every closure inlined,
//! not a call per adapter. Unlike [`core::iter`] there is no `next`:
//! [`Iterable::expand`](super::Iterable::expand) drives the whole loop at once, and adapters that
//! need more than that use random access through [`IndexedIteratorExpand`]. This is why `zip`,
//! `enumerate`, `rev`, `take` and `skip` can't follow a `filter` in a chain. Reorder the chain
//! instead, as in `.enumerate().filter(..)`.
//!
//! Random access also means `map` doesn't run its closure for an item that a later `skip` drops,
//! which it would in [`core::iter`]. This only matters for a closure with side effects.
//!
//! Methods that return an [`Option`], such as `min`, `max`, `find` and
//! `position`, aren't provided yet. `fold` and `any` cover most of their uses.

mod adapters;
mod range;
mod traits;

pub use adapters::*;
pub use traits::*;

/// Implements `Iterable` and [`IteratorExpand`] for an [`IndexedIteratorExpand`], which can't be
/// done with a blanket impl because `Iterable` is also implemented for other types.
macro_rules! impl_iterable_indexed {
    ([$($generics:tt)*] $ty:ty, $item:ty) => {
        impl<$($generics)*> $crate::frontend::Iterable for $ty {
            type Item = $item;

            fn expand(
                self,
                scope: &$crate::ir::Scope,
                body: &mut dyn FnMut(&$crate::ir::Scope, Self::Item),
            ) {
                $crate::frontend::iter::expand_indexed(self, scope, body);
            }

            fn expand_unroll(
                self,
                _scope: &$crate::ir::Scope,
                _body: &mut dyn FnMut(&$crate::ir::Scope, Self::Item),
            ) {
                unimplemented!("Can't unroll {}", stringify!($ty))
            }
        }

        impl<$($generics)*> $crate::frontend::iter::IteratorExpand for $ty {}
    };
}

pub(crate) use impl_iterable_indexed;
