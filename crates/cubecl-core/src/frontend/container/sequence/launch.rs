use alloc::{rc::Rc, vec::Vec};
use core::cell::RefCell;

use cubecl_runtime::runtime::Runtime;
use cubecl_zspace::SmallVec;

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    prelude::LaunchArg,
};

use super::{Sequence, SequenceExpand};

pub struct SequenceArg<R: Runtime, T: LaunchArg> {
    pub values: SmallVec<[T::RuntimeArg<R>; 5]>,
}

impl<R: Runtime, T: LaunchArg> Default for SequenceArg<R, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Runtime, T: LaunchArg> SequenceArg<R, T> {
    pub fn new() -> Self {
        Self {
            values: SmallVec::new(),
        }
    }
    pub fn push(&mut self, arg: T::RuntimeArg<R>) {
        self.values.push(arg);
    }
}

pub struct SequenceCompilationArg<C: LaunchArg> {
    pub values: SmallVec<[C::CompilationArg; 5]>,
}

impl<C: LaunchArg> Clone for SequenceCompilationArg<C> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
        }
    }
}

impl<C: LaunchArg> core::hash::Hash for SequenceCompilationArg<C> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.values.hash(state)
    }
}

impl<C: LaunchArg> core::cmp::PartialEq for SequenceCompilationArg<C> {
    fn eq(&self, other: &Self) -> bool {
        self.values.eq(&other.values)
    }
}

impl<C: LaunchArg> core::fmt::Debug for SequenceCompilationArg<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Sequence ")?;
        self.values.fmt(f)
    }
}
impl<C: LaunchArg> core::cmp::Eq for SequenceCompilationArg<C> {}

impl<C: LaunchArg> LaunchArg for Sequence<C> {
    type RuntimeArg<R: Runtime> = SequenceArg<R, C>;
    type CompilationArg = SequenceCompilationArg<C>;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        arg.values
            .into_iter()
            .map(|arg| C::register(arg, launcher))
            .collect()
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> SequenceExpand<C> {
        let values = arg
            .values
            .iter()
            .map(|value| C::expand(value, builder))
            .collect::<Vec<_>>();

        SequenceExpand {
            values: Rc::new(RefCell::new(values)),
        }
    }

    fn expand_output(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> SequenceExpand<C> {
        let values = arg
            .values
            .iter()
            .map(|value| C::expand_output(value, builder))
            .collect::<Vec<_>>();

        SequenceExpand {
            values: Rc::new(RefCell::new(values)),
        }
    }
}

impl<R: Runtime, E: LaunchArg> FromIterator<E::RuntimeArg<R>> for SequenceArg<R, E> {
    fn from_iter<T: IntoIterator<Item = E::RuntimeArg<R>>>(iter: T) -> Self {
        SequenceArg {
            values: iter.into_iter().collect(),
        }
    }
}

impl<E: LaunchArg> FromIterator<E::CompilationArg> for SequenceCompilationArg<E> {
    fn from_iter<T: IntoIterator<Item = E::CompilationArg>>(iter: T) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }
}
