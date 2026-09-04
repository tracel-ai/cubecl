use alloc::vec::Vec;

use cubecl_zspace::SmallVec;

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    prelude::{CubeType, LaunchArg},
};

use super::{Sequence, SequenceExpand};

pub struct SequenceArg<T: LaunchArg> {
    pub values: SmallVec<[T::RuntimeArg; 5]>,
}

impl<T: LaunchArg> Default for SequenceArg<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: LaunchArg> SequenceArg<T> {
    pub fn new() -> Self {
        Self {
            values: SmallVec::new(),
        }
    }
    pub fn push(&mut self, arg: T::RuntimeArg) {
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

impl<C: LaunchArg + CubeType + 'static> LaunchArg for Sequence<C> {
    type RuntimeArg = SequenceArg<C>;
    type CompilationArg = SequenceCompilationArg<C>;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
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

        SequenceExpand { values }
    }
}

impl<E: LaunchArg> FromIterator<E::RuntimeArg> for SequenceArg<E> {
    fn from_iter<T: IntoIterator<Item = E::RuntimeArg>>(iter: T) -> Self {
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

impl<E: LaunchArg, const N: usize> From<[E::RuntimeArg; N]> for SequenceArg<E> {
    fn from(value: [E::RuntimeArg; N]) -> Self {
        let mut arg = SequenceArg::<E> {
            values: SmallVec::new(),
        };
        for v in value {
            arg.values.push(v)
        }
        arg
    }
}
