use std::{cell::RefCell, rc::Rc};

use crate::{
    compute::KernelBuilder,
    prelude::{ArgSettings, LaunchArg, LaunchArgExpand},
    Runtime,
};

use super::{Sequence, SequenceExpand};

pub struct SequenceArg<'a, R: Runtime, T: LaunchArg> {
    values: Vec<T::RuntimeArg<'a, R>>,
}

impl<'a, R: Runtime, T: LaunchArg> SequenceArg<'a, R, T> {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }
    pub fn push(&mut self, arg: T::RuntimeArg<'a, R>) {
        self.values.push(arg);
    }
}

pub struct SequenceCompilationArg<C: LaunchArg> {
    values: Vec<C::CompilationArg>,
}

impl<C: LaunchArg> Clone for SequenceCompilationArg<C> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
        }
    }
}

impl<C: LaunchArg> core::hash::Hash for SequenceCompilationArg<C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.values.hash(state)
    }
}

impl<C: LaunchArg> core::cmp::PartialEq for SequenceCompilationArg<C> {
    fn eq(&self, other: &Self) -> bool {
        self.values.eq(&other.values)
    }
}

impl<C: LaunchArg> core::fmt::Debug for SequenceCompilationArg<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("SequenceCompilationArg {:?}", self.values))
    }
}
impl<C: LaunchArg> core::cmp::Eq for SequenceCompilationArg<C> {}

impl<C: LaunchArg> LaunchArg for Sequence<C> {
    type RuntimeArg<'a, R: Runtime> = SequenceArg<'a, R, C>;

    fn compilation_arg<'a, R: Runtime>(
        runtime_arg: &Self::RuntimeArg<'a, R>,
    ) -> Self::CompilationArg {
        SequenceCompilationArg {
            values: runtime_arg
                .values
                .iter()
                .map(|value| C::compilation_arg(value))
                .collect(),
        }
    }
}

impl<'a, R: Runtime, T: LaunchArg> ArgSettings<R> for SequenceArg<'a, R, T> {
    fn register(&self, launcher: &mut crate::prelude::KernelLauncher<R>) {
        self.values.iter().for_each(|arg| arg.register(launcher));
    }
}

impl<C: LaunchArg> LaunchArgExpand for Sequence<C> {
    type CompilationArg = SequenceCompilationArg<C>;

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
