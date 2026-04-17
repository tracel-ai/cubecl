use std::{marker::PhantomData, sync::Arc};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic, ir::Scope, unexpanded};

use crate::tensor::layout::{Coordinates, Layout, LayoutExpand};

/// A virtual layout, to carry a layout without the need for generic parameters everywhere.
/// `C` represents the coordinate space of the underlying layout.
#[derive(Clone)]
pub struct VirtualLayout<C: Coordinates, S: Coordinates> {
    _coords: PhantomData<(C, S)>,
}

impl<C: Coordinates, S: Coordinates> Copy for VirtualLayout<C, S> {}
unsafe impl<C: Coordinates, S: Coordinates> Send for VirtualLayout<C, S> {}
unsafe impl<C: Coordinates, S: Coordinates> Sync for VirtualLayout<C, S> {}

#[derive(Clone)]
pub struct VirtualLayoutExpand<C: Coordinates, S: Coordinates> {
    pub(crate) state: Arc<dyn VirtualLayoutOperationsExpand<C, S>>,
}

#[cube]
impl<C: Coordinates, S: Coordinates> VirtualLayout<C, S> {
    /// Virtual version of [`Layout::to_source_pos`]
    #[allow(unused)]
    pub fn to_source_pos(&self, pos: C) -> S {
        intrinsic!(|scope| { self.state.__expand_to_source_pos_method(scope, pos) })
    }

    /// Virtual version of [`Layout::to_source_pos_checked`]
    #[allow(unused)]
    pub fn to_source_pos_checked(&self, pos: C) -> (S, bool) {
        intrinsic!(|scope| { self.state.__expand_to_source_pos_checked_method(scope, pos) })
    }

    /// Virtual version of [`Layout::shape`]
    pub fn shape(&self) -> C {
        intrinsic!(|scope| { self.state.__expand_shape_method(scope) })
    }

    /// Virtual version of [`Layout::is_in_bounds`]
    #[allow(unused)]
    pub fn is_in_bounds(&self, pos: C) -> bool {
        intrinsic!(|scope| { self.state.__expand_is_in_bounds_method(scope, pos) })
    }
}

impl<C: Coordinates, S: Coordinates> VirtualLayout<C, S> {
    /// Create a new virtual layout from a concrete one
    pub fn new<L: Layout<Coordinates = C, SourceCoordinates = S>>(
        _layout: L,
    ) -> VirtualLayout<C, S> {
        unexpanded!()
    }

    /// Expand function of [`VirtualLayout::`__`expand_new`]
    pub fn __expand_new<L: Layout<Coordinates = C, SourceCoordinates = S> + 'static>(
        _scope: &mut Scope,
        layout: L::ExpandType,
    ) -> VirtualLayoutExpand<C, S> {
        VirtualLayoutExpand::new::<L::ExpandType>(layout)
    }
}

impl<C: Coordinates, S: Coordinates> VirtualLayoutExpand<C, S> {
    /// Create a new virtual layout from a concrete one
    pub fn new<L: VirtualLayoutOperationsExpand<C, S> + 'static>(
        layout: L,
    ) -> VirtualLayoutExpand<C, S> {
        VirtualLayoutExpand::<C, S> {
            state: Arc::new(layout),
        }
    }
}

impl<C: Coordinates, S: Coordinates> CubeType for VirtualLayout<C, S> {
    type ExpandType = VirtualLayoutExpand<C, S>;
}

impl<C: Coordinates, S: Coordinates> IntoMut for VirtualLayoutExpand<C, S> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<C: Coordinates, S: Coordinates> CubeDebug for VirtualLayoutExpand<C, S> {}

// We need to seal the trait to allow us to blanket implement `From<L>` below
mod private {
    pub trait Sealed {}
}
pub trait VirtualLayoutOperationsExpand<C: CubeType, S: CubeType>: private::Sealed {
    fn __expand_to_source_pos_method(
        &self,
        scope: &mut Scope,
        pos: <C as CubeType>::ExpandType,
    ) -> <S as CubeType>::ExpandType;
    fn __expand_to_source_pos_checked_method(
        &self,
        scope: &mut Scope,
        pos: <C as CubeType>::ExpandType,
    ) -> <(S, bool) as CubeType>::ExpandType;
    fn __expand_shape_method(&self, scope: &mut Scope) -> <C as CubeType>::ExpandType;
    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: <C as CubeType>::ExpandType,
    ) -> NativeExpand<bool>;
}

impl<L: LayoutExpand> private::Sealed for L {}
impl<L: LayoutExpand> VirtualLayoutOperationsExpand<L::Coordinates, L::SourceCoordinates> for L {
    fn __expand_to_source_pos_method(
        &self,
        scope: &mut Scope,
        pos: <L::Coordinates as CubeType>::ExpandType,
    ) -> <L::SourceCoordinates as CubeType>::ExpandType {
        <L as LayoutExpand>::__expand_to_source_pos_method(self.clone(), scope, pos)
    }

    fn __expand_to_source_pos_checked_method(
        &self,
        scope: &mut Scope,
        pos: <L::Coordinates as CubeType>::ExpandType,
    ) -> <(L::SourceCoordinates, bool) as CubeType>::ExpandType {
        <L as LayoutExpand>::__expand_to_source_pos_checked_method(self.clone(), scope, pos)
    }

    fn __expand_shape_method(&self, scope: &mut Scope) -> <L::Coordinates as CubeType>::ExpandType {
        <L as LayoutExpand>::__expand_shape_method(self.clone(), scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: <L::Coordinates as CubeType>::ExpandType,
    ) -> NativeExpand<bool> {
        <L as LayoutExpand>::__expand_is_in_bounds_method(self.clone(), scope, pos)
    }
}

impl<C: Coordinates, S: Coordinates, L: VirtualLayoutOperationsExpand<C, S> + 'static> From<L>
    for VirtualLayoutExpand<C, S>
{
    fn from(value: L) -> Self {
        VirtualLayoutExpand::new(value)
    }
}

impl<L: Layout + 'static> From<L> for VirtualLayout<L::Coordinates, L::SourceCoordinates> {
    fn from(_value: L) -> Self {
        VirtualLayout {
            _coords: PhantomData,
        }
    }
}

mod launch {
    use alloc::rc::Rc;
    use core::cell::RefCell;

    use cubecl_core::{
        format::DebugRaw,
        hash::{StableHash, StableHasher},
    };

    use super::*;

    type ExpandFn<C, S> =
        Rc<RefCell<dyn FnMut(&mut KernelBuilder, bool) -> VirtualLayoutExpand<C, S> + Send>>;

    pub struct VirtualLayoutLaunch<C: Coordinates, S: Coordinates, R: Runtime> {
        _phantom_runtime: core::marker::PhantomData<R>,
        #[allow(clippy::type_complexity)]
        register: Box<
            dyn FnOnce(&mut KernelLauncher<R>) -> VirtualLayoutCompilationArg<C, S> + Send + Sync,
        >,
    }

    impl<C: Coordinates, S: Coordinates, R: Runtime> VirtualLayoutLaunch<C, S, R> {
        pub fn new<L: Layout<Coordinates = C, SourceCoordinates = S> + LaunchArg>(
            layout: L::RuntimeArg<R>,
        ) -> Self {
            Self {
                _phantom_runtime: PhantomData,
                register: Box::new(move |launcher| {
                    let comp_arg = L::register(layout, launcher);
                    let comp_arg_2 = comp_arg.clone();
                    let expand = move |builder: &mut KernelBuilder, is_out: bool| {
                        let expand = match is_out {
                            true => L::expand_output(&comp_arg_2, builder),
                            false => L::expand(&comp_arg_2, builder),
                        };
                        VirtualLayoutExpand::new(expand)
                    };
                    VirtualLayoutCompilationArg::new::<L::CompilationArg>(
                        comp_arg,
                        Rc::new(RefCell::new(expand)),
                    )
                }),
            }
        }
    }

    #[derive(Clone)]
    pub struct VirtualLayoutCompilationArg<C: Coordinates, S: Coordinates> {
        type_name: String,
        debug: Rc<dyn core::fmt::Debug>,
        hash: StableHash,
        expand: ExpandFn<C, S>,
    }

    // SAFETY: The struct is readonly, so `Sync` is safe to implement
    unsafe impl<C: Coordinates, S: Coordinates> Send for VirtualLayoutCompilationArg<C, S> {}
    unsafe impl<C: Coordinates, S: Coordinates> Sync for VirtualLayoutCompilationArg<C, S> {}

    impl<C: Coordinates, S: Coordinates> VirtualLayoutCompilationArg<C, S> {
        pub fn new<L: CompilationArg + 'static>(arg: L, expand: ExpandFn<C, S>) -> Self {
            // Hash ahead of time so we don't need to store the actual data, which would be far
            // more complex
            let hash = StableHasher::hash_one(&arg);
            Self {
                type_name: core::any::type_name::<L>().to_string(),
                debug: Rc::new(arg),
                hash,
                expand,
            }
        }
    }

    impl<C: Coordinates, S: Coordinates> PartialEq for VirtualLayoutCompilationArg<C, S> {
        fn eq(&self, other: &Self) -> bool {
            self.type_name == other.type_name && self.hash == other.hash
        }
    }
    impl<C: Coordinates, S: Coordinates> Eq for VirtualLayoutCompilationArg<C, S> {}

    impl<C: Coordinates, S: Coordinates> core::hash::Hash for VirtualLayoutCompilationArg<C, S> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.type_name.hash(state);
            self.hash.hash(state);
        }
    }

    impl<C: Coordinates, S: Coordinates> core::fmt::Debug for VirtualLayoutCompilationArg<C, S> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct(stringify!(VirtualLayout))
                .field("type", &DebugRaw(&self.type_name))
                .field("value", &self.debug)
                .finish()
        }
    }

    impl<C: Coordinates + 'static, S: Coordinates + 'static> LaunchArg for VirtualLayout<C, S> {
        type RuntimeArg<R: Runtime> = VirtualLayoutLaunch<C, S, R>;
        type CompilationArg = VirtualLayoutCompilationArg<C, S>;

        fn register<R: Runtime>(
            arg: Self::RuntimeArg<R>,
            launcher: &mut KernelLauncher<R>,
        ) -> Self::CompilationArg {
            let func = arg.register;
            func(launcher)
        }
        fn expand(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            let mut expand = arg.expand.borrow_mut();
            expand(builder, false)
        }
        fn expand_output(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            let mut expand = arg.expand.borrow_mut();
            expand(builder, true)
        }
    }
}

pub use launch::*;
