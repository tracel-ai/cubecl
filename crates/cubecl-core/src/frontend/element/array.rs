use std::marker::PhantomData;

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    frontend::CubeType,
    ir::{Item, Vectorization},
    unexpanded, KernelSettings, Runtime,
};
use crate::{
    frontend::{indexation::Index, CubeContext},
    prelude::{assign, index, index_assign, Comptime},
};

use super::{
    ArgSettings, CubePrimitive, ExpandElement, ExpandElementBaseInit, ExpandElementTyped,
    LaunchArg, LaunchArgExpand, TensorHandleRef, UInt,
};

/// A contiguous array of elements.
pub struct Array<E> {
    _val: PhantomData<E>,
}

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<T: CubePrimitive + Clone> Array<T> {
    pub fn new<S: Index>(_size: S) -> Self {
        Array { _val: PhantomData }
    }

    pub fn vectorized<S: Index>(_size: S, _vectorization_factor: UInt) -> Self {
        Array { _val: PhantomData }
    }

    pub fn __expand_new<S: Index>(
        context: &mut CubeContext,
        size: S,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar(value) => value.as_u32(),
            _ => panic!("Array need constant initialization value"),
        };
        context
            .create_local_array(Item::new(T::as_elem()), size)
            .into()
    }

    pub fn __expand_vectorized<S: Index>(
        context: &mut CubeContext,
        size: S,
        vectorization_factor: UInt,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar(value) => value.as_u32(),
            _ => panic!("Shared memory need constant initialization value"),
        };
        context
            .create_local_array(
                Item::vectorized(T::as_elem(), vectorization_factor.val as u8),
                size,
            )
            .into()
    }

    pub fn to_vectorized(self, _vectorization_factor: Comptime<UInt>) -> T {
        unexpanded!()
    }
}

impl<C: CubePrimitive> ExpandElementTyped<Array<C>> {
    pub fn __expand_to_vectorized_method(
        self,
        context: &mut CubeContext,
        vectorization_factor: UInt,
    ) -> ExpandElementTyped<C> {
        let factor = vectorization_factor.val;
        let var = self.expand.clone();
        let new_var = context.create_local(Item::vectorized(var.item().elem(), factor as u8));

        if vectorization_factor.val == 1 {
            let element = index::expand(context, self.clone(), ExpandElementTyped::from_lit(0u32));
            assign::expand(context, element, new_var.clone());
        } else {
            for i in 0..factor {
                let expand: Self = self.expand.clone().into();
                let element = index::expand(context, expand, ExpandElementTyped::from_lit(i));
                index_assign::expand::<Array<C>>(
                    context,
                    new_var.clone().into(),
                    ExpandElementTyped::from_lit(i),
                    element,
                );
            }
        }
        new_var.into()
    }
}

impl<C: CubeType> CubeType for &Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<C: CubeType> ExpandElementBaseInit for Array<C> {
    fn init_elem(_context: &mut crate::prelude::CubeContext, elem: ExpandElement) -> ExpandElement {
        // The type can't be deeply cloned/copied.
        elem
    }
}

impl<E: CubeType> Array<E> {
    /// Obtain the array length
    pub fn len(&self) -> UInt {
        unexpanded!()
    }
}

impl<C: CubePrimitive> LaunchArg for Array<C> {
    type RuntimeArg<'a, R: Runtime> = ArrayArg<'a, R>;
}

impl<C: CubePrimitive> LaunchArgExpand for Array<C> {
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Array<C>> {
        builder
            .input_array(Item::vectorized(C::as_elem(), vectorization))
            .into()
    }
    fn expand_output(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Array<C>> {
        builder
            .output_array(Item::vectorized(C::as_elem(), vectorization))
            .into()
    }
}

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle).
pub struct ArrayHandleRef<'a, R: Runtime> {
    pub handle: &'a cubecl_runtime::server::Handle<R::Server>,
    pub length: [usize; 1],
}

pub enum ArrayArg<'a, R: Runtime> {
    /// The array is passed with an array handle.
    Handle {
        /// The array handle.
        handle: ArrayHandleRef<'a, R>,
        /// The vectorization factor.
        vectorization_factor: u8,
    },
    /// The array is aliasing another input array.
    Alias {
        /// The position of the input array.
        input_pos: usize,
    },
}

impl<'a, R: Runtime> ArgSettings<R> for ArrayArg<'a, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        if let ArrayArg::Handle {
            handle,
            vectorization_factor: _,
        } = self
        {
            launcher.register_array(handle)
        }
    }

    fn configure_input(&self, position: usize, settings: KernelSettings) -> KernelSettings {
        match self {
            Self::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_input(position, *vectorization_factor),
            Self::Alias { input_pos: _ } => {
                panic!("Not yet supported, only output can be aliased for now.");
            }
        }
    }

    fn configure_output(&self, position: usize, mut settings: KernelSettings) -> KernelSettings {
        match self {
            Self::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_output(position, *vectorization_factor),
            Self::Alias { input_pos } => {
                settings.mappings.push(crate::InplaceMapping {
                    pos_input: *input_pos,
                    pos_output: position,
                });
                settings
            }
        }
    }
}

impl<'a, R: Runtime> ArrayArg<'a, R> {
    /// Create a new array argument.
    ///
    /// Equivalent to using the [vectorized constructor](Self::vectorized) with a vectorization
    /// factor of 1.
    pub fn new(handle: &'a cubecl_runtime::server::Handle<R::Server>, length: usize) -> Self {
        ArrayArg::Handle {
            handle: ArrayHandleRef::new(handle, length),
            vectorization_factor: 1,
        }
    }
    /// Create a new array argument specified with its vectorization factor.
    pub fn vectorized(
        vectorization_factor: u8,
        handle: &'a cubecl_runtime::server::Handle<R::Server>,
        length: usize,
    ) -> Self {
        ArrayArg::Handle {
            handle: ArrayHandleRef::new(handle, length),
            vectorization_factor,
        }
    }
}

impl<'a, R: Runtime> ArrayHandleRef<'a, R> {
    pub fn new(handle: &'a cubecl_runtime::server::Handle<R::Server>, length: usize) -> Self {
        Self {
            handle,
            length: [length],
        }
    }

    pub fn as_tensor(&self) -> TensorHandleRef<'_, R> {
        let shape = &self.length;

        TensorHandleRef {
            handle: self.handle,
            strides: &[1],
            shape,
        }
    }
}
