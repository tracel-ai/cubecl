use alloc::{boxed::Box, vec::Vec};
use core::marker::PhantomData;

use crate::Runtime;
use crate::prelude::{ArrayArg, TensorArg, TensorMapArg, TensorMapKind};
use crate::{InfoBuilder, KernelSettings, ScalarArgType};
#[cfg(feature = "std")]
use core::cell::RefCell;
use cubecl_ir::{AddressType, Scope, StorageType, Type};
use cubecl_runtime::server::{Binding, CubeCount, TensorMapBinding};
use cubecl_runtime::{
    client::ComputeClient,
    kernel::{CubeKernel, KernelTask},
    server::KernelArguments,
};

#[cfg(feature = "std")]
std::thread_local! {
    static INFO: RefCell<InfoBuilder> = RefCell::new(InfoBuilder::default());
    // Only used for resolving types
    static SCOPE: RefCell<Scope> = RefCell::new(Scope::root(false));
}

/// Prepare a kernel for [launch](KernelLauncher::launch).
pub struct KernelLauncher<R: Runtime> {
    buffers: Vec<Binding>,
    tensor_maps: Vec<TensorMapBinding>,
    address_type: AddressType,
    pub settings: KernelSettings,
    #[cfg(not(feature = "std"))]
    info: InfoBuilder,
    #[cfg(not(feature = "std"))]
    pub scope: Scope,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> KernelLauncher<R> {
    #[cfg(feature = "std")]
    pub fn with_scope<T>(&mut self, fun: impl FnMut(&mut Scope) -> T) -> T {
        SCOPE.with_borrow_mut(fun)
    }

    #[cfg(not(feature = "std"))]
    pub fn with_scope<T>(&mut self, mut fun: impl FnMut(&mut Scope) -> T) -> T {
        fun(&mut self.scope)
    }

    #[cfg(feature = "std")]
    fn with_info<T>(&mut self, fun: impl FnMut(&mut InfoBuilder) -> T) -> T {
        INFO.with_borrow_mut(fun)
    }

    #[cfg(not(feature = "std"))]
    fn with_info<T>(&mut self, mut fun: impl FnMut(&mut InfoBuilder) -> T) -> T {
        fun(&mut self.info)
    }

    /// Register a scalar to be launched.
    pub fn register_scalar<C: ScalarArgType>(&mut self, scalar: C) {
        self.with_info(|info| info.scalars.push(scalar));
    }

    /// Register a scalar to be launched from raw data.
    pub fn register_scalar_raw(&mut self, bytes: &[u8], dtype: StorageType) {
        self.with_info(|info| info.scalars.push_raw(bytes, dtype));
    }

    /// Launch the kernel.
    #[track_caller]
    pub fn launch<K: CubeKernel>(
        self,
        cube_count: CubeCount,
        kernel: K,
        client: &ComputeClient<R>,
    ) {
        let bindings = self.into_bindings();
        let kernel = Box::new(KernelTask::<R::Compiler, K>::new(kernel));

        client.launch(kernel, cube_count, bindings)
    }

    /// Launch the kernel without check bounds.
    ///
    /// # Safety
    ///
    /// The kernel must not:
    /// - Contain any out of bounds reads or writes. Doing so is immediate UB.
    /// - Contain any loops that never terminate. These may be optimized away entirely or cause
    ///   other unpredictable behaviour.
    #[track_caller]
    pub unsafe fn launch_unchecked<K: CubeKernel>(
        self,
        cube_count: CubeCount,
        kernel: K,
        client: &ComputeClient<R>,
    ) {
        unsafe {
            let bindings = self.into_bindings();
            let kernel = Box::new(KernelTask::<R::Compiler, K>::new(kernel));

            client.launch_unchecked(kernel, cube_count, bindings)
        }
    }

    /// We need to create the bindings in the same order they are defined in the compilation step.
    ///
    /// The function [`crate::KernelIntegrator::integrate`] stars by registering the input tensors followed
    /// by the output tensors. Then the tensor metadata, and the scalars at the end. The scalars
    /// are registered in the same order they are added. This is why we store the scalar data type
    /// in the `scalar_order` vector, so that we can register them in the same order.
    ///
    /// Also returns an ordered list of constant bindings. The ordering between constants and tensors
    /// is up to the runtime.
    fn into_bindings(mut self) -> KernelArguments {
        let mut bindings = KernelArguments::new();
        let address_type = self.address_type;
        let info = self.with_info(|info| info.finish(address_type));

        bindings.buffers = self.buffers;
        bindings.tensor_maps = self.tensor_maps;
        bindings.info = info;

        bindings
    }
}

// Tensors/arrays
impl<R: Runtime> KernelLauncher<R> {
    /// Push a new input tensor to the state.
    pub fn register_tensor(&mut self, tensor: TensorArg<R>, ty: Type) {
        if let Some(tensor) = self.process_tensor(tensor, ty) {
            self.buffers.push(tensor);
        }
    }

    fn process_tensor(&mut self, tensor: TensorArg<R>, ty: Type) -> Option<Binding> {
        let tensor = match tensor {
            TensorArg::Handle { handle, .. } => handle,
            TensorArg::Alias { .. } => return None,
        };

        let elem_size = ty.size();
        let vectorization = ty.vector_size();

        let buffer_len = tensor.handle.size_in_used() / elem_size as u64;
        let len = tensor.shape.iter().product::<usize>() / vectorization;
        let address_type = self.address_type;
        self.with_info(|info| {
            info.metadata.register_tensor(
                tensor.strides.len() as u64,
                buffer_len,
                len as u64,
                tensor.shape.clone(),
                tensor.strides.clone(),
                address_type,
            )
        });
        Some(tensor.handle)
    }

    /// Push a new input array to the state.
    pub fn register_array(&mut self, array: ArrayArg<R>, ty: Type) {
        if let Some(tensor) = self.process_array(array, ty) {
            self.buffers.push(tensor);
        }
    }

    fn process_array(&mut self, array: ArrayArg<R>, ty: Type) -> Option<Binding> {
        let array = match array {
            ArrayArg::Handle { handle, .. } => handle,
            ArrayArg::Alias { .. } => return None,
        };

        let elem_size = ty.size();
        let vectorization = ty.vector_size();

        let buffer_len = array.handle.size_in_used() / elem_size as u64;
        let address_type = self.address_type;
        self.with_info(|info| {
            info.metadata.register_array(
                buffer_len,
                array.length[0] as u64 / vectorization as u64,
                address_type,
            )
        });
        Some(array.handle)
    }

    /// Push a new tensor to the state.
    pub fn register_tensor_map<K: TensorMapKind>(&mut self, map: TensorMapArg<R, K>, ty: Type) {
        let binding = self
            .process_tensor(map.tensor, ty)
            .expect("Can't use alias for TensorMap");

        let map = map.metadata.clone();
        self.tensor_maps.push(TensorMapBinding { binding, map });
    }
}

impl<R: Runtime> KernelLauncher<R> {
    pub fn new(settings: KernelSettings) -> Self {
        Self {
            address_type: settings.address_type,
            settings,
            buffers: Vec::new(),
            tensor_maps: Vec::new(),
            _runtime: PhantomData,
            #[cfg(not(feature = "std"))]
            info: InfoBuilder::default(),
            #[cfg(not(feature = "std"))]
            scope: Scope::root(false),
        }
    }
}
