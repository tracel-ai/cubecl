use alloc::{boxed::Box, vec::Vec};
use core::marker::PhantomData;

use crate::Runtime;
use crate::prelude::{BufferArg, TensorArg, TensorMapArg, TensorMapKind};
use crate::{InfoBuilder, ScalarArgType};
#[cfg(feature = "std")]
use core::cell::RefCell;
use cubecl_ir::{AddressType, ElemType, Scope, settings::KernelSettings};
use cubecl_runtime::server::{
    BufferBinding, CubeCount, KernelResource, TensorMapBinding, WriteMask,
};
use cubecl_runtime::{
    client::ComputeClient,
    kernel::{CubeKernel, KernelTask},
    server::KernelArguments,
};

#[cfg(feature = "std")]
std::thread_local! {
    static INFO: RefCell<InfoBuilder> = RefCell::new(InfoBuilder::default());
    // Only used for resolving types
    static SCOPE: RefCell<Scope> = RefCell::new(Scope::dummy());
}

/// Prepare a kernel for [launch](KernelLauncher::launch).
pub struct KernelLauncher<R: Runtime> {
    resources: Vec<KernelResource>,
    /// Which of `resources` the kernel writes, one bit per resource.
    writes: WriteMask,
    /// Whether the argument being registered right now is one the kernel
    /// writes. Set by [`arg_writes`](Self::arg_writes) before each argument, so
    /// everything that argument registers inherits it — a `&mut` struct of
    /// tensors registers several resources and the kernel may write any of
    /// them.
    arg_writes: bool,
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
    pub fn with_scope<T>(&mut self, fun: impl FnMut(&Scope) -> T) -> T {
        SCOPE.with_borrow(fun)
    }

    #[cfg(not(feature = "std"))]
    pub fn with_scope<T>(&mut self, mut fun: impl FnMut(&Scope) -> T) -> T {
        fun(&self.scope)
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
    pub fn register_scalar_raw(&mut self, bytes: &[u8], dtype: ElemType) {
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

        bindings.resources = self.resources;
        bindings.writes = self.writes;
        bindings.info = info;

        bindings
    }
}

// Tensors/arrays
impl<R: Runtime> KernelLauncher<R> {
    /// Declare whether the argument registered next is one the kernel writes.
    ///
    /// `&mut` in the `#[cube]` signature is how a kernel says it produces a
    /// value — the macro has no other way to tell, since `&[T]` and `&mut [T]`
    /// share one [`LaunchArg`](crate::prelude::LaunchArg) impl and the
    /// distinction is gone by the time this runs.
    ///
    /// It stays set for everything that argument registers, which is the safe
    /// direction: a `&mut` struct of tensors names all of them, and the kernel
    /// may write any.
    pub fn arg_writes(&mut self, writes: bool) {
        self.arg_writes = writes;
    }

    /// Record a resource and whether the kernel writes it, so the two can never
    /// fall out of step.
    fn push_resource(&mut self, resource: KernelResource) {
        self.resources.push(resource);
        self.writes.push(self.arg_writes);
    }

    /// An output that aliases an input writes that input's buffer in place, so
    /// the input is written however its own argument was declared. Missing this
    /// would leave an in-place kernel's only buffer unnamed when it fails.
    fn alias_writes(&mut self, input_pos: usize) {
        if self.arg_writes {
            self.writes.set(input_pos);
        }
    }

    /// Push a new input tensor to the state.
    pub fn register_tensor(&mut self, tensor: TensorArg<R>, elem_size: usize) {
        if let Some(tensor) = self.process_tensor(tensor, elem_size) {
            self.push_resource(KernelResource::Buffer(tensor));
        }
    }

    fn process_tensor(&mut self, tensor: TensorArg<R>, elem_size: usize) -> Option<BufferBinding> {
        let tensor = match tensor {
            TensorArg::Handle { handle, .. } => handle,
            TensorArg::Alias { input_pos, .. } => {
                self.alias_writes(input_pos);
                return None;
            }
        };

        let buffer_len = tensor.handle.size_in_used() / elem_size as u64;
        let address_type = self.address_type;

        self.with_info(|info| {
            info.metadata.register_tensor(
                buffer_len,
                tensor.shape.clone(),
                tensor.strides.clone(),
                address_type,
            )
        });
        Some(tensor.handle)
    }

    /// Push a new input array to the state.
    pub fn register_buffer(&mut self, array: BufferArg<R>, elem_size: usize) {
        if let Some(tensor) = self.process_buffer(array, elem_size) {
            self.push_resource(KernelResource::Buffer(tensor));
        }
    }

    fn process_buffer(&mut self, array: BufferArg<R>, elem_size: usize) -> Option<BufferBinding> {
        let array = match array {
            BufferArg::Handle { handle, .. } => handle,
            BufferArg::Alias { input_pos, .. } => {
                self.alias_writes(input_pos);
                return None;
            }
        };

        let buffer_len = array.handle.size_in_used() / elem_size as u64;
        let address_type = self.address_type;
        self.with_info(|info| info.metadata.register_buffer(buffer_len, address_type));
        Some(array.handle)
    }

    /// Push a new tensor to the state.
    pub fn register_tensor_map<K: TensorMapKind>(
        &mut self,
        map: TensorMapArg<R, K>,
        elem_size: usize,
    ) {
        let binding = self
            .process_tensor(map.tensor, elem_size)
            .expect("Can't use alias for TensorMap");

        let map = map.metadata.clone();
        self.push_resource(KernelResource::TensorMap(TensorMapBinding { binding, map }));
    }
}

impl<R: Runtime> KernelLauncher<R> {
    pub fn new(settings: KernelSettings) -> Self {
        Self {
            address_type: settings.address_type,
            settings,
            resources: Vec::new(),
            writes: WriteMask::default(),
            arg_writes: false,
            _runtime: PhantomData,
            #[cfg(not(feature = "std"))]
            info: InfoBuilder::default(),
            #[cfg(not(feature = "std"))]
            scope: Scope::dummy(),
        }
    }
}
