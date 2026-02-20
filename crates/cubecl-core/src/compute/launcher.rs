use alloc::{boxed::Box, collections::BTreeMap, vec, vec::Vec};
use core::marker::PhantomData;

use crate::prelude::{ArrayArg, TensorArg, TensorMapArg, TensorMapKind};
use crate::{CubeScalar, KernelSettings};
use crate::{MetadataBuilder, Runtime};
#[cfg(feature = "std")]
use core::cell::RefCell;
use cubecl_ir::{AddressType, StorageType};
use cubecl_runtime::server::{CubeCount, Handle, ScalarBinding, TensorMapBinding};
use cubecl_runtime::{
    client::ComputeClient,
    kernel::{CubeKernel, KernelTask},
    server::Bindings,
};

/// Prepare a kernel for [launch](KernelLauncher::launch).
pub struct KernelLauncher<R: Runtime> {
    tensors: TensorState<R>,
    scalars: ScalarState,
    pub settings: KernelSettings,
    runtime: PhantomData<R>,
}

impl<R: Runtime> KernelLauncher<R> {
    /// Register a tensor to be launched.
    pub fn register_tensor(&mut self, tensor: &TensorArg<'_, R>) {
        self.tensors.push_tensor(tensor);
    }

    /// Register a mapped tensor to be launched.
    pub fn register_tensor_map<K: TensorMapKind>(&mut self, tensor: &TensorMapArg<'_, R, K>) {
        self.tensors.push_tensor_map(tensor);
    }

    /// Register an input array to be launched.
    pub fn register_array(&mut self, array: &ArrayArg<'_, R>) {
        self.tensors.push_array(array);
    }

    /// Register a scalar to be launched.
    pub fn register_scalar<C: CubeScalar>(&mut self, scalar: C) {
        self.scalars.push(scalar);
    }

    /// Register a scalar to be launched from raw data.
    pub fn register_scalar_raw(&mut self, bytes: &[u8], dtype: StorageType) {
        self.scalars.push_raw(bytes, dtype);
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
    fn into_bindings(self) -> Bindings {
        let mut bindings = Bindings::new();

        self.tensors.register(&mut bindings);
        self.scalars.register(&mut bindings);

        bindings
    }
}

#[cfg(feature = "std")]
std::thread_local! {
    static METADATA: RefCell<MetadataBuilder> = RefCell::new(MetadataBuilder::default());
}

/// Handles the tensor state.
pub enum TensorState<R: Runtime> {
    /// No tensor is registered yet.
    Empty { addr_type: AddressType },
    /// The registered tensors.
    Some {
        buffers: Vec<Handle>,
        tensor_maps: Vec<TensorMapBinding>,
        addr_type: AddressType,
        runtime: PhantomData<R>,
        #[cfg(not(feature = "std"))]
        metadata: MetadataBuilder,
    },
}

/// Handles the scalar state of an element type
///
/// The scalars are grouped to reduce the number of buffers needed to send data to the compute device.
#[derive(Default, Clone)]
pub struct ScalarState {
    data: BTreeMap<StorageType, ScalarValues>,
}

/// Stores the data and type for a scalar arg
pub type ScalarValues = Vec<u8>;

impl<R: Runtime> TensorState<R> {
    fn maybe_init(&mut self) {
        if let TensorState::Empty { addr_type } = self {
            *self = TensorState::Some {
                buffers: Vec::new(),
                tensor_maps: Vec::new(),
                addr_type: *addr_type,
                runtime: PhantomData,
                #[cfg(not(feature = "std"))]
                metadata: MetadataBuilder::default(),
            };
        }
    }

    #[cfg(feature = "std")]
    fn with_metadata<T>(&mut self, fun: impl FnMut(&mut MetadataBuilder) -> T) -> T {
        METADATA.with_borrow_mut(fun)
    }

    #[cfg(not(feature = "std"))]
    fn with_metadata<T>(&mut self, mut fun: impl FnMut(&mut MetadataBuilder) -> T) -> T {
        self.maybe_init();
        let TensorState::Some { metadata, .. } = self else {
            panic!("Should be init");
        };
        fun(metadata)
    }

    fn buffers(&mut self) -> &mut Vec<Handle> {
        self.maybe_init();
        let TensorState::Some { buffers, .. } = self else {
            panic!("Should be init");
        };
        buffers
    }

    fn tensor_maps(&mut self) -> &mut Vec<TensorMapBinding> {
        self.maybe_init();
        let TensorState::Some { tensor_maps, .. } = self else {
            panic!("Should be init");
        };
        tensor_maps
    }

    fn address_type(&self) -> AddressType {
        match self {
            TensorState::Empty { addr_type } => *addr_type,
            TensorState::Some { addr_type, .. } => *addr_type,
        }
    }

    /// Push a new input tensor to the state.
    pub fn push_tensor(&mut self, tensor: &TensorArg<'_, R>) {
        if let Some(tensor) = self.process_tensor(tensor) {
            self.buffers().push(tensor);
        }
    }

    fn process_tensor(&mut self, tensor: &TensorArg<'_, R>) -> Option<Handle> {
        let (tensor, vectorization) = match tensor {
            TensorArg::Handle {
                handle,
                line_size: vectorization_factor,
                ..
            } => (handle, vectorization_factor),
            TensorArg::Alias { .. } => return None,
        };

        let elem_size = tensor.elem_size * *vectorization;
        let buffer_len = tensor.handle.size() / elem_size as u64;
        let len = tensor.shape.iter().product::<usize>() / *vectorization;
        let address_type = self.address_type();
        self.with_metadata(|meta| {
            meta.register_tensor(
                tensor.strides.len() as u64,
                buffer_len,
                len as u64,
                tensor.shape.clone(),
                tensor.strides.clone(),
                address_type,
            )
        });
        Some(tensor.handle.clone())
    }

    /// Push a new input array to the state.
    pub fn push_array(&mut self, array: &ArrayArg<'_, R>) {
        if let Some(tensor) = self.process_array(array) {
            self.buffers().push(tensor);
        }
    }

    fn process_array(&mut self, array: &ArrayArg<'_, R>) -> Option<Handle> {
        let (array, vectorization) = match array {
            ArrayArg::Handle {
                handle,
                line_size: vectorization_factor,
                ..
            } => (handle, vectorization_factor),
            ArrayArg::Alias { .. } => return None,
        };

        let elem_size = array.elem_size * *vectorization;
        let buffer_len = array.handle.size() / elem_size as u64;
        let address_type = self.address_type();
        self.with_metadata(|meta| {
            meta.register_array(
                buffer_len,
                array.length[0] as u64 / *vectorization as u64,
                address_type,
            )
        });
        Some(array.handle.clone())
    }

    /// Push a new tensor to the state.
    pub fn push_tensor_map<K: TensorMapKind>(&mut self, map: &TensorMapArg<'_, R, K>) {
        let binding = self
            .process_tensor(&map.tensor)
            .expect("Can't use alias for TensorMap");

        let map = map.metadata.clone();
        self.tensor_maps().push(TensorMapBinding { binding, map });
    }

    fn register(mut self, bindings_global: &mut Bindings) {
        let metadata = matches!(self, Self::Some { .. }).then(|| {
            let addr_type = self.address_type();
            self.with_metadata(|meta| meta.finish(addr_type))
        });
        if let Self::Some {
            buffers,
            tensor_maps,
            ..
        } = self
        {
            let metadata = metadata.unwrap();

            bindings_global.handles = buffers;
            bindings_global.tensor_maps = tensor_maps;
            bindings_global.metadata = metadata;
        }
    }
}

impl ScalarState {
    /// Add a new scalar value to the state.
    pub fn push<T: CubeScalar>(&mut self, val: T) {
        let val = [val];
        let bytes = T::as_bytes(&val);
        self.data
            .entry(T::cube_type())
            .or_default()
            .extend(bytes.iter().copied());
    }

    /// Add a new raw value to the state.
    pub fn push_raw(&mut self, bytes: &[u8], dtype: StorageType) {
        self.data
            .entry(dtype)
            .or_default()
            .extend(bytes.iter().copied());
    }

    fn register(&self, bindings: &mut Bindings) {
        for (ty, values) in self.data.iter() {
            let len = values.len() / ty.size();
            let len_u64 = len.div_ceil(size_of::<u64>() / ty.size());

            let mut data = vec![0; len_u64];
            let slice = bytemuck::cast_slice_mut::<u64, u8>(&mut data);
            slice[0..values.len()].copy_from_slice(values);
            bindings
                .scalars
                .insert(*ty, ScalarBinding::new(*ty, len, data));
        }
    }
}

impl<R: Runtime> KernelLauncher<R> {
    pub fn new(settings: KernelSettings) -> Self {
        Self {
            tensors: TensorState::Empty {
                addr_type: settings.address_type,
            },
            scalars: Default::default(),
            settings,
            runtime: PhantomData,
        }
    }
}
