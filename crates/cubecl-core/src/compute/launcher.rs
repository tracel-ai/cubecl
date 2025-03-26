use std::marker::PhantomData;

use crate::prelude::{ArrayArg, TensorArg, TensorMapArg};
use crate::{Kernel, Runtime};
use crate::{KernelSettings, prelude::CubePrimitive};
use crate::{
    MetadataBuilder,
    ir::{Elem, FloatKind, IntKind},
};
use crate::{compute::KernelTask, ir::UIntKind};
use bytemuck::NoUninit;
use cubecl_runtime::server::{Binding, CubeCount, ScalarBinding, TensorMapBinding};
use cubecl_runtime::{client::ComputeClient, server::Bindings};

/// Prepare a kernel for [launch](KernelLauncher::launch).
pub struct KernelLauncher<R: Runtime> {
    tensors: TensorState<R>,
    scalar_bf16: ScalarState<half::bf16>,
    scalar_f16: ScalarState<half::f16>,
    scalar_f32: ScalarState<f32>,
    scalar_f64: ScalarState<f64>,
    scalar_u64: ScalarState<u64>,
    scalar_u32: ScalarState<u32>,
    scalar_u16: ScalarState<u16>,
    scalar_u8: ScalarState<u8>,
    scalar_i64: ScalarState<i64>,
    scalar_i32: ScalarState<i32>,
    scalar_i16: ScalarState<i16>,
    scalar_i8: ScalarState<i8>,
    scalar_order: Vec<Elem>,
    pub settings: KernelSettings,
    runtime: PhantomData<R>,
}

impl<R: Runtime> KernelLauncher<R> {
    /// Register an input tensor to be launched.
    pub fn register_input_tensor(&mut self, tensor: &TensorArg<'_, R>) {
        self.tensors.push_input_tensor(tensor);
    }

    /// Register an output tensor to be launched.
    pub fn register_output_tensor(&mut self, tensor: &TensorArg<'_, R>) {
        self.tensors.push_output_tensor(tensor);
    }

    /// Register a mapped tensor to be launched.
    pub fn register_tensor_map(&mut self, tensor: &TensorMapArg<'_, R>) {
        self.tensors.push_tensor_map(tensor);
    }

    /// Register an input array to be launched.
    pub fn register_input_array(&mut self, array: &ArrayArg<'_, R>) {
        self.tensors.push_input_array(array);
    }

    /// Register an output array to be launched.
    pub fn register_output_array(&mut self, array: &ArrayArg<'_, R>) {
        self.tensors.push_output_array(array);
    }

    /// Register a u8 scalar to be launched.
    pub fn register_u8(&mut self, scalar: u8) {
        self.register_scalar(Elem::UInt(UIntKind::U8));
        self.scalar_u8.push(scalar);
    }

    /// Register a u16 scalar to be launched.
    pub fn register_u16(&mut self, scalar: u16) {
        self.register_scalar(Elem::UInt(UIntKind::U16));
        self.scalar_u16.push(scalar);
    }

    /// Register a u32 scalar to be launched.
    pub fn register_u32(&mut self, scalar: u32) {
        self.register_scalar(Elem::UInt(UIntKind::U32));
        self.scalar_u32.push(scalar);
    }

    /// Register a u64 scalar to be launched.
    pub fn register_u64(&mut self, scalar: u64) {
        self.register_scalar(Elem::UInt(UIntKind::U64));
        self.scalar_u64.push(scalar);
    }

    /// Register a i8 scalar to be launched.
    pub fn register_i8(&mut self, scalar: i8) {
        self.register_scalar(Elem::Int(IntKind::I8));
        self.scalar_i8.push(scalar);
    }

    /// Register a i16 scalar to be launched.
    pub fn register_i16(&mut self, scalar: i16) {
        self.register_scalar(Elem::Int(IntKind::I16));
        self.scalar_i16.push(scalar);
    }

    /// Register a i32 scalar to be launched.
    pub fn register_i32(&mut self, scalar: i32) {
        self.register_scalar(Elem::Int(IntKind::I32));
        self.scalar_i32.push(scalar);
    }

    /// Register a i64 scalar to be launched.
    pub fn register_i64(&mut self, scalar: i64) {
        self.register_scalar(Elem::Int(IntKind::I64));
        self.scalar_i64.push(scalar);
    }

    /// Register a bf16 scalar to be launched.
    pub fn register_bf16(&mut self, scalar: half::bf16) {
        self.register_scalar(Elem::Float(FloatKind::BF16));
        self.scalar_bf16.push(scalar);
    }

    /// Register a f16 scalar to be launched.
    pub fn register_f16(&mut self, scalar: half::f16) {
        self.register_scalar(Elem::Float(FloatKind::F16));
        self.scalar_f16.push(scalar);
    }

    /// Register a f32 scalar to be launched.
    pub fn register_f32(&mut self, scalar: f32) {
        self.register_scalar(Elem::Float(FloatKind::F32));
        self.scalar_f32.push(scalar);
    }

    /// Register a f64 scalar to be launched.
    pub fn register_f64(&mut self, scalar: f64) {
        self.register_scalar(Elem::Float(FloatKind::F64));
        self.scalar_f64.push(scalar);
    }

    /// Launch the kernel.
    pub fn launch<K: Kernel>(
        self,
        cube_count: CubeCount,
        kernel: K,
        client: &ComputeClient<R::Server, R::Channel>,
    ) {
        let bindings = self.into_bindings();
        let kernel = Box::new(KernelTask::<R::Compiler, K>::new(kernel));

        client.execute(kernel, cube_count, bindings);
    }

    /// Launch the kernel without check bounds.
    ///
    /// # Safety
    ///
    /// The kernel must not:
    /// - Contain any out of bounds reads or writes. Doing so is immediate UB.
    /// - Contain any loops that never terminate. These may be optimized away entirely or cause
    ///   other unpredictable behaviour.
    pub unsafe fn launch_unchecked<K: Kernel>(
        self,
        cube_count: CubeCount,
        kernel: K,
        client: &ComputeClient<R::Server, R::Channel>,
    ) {
        unsafe {
            let bindings = self.into_bindings();
            let kernel = Box::new(KernelTask::<R::Compiler, K>::new(kernel));

            client.execute_unchecked(kernel, cube_count, bindings);
        }
    }

    /// We need to create the bindings in the same order they are defined in the compilation step.
    ///
    /// The function [crate::KernelIntegrator::integrate] stars by registering the input tensors followed
    /// by the output tensors. Then the tensor metadata, and the scalars at the end. The scalars
    /// are registered in the same order they are added. This is why we store the scalar data type
    /// in the `scalar_order` vector, so that we can register them in the same order.
    ///
    /// Also returns an ordered list of constant bindings. The ordering between constants and tensors
    /// is up to the runtime.
    fn into_bindings(self) -> Bindings {
        let mut bindings = Bindings::new();

        self.tensors.register(&mut bindings);

        self.scalar_u8.register(&mut bindings);
        self.scalar_u16.register(&mut bindings);
        self.scalar_u32.register(&mut bindings);
        self.scalar_u64.register(&mut bindings);
        self.scalar_i8.register(&mut bindings);
        self.scalar_i16.register(&mut bindings);
        self.scalar_i32.register(&mut bindings);
        self.scalar_i64.register(&mut bindings);
        self.scalar_f16.register(&mut bindings);
        self.scalar_bf16.register(&mut bindings);
        self.scalar_f32.register(&mut bindings);
        self.scalar_f64.register(&mut bindings);

        bindings.scalars.sort_by_key(|it| it.elem);

        bindings
    }

    fn register_scalar(&mut self, elem: Elem) {
        if !self.scalar_order.contains(&elem) {
            self.scalar_order.push(elem);
        }
    }
}

/// Handles the tensor state.
pub enum TensorState<R: Runtime> {
    /// No tensor is registered yet.
    Empty,
    /// The registered tensors.
    Some {
        inputs: Vec<Binding>,
        outputs: Vec<Binding>,
        tensor_maps: Vec<TensorMapBinding>,
        metadata: MetadataBuilder,
        runtime: PhantomData<R>,
    },
}

/// Handles the scalar state of an element type
///
/// The scalars are grouped to reduce the number of buffers needed to send data to the compute device.
pub enum ScalarState<T> {
    /// No scalar of that type is registered yet.
    Empty,
    /// The registered scalars.
    Some(Vec<T>),
}

impl<R: Runtime> TensorState<R> {
    fn maybe_init(&mut self) {
        if matches!(self, TensorState::Empty) {
            *self = TensorState::Some {
                inputs: Vec::new(),
                outputs: Vec::new(),
                tensor_maps: Vec::new(),
                metadata: MetadataBuilder::default(),
                runtime: PhantomData,
            };
        }
    }

    fn inputs(&mut self) -> &mut Vec<Binding> {
        self.maybe_init();
        let TensorState::Some { inputs, .. } = self else {
            panic!("Should be init");
        };
        inputs
    }

    fn outputs(&mut self) -> &mut Vec<Binding> {
        self.maybe_init();
        let TensorState::Some { outputs, .. } = self else {
            panic!("Should be init");
        };
        outputs
    }

    fn tensor_maps(&mut self) -> &mut Vec<TensorMapBinding> {
        self.maybe_init();
        let TensorState::Some { tensor_maps, .. } = self else {
            panic!("Should be init");
        };
        tensor_maps
    }

    fn metadata(&mut self) -> &mut MetadataBuilder {
        self.maybe_init();
        let TensorState::Some { metadata, .. } = self else {
            panic!("Should be init");
        };
        metadata
    }

    /// Push a new input tensor to the state.
    pub fn push_input_tensor(&mut self, tensor: &TensorArg<'_, R>) {
        if let Some(tensor) = self.process_tensor(tensor) {
            self.inputs().push(tensor);
        }
    }

    /// Push a new input tensor to the state.
    pub fn push_output_tensor(&mut self, tensor: &TensorArg<'_, R>) {
        if let Some(tensor) = self.process_tensor(tensor) {
            self.outputs().push(tensor);
        }
    }

    fn process_tensor(&mut self, tensor: &TensorArg<'_, R>) -> Option<Binding> {
        let (tensor, vectorization) = match tensor {
            TensorArg::Handle {
                handle,
                vectorization_factor,
                ..
            } => (handle, vectorization_factor),
            TensorArg::Alias { .. } => return None,
        };

        let elem_size = tensor.elem_size * *vectorization as usize;
        let buffer_len = tensor.handle.size() / elem_size as u64;
        let len = tensor.shape.iter().product::<usize>() / *vectorization as usize;
        self.metadata().with_tensor(
            tensor.strides.len() as u32,
            buffer_len as u32,
            len as u32,
            tensor.shape.iter().map(|it| *it as u32).collect(),
            tensor.strides.iter().map(|it| *it as u32).collect(),
        );
        Some(tensor.handle.clone().binding())
    }

    /// Push a new input array to the state.
    pub fn push_input_array(&mut self, array: &ArrayArg<'_, R>) {
        if let Some(tensor) = self.process_array(array) {
            self.inputs().push(tensor);
        }
    }

    /// Push a new input array to the state.
    pub fn push_output_array(&mut self, array: &ArrayArg<'_, R>) {
        if let Some(tensor) = self.process_array(array) {
            self.outputs().push(tensor);
        }
    }

    fn process_array(&mut self, array: &ArrayArg<'_, R>) -> Option<Binding> {
        let (array, vectorization) = match array {
            ArrayArg::Handle {
                handle,
                vectorization_factor,
                ..
            } => (handle, vectorization_factor),
            ArrayArg::Alias { .. } => return None,
        };

        let elem_size = array.elem_size * *vectorization as usize;
        let buffer_len = array.handle.size() / elem_size as u64;
        self.metadata()
            .with_array(buffer_len as u32, array.length[0] as u32);
        Some(array.handle.clone().binding())
    }

    /// Push a new tensor to the state.
    pub fn push_tensor_map(&mut self, map: &TensorMapArg<'_, R>) {
        let binding = self
            .process_tensor(&map.tensor)
            .expect("Can't use alias for TensorMap");

        let map = map.metadata.clone();
        self.tensor_maps().push(TensorMapBinding { binding, map });
    }

    fn register(self, bindings_global: &mut Bindings) {
        if let Self::Some {
            inputs,
            outputs,
            tensor_maps,
            metadata,
            ..
        } = self
        {
            let metadata = metadata.finish();

            bindings_global.inputs = inputs;
            bindings_global.outputs = outputs;
            bindings_global.tensor_maps = tensor_maps;
            bindings_global.metadata = metadata;
        }
    }
}

impl<T: NoUninit + CubePrimitive> ScalarState<T> {
    /// Add a new scalar value to the state.
    pub fn push(&mut self, val: T) {
        match self {
            ScalarState::Empty => *self = Self::Some(vec![val]),
            ScalarState::Some(values) => values.push(val),
        }
    }

    fn register(self, bindings: &mut Bindings) {
        if let ScalarState::Some(mut values) = self {
            // Can't use to_vec because it misaligns the pointer. We want to preserve the alignment
            // of `T`
            let bytes_len = values.len() * size_of::<T>();
            let bytes_capacity = values.capacity() * size_of::<T>();
            let ptr = values.as_mut_ptr() as *mut u8;
            core::mem::forget(values);
            let data = unsafe { Vec::from_raw_parts(ptr, bytes_len, bytes_capacity) };
            let elem = T::as_elem_native_unchecked();
            bindings.scalars.push(ScalarBinding::new(elem, data));
        }
    }
}

impl<R: Runtime> Default for KernelLauncher<R> {
    fn default() -> Self {
        Self {
            tensors: TensorState::Empty,
            scalar_bf16: ScalarState::Empty,
            scalar_f16: ScalarState::Empty,
            scalar_f32: ScalarState::Empty,
            scalar_f64: ScalarState::Empty,
            scalar_u64: ScalarState::Empty,
            scalar_u32: ScalarState::Empty,
            scalar_u16: ScalarState::Empty,
            scalar_u8: ScalarState::Empty,
            scalar_i64: ScalarState::Empty,
            scalar_i32: ScalarState::Empty,
            scalar_i16: ScalarState::Empty,
            scalar_i8: ScalarState::Empty,
            scalar_order: Vec::new(),
            settings: Default::default(),
            runtime: PhantomData,
        }
    }
}
