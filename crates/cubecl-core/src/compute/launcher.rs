use std::marker::PhantomData;

use crate::KernelSettings;
use crate::prelude::{ArrayArg, TensorArg, TensorMapArg};
use crate::{Kernel, Runtime};
use crate::{
    MetadataBuilder,
    ir::{Elem, FloatKind, IntKind},
};
use crate::{compute::KernelTask, ir::UIntKind};
use bytemuck::NoUninit;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::{Binding, ConstBinding, CubeCount};

/// Prepare a kernel for [launch](KernelLauncher::launch).
pub struct KernelLauncher<R: Runtime> {
    tensors: TensorState<R>,
    constants: ConstantState<R>,
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
    /// Register a tensor to be launched.
    pub fn register_tensor(&mut self, tensor: &TensorArg<'_, R>) {
        self.tensors.push_tensor(tensor);
    }

    /// Register a mapped tensor to be launched.
    pub fn register_tensor_map(&mut self, tensor: &TensorMapArg<'_, R>) {
        self.constants.push_tensor_map(tensor);
    }

    /// Register an array to be launched.
    pub fn register_array(&mut self, array: &ArrayArg<'_, R>) {
        self.tensors.push_array(array);
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
        let (constants, bindings) = self.into_bindings(client);

        let kernel = Box::new(KernelTask::<R::Compiler, K>::new(kernel));

        client.execute(kernel, cube_count, constants, bindings);
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
            let (constants, bindings) = self.into_bindings(client);

            let kernel = Box::new(KernelTask::<R::Compiler, K>::new(kernel));

            client.execute_unchecked(kernel, cube_count, constants, bindings);
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
    fn into_bindings(
        mut self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> (Vec<ConstBinding>, Vec<Binding>) {
        let constants = self.constants.bindings();
        let mut bindings = Vec::new();

        self.tensors.register(client, &mut bindings);

        for elem in self.scalar_order.drain(..) {
            match elem {
                Elem::Float(kind) | Elem::AtomicFloat(kind) => match kind {
                    FloatKind::F16 => self.scalar_f16.register::<R>(client, &mut bindings),
                    FloatKind::BF16 => self.scalar_bf16.register::<R>(client, &mut bindings),
                    FloatKind::TF32 => self.scalar_f32.register::<R>(client, &mut bindings),
                    FloatKind::Flex32 => self.scalar_f32.register::<R>(client, &mut bindings),
                    FloatKind::F32 => self.scalar_f32.register::<R>(client, &mut bindings),
                    FloatKind::F64 => self.scalar_f64.register::<R>(client, &mut bindings),
                },
                Elem::Int(kind) => match kind {
                    IntKind::I8 => self.scalar_i8.register::<R>(client, &mut bindings),
                    IntKind::I16 => self.scalar_i16.register::<R>(client, &mut bindings),
                    IntKind::I32 => self.scalar_i32.register::<R>(client, &mut bindings),
                    IntKind::I64 => self.scalar_i64.register::<R>(client, &mut bindings),
                },
                Elem::AtomicInt(kind) => match kind {
                    IntKind::I8 => self.scalar_i8.register::<R>(client, &mut bindings),
                    IntKind::I16 => self.scalar_i16.register::<R>(client, &mut bindings),
                    IntKind::I32 => self.scalar_i32.register::<R>(client, &mut bindings),
                    IntKind::I64 => self.scalar_i64.register::<R>(client, &mut bindings),
                },
                Elem::UInt(kind) | Elem::AtomicUInt(kind) => match kind {
                    UIntKind::U8 => self.scalar_u8.register::<R>(client, &mut bindings),
                    UIntKind::U16 => self.scalar_u16.register::<R>(client, &mut bindings),
                    UIntKind::U32 => self.scalar_u32.register::<R>(client, &mut bindings),
                    UIntKind::U64 => self.scalar_u64.register::<R>(client, &mut bindings),
                },
                Elem::Bool => panic!("Bool can't be passed as bindings."),
            }
        }

        (constants, bindings)
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
        bindings: Vec<Binding>,
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

pub struct ConstantState<R: Runtime> {
    bindings: Vec<ConstBinding>,
    _ty: PhantomData<R>,
}

impl<R: Runtime> Default for ConstantState<R> {
    fn default() -> Self {
        Self {
            bindings: Default::default(),
            _ty: PhantomData,
        }
    }
}

impl<R: Runtime> ConstantState<R> {
    /// Push a new tensor to the state.
    pub fn push_tensor_map(&mut self, map: &TensorMapArg<'_, R>) {
        let tensor = match &map.tensor {
            TensorArg::Handle { handle, .. } => handle,
            TensorArg::Alias { .. } => panic!("Can't use aliased tensor for tensor map"),
        };

        let binding = tensor.handle.clone().binding();
        let map = map.metadata.clone();
        self.bindings.push(ConstBinding::TensorMap { binding, map });
    }

    fn bindings(&self) -> Vec<ConstBinding> {
        self.bindings.clone()
    }
}

impl<R: Runtime> TensorState<R> {
    /// Push a new tensor to the state.
    pub fn push_tensor(&mut self, tensor: &TensorArg<'_, R>) {
        let (tensor, vectorization) = match tensor {
            TensorArg::Handle {
                handle,
                vectorization_factor,
                ..
            } => (handle, vectorization_factor),
            TensorArg::Alias { .. } => return,
        };

        if let TensorState::Empty = self {
            *self = TensorState::Some {
                bindings: Vec::with_capacity(1),
                metadata: MetadataBuilder::default(),
                runtime: PhantomData,
            };
        };

        let TensorState::Some {
            bindings, metadata, ..
        } = self
        else {
            panic!("Should be init")
        };

        let elem_size = tensor.elem_size * *vectorization as usize;
        let buffer_len = tensor.handle.size() / elem_size as u64;
        let len = tensor.shape.iter().product::<usize>() / *vectorization as usize;
        bindings.push(tensor.handle.clone().binding());
        metadata.with_tensor(
            tensor.strides.len() as u32,
            buffer_len as u32,
            len as u32,
            tensor.shape.iter().map(|it| *it as u32).collect(),
            tensor.strides.iter().map(|it| *it as u32).collect(),
        );
    }

    /// Push a new array to the state.
    pub fn push_array(&mut self, array: &ArrayArg<'_, R>) {
        let (array, vectorization) = match array {
            ArrayArg::Handle {
                handle,
                vectorization_factor,
                ..
            } => (handle, vectorization_factor),
            ArrayArg::Alias { .. } => return,
        };

        if let TensorState::Empty = self {
            *self = TensorState::Some {
                bindings: Vec::with_capacity(1),
                metadata: MetadataBuilder::default(),
                runtime: PhantomData,
            };
        };

        let TensorState::Some {
            bindings, metadata, ..
        } = self
        else {
            panic!("Should be init")
        };

        let elem_size = array.elem_size * *vectorization as usize;
        let buffer_len = array.handle.size() / elem_size as u64;
        bindings.push(array.handle.clone().binding());
        metadata.with_array(buffer_len as u32, array.length[0] as u32);
    }

    fn register(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
        bindings_global: &mut Vec<Binding>,
    ) {
        if let Self::Some {
            bindings,
            metadata,
            runtime: _,
        } = self
        {
            let metadata = metadata.finish();

            bindings_global.extend(bindings);
            bindings_global.push(client.create(bytemuck::cast_slice(&metadata)).binding());
        } else {
            bindings_global.push(client.create(&[0]).binding());
        }
    }
}

impl<T: NoUninit> ScalarState<T> {
    /// Add a new scalar value to the state.
    pub fn push(&mut self, val: T) {
        match self {
            ScalarState::Empty => *self = Self::Some(vec![val]),
            ScalarState::Some(values) => values.push(val),
        }
    }

    fn register<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        bindings: &mut Vec<Binding>,
    ) {
        match self {
            ScalarState::Empty => (),
            ScalarState::Some(values) => {
                let handle = client.create(bytemuck::cast_slice(values));
                bindings.push(handle.binding());
            }
        }
    }
}

impl<R: Runtime> Default for KernelLauncher<R> {
    fn default() -> Self {
        Self {
            tensors: TensorState::Empty,
            constants: ConstantState::default(),
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
