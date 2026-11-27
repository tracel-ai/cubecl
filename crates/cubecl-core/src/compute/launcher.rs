use alloc::{boxed::Box, vec::Vec};

use crate::frontend::into_kernel_binding;
use crate::prelude::{BufferArg, InputScalar, PeerArg, TensorArg, TensorMapArg, TensorMapKind};
use crate::{InfoBuilder, MetadataOverflow, ScalarArgType};
use alloc::string::{String, ToString};
#[cfg(feature = "std")]
use core::cell::RefCell;
use cubecl_ir::{
    AddressType, ElemType, Scope,
    settings::{KernelSettings, NarrowBindings},
};
use cubecl_runtime::kernel::BufferIOAttr;
use cubecl_runtime::server::{
    ArgumentOverflow, BufferBinding, CubeCount, KernelResource, LaunchError, PeerAccess,
    PeerBinding, TensorMapBinding,
};
use cubecl_runtime::{client::Client, kernel::CubeKernel, server::KernelArguments};

#[cfg(feature = "std")]
std::thread_local! {
    static INFO: RefCell<InfoBuilder> = RefCell::new(InfoBuilder::default());
    // Only used for resolving types
    static SCOPE: RefCell<Scope> = RefCell::new(Scope::dummy());
}

/// Prepare a kernel for [launch](KernelLauncher::launch).
pub struct KernelLauncher {
    resources: Vec<KernelResource>,
    /// What the caller declared each resource is for, indexed like
    /// `resources` — see [`declare_io`](Self::declare_io).
    declared_io: Vec<BufferIOAttr>,
    /// The declaration the next registered resources fall under.
    declaring: BufferIOAttr,
    /// The name of the argument being registered, for the errors registering
    /// it can raise — see [`name_argument`](Self::name_argument).
    argument: &'static str,
    /// The first value registered that the kernel's address type cannot
    /// hold. The launch carries it to the server, which refuses to run it.
    overflow: Option<ArgumentOverflow>,
    address_type: AddressType,
    pub settings: KernelSettings,
    #[cfg(not(feature = "std"))]
    info: InfoBuilder,
    #[cfg(not(feature = "std"))]
    pub scope: Scope,
}

impl KernelLauncher {
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

    /// Register a `usize` or `isize` scalar, read by the kernel as `dtype`, the unsigned or
    /// signed type of its address type.
    ///
    /// A value `dtype` cannot hold is not truncated: a zero stands in for it, so the arguments
    /// keep the layout the kernel expects, and the launch is refused with an
    /// [`ArgumentOverflow`] naming it.
    pub fn register_address_scalar<E: num_traits::ToPrimitive + Copy>(
        &mut self,
        value: E,
        dtype: ElemType,
        registered_as: ElemType,
    ) {
        let scalar = InputScalar::try_new(value, dtype).unwrap_or_else(|| {
            let value = value.to_i128().expect("a `usize` or `isize` is an `i128`");
            self.record_overflow("value".to_string(), value, dtype.to_string());
            InputScalar::new(0, dtype)
        });
        self.register_scalar_raw(scalar.as_bytes(), registered_as);
    }

    /// Name the argument registered from here on, until the next name, in the
    /// errors registering it can raise.
    ///
    /// The generated launch functions call this before each argument with the
    /// name the kernel's signature gives it.
    pub fn name_argument(&mut self, argument: &'static str) {
        self.argument = argument;
    }

    /// Keep the first value the kernel's address type cannot hold.
    fn record_overflow(&mut self, quantity: String, value: i128, ty: String) {
        if self.overflow.is_some() {
            return;
        }
        let argument = match self.argument {
            "" => alloc::format!("#{}", self.resources.len()),
            name => name.to_string(),
        };
        self.overflow = Some(ArgumentOverflow {
            kernel: String::new(),
            argument,
            quantity,
            value,
            ty,
        });
    }

    /// Keep a metadata value the kernel's address type cannot hold.
    fn record_metadata_overflow(&mut self, overflow: MetadataOverflow) {
        let value = i128::try_from(overflow.value).unwrap_or(i128::MAX);
        let ty = self.address_type.unsigned_type().to_string();
        self.record_overflow(overflow.quantity.to_string(), value, ty);
    }

    /// Launch the kernel.
    ///
    /// A value registered that the kernel's address type cannot hold does not
    /// panic here: the launch is refused on the device, and its outputs fail
    /// to read on an [`ArgumentOverflow`] naming the argument and the value.
    #[track_caller]
    pub fn launch<K: CubeKernel>(mut self, cube_count: CubeCount, kernel: K, client: &Client) {
        if let Some(overflow) = &mut self.overflow {
            overflow.kernel = kernel.name().to_string();
        }
        let bindings = self.into_bindings();
        let kernel = Box::new(kernel);

        client.launch(kernel, cube_count, bindings)
    }

    /// Drop a launcher that will never launch, releasing what it registered.
    ///
    /// With `std` a launcher's scalars and metadata accumulate in a
    /// thread-local [`InfoBuilder`] that only building the bindings drains, so
    /// a launcher built to register arguments and then dropped — what the
    /// `create_dummy_kernel` launch variant does — would leave that state
    /// behind for the next real launch on the same thread to pick up as extra
    /// arguments. Discarding drains it instead.
    pub fn discard(self) {
        let _ = self.into_bindings();
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
        bindings.declared_io = self.declared_io;
        bindings.info = info;
        bindings.argument_error = self.overflow.map(LaunchError::from);

        bindings
    }
}

// Tensors/arrays
impl KernelLauncher {
    /// Declare what the kernel does with the buffers registered from here on,
    /// until the next declaration.
    ///
    /// The generated launch functions call this before each argument with
    /// what the signature proves — `&Tensor` cannot be written, `&mut Tensor`
    /// may be read — so a launch that fails before running, a kernel that
    /// does not compile above all, taints only the buffers the kernel could
    /// have written. The compiled kernel's own answer still wins once it
    /// exists; this one is the answer that survives compilation failing. A
    /// launcher that never declares leaves every resource
    /// [`ReadWrite`](BufferIOAttr::ReadWrite), the loud fallback.
    pub fn declare_io(&mut self, io: BufferIOAttr) {
        self.declaring = io;
    }

    /// Declare `io` for the buffers `register` registers, then restore the declaration in force.
    ///
    /// For an argument that knows more about one of its buffers than the signature it arrives
    /// through: an atomic buffer is written through `&`, and the table a tile operand is read
    /// through is never written through `&mut`.
    pub fn declaring_io<T>(
        &mut self,
        io: BufferIOAttr,
        register: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let in_force = core::mem::replace(&mut self.declaring, io);
        let registered = register(self);
        self.declaring = in_force;
        registered
    }

    /// Register what `register` registers, declared written where `atomic`: an atomic buffer is
    /// written through `&`, so the signature proves nothing about it, as with a tensor map.
    pub(crate) fn registering_elem<T>(
        &mut self,
        atomic: bool,
        register: impl FnOnce(&mut Self) -> T,
    ) -> T {
        match atomic {
            true => self.declaring_io(BufferIOAttr::ReadWrite, register),
            false => register(self),
        }
    }

    /// An aliasing argument writes the buffer it aliases in place, however
    /// that buffer's own argument was declared — the aliased buffer usually
    /// arrives through a `&Tensor`, and it is the one buffer an in-place
    /// kernel exists to produce. The alias registers no resource of its own,
    /// so its declaration lands on the buffer at `input_pos` instead: a
    /// declaration built from each signature position alone would call that
    /// buffer read-only and leave the in-place output unnamed by a failure,
    /// which is silent garbage on a read.
    fn alias_io(&mut self, input_pos: usize) {
        if self.declaring.is_writable()
            && let Some(io) = self.declared_io.get_mut(input_pos)
        {
            *io = BufferIOAttr::ReadWrite;
        }
    }

    /// Marks `resource`, about to take the next binding position, in the settings'
    /// [narrow bindings](NarrowBindings) when it is a buffer that [`is_narrow`].
    fn mark_narrow(&mut self, resource: &KernelResource) {
        let KernelResource::Buffer(binding) = resource else {
            return;
        };
        let device_marks = self.with_scope(|scope| {
            scope
                .state()
                .device_properties
                .as_ref()
                .is_some_and(|properties| properties.features.narrow_bindings)
        });
        if is_narrow(device_marks, self.address_type, binding.size_in_used()) {
            self.settings
                .narrow_bindings
                .insert_position(self.resources.len());
        }
    }

    /// Record a resource.
    fn push_resource(&mut self, resource: KernelResource) {
        self.mark_narrow(&resource);
        let io = match &resource {
            // A tensor map's global side is written through TMA operations no
            // signature shows — a map registered from a `&TensorMap` can
            // still be a store's destination — so the declaration is clamped
            // to the same answer the visibility analysis gives it.
            KernelResource::TensorMap(_) => BufferIOAttr::ReadWrite,
            KernelResource::Buffer(_) | KernelResource::Peer(_) => self.declaring,
        };
        self.declared_io.push(io);
        self.resources.push(resource);
    }

    /// Push a new input tensor to the state.
    pub fn register_tensor(&mut self, tensor: TensorArg, elem_size: usize) {
        if let Some(tensor) = self.process_tensor(tensor, elem_size) {
            self.push_resource(KernelResource::Buffer(tensor));
        }
    }

    fn process_tensor(&mut self, tensor: TensorArg, elem_size: usize) -> Option<BufferBinding> {
        let tensor = match tensor {
            TensorArg::Handle { handle, .. } => handle,
            TensorArg::Alias { input_pos, .. } => {
                self.alias_io(input_pos);
                return None;
            }
        };

        let buffer_len = tensor.handle.size_in_used() / elem_size as u64;
        let address_type = self.address_type;

        let registered = self.with_info(|info| {
            info.metadata.register_tensor(
                buffer_len,
                tensor.shape.clone(),
                tensor.strides.clone(),
                address_type,
            )
        });
        if let Err(overflow) = registered {
            self.record_metadata_overflow(overflow);
        }
        Some(tensor.handle)
    }

    /// Push a new input array to the state.
    ///
    /// `elem_size` is the size in bytes of the kernel's element, a whole vector of `vector_size`
    /// scalars for a vectorized buffer.
    pub fn register_buffer(&mut self, array: BufferArg, elem_size: usize, vector_size: usize) {
        if let Some(tensor) = self.process_buffer(array, elem_size, vector_size) {
            self.push_resource(KernelResource::Buffer(tensor));
        }
    }

    /// The kernel sees a buffer's declared length, counted in its own elements: a vectorized
    /// buffer of `length` scalars holds `length / vector_size` whole vectors, and a trailing
    /// partial vector is out of its reach, as it is on a vectorized tensor. An alias registers no
    /// length of its own, the kernel reads the one of the buffer it aliases.
    fn process_buffer(
        &mut self,
        array: BufferArg,
        elem_size: usize,
        vector_size: usize,
    ) -> Option<BufferBinding> {
        let (handle, buffer_len) = match array {
            BufferArg::Handle { handle, .. } => into_kernel_binding(handle, elem_size, vector_size),
            BufferArg::Alias { input_pos, .. } => {
                self.alias_io(input_pos);
                return None;
            }
        };

        let address_type = self.address_type;
        let registered =
            self.with_info(|info| info.metadata.register_buffer(buffer_len, address_type));
        if let Err(overflow) = registered {
            self.record_metadata_overflow(overflow);
        }
        Some(handle)
    }

    /// Push a buffer in memory shared with peers to the state. It is bound as a peer binding
    /// whichever device it lives on, which is what tells the client the kernel declared it
    /// shared; on its own device, the client binds it as an ordinary buffer.
    pub fn register_peer(
        &mut self,
        buffer: PeerArg,
        elem_size: usize,
        vector_size: usize,
        required_access: PeerAccess,
    ) {
        let binding = self
            .process_buffer(
                BufferArg::Handle {
                    handle: buffer.handle,
                },
                elem_size,
                vector_size,
            )
            .expect("a handle is no alias");
        let peer = PeerBinding::new(binding, required_access)
            .expect("a peer argument is built from a handle shared with its peers");
        self.push_resource(KernelResource::Peer(peer));
    }

    /// Push a new tensor to the state.
    pub fn register_tensor_map<K: TensorMapKind>(
        &mut self,
        map: TensorMapArg<K>,
        elem_size: usize,
    ) {
        let binding = self
            .process_tensor(map.tensor, elem_size)
            .expect("Can't use alias for TensorMap");

        let map = map.metadata.clone();
        self.push_resource(KernelResource::TensorMap(TensorMapBinding { binding, map }));
    }
}

impl KernelLauncher {
    pub fn new(settings: KernelSettings) -> Self {
        Self {
            address_type: settings.address_type,
            settings,
            resources: Vec::new(),
            declared_io: Vec::new(),
            declaring: BufferIOAttr::ReadWrite,
            argument: "",
            overflow: None,
            #[cfg(not(feature = "std"))]
            info: InfoBuilder::default(),
            #[cfg(not(feature = "std"))]
            scope: Scope::dummy(),
        }
    }
}

/// Whether a buffer of `size_bytes` is a [narrow binding](NarrowBindings): the device addresses
/// narrow bindings (`device_marks`), the kernel computes its addresses in 32 bits and every byte
/// offset into the buffer fits 32 bits.
fn is_narrow(device_marks: bool, address_type: AddressType, size_bytes: u64) -> bool {
    device_marks && address_type == AddressType::U32 && NarrowBindings::fits_size_bytes(size_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_ir::settings::{Dim3, ExecutionMode};

    #[test]
    fn a_small_buffer_on_a_device_that_marks_is_narrow() {
        assert!(is_narrow(true, AddressType::U32, NarrowBindings::MAX_BYTES));
    }

    #[test]
    fn a_device_that_does_not_mark_has_no_narrow_binding() {
        assert!(!is_narrow(false, AddressType::U32, 4));
    }

    #[test]
    fn a_64_bit_kernel_has_no_narrow_binding() {
        assert!(!is_narrow(true, AddressType::U64, 4));
    }

    #[test]
    fn a_buffer_past_four_gib_is_not_narrow() {
        assert!(!is_narrow(
            true,
            AddressType::U32,
            NarrowBindings::MAX_BYTES + 1
        ));
    }

    fn settings() -> KernelSettings {
        KernelSettings::new(Dim3::new_single(), ExecutionMode::Checked, AddressType::U32)
    }

    fn info_of(launcher: KernelLauncher) -> Vec<u64> {
        launcher.into_bindings().info.data
    }

    /// `create_dummy_kernel` registers arguments into a launcher it never
    /// launches. With `std` those registrations land in a thread-local that
    /// only building the bindings drains, so the launcher has to be discarded
    /// rather than dropped — otherwise the next real launch on the same
    /// thread inherits them as extra arguments.
    #[test]
    fn a_discarded_launcher_leaves_nothing_for_the_next_launch() {
        let empty = info_of(KernelLauncher::new(settings()));

        // A registered scalar is visible in the info a launcher produces, so
        // the equality below is a real claim about the thread-local, not a
        // comparison of two things that could never differ.
        let mut registered = KernelLauncher::new(settings());
        registered.register_scalar(1u32);
        assert_ne!(info_of(registered), empty);

        let mut dummy = KernelLauncher::new(settings());
        dummy.register_scalar(1u32);
        dummy.discard();

        assert_eq!(
            info_of(KernelLauncher::new(settings())),
            empty,
            "a discarded launcher left its scalars behind for the next launch"
        );
    }
}
