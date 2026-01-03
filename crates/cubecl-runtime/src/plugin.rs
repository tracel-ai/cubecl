#![allow(missing_docs)]
//! This is a set of traits which need to be implemented when adding a `Plugin` to a backend.
//! The `ComputeClient` got two additional functions to execute `Plugin`'s.
//! One for type initialisation and one for passing a function to the `ComputeServer`.
//! These traits need to be implemented carefully and with synchronization and cleanup in mind,
//! especially when considering to handle a `Plugin` over a continuous runtime.

use cubecl_common::stream_id::StreamId;
use crate::server::ComputeServer;

/// This trait defines all types and the function trait for a `Plugin`.
pub trait Plugin: Send + Sync + 'static {
    /// This is a structure of some kind
    /// representing all data the client layer needs to provide
    /// for executing functions of the `Plugin` on a `ComputeServer`.
    type ClientHandle: Send;
    /// This is a type which needs to be build by a `ComputeServer`.
    type ServerHandle: Send;
    /// A `Plugin` wants to add additional functions to a `ComputeServer`.
    /// Since we know how the `ServerHandle` will look the `Plugin` will
    /// be able to define additional functions which can be executed over the
    /// `ServerHandle` by a `ComputeServer`
    type Fns: FnOnce(Self::ServerHandle) -> Result<Self::ReturnVal, PluginError>;
    /// A `Plugin` might need an additional type which needs to be
    /// initialised after the `ComputeServer` is loaded.
    /// This type can be used to define initialisation parameters.
    type InitType: PluginType;
    /// For the case the `Plugin` needs a return type.
    type ReturnVal;
    /// Extension name for complex runtime operations.
    const EXTENSION_NAME: &'static str;
}

/// Type we want for initialisation.
pub trait PluginType {
    type Insert: Send + Sync;

    fn init(self) -> Self::Insert;
}

/// These are the functions used by the `ComputeClient` implemented over `ComputeServer`.
pub trait SupportsPlugin<SP: Plugin>: ComputeServer {
    /// The `ComputeServer` should be able to initialise the `Plugin`.
    fn init_type(&mut self, plugin_type: SP::InitType, stream: StreamId) -> Result<(), PluginError>;

    /// And should be able to take the `Plugin`'s fns plus `ClientHandle` and
    /// `StreamId` to build the `ServerHandle` and execute the function.
    fn plugin_fn<F>(
        &mut self,
        client_handle: SP::ClientHandle,
        stream_id: StreamId,
        op: SP::Fns,
    ) -> Result<SP::ReturnVal, PluginError>;
}

#[derive(Debug)]
pub enum PluginError {
    NotSupported(&'static str),
    NotInitialized(String),
    ExecutionFailed(String),
    InvalidHandle(String),
}
