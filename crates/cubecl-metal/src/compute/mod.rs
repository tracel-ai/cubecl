pub(crate) mod artifact;
pub(crate) mod capture;
pub mod context;
pub(crate) mod driver;
pub(crate) mod pipelines;
pub mod server;
pub mod stream;

pub use context::MetalContext;
pub use pipelines::MetalCompiledKernel;
pub use server::MetalServer;
pub use stream::{MetalEvent, MetalStream, MetalStreamBackend};

/// One unit of work against the device — the shared
/// [`Command`](cubecl_server::command::Command), driven by [`driver::Metal`].
pub(crate) type Command<'a> = cubecl_server::command::Command<'a, driver::Metal>;

/// The graphs this device has captured, driven by [`driver::Metal`].
pub(crate) type Captures = cubecl_server::command::Captures<driver::Metal>;

/// A capture window on one stream, driven by [`driver::Metal`].
pub(crate) type Window<'c, 'a> = cubecl_server::command::Window<'c, 'a, driver::Metal>;
