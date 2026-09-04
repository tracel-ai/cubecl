mod server;

pub(crate) mod capture;
pub(crate) mod context;
pub(crate) mod driver;
pub(crate) mod events;
pub(crate) mod storage;
pub(crate) mod stream;

pub use server::*;
pub use storage::*;

/// One unit of work against the device — the shared
/// [`Command`](cubecl_server::command::Command), driven by [`driver::Hip`].
pub(crate) type Command<'a> = cubecl_server::command::Command<'a, driver::Hip>;

/// The graphs this device has captured, driven by [`driver::Hip`].
pub(crate) type Captures = cubecl_server::command::Captures<driver::Hip>;

/// A capture window on one stream, driven by [`driver::Hip`].
pub(crate) type Window<'a> = cubecl_server::command::Window<'a, driver::Hip>;
