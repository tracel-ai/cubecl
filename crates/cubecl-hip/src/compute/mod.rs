mod server;

pub(crate) mod capture;
pub(crate) mod context;
pub(crate) mod driver;
pub(crate) mod fence;
pub(crate) mod io;
pub(crate) mod storage;
pub(crate) mod stream;

pub use server::*;
pub use storage::*;

/// One unit of work against the device — the shared
/// [`Command`](cubecl_runtime::command::Command), driven by [`driver::Hip`].
pub(crate) type Command<'a> = cubecl_runtime::command::Command<'a, driver::Hip>;
