pub use cubecl_server::device::AmdDevice;

// It is not clear if AMD has a limit on the number of bindings it can hold at
// any given time, but it's highly unlikely that it's more than this. We can
// also assume that we'll never have more than this many bindings in flight,
// so it's 'safe' to store only this many bindings.
pub const AMD_MAX_BINDINGS: u32 = 1024;
