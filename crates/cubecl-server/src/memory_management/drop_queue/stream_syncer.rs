use crate::server::ServerError;

/// A fence on one stream, which syncs with every command submitted to that
/// stream before the fence was recorded.
pub trait StreamSyncer: Sized {
    /// Block the host until the fence's stream reaches the fence.
    ///
    /// # Guarantees
    ///
    /// `Ok` only once every command submitted to the fence's stream before it
    /// was recorded completed. Other streams' work is not synced with.
    ///
    /// # Errors
    ///
    /// The fault the sync reveals, when the stream itself failed.
    fn sync_host(self) -> Result<(), ServerError>;
}
