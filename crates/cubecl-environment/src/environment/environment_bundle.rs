//! An environment serialized whole.

use crate::bytes::Bytes;

/// An environment serialized whole: what [`EnvironmentId::save`](super::EnvironmentId::save) answers and
/// an [`EnvironmentOrigin::Bundle`](super::EnvironmentOrigin::Bundle) is created from. The bytes are a
/// [`BundleFormat::Sqlite`](crate::bundle::BundleFormat) bundle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentBundle {
    /// The serialized environment.
    pub bytes: Bytes,
}
