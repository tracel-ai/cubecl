//! Who wrote a namespace, read off its first segment.

use super::WorkLaunchStore;

/// The first segment of a namespace: who wrote it — `autotune`, a backend's
/// compiler, the records, an application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamespaceRoot<'a>(&'a str);

impl<'a> NamespaceRoot<'a> {
    /// The root of `namespace`.
    pub fn new(namespace: &'a str) -> Self {
        Self(namespace.split('/').next().unwrap_or_default())
    }

    /// The root of the records, which every layout of them shares.
    pub fn records() -> Self {
        Self::new(cubecl_environment::records::ROOT)
    }

    /// The root as written.
    pub fn as_str(self) -> &'a str {
        self.0
    }

    /// Whether the namespace may hold compiled artifacts: everything but the
    /// measurements, the records and the work's launches cubecl keeps beside
    /// them.
    pub fn holds_artifacts(self) -> bool {
        ![
            crate::tune::ROOT,
            crate::throughput::ROOT,
            Self::records().0,
            WorkLaunchStore::ROOT,
        ]
        .contains(&self.0)
    }
}
