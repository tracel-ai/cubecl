use alloc::format;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, Ordering};
use cubecl_environment::persistence::Namespace;

/// The version a scope's namespaces carry, as a numbered layout of what is stored under it.
///
/// The crate version in a namespace does not move when a layout changes inside a release, so the
/// layout carries a number of its own, and namespaces under an earlier one are dropped.
pub(crate) struct NamespaceVersion {
    scope: &'static str,
    marker: &'static str,
    current: u32,
    dropped: AtomicBool,
}

impl NamespaceVersion {
    pub(crate) const fn new(scope: &'static str, marker: &'static str, current: u32) -> Self {
        Self {
            scope,
            marker,
            current,
            dropped: AtomicBool::new(false),
        }
    }

    /// The namespace `path` is stored under in the current version.
    pub(crate) fn namespace(&self, path: &str) -> Namespace {
        Namespace::scoped(
            self.scope,
            format!("{}{}/{path}", self.marker, self.current),
        )
    }

    /// Purge, once per process, the namespaces this crate version wrote under an earlier layout.
    pub(crate) fn drop_earlier(&self) {
        if self.dropped.swap(true, Ordering::Relaxed) {
            return;
        }

        let scope = String::from(Namespace::scoped(self.scope, "").as_str());
        for candidate in cubecl_environment::persistence::namespaces() {
            if self.is_earlier(&candidate, &scope) {
                cubecl_environment::persistence::open(&candidate).purge();
            }
        }
    }

    /// Only this crate version's, and only earlier: another version may belong to a cubecl still
    /// in use, and a later layout to a build running right now. A namespace with no layout number
    /// at all predates them.
    fn is_earlier(&self, candidate: &str, scope: &str) -> bool {
        let Some(path) = candidate.strip_prefix(scope) else {
            return false;
        };

        let version = path
            .strip_prefix(self.marker)
            .and_then(|path| path.split('/').next())
            .and_then(|version| version.parse::<u32>().ok())
            .unwrap_or(0);

        version < self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn only_this_crate_version_s_earlier_layouts_are_dropped() {
        let probes = NamespaceVersion::new("throughput", "probe-v", 2);
        let stale = |ns: &str| probes.is_earlier(ns, "throughput/0.11.0/");

        assert!(stale(
            "throughput/0.11.0/probe-v1/cuda_ptx-sm75_rtx2060_par30"
        ));
        assert!(stale("throughput/0.11.0/cuda_dev0"));

        assert!(!stale(
            "throughput/0.11.0/probe-v2/cuda_ptx-sm75_rtx2060_par30"
        ));
        assert!(!stale(
            "throughput/0.11.0/probe-v3/cuda_ptx-sm75_rtx2060_par30"
        ));
        assert!(!stale(
            "throughput/0.10.0/probe-v1/cuda_ptx-sm75_rtx2060_par30"
        ));
        assert!(!stale("autotune/0.11.0/device-0-0-cpu/matmul"));
    }

    #[test]
    fn the_namespace_carries_its_version() {
        let probes = NamespaceVersion::new("throughput", "probe-v", 2);
        let scope = Namespace::scoped("throughput", "").as_str().to_string();

        assert_eq!(
            probes.namespace("cuda_ptx-sm75_rtx2060_par30").as_str(),
            format!("{scope}probe-v2/cuda_ptx-sm75_rtx2060_par30")
        );
    }
}
