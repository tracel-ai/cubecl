use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ops::Range;
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

        let scope = Namespace::scoped(self.scope, "");
        for range in self.earlier(scope.as_str()) {
            cubecl_environment::persistence::purge_range(range.start.as_str()..range.end.as_str());
        }
    }

    /// The namespaces under `scope` an earlier layout wrote: every one without the marker, and
    /// every one under a lower number. Only this crate version's, and only earlier: another version
    /// may belong to a cubecl still in use, and a later layout to a build running right now.
    fn earlier(&self, scope: &str) -> Vec<Range<String>> {
        let marked = format!("{scope}{}", self.marker);
        let mut ranges = Vec::from([
            scope.to_string()..marked.clone(),
            past(&marked)..past(scope),
        ]);
        ranges.extend((1..self.current).map(|version| {
            let layout = format!("{marked}{version}/");
            let end = past(&layout);
            layout..end
        }));
        ranges
    }
}

/// The first string past every one starting with `prefix`: its last byte raised, which stays a
/// character since every prefix here is ASCII.
fn past(prefix: &str) -> String {
    let mut bytes = prefix.as_bytes().to_vec();
    *bytes.last_mut().expect("a namespace prefix is never empty") += 1;
    String::from_utf8(bytes).expect("an ASCII byte raised is still ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_this_crate_version_s_earlier_layouts_are_dropped() {
        let probes = NamespaceVersion::new("throughput", "probe-v", 2);
        let earlier = probes.earlier("throughput/0.11.0/");
        let stale = |ns: &str| earlier.iter().any(|range| range.contains(&ns.to_string()));

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
        assert!(!stale("throughput/0.11.0-pre/cuda_dev0"));
    }

    #[test]
    fn a_layout_number_is_dropped_whole_and_alone() {
        let tables = NamespaceVersion::new("autotune", "layout-v", 11);
        let earlier = tables.earlier("autotune/0.11.0/");
        let stale = |ns: &str| earlier.iter().any(|range| range.contains(&ns.to_string()));

        assert!(stale(
            "autotune/0.11.0/layout-v1/cuda_ptx-sm89_l4_par58/matmul"
        ));
        assert!(stale(
            "autotune/0.11.0/layout-v10/cuda_ptx-sm89_l4_par58/matmul"
        ));
        assert!(!stale(
            "autotune/0.11.0/layout-v11/cuda_ptx-sm89_l4_par58/matmul"
        ));
        assert!(!stale(
            "autotune/0.11.0/layout-v12/cuda_ptx-sm89_l4_par58/matmul"
        ));
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
