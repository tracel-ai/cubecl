//! What an environment holds, counted.

use crate::persistence::{NamespaceContent, NamespaceRoot, NamespaceSummary};

/// What an environment holds, counted by what its entries are: read off the
/// namespaces alone, without decoding an entry, so it costs one query however
/// large the environment is.
///
/// A value, like [`Pruned`](super::Pruned): it crosses the network unchanged,
/// so a remote environment is counted where it lives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentSummary {
    /// The compiled kernels, one per artifact a compiler stored. The entries
    /// that map a source to a kernel name one and are not counted.
    pub kernels: u64,
    /// The autotune answers, one per key a tuner stored the fastest candidate
    /// for.
    pub autotune_answers: u64,
    /// The build records, each one entry.
    pub records: u64,
}

impl EnvironmentSummary {
    /// Counts `namespaces`, each by its [root](NamespaceRoot). An
    /// application's own namespaces, the throughput measurements and the
    /// work's launches count as none of the three.
    pub fn new(namespaces: &[NamespaceSummary]) -> Self {
        let mut summary = Self::default();
        for namespace in namespaces {
            let count = match NamespaceRoot::new(&namespace.namespace).content() {
                NamespaceContent::Kernels => &mut summary.kernels,
                NamespaceContent::AutotuneAnswers => &mut summary.autotune_answers,
                NamespaceContent::Records => &mut summary.records,
                NamespaceContent::KernelSources | NamespaceContent::Other => continue,
            };
            *count += namespace.entries;
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::ArtifactRoot;
    use alloc::vec::Vec;

    /// Each entry counts once, as what its namespace's root says it is: the
    /// artifacts of every compiler as kernels, their source maps as nothing,
    /// and an application's namespace, even one named like cubecl's, as
    /// nothing either.
    #[test]
    fn entries_count_by_their_namespace_root() {
        let mut namespaces = Vec::new();
        for root in ArtifactRoot::ALL {
            namespaces.push(namespace(&[root.as_str(), "0.12.0/fingerprint"], 2));
        }
        namespaces.extend([
            namespace(&["autotune", "0.12.0/layout-v1/hardware/matmul"], 5),
            namespace(&["autotune", "0.12.0/layout-v1/hardware/reduce"], 1),
            namespace(&["records", "v1/sessions"], 3),
            namespace(&["records", "v1/spans"], 4),
            namespace(&["launched", "0.12.0/00ff"], 11),
            namespace(&["throughput", "0.12.0/probe-v2"], 13),
            namespace(&["my-application", "0.1.0/settings"], 17),
            namespace(&["autotune-notes", "0.1.0"], 19),
        ]);

        let summary = EnvironmentSummary::new(&namespaces);

        let sources = 2;
        let kernels = 2 * ArtifactRoot::ALL.len() as u64 - 2 * sources;
        assert_eq!(
            summary,
            EnvironmentSummary {
                kernels,
                autotune_answers: 6,
                records: 7,
            }
        );
    }

    fn namespace(segments: &[&str], entries: u64) -> NamespaceSummary {
        NamespaceSummary {
            namespace: segments.join("/"),
            entries,
            bytes: entries * 10,
        }
    }
}
