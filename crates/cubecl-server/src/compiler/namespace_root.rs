//! Who wrote a namespace, read off its first segment.

/// The first segment of a namespace: who wrote it — `autotune`, a backend's
/// compiler, the records, an application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamespaceRoot<'a>(&'a str);

impl<'a> NamespaceRoot<'a> {
    /// The root of `namespace`.
    pub fn new(namespace: &'a str) -> Self {
        Self(namespace.split('/').next().unwrap_or_default())
    }

    /// The root as written.
    pub fn as_str(self) -> &'a str {
        self.0
    }

    /// Whether the namespace is one a compiler keeps its artifacts under,
    /// as [`compilation_store`](super::compilation_store) opens them. Only
    /// those: an application keeps its own data in the same environment, and
    /// nothing that reads artifacts may mistake an entry of it for one.
    pub fn holds_artifacts(self) -> bool {
        ArtifactRoot::ALL.iter().any(|root| root.as_str() == self.0)
    }
}

/// The root of each namespace a compiler keeps artifacts under: the one list
/// [`compilation_store`](super::compilation_store) opens a store from and
/// [`NamespaceRoot::holds_artifacts`] tells those namespaces apart by. A
/// backend that keeps artifacts adds its root here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactRoot {
    /// The CPU runtime's object code.
    Cpu,
    /// CUDA's artifacts, by kernel.
    Cuda,
    /// CUDA's map from a source to the kernel that first finalized it.
    CudaSources,
    /// HIP's artifacts, by kernel.
    Hip,
    /// HIP's map from a source to the kernel that first finalized it.
    HipSources,
    /// Metal's shaders.
    Metal,
    /// The SPIR-V wgpu compiles for Vulkan.
    Vulkan,
}

impl ArtifactRoot {
    /// Every root, each once.
    pub const ALL: [Self; 7] = [
        Self::Cpu,
        Self::Cuda,
        Self::CudaSources,
        Self::Hip,
        Self::HipSources,
        Self::Metal,
        Self::Vulkan,
    ];

    /// The root as stored: the first segment of the namespace.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            // Stored under this spelling since before the sources were named:
            // renaming it would orphan every map already written.
            Self::CudaSources => "cuda-second-line",
            Self::Hip => "hip",
            Self::HipSources => "hip-second-line",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only a compiler's namespace holds artifacts: the measurements, the
    /// records, the work's launches and an application's data never do.
    #[test]
    fn only_a_compilers_namespace_holds_artifacts() {
        for root in ArtifactRoot::ALL {
            let namespace = alloc::format!("{}/0.12.0/fingerprint", root.as_str());
            assert!(NamespaceRoot::new(&namespace).holds_artifacts(), "{root:?}");
        }
        for namespace in [
            "autotune/0.12.0/matmul",
            "throughput/0.12.0/probe",
            "records/v1/sessions",
            "launched/0.12.0/00ff",
            "my-application/0.1.0/settings",
        ] {
            assert!(
                !NamespaceRoot::new(namespace).holds_artifacts(),
                "{namespace}"
            );
        }
    }
}
