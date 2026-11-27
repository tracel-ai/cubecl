//! What names an autotune table.

use alloc::string::{String, ToString};

use super::{Namespace, NamespaceRoot};

/// What the segment numbering an autotune table layout starts with, as in
/// `layout-v1`.
pub const LAYOUT_PREFIX: &str = "layout-v";

/// The layout autotune tables are named by. Bumped when they are named another
/// way.
const LAYOUT: u32 = 1;

/// The cubecl version whose tuners read the tables it names.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What names the table one tuner of one cubecl version stores its answers in
/// on one hardware.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TuneTableName {
    /// The cubecl version whose tuners read it.
    pub version: String,
    /// The hardware the answers were measured on, as its `HardwareIdentity`
    /// displays.
    pub hardware: String,
    /// The tuner, named by the code that declared it.
    pub tuner: String,
}

impl TuneTableName {
    /// The table `tuner` stores its answers in on `hardware` in this cubecl
    /// version.
    pub fn current(hardware: impl Into<String>, tuner: impl Into<String>) -> Self {
        Self {
            version: VERSION.to_string(),
            hardware: hardware.into(),
            tuner: tuner.into(),
        }
    }

    /// The name `namespace` was written from, whichever cubecl version wrote
    /// it, or `None` when it is no autotune table of the current layout.
    pub fn parse(namespace: &str) -> Option<Self> {
        let path = namespace
            .strip_prefix(NamespaceRoot::AUTOTUNE)?
            .strip_prefix('/')?;
        let (version, path) = path.split_once('/')?;
        let layout = alloc::format!("{LAYOUT_PREFIX}{LAYOUT}/");
        let (hardware, tuner) = path.strip_prefix(layout.as_str())?.split_once('/')?;
        Some(Self {
            version: version.to_string(),
            hardware: hardware.to_string(),
            tuner: tuner.to_string(),
        })
    }

    /// `autotune/<version>/layout-v<layout>/<hardware>/<tuner>`.
    pub fn namespace(&self) -> Namespace {
        let Self {
            version,
            hardware,
            tuner,
        } = self;
        let root = NamespaceRoot::AUTOTUNE;
        Namespace::from(alloc::format!(
            "{root}/{version}/{LAYOUT_PREFIX}{LAYOUT}/{hardware}/{tuner}"
        ))
    }

    /// Whether the tuners of this cubecl version read it.
    pub fn is_current(&self) -> bool {
        self.version == VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_reads_back_from_its_namespace() {
        let name = TuneTableName {
            version: "0.1.0".into(),
            ..TuneTableName::current("cuda_ptx-sm89_NVIDIA-L4_par58", "matmul-tune-gemm")
        };
        assert_eq!(TuneTableName::parse(name.namespace().as_str()), Some(name));
    }

    #[test]
    fn a_current_name_is_where_this_version_stores() {
        let name = TuneTableName::current("hardware", "tuner");
        assert_eq!(
            name.namespace(),
            Namespace::scoped(NamespaceRoot::AUTOTUNE, "layout-v1/hardware/tuner")
        );
    }

    #[test]
    fn a_namespace_of_another_layout_or_root_names_no_table() {
        for namespace in [
            "autotune/0.12.0/layout-v0/hardware/tuner",
            "autotune/0.12.0/layout-v10/hardware/tuner",
            "autotune/0.12.0/device/tuner",
            "autotune/0.12.0/layout-v1/hardware",
            "cuda/0.12.0/layout-v1/hardware/tuner",
        ] {
            assert_eq!(TuneTableName::parse(namespace), None, "{namespace}");
        }
    }
}
