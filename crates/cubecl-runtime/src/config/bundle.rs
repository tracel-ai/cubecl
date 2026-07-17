use alloc::string::String;
use alloc::vec::Vec;

/// Configuration for environment bundles: directories with pre-warmed caches
/// (autotune results, compiled kernels) produced by `cubecl` bundle export.
///
/// ```toml
/// [bundle]
/// paths = ["./bundles/h100-linux"]
/// ```
///
/// The `CUBECL_BUNDLE` environment variable appends additional paths
/// (separated like `PATH`). Bundles can also be installed programmatically
/// with `cubecl_environment::bundle::install`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BundleConfig {
    /// Bundle directories loaded when the configuration is first read.
    ///
    /// Any bundle that fails to load is skipped with a warning.
    #[serde(default)]
    pub paths: Vec<String>,
}
