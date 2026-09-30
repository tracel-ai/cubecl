//! Experimental kernels must not enter the default persistent compilation cache.

#![cfg(compilation_cache)]

use cubecl_server::{
    compiler::compilation_store,
    config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig},
};
use std::process::Command;

#[test]
fn experiment_does_not_share_the_default_cache() {
    let directory = tempfile::tempdir().unwrap();
    for (enable, force_off, cache_expected) in [
        (None, false, true),
        (Some("0"), false, true),
        (Some("1"), false, false),
        (Some("1"), true, false),
    ] {
        // Process isolation avoids mutating the test runner's environment or
        // once-initialized runtime configuration while other tests are active.
        let mut child = Command::new(std::env::current_exe().unwrap());
        child
            .args(["--exact", "cache_policy_probe"])
            .env("CUBECL_TEST_QUOTIENT_CACHE_ROOT", directory.path())
            .env(
                "CUBECL_TEST_QUOTIENT_CACHE_EXPECTED",
                cache_expected.to_string(),
            )
            .env_remove("CUBECL_ENABLE_QUOTIENT_RANGE")
            .env_remove("CUBECL_DISABLE_QUOTIENT_RANGE");
        if let Some(value) = enable {
            child.env("CUBECL_ENABLE_QUOTIENT_RANGE", value);
        }
        if force_off {
            child.env("CUBECL_DISABLE_QUOTIENT_RANGE", "1");
        }
        let result = child.output().unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[test]
fn cache_policy_probe() {
    let Some(root) = std::env::var_os("CUBECL_TEST_QUOTIENT_CACHE_ROOT") else {
        return;
    };
    let mut config = CubeClRuntimeConfig::default();
    config.compilation.cache = true;
    config.environment.path = CacheConfig::Directory(root.into());
    CubeClRuntimeConfig::set(config);
    let store = compilation_store::<u64, Vec<u8>>("quotient-range-policy-test", "device");
    let expected = std::env::var("CUBECL_TEST_QUOTIENT_CACHE_EXPECTED").unwrap() == "true";
    assert_eq!(store.is_some(), expected);
}
