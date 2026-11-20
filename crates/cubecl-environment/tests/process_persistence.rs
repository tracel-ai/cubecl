#![cfg(all(feature = "persistence", not(target_family = "wasm")))]
//! Cached values remain available after the writer process exits.

use cubecl_environment::persistence::{Namespace, Store, StoreOptions};

#[test]
fn cache_survives_process_exit() {
    if let Some(root) = std::env::var_os("CUBECL_TEST_PROCESS_CACHE_ROOT") {
        cubecl_environment::environment::set_root(root);
        let mut store: Store<String, u32> =
            Store::new(StoreOptions::new().storage(Namespace::scoped("process-persistence", "v1")));
        let key = String::from("kernel");
        match std::env::var("CUBECL_TEST_PROCESS_CACHE_MODE")
            .unwrap()
            .as_str()
        {
            "write" => store.insert(key, 42).unwrap(),
            "read" => assert_eq!(store.get(&key).copied(), Some(42)),
            mode => panic!("unsupported cache test mode: {mode}"),
        }
        return;
    }

    let root = tempfile::tempdir().unwrap();
    for mode in ["write", "read"] {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "cache_survives_process_exit", "--nocapture"])
            .env("CUBECL_TEST_PROCESS_CACHE_ROOT", root.path())
            .env("CUBECL_TEST_PROCESS_CACHE_MODE", mode)
            .status()
            .unwrap();
        assert!(status.success(), "cache {mode} process failed");
    }
}
