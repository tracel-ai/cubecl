#![cfg(all(feature = "persistence", not(target_family = "wasm")))]
//! Cached values reach other processes sharing a cache root, both after the
//! writer exits and while it still holds the database open.
//!
//! Each test runs this binary again as its child processes, one role each,
//! told apart by [`ROLE_VAR`]: the parent never opens the store itself.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};

use cubecl_environment::persistence::{Namespace, Store, StoreOptions};

/// The cache root a child process opens.
const ROOT_VAR: &str = "CUBECL_TEST_PROCESS_CACHE_ROOT";
/// What a child process does: `write`, `hold` or `read`.
const ROLE_VAR: &str = "CUBECL_TEST_PROCESS_CACHE_ROLE";
/// The key every role writes or reads.
const KEY: &str = "kernel";
/// The value every role writes or expects.
const VALUE: u32 = 42;
/// The line a holder prints once its value is stored.
const READY: &str = "stored";

#[test]
fn cache_survives_process_exit() {
    if run_child_role() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    run_to_completion(root.path(), "cache_survives_process_exit", "write");
    run_to_completion(root.path(), "cache_survives_process_exit", "read");
}

/// Where the engine won't share a WAL between processes (Windows, network
/// file systems), the second process opens the file read-only: it still
/// reads what the first one stored.
#[test]
fn cache_reaches_a_process_opened_while_the_writer_runs() {
    if run_child_role() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let test_name = "cache_reaches_a_process_opened_while_the_writer_runs";

    let mut holder = child(root.path(), test_name, "hold")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut holder_output = BufReader::new(holder.stdout.take().unwrap());
    let mut line = String::new();
    while line.trim() != READY {
        line.clear();
        let read = holder_output.read_line(&mut line).unwrap();
        assert_ne!(read, 0, "the holder exited before storing its value");
    }

    run_to_completion(root.path(), test_name, "read");

    // Closing its stdin lets the holder exit.
    drop(holder.stdin.take());
    assert!(
        holder.wait().unwrap().success(),
        "cache hold process failed"
    );
}

/// Plays the role [`ROLE_VAR`] names, and whether this process is a child
/// that did so.
fn run_child_role() -> bool {
    let Some(root) = std::env::var_os(ROOT_VAR) else {
        return false;
    };
    cubecl_environment::environment::set_root(root);
    let mut store: Store<String, u32> =
        Store::new(StoreOptions::new().storage(Namespace::scoped("process-persistence", "v1")));
    match std::env::var(ROLE_VAR).unwrap().as_str() {
        "write" => store.insert(KEY.to_string(), VALUE).unwrap(),
        "hold" => hold(&mut store),
        "read" => assert_eq!(store.get(&KEY.to_string()).copied(), Some(VALUE)),
        role => panic!("unsupported cache test role: {role}"),
    }
    true
}

/// Stores the value, says so, and keeps the database open until stdin closes.
fn hold(store: &mut Store<String, u32>) {
    store.insert(KEY.to_string(), VALUE).unwrap();
    println!("{READY}");
    std::io::stdin().read_to_end(&mut Vec::new()).unwrap();
}

/// This test binary, run as the child playing `role` in the test `test_name`.
fn child(root: &Path, test_name: &str, role: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", test_name, "--nocapture"])
        .env(ROOT_VAR, root)
        .env(ROLE_VAR, role);
    command
}

fn run_to_completion(root: &Path, test_name: &str, role: &str) {
    let status = child(root, test_name, role).status().unwrap();
    assert!(status.success(), "cache {role} process failed");
}
