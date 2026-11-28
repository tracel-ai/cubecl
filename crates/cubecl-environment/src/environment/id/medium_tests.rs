//! Saves of an environment under a root of the test's own.

use super::*;
use crate::environment::{SaveOptions, SavedContent, activate, set_root};
use crate::persistence::{Namespace, Store, StoreOptions};
use alloc::string::{String, ToString};
use alloc::vec;
use serial_test::serial;

/// A save of the used tunes that would hold none fails, rather than
/// handing back a bundle that brings nothing.
#[test]
#[serial]
fn a_save_of_the_used_answers_fails_when_the_work_used_none() {
    let root = tempfile::tempdir().expect("a temp dir");
    set_root(root.path());
    let id = EnvironmentId::create("unused", EnvironmentOrigin::Empty).unwrap();
    id.activate().unwrap();
    let mut table: Store<String, u32> =
        Store::new(StoreOptions::new().storage(Namespace::scoped("autotune", "hardware/tuner")));
    table.insert("key".to_string(), 1).unwrap();
    activate(crate::environment::DEFAULT);

    let saved = id.save(SaveOptions::new().content(SavedContent::UsedTunes));

    assert!(matches!(saved, Err(EnvironmentError::NoTunesUsed { .. })));
}
