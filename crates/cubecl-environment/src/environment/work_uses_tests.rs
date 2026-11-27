//! What a database's records of the work read as, over a database file.

use super::{BuildStamp, RecordedUses, UsedContent};
use crate::persistence::{Access, Database, Origin, encode};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use tempfile::TempDir;

const BUILD: BuildStamp = BuildStamp { hash: 7 };
const OTHER: BuildStamp = BuildStamp { hash: 8 };

/// A database of its own, in a directory the test holds.
fn database() -> (TempDir, Database) {
    let directory = tempfile::tempdir().unwrap();
    let database = Database::open_at(&directory.path().join("uses.db"), Access::ReadWrite).unwrap();
    (directory, database)
}

/// Record `key` as used under `namespace`, as a store writes it.
fn plant(database: &Database, namespace: &str, key: u128) {
    database.insert(namespace, &encode(&key), &encode(&()), Origin::Local);
}

fn kernels(build: BuildStamp) -> String {
    UsedContent::Kernels.namespace(build).to_string()
}

fn matmul() -> UsedContent {
    UsedContent::Entries {
        namespace: "autotune/0.1.0/matmul".to_string(),
    }
}

/// Only this build's record is read as its uses.
#[test]
fn only_this_builds_record_reads_as_used() {
    let (_directory, database) = database();
    plant(&database, &kernels(BUILD), 1);
    plant(&database, &kernels(OTHER), 2);

    let uses = RecordedUses::read(&database, BUILD);

    assert!(uses.used(&UsedContent::Kernels, &1_u128));
    assert!(!uses.used(&UsedContent::Kernels, &2_u128));
}

/// A record of the entries of a namespace is read as that content, not as
/// the kernels.
#[test]
fn an_entries_record_is_read_as_its_namespace() {
    let (_directory, database) = database();
    plant(&database, matmul().namespace(BUILD).as_str(), 1);

    let uses = RecordedUses::read(&database, BUILD);

    assert!(uses.used(&matmul(), &1_u128));
    assert!(!uses.recorded(&UsedContent::Kernels));
}

/// A row of this build's kernels record counts as a record even when it is
/// no key.
#[test]
fn a_row_that_is_no_key_still_counts_as_recorded() {
    let (_directory, database) = database();
    database.insert(&kernels(BUILD), b"not a key", b"", Origin::Local);

    let uses = RecordedUses::read(&database, BUILD);

    assert!(uses.recorded(&UsedContent::Kernels));
}

/// Purging other builds' kernels records removes those of another build,
/// of another version and under the legacy root, and nothing else: this
/// build's, another content's, and a namespace that only starts like one.
#[test]
fn purging_other_builds_removes_exactly_their_records_of_the_content() {
    let (_directory, database) = database();
    let version = env!("CARGO_PKG_VERSION");
    let gone = [
        kernels(OTHER),
        alloc::format!("used/0.0.0-other/{BUILD}/kernels"),
        alloc::format!("launched/{version}/{OTHER}"),
    ];
    let kept = [
        kernels(BUILD),
        matmul().namespace(OTHER).to_string(),
        alloc::format!("{}-extra", kernels(OTHER)),
    ];
    for namespace in gone.iter().chain(&kept) {
        plant(&database, namespace, 1);
    }

    RecordedUses::read(&database, BUILD).purge_other_builds(&database, &UsedContent::Kernels);

    let mut left = database.namespaces();
    left.sort();
    let mut expected: Vec<String> = kept.to_vec();
    expected.sort();
    assert_eq!(left, expected);
}
