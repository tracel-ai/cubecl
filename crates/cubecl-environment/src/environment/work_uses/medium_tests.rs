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

fn conv() -> UsedContent {
    UsedContent::Entries {
        namespace: "autotune/0.1.0/conv".to_string(),
    }
}

/// The namespace of an autotune table `content` names.
fn table(content: &UsedContent) -> &str {
    match content {
        UsedContent::Entries { namespace } => namespace,
        UsedContent::Kernels => unreachable!("the kernels should not be asked for a table"),
    }
}

/// The keys `namespace` holds, decoded, in order.
fn keys(database: &Database, namespace: &str) -> Vec<u128> {
    let mut keys = Vec::new();
    database.scan(namespace, &mut |key, _| {
        keys.push(ciborium::from_reader(key).unwrap());
    });
    keys.sort();
    keys
}

/// In a table this build used, a tune it did not use goes and one it
/// used stays, and the tally counts both.
#[test]
fn a_prune_removes_the_unused_answers_of_a_used_table() {
    let (_directory, database) = database();
    plant(&database, table(&matmul()), 1);
    plant(&database, table(&matmul()), 2);
    plant(&database, matmul().namespace(BUILD).as_str(), 1);

    let tally = RecordedUses::read(&database, BUILD).prune_unused_tunes(&database);

    assert_eq!(keys(&database, table(&matmul())), [1]);
    assert_eq!((tally.kept, tally.removed), (1, 1));
}

/// A table this build never used may be another hardware's, another
/// version's or a tuner's it never ran: it stays whole.
#[test]
fn a_prune_keeps_a_table_the_build_never_used() {
    let (_directory, database) = database();
    plant(&database, table(&conv()), 2);
    plant(&database, matmul().namespace(BUILD).as_str(), 1);

    RecordedUses::read(&database, BUILD).prune_unused_tunes(&database);

    assert_eq!(keys(&database, table(&conv())), [2]);
}

/// With no use recorded, a prune knows nothing of what the work needs: no
/// tune goes, and no other build's record of one either.
#[test]
fn a_prune_of_a_build_that_used_no_answer_changes_nothing() {
    let (_directory, database) = database();
    plant(&database, table(&matmul()), 2);
    plant(&database, matmul().namespace(OTHER).as_str(), 2);

    RecordedUses::read(&database, BUILD).prune_unused_tunes(&database);

    let mut left = database.namespaces();
    left.sort();
    let mut expected = [
        table(&matmul()).to_string(),
        matmul().namespace(OTHER).to_string(),
    ];
    expected.sort();
    assert_eq!(left, expected);
}

/// Other builds' records of a table this build pruned go with its unused
/// tunes.
#[test]
fn a_prune_removes_other_builds_records_of_a_table_it_prunes() {
    let (_directory, database) = database();
    plant(&database, table(&matmul()), 1);
    plant(&database, matmul().namespace(BUILD).as_str(), 1);
    plant(&database, matmul().namespace(OTHER).as_str(), 1);

    RecordedUses::read(&database, BUILD).prune_unused_tunes(&database);

    assert!(
        !database
            .namespaces()
            .contains(&matmul().namespace(OTHER).to_string())
    );
}

/// What a save of the used tunes keeps: a table the build never used is
/// emptied, unlike in a prune.
#[test]
fn keeping_only_the_used_answers_empties_a_table_the_build_never_used() {
    let (_directory, database) = database();
    plant(&database, table(&matmul()), 1);
    plant(&database, table(&conv()), 2);
    plant(&database, matmul().namespace(BUILD).as_str(), 1);

    let kept = RecordedUses::read(&database, BUILD).keep_only_used_tunes(&database);

    assert_eq!(kept, 1);
    assert_eq!(keys(&database, table(&conv())), []);
}
