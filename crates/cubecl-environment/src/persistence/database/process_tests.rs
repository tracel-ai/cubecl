//! Native cache contracts that require genuinely separate engine registries.

use super::*;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

const CHILD_ROLE: &str = "CUBECL_TEST_CACHE_PROCESS_ROLE";
const CHILD_PATH: &str = "CUBECL_TEST_CACHE_PROCESS_PATH";

fn wait_until(mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !ready() {
        assert!(Instant::now() < deadline, "cache subprocess timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Reap a subprocess even when an assertion fails before its next signal.
struct CacheProcess(Child);

impl CacheProcess {
    fn start(location: &str, role: &str) -> Self {
        Self(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "persistence::database::process_tests::cache_process",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_ROLE, role)
                .env(CHILD_PATH, location)
                .spawn()
                .unwrap(),
        )
    }

    fn wait_for(&mut self, path: &Path) {
        wait_until(|| {
            if path.exists() {
                return true;
            }
            assert!(
                self.0.try_wait().unwrap().is_none(),
                "cache subprocess exited before {}",
                path.display()
            );
            false
        });
    }

    fn finish(mut self) {
        wait_until(|| match self.0.try_wait().unwrap() {
            Some(status) => {
                assert!(status.success(), "cache subprocess failed: {status}");
                true
            }
            None => false,
        });
    }
}

impl Drop for CacheProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn marker(location: &str, name: &str) -> PathBuf {
    Path::new(location).with_extension(name)
}

fn notes(location: &str) -> (sql::Database, sql::Connection) {
    let database = crate::future::block_on(sql::Database::open(location, Opening::Shared)).unwrap();
    let connection = connect(&database).unwrap();
    connection
        .execute("CREATE TABLE notes (body BLOB)", &[])
        .unwrap();
    connection
        .execute("INSERT INTO notes VALUES ('initial')", &[])
        .unwrap();
    (database, connection)
}

fn count(connection: &sql::Connection) -> i64 {
    let mut count = 0;
    connection
        .query("SELECT count(*) FROM notes", &[], &mut |row| {
            count = row.integer(0)?;
            Ok::<_, SqlError>(())
        })
        .unwrap();
    count
}

#[test]
#[ignore = "subprocess helper for the native cache tests"]
fn cache_process() {
    let Ok(role) = std::env::var(CHILD_ROLE) else {
        return;
    };
    let location = std::env::var(CHILD_PATH).unwrap();
    match role.as_str() {
        "reader" => {
            let database =
                crate::future::block_on(sql::Database::open(&location, Opening::ReadOnly)).unwrap();
            let connection = connect(&database).unwrap();
            assert_eq!(count(&connection), 1);
            std::fs::write(marker(&location, "ready"), []).unwrap();
            wait_until(|| marker(&location, "committed").exists());
            assert_eq!(
                count(&connection),
                2,
                "reader missed the other process's commit"
            );
        }
        "writer" => {
            let database =
                crate::future::block_on(sql::Database::open(&location, Opening::Shared)).unwrap();
            let connection = connect(&database).unwrap();
            connection
                .execute("INSERT INTO notes VALUES ('child')", &[])
                .unwrap();
            assert_eq!(count(&connection), 2);
        }
        #[cfg(windows)]
        "process-local" => {
            assert!(
                crate::future::block_on(sql::Database::open(&location, Opening::ProcessLocal))
                    .is_err(),
                "process-local recovery must not bypass a live shared writer"
            );
        }
        #[cfg(windows)]
        "recover-held" => {
            let recovered = crate::future::block_on(recover_long_log(&location));
            assert_eq!(
                recovered,
                LogRecovery::Deferred,
                "a reader still pins the log"
            );
        }
        #[cfg(windows)]
        "recover-idle" => {
            assert_eq!(
                crate::future::block_on(recover_long_log(&location)),
                LogRecovery::Restarted
            );
        }
        other => panic!("unknown cache subprocess role: {other}"),
    }
}

#[test]
fn a_reader_observes_later_commits_from_another_process() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cache.db");
    let location = path.to_str().unwrap();
    let (_database, connection) = notes(location);
    let mut reader = CacheProcess::start(location, "reader");
    reader.wait_for(&marker(location, "ready"));
    connection
        .execute("INSERT INTO notes VALUES ('later')", &[])
        .unwrap();
    std::fs::write(marker(location, "committed"), []).unwrap();
    reader.finish();
}

#[test]
fn a_second_process_can_write_while_the_first_stays_open() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cache.db");
    let location = path.to_str().unwrap();
    let (_database, connection) = notes(location);
    CacheProcess::start(location, "writer").finish();
    assert_eq!(count(&connection), 2);
}

#[test]
#[cfg(windows)]
fn process_local_recovery_cannot_bypass_a_shared_writer() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cache.db");
    let location = path.to_str().unwrap();
    let (_database, _connection) = notes(location);
    CacheProcess::start(location, "process-local").finish();
}

#[cfg(windows)]
fn recovery_with_an_open_writer(hold_read: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cache.db");
    let location = path.to_str().unwrap();
    let (database, writer) = notes(location);
    let reader = connect(&database).unwrap();
    reader.execute("BEGIN", &[]).unwrap();
    assert_eq!(count(&reader), 1);

    // Grow a real log beyond the recovery threshold without crossing Turso's
    // shared index limit. Pin the read while growing, then retain it only for
    // the deferral case.
    let value = alloc::vec![7u8; 512 * 1024];
    let frame_bytes = page_size_on_disk(location).unwrap() + 24;
    let mut entries = 1;
    let deadline = Instant::now() + Duration::from_secs(30);
    while log_frames(location, frame_bytes) < LOG_RESTART_FRAMES {
        assert!(Instant::now() < deadline, "fixture log did not grow");
        writer.execute("BEGIN IMMEDIATE", &[]).unwrap();
        for _ in 0..32 {
            writer
                .execute("INSERT INTO notes VALUES (?1)", &[value.as_slice().into()])
                .unwrap();
            entries += 1;
        }
        writer.execute("COMMIT", &[]).unwrap();
    }
    let frames = log_frames(location, frame_bytes);
    if !hold_read {
        reader.execute("COMMIT", &[]).unwrap();
    }
    CacheProcess::start(
        location,
        if hold_read {
            "recover-held"
        } else {
            "recover-idle"
        },
    )
    .finish();
    assert!(path.exists(), "recovery must not rename a live database");
    assert!(
        Path::new(&format!("{location}-tshm")).exists(),
        "recovery must retain shared coordination"
    );
    if hold_read {
        assert_eq!(
            log_frames(location, frame_bytes),
            frames,
            "the pinned log must not be truncated"
        );
        assert_eq!(count(&reader), 1, "the reader's snapshot must remain valid");
        reader.execute("COMMIT", &[]).unwrap();
    }
    writer
        .execute("INSERT INTO notes VALUES ('after recovery')", &[])
        .unwrap();
    assert_eq!(
        count(&writer),
        entries + 1,
        "the original writer remains usable"
    );
}

#[test]
#[cfg(windows)]
fn recovery_keeps_shared_coordination_for_a_live_writer() {
    recovery_with_an_open_writer(false);
}

#[test]
#[cfg(windows)]
fn recovery_defers_while_another_process_holds_a_read() {
    recovery_with_an_open_writer(true);
}
