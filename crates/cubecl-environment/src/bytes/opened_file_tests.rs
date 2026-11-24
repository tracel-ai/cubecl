use super::{Bytes, FileReadCounter, OpenedFile, Reader};
use std::{fs::File, io::Write, path::PathBuf, sync::Barrier, vec::Vec};
use tempfile::TempDir;

/// The bytes the test file holds: each one differs from its neighbors, so a
/// window read from the wrong offset shows.
fn content(len: usize) -> Vec<u8> {
    (0..len).map(|index| (index % 251) as u8).collect()
}

/// A file holding `bytes` in a fresh directory, and the file opened at it.
fn opened_with(bytes: &[u8]) -> (TempDir, PathBuf, OpenedFile) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tensors");
    File::create(&path).unwrap().write_all(bytes).unwrap();
    let opened = OpenedFile::new(File::open(&path).unwrap(), &path);
    (dir, path, opened)
}

#[test]
fn window_of_an_opened_file_reads_its_bytes() {
    let expected = content(300);
    let (_dir, _path, opened) = opened_with(&expected);
    let window = Bytes::from_opened_file(&opened, 50, 100);

    assert_eq!(&window[..], &expected[100..150]);
}

#[test]
fn reads_of_an_opened_file_are_counted_under_its_path() {
    let (_dir, path, opened) = opened_with(&content(200));
    let counter = FileReadCounter::new(&path);
    let window = Bytes::from_opened_file(&opened, 80, 40);

    window.read(Reader::new()).unwrap();

    assert_eq!(counter.read(), 80);
}

#[cfg(unix)]
#[test]
fn window_reads_the_opened_file_after_its_path_is_removed() {
    let expected = content(200);
    let (_dir, path, opened) = opened_with(&expected);
    let window = Bytes::from_opened_file(&opened, 80, 40);
    std::fs::remove_file(&path).unwrap();

    assert_eq!(&window[..], &expected[40..120]);
}

#[test]
fn window_reads_the_opened_file_after_another_is_renamed_over_its_path() {
    let expected = content(200);
    let (dir, path, opened) = opened_with(&expected);
    let window = Bytes::from_opened_file(&opened, 80, 40);
    let other = dir.path().join("other");
    std::fs::write(&other, [0u8; 200]).unwrap();
    std::fs::rename(&other, &path).unwrap();

    assert_eq!(&window[..], &expected[40..120]);
}

#[test]
fn windows_read_concurrently_get_their_own_bytes() {
    const WINDOW_COUNT: usize = 16;
    const WINDOW_BYTES: usize = 64 * 1024;
    let expected = content(WINDOW_COUNT * WINDOW_BYTES);
    let (_dir, _path, opened) = opened_with(&expected);
    let windows: Vec<Bytes> = (0..WINDOW_COUNT)
        .map(|index| {
            let offset = (index * WINDOW_BYTES) as u64;
            Bytes::from_opened_file(&opened, WINDOW_BYTES as u64, offset)
        })
        .collect();
    let start = Barrier::new(WINDOW_COUNT);

    let read: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let readers: Vec<_> = windows
            .iter()
            .map(|window| {
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    window[..].to_vec()
                })
            })
            .collect();
        readers
            .into_iter()
            .map(|reader| reader.join().unwrap())
            .collect()
    });

    let expected: Vec<&[u8]> = expected.chunks(WINDOW_BYTES).collect();
    assert_eq!(read, expected);
}
