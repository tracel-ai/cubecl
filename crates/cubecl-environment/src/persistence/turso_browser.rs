use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};

use hashbrown::HashMap;
use opfs::{
    CreateWritableOptions, DirectoryHandle as _, FileHandle as _, GetFileHandleOptions,
    WritableFileStream as _, WriteCommandType, WriteParams,
};
use turso::core::io::FileSyncType;
use turso::core::{
    Buffer, Clock, Completion, File, IO, MonotonicInstant, OpenFlags, WallClockInstant,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, spawn_local};

use crate::sync::Mutex;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["navigator", "storage"], js_name = getDirectory)]
    fn get_directory() -> js_sys::Promise;
}

#[derive(Debug)]
enum Operation {
    Read {
        position: usize,
        completion: Completion,
    },
    Write {
        position: usize,
        buffer: Arc<Buffer>,
        completion: Completion,
    },
    Truncate {
        size: usize,
        completion: Completion,
    },
    Sync {
        completion: Completion,
    },
    Reset,
}

#[derive(Debug)]
struct BrowserFile {
    operations: async_channel::Sender<Operation>,
    size: Arc<AtomicUsize>,
}

impl BrowserFile {
    fn new(handle: opfs::web::FileHandle, size: usize) -> Arc<Self> {
        let (sender, receiver) = async_channel::unbounded();
        let size = Arc::new(AtomicUsize::new(size));
        let file = Arc::new(Self {
            operations: sender,
            size: size.clone(),
        });

        spawn_local(run_file(handle, size, receiver));
        file
    }

    fn schedule(&self, operation: Operation) -> turso::core::Result<()> {
        self.operations.try_send(operation).map_err(|error| {
            turso::core::LimboError::InternalError(format!(
                "unable to schedule browser database I/O: {error}"
            ))
        })
    }
}

async fn run_file(
    handle: opfs::web::FileHandle,
    size: Arc<AtomicUsize>,
    operations: async_channel::Receiver<Operation>,
) {
    while let Ok(operation) = operations.recv().await {
        match operation {
            Operation::Read {
                position,
                completion,
            } => {
                let buffer = completion.as_read().buf_arc();
                let end = position.saturating_add(buffer.len());
                match handle.read_range(position..end).await {
                    Ok(bytes) => {
                        let len = bytes.len();
                        buffer.as_mut_slice()[..len].copy_from_slice(&bytes);
                        completion.complete(len as i32);
                    }
                    Err(error) => fail(completion, "read", error),
                }
            }
            Operation::Write {
                position,
                buffer,
                completion,
            } => {
                let len = buffer.len();
                let mut handle = handle.clone();
                let result = async {
                    let mut writer = handle
                        .create_writable_with_options(&CreateWritableOptions {
                            keep_existing_data: true,
                        })
                        .await?;
                    writer
                        .write_with_params(&WriteParams {
                            command_type: WriteCommandType::Write,
                            data: Some(buffer.as_slice().to_vec()),
                            position: Some(position),
                            size: None,
                        })
                        .await?;
                    writer.close().await
                }
                .await;

                match result {
                    Ok(()) => {
                        size.fetch_max(position.saturating_add(len), Ordering::Relaxed);
                        completion.complete(len as i32);
                    }
                    Err(error) => fail(completion, "write", error),
                }
            }
            Operation::Truncate {
                size: next_size,
                completion,
            } => {
                let mut handle = handle.clone();
                let result = async {
                    let mut writer = handle
                        .create_writable_with_options(&CreateWritableOptions {
                            keep_existing_data: true,
                        })
                        .await?;
                    writer.truncate(next_size).await?;
                    writer.close().await
                }
                .await;

                match result {
                    Ok(()) => {
                        size.store(next_size, Ordering::Relaxed);
                        completion.complete(0);
                    }
                    Err(error) => fail(completion, "truncate", error),
                }
            }
            Operation::Sync { completion } => completion.complete(0),
            Operation::Reset => {
                let mut handle = handle.clone();
                let result = async {
                    let mut writer = handle
                        .create_writable_with_options(&CreateWritableOptions {
                            keep_existing_data: true,
                        })
                        .await?;
                    writer.truncate(0).await?;
                    writer.close().await
                }
                .await;

                match result {
                    Ok(()) => size.store(0, Ordering::Relaxed),
                    Err(error) => {
                        log::warn!("Unable to reset browser database file: {error:?}")
                    }
                }
            }
        }
    }
}

fn fail(completion: Completion, operation: &str, error: JsValue) {
    log::warn!("Browser database {operation} failed: {error:?}");
    completion.complete(-1);
}

impl File for BrowserFile {
    fn lock_file(&self, _exclusive: bool) -> turso::core::Result<()> {
        Ok(())
    }

    fn unlock_file(&self) -> turso::core::Result<()> {
        Ok(())
    }

    fn pread(&self, position: u64, completion: Completion) -> turso::core::Result<Completion> {
        self.schedule(Operation::Read {
            position: position as usize,
            completion: completion.clone(),
        })?;
        Ok(completion)
    }

    fn pwrite(
        &self,
        position: u64,
        buffer: Arc<Buffer>,
        completion: Completion,
    ) -> turso::core::Result<Completion> {
        self.schedule(Operation::Write {
            position: position as usize,
            buffer,
            completion: completion.clone(),
        })?;
        Ok(completion)
    }

    fn sync(
        &self,
        completion: Completion,
        _sync_type: FileSyncType,
    ) -> turso::core::Result<Completion> {
        self.schedule(Operation::Sync {
            completion: completion.clone(),
        })?;
        Ok(completion)
    }

    fn size(&self) -> turso::core::Result<u64> {
        Ok(self.size.load(Ordering::Relaxed) as u64)
    }

    fn truncate(&self, size: u64, completion: Completion) -> turso::core::Result<Completion> {
        self.schedule(Operation::Truncate {
            size: size as usize,
            completion: completion.clone(),
        })?;
        Ok(completion)
    }
}

#[derive(Debug)]
pub struct BrowserIo {
    files: Mutex<HashMap<String, Arc<BrowserFile>>>,
}

impl BrowserIo {
    pub async fn new(paths: &[&str]) -> Result<Self, String> {
        let root = JsFuture::from(get_directory())
            .await
            .map_err(|error| format!("unable to open OPFS: {error:?}"))?
            .unchecked_into::<web_sys::FileSystemDirectoryHandle>();
        let root = opfs::web::DirectoryHandle::from(root);
        let mut files = HashMap::new();

        for path in paths {
            let handle = root
                .get_file_handle_with_options(path, &GetFileHandleOptions { create: true })
                .await
                .map_err(|error| format!("unable to open OPFS file {path}: {error:?}"))?;
            let size = handle
                .size()
                .await
                .map_err(|error| format!("unable to read OPFS file size for {path}: {error:?}"))?;
            files.insert((*path).to_string(), BrowserFile::new(handle, size));
        }

        Ok(Self {
            files: Mutex::new(files),
        })
    }
}

// The engine's own instants are `std::time`, which panics on this target;
// it consults the clock when a statement is busy or has a deadline, so a
// second write landing during the first one's I/O would take the page down.
impl Clock for BrowserIo {
    fn current_time_monotonic(&self) -> MonotonicInstant {
        // Milliseconds since the page started, with a fractional part.
        let millis = js_sys::Reflect::get(&js_sys::global(), &"performance".into())
            .ok()
            .and_then(|performance| {
                js_sys::Reflect::get(&performance, &"now".into())
                    .ok()
                    .and_then(|now| now.dyn_into::<js_sys::Function>().ok())
                    .and_then(|now| now.call0(&performance).ok())
            })
            .and_then(|value| value.as_f64())
            .unwrap_or_else(js_sys::Date::now);
        MonotonicInstant::from_nanos((millis * 1_000_000.0) as u128)
    }

    fn current_time_wall_clock(&self) -> WallClockInstant {
        let millis = js_sys::Date::now();
        let secs = (millis / 1000.0).floor();
        WallClockInstant {
            secs: secs as i64,
            micros: ((millis - secs * 1000.0) * 1000.0) as u32,
        }
    }
}

impl IO for BrowserIo {
    fn open_file(
        &self,
        path: &str,
        _flags: OpenFlags,
        _direct: bool,
    ) -> turso::core::Result<Arc<dyn File>> {
        self.files
            .lock()
            .get(path)
            .cloned()
            .map(|file| file as Arc<dyn File>)
            .ok_or_else(|| {
                turso::core::LimboError::InternalError(format!(
                    "browser database file was not registered: {path}"
                ))
            })
    }

    fn remove_file(&self, path: &str) -> turso::core::Result<()> {
        let file = self.files.lock().get(path).cloned().ok_or_else(|| {
            turso::core::LimboError::InternalError(format!(
                "browser database file was not registered: {path}"
            ))
        })?;
        file.size.store(0, Ordering::Relaxed);
        file.operations.try_send(Operation::Reset).map_err(|error| {
            turso::core::LimboError::InternalError(format!(
                "unable to reset browser database file: {error}"
            ))
        })
    }

    fn file_id(&self, path: &str) -> turso::core::Result<turso::core::io::FileId> {
        Ok(turso::core::io::FileId::from_path_hash(path))
    }
}
