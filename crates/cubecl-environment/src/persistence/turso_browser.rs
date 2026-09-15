//! The engine's I/O in the browser: the origin's private file system, through
//! synchronous access handles.
//!
//! The engine waits on a completion synchronously in places — a connection
//! reads the file header before it can do anything else, a checkpoint waits
//! for its writes — and a worker cannot turn its event loop while it waits,
//! so an operation that completes on the event loop is one the engine waits
//! for forever. Every operation here completes before it returns: OPFS hands
//! a dedicated worker a synchronous access handle to a file, and reading,
//! writing, truncating and flushing through it are ordinary calls. The handle
//! is the file's exclusive lock too, which is what one tab per environment
//! asks for; a tab that finds the file taken is told so.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;

use hashbrown::HashMap;
use turso::core::io::FileSyncType;
use turso::core::{
    Buffer, Clock, Completion, File, IO, MonotonicInstant, OpenFlags, WallClockInstant,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetFileOptions,
    FileSystemReadWriteOptions, FileSystemSyncAccessHandle,
};

use crate::sync::Mutex;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["navigator", "storage"], js_name = getDirectory)]
    fn get_directory() -> js_sys::Promise;

    #[wasm_bindgen(catch, js_namespace = ["navigator", "locks"], js_name = request)]
    fn request_lock(
        name: &str,
        options: &JsValue,
        callback: &js_sys::Function,
    ) -> Result<js_sys::Promise, JsValue>;
}

/// Takes the page's exclusive lock on `name` for the rest of its life, or
/// reports that another tab holds it.
///
/// The access handles below lock their files too, but only once the files
/// are opened, one at a time, and a tab that loses the race on the second
/// file would have the first. The Web Locks API names the environment as a
/// whole; a lock is released when the tab goes away, so a crashed tab never
/// wedges the others.
///
/// A browser without the API grants nothing and refuses nothing, which
/// leaves it exactly as unprotected as it was.
async fn hold_lock(name: &str) -> Result<(), String> {
    let (granted, was_granted) = async_channel::bounded::<bool>(1);

    // The lock is held for as long as the promise the callback returns stays
    // pending, so a granted lock returns one that never settles; the
    // request's own promise then never settles either, which is why the
    // outcome travels through the channel instead.
    let callback = Closure::once_into_js(move |lock: JsValue| -> js_sys::Promise {
        let held = !lock.is_null() && !lock.is_undefined();
        let _ = granted.try_send(held);
        if held {
            js_sys::Promise::new(&mut |_, _| {})
        } else {
            js_sys::Promise::resolve(&JsValue::UNDEFINED)
        }
    });

    let options = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&options, &"ifAvailable".into(), &JsValue::TRUE);

    let name = format!("cubecl-environment:{name}");
    match request_lock(&name, &options, callback.unchecked_ref()) {
        Ok(_pending) => {}
        Err(error) => {
            log::debug!("Web Locks unavailable ({error:?}); opening {name} unlocked");
            return Ok(());
        }
    }

    match was_granted.recv().await {
        Ok(true) => Ok(()),
        Ok(false) => Err(format!("another tab holds the environment {name}")),
        Err(_) => Err(format!("the lock request for {name} was dropped")),
    }
}

/// One file of the environment, held open through its synchronous access
/// handle for as long as the I/O lives.
///
/// The handle is a JavaScript value, which is neither `Send` nor `Sync`; the
/// engine asks for both of its files. This target has one thread, and every
/// call lands on the worker that opened the handle, which is what the two
/// traits promise.
struct BrowserFile {
    path: String,
    handle: FileSystemSyncAccessHandle,
}

unsafe impl Send for BrowserFile {}
unsafe impl Sync for BrowserFile {}

impl core::fmt::Debug for BrowserFile {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BrowserFile")
            .field("path", &self.path)
            .finish()
    }
}

impl BrowserFile {
    async fn open(root: &FileSystemDirectoryHandle, path: &str) -> Result<Self, String> {
        let options = FileSystemGetFileOptions::new();
        options.set_create(true);
        let handle = JsFuture::from(root.get_file_handle_with_options(path, &options))
            .await
            .map_err(|error| format!("unable to open OPFS file {path}: {error:?}"))?
            .unchecked_into::<FileSystemFileHandle>();
        let handle = JsFuture::from(handle.create_sync_access_handle())
            .await
            .map_err(|error| {
                format!("unable to take OPFS file {path}, which another tab may hold: {error:?}")
            })?
            .unchecked_into::<FileSystemSyncAccessHandle>();

        Ok(Self {
            path: path.to_string(),
            handle,
        })
    }

    fn at(position: u64) -> FileSystemReadWriteOptions {
        let options = FileSystemReadWriteOptions::new();
        options.set_at(position as f64);
        options
    }

    /// Reports `result` on `completion`: the count it carries, or a failure
    /// the engine reads as `-1`, logged here since the completion carries no
    /// message.
    fn report(&self, operation: &str, completion: Completion, result: Result<f64, JsValue>) {
        match result {
            Ok(count) => completion.complete(count as i32),
            Err(error) => {
                log::warn!(
                    "Browser database {operation} on {} failed: {error:?}",
                    self.path
                );
                completion.complete(-1);
            }
        }
    }
}

impl Drop for BrowserFile {
    fn drop(&mut self) {
        self.handle.close();
    }
}

impl File for BrowserFile {
    fn lock_file(&self, _exclusive: bool) -> turso::core::Result<()> {
        Ok(())
    }

    fn unlock_file(&self) -> turso::core::Result<()> {
        Ok(())
    }

    fn pread(&self, position: u64, completion: Completion) -> turso::core::Result<Completion> {
        let buffer = completion.as_read().buf_arc();
        let result = self
            .handle
            .read_with_u8_array_and_options(buffer.as_mut_slice(), &Self::at(position));
        self.report("read", completion.clone(), result);
        Ok(completion)
    }

    fn pwrite(
        &self,
        position: u64,
        buffer: Arc<Buffer>,
        completion: Completion,
    ) -> turso::core::Result<Completion> {
        let result = self
            .handle
            .write_with_u8_array_and_options(buffer.as_slice(), &Self::at(position));
        self.report("write", completion.clone(), result);
        Ok(completion)
    }

    fn sync(
        &self,
        completion: Completion,
        _sync_type: FileSyncType,
    ) -> turso::core::Result<Completion> {
        let result = self.handle.flush().map(|()| 0.0);
        self.report("flush", completion.clone(), result);
        Ok(completion)
    }

    fn size(&self) -> turso::core::Result<u64> {
        self.handle
            .get_size()
            .map(|size| size as u64)
            .map_err(|error| {
                turso::core::LimboError::InternalError(format!(
                    "unable to read the size of browser database file {}: {error:?}",
                    self.path
                ))
            })
    }

    fn truncate(&self, size: u64, completion: Completion) -> turso::core::Result<Completion> {
        let result = self.handle.truncate_with_f64(size as f64).map(|()| 0.0);
        self.report("truncate", completion.clone(), result);
        Ok(completion)
    }
}

#[derive(Debug)]
pub struct BrowserIo {
    files: Mutex<HashMap<String, Arc<BrowserFile>>>,
}

impl BrowserIo {
    pub async fn new(paths: &[&str]) -> Result<Self, String> {
        // The first path names the database; the rest are its sidecars.
        if let Some(database) = paths.first() {
            hold_lock(database).await?;
        }

        let root = JsFuture::from(get_directory())
            .await
            .map_err(|error| format!("unable to open OPFS: {error:?}"))?
            .unchecked_into::<FileSystemDirectoryHandle>();
        let mut files = HashMap::new();

        for path in paths {
            let file = BrowserFile::open(&root, path).await?;
            files.insert((*path).to_string(), Arc::new(file));
        }

        Ok(Self {
            files: Mutex::new(files),
        })
    }

    fn file(&self, path: &str) -> turso::core::Result<Arc<BrowserFile>> {
        self.files.lock().get(path).cloned().ok_or_else(|| {
            turso::core::LimboError::InternalError(format!(
                "browser database file was not registered: {path}"
            ))
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
        Ok(self.file(path)? as Arc<dyn File>)
    }

    fn remove_file(&self, path: &str) -> turso::core::Result<()> {
        let file = self.file(path)?;
        file.handle.truncate_with_f64(0.0).map_err(|error| {
            turso::core::LimboError::InternalError(format!(
                "unable to reset browser database file {path}: {error:?}"
            ))
        })
    }

    fn file_id(&self, path: &str) -> turso::core::Result<turso::core::io::FileId> {
        Ok(turso::core::io::FileId::from_path_hash(path))
    }
}
