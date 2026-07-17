//! IndexedDB implementation of [`KvBackend`].
//!
//! IndexedDB is chosen over `localStorage` because it is available in Web
//! Workers (where wgpu applications commonly run) and has a much larger
//! quota.
//!
//! # Model
//!
//! One database `"cubecl"` with a single object store `"kv"`. Each entry is
//! stored under the key `"{prefix}{dedup_key}"` where `prefix` mirrors the
//! file system layout (`name/version/segments/`). The record value is the
//! exact serialized entry line (separator included), so hydration feeds the
//! same parser as the file backend.
//!
//! # Concurrency
//!
//! All JavaScript handles are confined to `spawn_local` tasks; the backend
//! itself only holds `Send` data. Writes are fire-and-forget: entry record
//! keys are unique and re-inserting an identical value is idempotent, so
//! cross-put ordering doesn't matter. There is no cross-tab lock: identical
//! keys are last-write-wins, and divergent values surface later as
//! `KeyOutOfSync`, which every caller already tolerates.
//!
//! # Durability
//!
//! Entries inserted immediately before the page closes may be lost. This is
//! acceptable for a cache.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::sync::Arc;

use js_sys::{Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use web_sys::{
    IdbDatabase, IdbFactory, IdbKeyRange, IdbOpenDbRequest, IdbRequest, IdbTransactionMode,
};

use super::backend::KvBackend;

const DB_NAME: &str = "cubecl";
const STORE_NAME: &str = "kv";

/// Content delivered asynchronously by the hydration task.
#[derive(Default, Debug)]
struct Inbox {
    hydrated: Vec<u8>,
    done: bool,
}

/// Browser storage backend, persisting entries to IndexedDB.
#[derive(Debug)]
pub struct BrowserBackend {
    prefix: String,
    inbox: Arc<spin::Mutex<Inbox>>,
}

impl BrowserBackend {
    /// Creates the backend and starts hydrating existing entries under
    /// `prefix` in the background.
    pub fn new(prefix: String) -> Self {
        let inbox = Arc::new(spin::Mutex::new(Inbox::default()));

        {
            let inbox = inbox.clone();
            let prefix = prefix.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let result = hydrate(&prefix, &inbox).await;
                let mut guard = inbox.lock();
                guard.done = true;
                if let Err(err) = result {
                    log::warn!(
                        "cubecl cache: browser storage hydration failed for '{prefix}': {err:?}; \
                         continuing memory-only"
                    );
                }
            });
        }

        Self { prefix, inbox }
    }
}

impl KvBackend for BrowserBackend {
    fn lock(&mut self) -> Option<Vec<u8>> {
        let mut guard = self.inbox.lock();
        if guard.hydrated.is_empty() {
            None
        } else {
            Some(core::mem::take(&mut guard.hydrated))
        }
    }

    fn unlock(&mut self) {}

    fn append(&mut self, dedup_key: &str, bytes: &[u8]) {
        let record_key = format!("{}{}", self.prefix, dedup_key);
        let bytes = bytes.to_vec();

        wasm_bindgen_futures::spawn_local(async move {
            if let Err(err) = put(&record_key, &bytes).await {
                log::warn!("cubecl cache: browser storage put('{record_key}') failed: {err:?}");
            }
        });
    }

    fn hydrating(&self) -> bool {
        !self.inbox.lock().done
    }

    fn has_pending(&self) -> bool {
        let inbox = self.inbox.lock();
        !inbox.done || !inbox.hydrated.is_empty()
    }

    fn describe(&self) -> String {
        format!("browser storage (indexeddb: {}/{}*)", DB_NAME, self.prefix)
    }
}

/// Fetches the `indexedDB` factory from the global scope, working in both
/// `Window` and `WorkerGlobalScope` contexts.
fn factory() -> Result<IdbFactory, JsValue> {
    let global = js_sys::global();
    let value = Reflect::get(&global, &JsValue::from_str("indexedDB"))?;

    value
        .dyn_into::<IdbFactory>()
        .map_err(|_| JsValue::from_str("indexedDB is not available in this context"))
}

/// Resolves an [`IdbRequest`] into its result, bridging callbacks to a future.
async fn request_result(request: &IdbRequest) -> Result<JsValue, JsValue> {
    let (tx, rx) = oneshot::channel::<Result<(), JsValue>>();
    let tx = alloc::rc::Rc::new(core::cell::RefCell::new(Some(tx)));

    let on_success = {
        let tx = tx.clone();
        Closure::once(move |_event: web_sys::Event| {
            if let Some(tx) = tx.borrow_mut().take() {
                tx.send(Ok(())).ok();
            }
        })
    };
    let on_error = {
        let tx = tx.clone();
        Closure::once(move |event: web_sys::Event| {
            if let Some(tx) = tx.borrow_mut().take() {
                tx.send(Err(JsValue::from(event))).ok();
            }
        })
    };

    request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
    request.set_onerror(Some(on_error.as_ref().unchecked_ref()));

    let outcome = rx
        .await
        .map_err(|_| JsValue::from_str("request callback dropped"))?;

    request.set_onsuccess(None);
    request.set_onerror(None);

    outcome?;
    request.result()
}

/// Opens (and on first use creates) the database.
async fn open_db() -> Result<IdbDatabase, JsValue> {
    let request: IdbOpenDbRequest = factory()?.open_with_u32(DB_NAME, 1)?;

    // On the first open the object store must be created during "upgradeneeded".
    let on_upgrade = Closure::once(move |event: web_sys::Event| {
        let request: IdbOpenDbRequest = match event.target() {
            Some(target) => match target.dyn_into() {
                Ok(request) => request,
                Err(_) => return,
            },
            None => return,
        };
        if let Ok(result) = request.result()
            && let Ok(db) = result.dyn_into::<IdbDatabase>()
            && !db.object_store_names().contains(STORE_NAME)
        {
            db.create_object_store(STORE_NAME).ok();
        }
    });
    request.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));

    let result = request_result(request.unchecked_ref()).await?;
    request.set_onupgradeneeded(None);

    result.dyn_into::<IdbDatabase>().map_err(JsValue::from)
}

/// Loads every record under `prefix` and pushes the concatenated bytes into
/// the inbox.
async fn hydrate(prefix: &str, inbox: &spin::Mutex<Inbox>) -> Result<(), JsValue> {
    let db = open_db().await?;

    let transaction = db.transaction_with_str(STORE_NAME)?;
    let store = transaction.object_store(STORE_NAME)?;

    // All keys starting with the prefix: [prefix, prefix + U+10FFFF).
    let upper = format!("{prefix}\u{10FFFF}");
    let range = IdbKeyRange::bound(&JsValue::from_str(prefix), &JsValue::from_str(&upper))?;

    let request = store.get_all_with_key(&range)?;
    let result = request_result(&request).await?;

    let values: js_sys::Array = result.dyn_into()?;
    let mut buffer = Vec::new();
    for value in values.iter() {
        match value.dyn_into::<Uint8Array>() {
            Ok(bytes) => buffer.extend(bytes.to_vec()),
            Err(value) => {
                log::warn!("cubecl cache: unexpected browser storage record type: {value:?}");
            }
        }
    }

    if !buffer.is_empty() {
        inbox.lock().hydrated.extend(buffer);
    }

    Ok(())
}

/// Writes one record.
async fn put(record_key: &str, bytes: &[u8]) -> Result<(), JsValue> {
    let db = open_db().await?;

    let transaction =
        db.transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)?;
    let store = transaction.object_store(STORE_NAME)?;

    let value = Uint8Array::from(bytes);
    let request = store.put_with_key(&value, &JsValue::from_str(record_key))?;
    request_result(&request).await?;

    Ok(())
}
