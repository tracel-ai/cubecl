//! IndexedDB implementation of [`Storage`].
//!
//! IndexedDB is chosen over `localStorage` because it is available in Web
//! Workers (where wgpu applications commonly run) and has a much larger
//! quota.
//!
//! # Model
//!
//! One database `"cubecl"` with a single object store `"kv"`. Each entry is
//! stored under the record key `"{namespace}/{hex key}"` and holds the raw value
//! bytes. The namespace's entries are mirrored in memory, loaded in the
//! background at open, and served from there.
//!
//! # Concurrency
//!
//! All JavaScript handles are confined to `spawn_local` tasks; the backend
//! itself only holds `Send` data. Writes are fire-and-forget: record keys are
//! unique and re-inserting an identical value is idempotent, so cross-put
//! ordering doesn't matter. There is no cross-tab lock, so a key written
//! concurrently by two tabs with different values is last-write-wins rather
//! than insert-only. Every caller already tolerates a stale cache entry.
//!
//! # Durability
//!
//! Entries inserted immediately before the page closes may be lost. This is
//! acceptable for a cache.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::bytes::Bytes;
use crate::sync::Arc;

use js_sys::{Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use web_sys::{
    IdbDatabase, IdbFactory, IdbKeyRange, IdbOpenDbRequest, IdbRequest, IdbTransactionMode,
};

use super::storage::{Insertion, Origin, Storage, entries};

const DB_NAME: &str = "cubecl";
const STORE_NAME: &str = "kv";

/// The mirrored content of one namespace.
#[derive(Default, Debug)]
struct State {
    entries: super::storage::Entries,
    loaded: bool,
}

/// Browser storage backend, persisting entries to IndexedDB.
#[derive(Debug)]
pub struct BrowserStorage {
    /// The record key prefix of this store, `"{store}/"`.
    prefix: String,
    state: Arc<spin::Mutex<State>>,
}

/// The storage serving `namespace` in browser storage.
pub(crate) fn open_storage(namespace: &str) -> Box<dyn Storage> {
    Box::new(BrowserStorage::new(format!("{namespace}/")))
}

impl BrowserStorage {
    /// Creates the storage and starts loading existing entries under
    /// `prefix` in the background.
    pub fn new(prefix: String) -> Self {
        let state = Arc::new(spin::Mutex::new(State::default()));

        {
            let state = state.clone();
            let prefix = prefix.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let result = load(&prefix, &state).await;
                state.lock().loaded = true;
                if let Err(err) = result {
                    log::warn!(
                        "cubecl cache: browser storage load failed for '{prefix}': {err:?}; \
                         continuing memory-only"
                    );
                }
            });
        }

        Self { prefix, state }
    }

    /// Mirrors one entry to IndexedDB. Fire-and-forget: the in-memory state is
    /// already authoritative for this process, and a failed put costs a
    /// recompute on the next page load.
    fn put_in_background(&self, key: &[u8], value: Bytes) {
        let record_key = format!("{}{}", self.prefix, to_hex(key));

        wasm_bindgen_futures::spawn_local(async move {
            if let Err(err) = put(&record_key, &value).await {
                log::warn!("cubecl cache: browser storage put('{record_key}') failed: {err:?}");
            }
        });
    }
}

impl Storage for BrowserStorage {
    fn get(&self, key: &[u8]) -> Option<Bytes> {
        entries::get(&self.state.lock().entries, key)
    }

    fn insert(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
        let result = entries::insert(&mut self.state.lock().entries, key, value.clone(), origin);

        // Only mirror what the arbitration actually accepted; a declined write
        // must not overwrite the stored record.
        if matches!(result, Insertion::Stored) {
            self.put_in_background(key, value);
        }

        result
    }

    fn replace(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
        let result = entries::replace(&mut self.state.lock().entries, key, value.clone(), origin);

        self.put_in_background(key, value);

        result
    }

    fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8])) {
        entries::scan(&self.state.lock().entries, visit)
    }

    fn loading(&self) -> bool {
        !self.state.lock().loaded
    }

    fn describe(&self) -> String {
        format!("browser storage (indexeddb: {}/{}*)", DB_NAME, self.prefix)
    }
}

fn to_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        hex.push(char::from_digit((byte >> 4) as u32, 16).expect("Nibble is a hex digit"));
        hex.push(char::from_digit((byte & 0xf) as u32, 16).expect("Nibble is a hex digit"));
    }
    hex
}

fn from_hex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }

    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let mut chars = hex.chars();
    while let (Some(high), Some(low)) = (chars.next(), chars.next()) {
        let high = high.to_digit(16)?;
        let low = low.to_digit(16)?;
        bytes.push((high * 16 + low) as u8);
    }

    Some(bytes)
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

/// Loads every record under `prefix` into the mirrored state.
async fn load(prefix: &str, state: &spin::Mutex<State>) -> Result<(), JsValue> {
    let db = open_db().await?;

    let transaction = db.transaction_with_str(STORE_NAME)?;
    let store = transaction.object_store(STORE_NAME)?;

    // All keys starting with the prefix: [prefix, prefix + U+10FFFF).
    let upper = format!("{prefix}\u{10FFFF}");
    let range = IdbKeyRange::bound(&JsValue::from_str(prefix), &JsValue::from_str(&upper))?;

    // Both requests report entries in record key order, so the two arrays line
    // up index by index.
    let keys: js_sys::Array = request_result(&store.get_all_keys_with_key(&range)?)
        .await?
        .dyn_into()?;
    let values: js_sys::Array = request_result(&store.get_all_with_key(&range)?)
        .await?
        .dyn_into()?;

    let mut entries = HashMap::new();
    for (key, value) in keys.iter().zip(values.iter()) {
        let Some(record_key) = key.as_string() else {
            log::warn!("cubecl cache: unexpected browser storage record key: {key:?}");
            continue;
        };
        let Some(key) = record_key.strip_prefix(prefix).and_then(from_hex) else {
            log::warn!("cubecl cache: unreadable browser storage record key '{record_key}'");
            continue;
        };

        match value.dyn_into::<Uint8Array>() {
            Ok(bytes) => {
                // Everything already durable is treated as local: an import
                // that reached storage is indistinguishable from a local
                // computation once the process restarts.
                entries.insert(key, (Bytes::from_bytes_vec(bytes.to_vec()), Origin::Local));
            }
            Err(value) => {
                log::warn!("cubecl cache: unexpected browser storage record type: {value:?}");
            }
        }
    }

    if !entries.is_empty() {
        // Entries inserted while the load was in flight are the fresher ones.
        let mut state = state.lock();
        for (key, value) in entries {
            state.entries.entry(key).or_insert(value);
        }
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
