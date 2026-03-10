use std::{collections::HashMap, sync::OnceLock};

use cubecl_core::{device::DeviceId, stub::Mutex};

#[derive(Debug, Hash, Eq, PartialEq)]
pub(crate) struct CudaCommId {
    pub str_id: String,
}

// TODO: Make sure that the peer ids are sorted.
impl From<Vec<DeviceId>> for CudaCommId {
    fn from(value: Vec<DeviceId>) -> Self {
        CudaCommId {
            str_id: value
                .iter()
                .map(|id| id.index_id.to_string())
                .collect::<Vec<String>>()
                .join(","),
        }
    }
}

/// Global state map from [`CudaCommId`] to boxed [`ncclUniqueId`].
static UNIQUE_IDS_MAP: OnceLock<Mutex<HashMap<CudaCommId, cudarc::nccl::sys::ncclUniqueId>>> =
    OnceLock::new();

pub(crate) fn get_nccl_comm_id(device_ids: Vec<DeviceId>) -> cudarc::nccl::sys::ncclUniqueId {
    let mut unique_ids_map = UNIQUE_IDS_MAP.get_or_init(Default::default).lock().unwrap();
    let comm_id = CudaCommId::from(device_ids);
    match unique_ids_map.get_mut(&comm_id) {
        Some(id) => id.clone(),
        None => {
            let id = cudarc::nccl::result::get_uniqueid().unwrap();
            unique_ids_map.insert(comm_id, id);
            id
        }
    }
}
