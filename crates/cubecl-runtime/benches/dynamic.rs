use std::collections::LinkedList;

use cubecl_runtime::{
    memory_management::{MemoryConfiguration, MemoryDeviceProperties, MemoryManagement},
    storage::BytesStorage,
};

const MB: u64 = 1024 * 1024;

fn main() {
    let start = std::time::Instant::now();
    let storage = BytesStorage::default();
    let config = MemoryConfiguration::default();
    let mem_props = MemoryDeviceProperties {
        max_page_size: 2048 * MB,
        alignment: 32,
    };
    let mut mm = MemoryManagement::from_configuration(storage, &mem_props, config);
    let mut handles = LinkedList::new();
    for _ in 0..100 * 2048 {
        if handles.len() >= 4000 {
            handles.pop_front();
        }
        let handle = mm.reserve(MB, None);
        handles.push_back(handle);
    }
    println!("{:?}", start.elapsed());
}
