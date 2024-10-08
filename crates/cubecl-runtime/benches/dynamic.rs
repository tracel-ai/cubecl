use std::collections::LinkedList;

use cubecl_runtime::{
    memory_management::{
        dynamic::DynamicMemoryManagement, MemoryConfiguration, MemoryDeviceProperties,
        MemoryManagement,
    },
    storage::BytesStorage,
};

const MB: usize = 1024 * 1024;

fn main() {
    let start = std::time::Instant::now();
    let storage = BytesStorage::default();
    let config = MemoryConfiguration::Default;
    let mem_props = MemoryDeviceProperties {
        max_page_size: 2048 * MB,
        alignment: 32,
    };
    let mut mm = DynamicMemoryManagement::from_configuration(storage, mem_props, config);
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
