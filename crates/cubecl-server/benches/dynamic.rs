use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::{
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, MemoryConfiguration, MemoryManagement, MemoryManagementOptions,
    },
    storage::BytesStorage,
};
use std::{collections::LinkedList, sync::Arc};

const MB: u64 = 1024 * 1024;

fn main() {
    let start = std::time::Instant::now();
    let storage = BytesStorage::default();
    let config = MemoryConfiguration::default();
    let mem_props = MemoryDeviceProperties::new(2048 * MB, 32);
    let logger = Arc::new(ServerLogger::default());
    let mut mm = MemoryManagement::from_configuration(
        storage,
        &mem_props,
        config,
        logger,
        MemoryManagementOptions::new("test"),
    );
    let mut handles = LinkedList::new();
    for _ in 0..100 * 2048 {
        if handles.len() >= 4000 {
            handles.pop_front();
        }
        let handle = mm.reserve(MB, PageUpdate::Allow, &mut ErrorGraph::default());
        handles.push_back(handle);
    }
    println!("{:?}", start.elapsed());
}
