use super::AllocationProperties;
use crate::bytes::{
    AllocationController,
    default_controller::{MAX_ALIGN, NativeAllocationController},
};
use core::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
};
use std::{
    io::{Read, Seek, SeekFrom},
    sync::{Arc, Mutex},
};

pub(crate) struct FileAllocationController {
    file: Arc<Mutex<std::fs::File>>,
    size: u64,
    offset: u64,
    controller: UnsafeCell<Option<Box<dyn AllocationController>>>,
    init: AtomicBool,
}

unsafe impl Send for FileAllocationController {}
unsafe impl Sync for FileAllocationController {}

impl FileAllocationController {
    pub fn new(file: Arc<Mutex<std::fs::File>>, size: u64, offset: u64) -> Self {
        Self {
            file,
            size,
            controller: UnsafeCell::new(None),
            offset,
            init: false.into(),
        }
    }
    fn init(&self) {
        if self.init.load(Ordering::Relaxed) {
            return;
        }

        let mut file = self.file.lock().unwrap();
        let mut buf = vec![0u8; self.size as usize];
        file.seek(SeekFrom::Start(self.offset)).unwrap();
        file.read(&mut buf).unwrap();

        let controller = NativeAllocationController::from_elems(buf);
        unsafe {
            *self.controller.get() = Some(Box::new(controller));
        };
        self.init.store(true, Ordering::Relaxed);

        core::mem::drop(file);
    }
}

impl AllocationController for FileAllocationController {
    fn alloc_align(&self) -> usize {
        MAX_ALIGN
    }

    fn properties(&self) -> AllocationProperties {
        AllocationProperties::File
    }

    unsafe fn memory_mut(&mut self) -> &mut [core::mem::MaybeUninit<u8>] {
        self.init();

        unsafe {
            let controller = self.controller.get();
            let option: &mut Option<Box<dyn AllocationController>> = controller.as_mut().unwrap();

            match option {
                Some(o) => o.memory_mut(),
                None => unreachable!(),
            }
        }
    }

    fn memory(&self) -> &[core::mem::MaybeUninit<u8>] {
        println!("Memory");
        self.init();

        unsafe {
            let controller = self.controller.get();
            let option: &Option<Box<dyn AllocationController>> = controller.as_ref().unwrap();

            match option {
                Some(o) => o.memory(),
                None => unreachable!(),
            }
        }
    }

    unsafe fn read_into(self: Box<Self>, buf: &mut [u8]) {
        if self.init.load(Ordering::Relaxed) {
            let len = buf.len();
            let memory = self.memory();
            let memory_slice = &memory[0..len];

            // SAFETY: By construction, bytes up to len are initialized.
            let data = unsafe {
                core::slice::from_raw_parts(memory_slice.as_ptr().cast(), memory_slice.len())
            };
            buf.copy_from_slice(data);
            return;
        }

        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(self.offset)).unwrap();
        file.read(buf).unwrap();
    }
}
