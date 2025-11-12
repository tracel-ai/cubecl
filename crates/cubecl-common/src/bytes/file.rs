use super::AllocationProperty;
use crate::bytes::{
    AllocationController,
    default_controller::{MAX_ALIGN, NativeAllocationController},
};
use core::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::Arc,
};

pub(crate) struct FileAllocationController {
    file: Arc<PathBuf>,
    size: u64,
    offset: u64,
    controller: UnsafeCell<Option<Box<dyn AllocationController>>>,
    init: AtomicBool,
}

unsafe impl Send for FileAllocationController {}
unsafe impl Sync for FileAllocationController {}

impl FileAllocationController {
    pub fn new<P: Into<PathBuf>>(file: P, size: u64, offset: u64) -> Self {
        Self {
            file: Arc::new(file.into()),
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

        let mut file = File::open(self.file.as_ref()).unwrap();
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

    fn split(
        &mut self,
        offset: usize,
    ) -> Result<(Box<dyn AllocationController>, Box<dyn AllocationController>), super::SplitError>
    {
        if self.size <= offset as u64 {
            return Err(super::SplitError::InvalidOffset);
        }

        let left = FileAllocationController::new(self.file.as_ref(), offset as u64, self.offset);
        let right = FileAllocationController::new(
            self.file.as_ref(),
            self.size - offset as u64,
            self.offset + offset as u64,
        );

        Ok((Box::new(left), Box::new(right)))
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::File
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

    fn duplicate(&self) -> Option<Box<dyn AllocationController>> {
        if self.init.load(Ordering::Relaxed) {
            return None;
        }

        let controller = Self::new(self.file.as_ref(), self.size, self.offset);
        Some(Box::new(controller))
    }

    unsafe fn copy_into(&self, buf: &mut [u8]) {
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

        let mut file = File::open(self.file.as_ref()).unwrap();
        file.seek(SeekFrom::Start(self.offset)).unwrap();
        file.read(buf).unwrap();
    }
}
