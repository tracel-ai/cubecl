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

/// The allocation is managed on a file.
///
/// # Safety
///
/// This implementation uses an [UnsafeCell] to copy the content of a file into an in-memory buffer
/// using the [NativeAllocationController]. It is safe because the controller can't be cloned or
/// sync between multiple threads. You can duplicate the file allocator, but every version of it
/// will have its own buffer.
///
/// # Notes
///
/// Because of that mechanism, dereferencing [crate::bytes::Bytes] isn't cost-free when using the
/// file allocator, since it's going to trigger a copy from the file system to an in-memory buffer.
pub(crate) struct FileAllocationController {
    file: Arc<PathBuf>,
    size: u64,
    offset: u64,
    controller: UnsafeCell<Option<Box<dyn AllocationController>>>,
    init: AtomicBool,
}

impl FileAllocationController {
    pub fn new<P: Into<PathBuf>>(file: P, size: u64, offset: u64) -> Self {
        Self::from_path_buf(Arc::new(file.into()), size, offset)
    }

    fn from_path_buf(file: Arc<PathBuf>, size: u64, offset: u64) -> Self {
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

        let mut file = File::open(self.file.as_ref()).unwrap();
        let mut buf = vec![0u8; self.size as usize];
        file.seek(SeekFrom::Start(self.offset)).unwrap();
        file.read_exact(&mut buf).unwrap();

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
        // Use `<` (not `<=`) to allow boundary splits where one side is empty.
        // This is symmetric: both offset==0 (empty left) and offset==size (empty right) are valid.
        if self.size < offset as u64 {
            return Err(super::SplitError::InvalidOffset);
        }

        let left =
            FileAllocationController::from_path_buf(self.file.clone(), offset as u64, self.offset);
        let right = FileAllocationController::from_path_buf(
            self.file.clone(),
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
        file.read_exact(buf).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::super::Bytes;
    use std::{io::Write, path::PathBuf};

    #[test]
    fn test_from_file() {
        let elems = (0..250).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);

        assert_eq!(&bytes, &bytes_file);
        core::mem::drop(dir);
    }

    #[test]
    fn test_split_file() {
        let elems = (0..250).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);
        let offset = 40;
        let (left, right) = bytes_file.split(offset).unwrap();

        let left_expected: &[u8] = &bytes[0..offset];
        let right_expected: &[u8] = &bytes[offset..];
        let left_actual: &[u8] = &left;
        let right_actual: &[u8] = &right;

        assert_eq!(left_expected, left_actual);
        assert_eq!(right_expected, right_actual);
        core::mem::drop(dir);
    }

    #[test]
    fn test_split_file_at_zero() {
        // Boundary case: split at 0 creates empty left, full right
        let elems: Vec<u8> = (0..100).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);
        let (left, right) = bytes_file.split(0).unwrap();

        assert_eq!(left.len(), 0);
        assert_eq!(&right[..], &bytes[..]);
        core::mem::drop(dir);
    }

    #[test]
    fn test_split_file_at_len() {
        // Boundary case: split at len creates full left, empty right
        let elems: Vec<u8> = (0..100).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);
        let len = bytes_file.len();
        let (left, right) = bytes_file.split(len).unwrap();

        assert_eq!(&left[..], &bytes[..]);
        assert_eq!(right.len(), 0);
        core::mem::drop(dir);
    }

    #[test]
    fn test_memory_mut_on_duplicated_file() {
        let elems = (0..250).collect();
        let (path, bytes, dir) = with_data(elems);

        let bytes_file = Bytes::from_file(&path, bytes.len() as u64, 0);

        let mut bytes_mut = bytes_file.clone();
        bytes_mut[0] = 5;

        let expected: &[u8] = &bytes[1..];
        let actual: &[u8] = &bytes_mut[1..];

        assert_eq!(&bytes, &bytes_file);
        assert_eq!(bytes_mut[0], 5);
        assert_eq!(expected, actual);
        core::mem::drop(dir);
    }

    fn with_data(elems: Vec<u8>) -> (PathBuf, Bytes, TempDir) {
        let bytes = Bytes::from_bytes_vec(elems);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        (path, bytes, dir)
    }
}
