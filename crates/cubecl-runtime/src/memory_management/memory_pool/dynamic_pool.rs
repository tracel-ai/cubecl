use crate::{
    memory_management::{BytesFormat, SliceHandle, SliceId},
    storage::{StorageHandle, StorageUtilization},
};
use core::fmt::{Debug, Display};
use hashbrown::HashMap;

/// A memory page is responsable to reserve slices of data based on a fixed storage buffer.
pub struct MemoryPage {
    storage: StorageHandle,
    slices: Vec<PageSlice>,
    slices_map: HashMap<SliceId, usize>,
}

impl MemoryPage {
    /// Creates a new memory page with the given storage and memory alignment.
    pub fn new(storage: StorageHandle) -> Self {
        let mut this = MemoryPage {
            storage: storage.clone(),
            slices: Vec::new(),
            slices_map: HashMap::new(),
        };

        let page = PageSlice {
            handle: SliceHandle::new(),
            storage,
        };
        let id = *page.handle.id();
        let index = 0;
        this.slices.insert(index, page);
        this.slices_map.insert(id, index);

        this
    }

    /// Gets the [summary](MemoryPageSummary) of the current memory page.
    pub fn summary(&self) -> MemoryPageSummary {
        let mut summary = MemoryPageSummary::default();

        for slice in self.slices.iter() {
            let is_free = slice.handle.is_free();
            if is_free {
                summary.amount_free += slice.storage.size();
                summary.num_free += 1;
            } else {
                summary.amount_full += slice.storage.size();
                summary.num_full += 1;
            }
            summary.blocks.push(MemoryBlock {
                is_free,
                size: slice.storage.size(),
            });
        }
        summary.amount_total = self.storage.size();
        summary.num_total = self.slices.len();

        summary
    }

    /// Reserves a slice of the given size if there is enough place in the page.
    pub fn reserve(&mut self, size: u64) -> Option<SliceHandle> {
        for (index, slice) in self.slices.iter().enumerate() {
            let can_use_slice = slice.storage.utilization.size >= size && slice.handle.is_free();
            if !can_use_slice {
                continue;
            }

            let can_be_splitted = slice.storage.utilization.size > size;
            let handle = slice.handle.clone();

            if can_be_splitted {
                let new_slice = PageSlice {
                    handle: SliceHandle::new(),
                    storage: slice.storage.offset_start(size),
                };
                self.add_new_slice(index, size, new_slice);
            }

            return Some(handle);
        }

        None
    }

    /// Gets the [storage handle](SliceHandle) with the correct offset and size using the slice
    /// binding.
    ///
    /// If the handle isn't returned, it means the binding isn't present in the given page.
    pub fn get(&self, binding: &super::SliceBinding) -> Option<&StorageHandle> {
        let index = self.slices_map.get(binding.id())?;
        self.slices.get(*index).map(|slice| &slice.storage)
    }

    /// Cleanups the current memory page by making sure adjacent slices are merged together into a
    /// single slice.
    ///
    /// This is necessary to allow bigger slices to be reserved on the current page.
    pub fn cleanup(&mut self) {
        let mut job = self.memory_job();
        let mut tasks = job.tasks.drain(..);

        let mut task = match tasks.next() {
            Some(task) => Some(task),
            None => return,
        };

        let mut slices_updated = Vec::with_capacity(self.slices.len() + 1);
        let mut slices_map_updated = HashMap::with_capacity(self.slices_map.len() + 1);

        let mut offset = 0;
        let mut size = 0;
        let mut index_current = 0;

        for (index, slice) in self.slices.drain(..).enumerate() {
            let status = match &mut task {
                Some(task) => task.register(index),
                None => MemoryTaskStatus::Ingoring,
            };

            match status {
                MemoryTaskStatus::StartMerging => {
                    offset = slice.storage.utilization.offset;
                    size = slice.storage.size();
                }
                MemoryTaskStatus::Merging => {
                    size += slice.storage.size();
                }
                MemoryTaskStatus::Ingoring => {
                    let id = *slice.handle.id();
                    slices_updated.push(slice);
                    slices_map_updated.insert(id, index_current);
                    index_current += 1;
                }
                MemoryTaskStatus::Completed => {
                    size += slice.storage.size();

                    let mut storage = self.storage.clone();
                    storage.utilization = StorageUtilization { offset, size };
                    let page = PageSlice {
                        handle: SliceHandle::new(),
                        storage,
                    };
                    let id = *page.handle.id();
                    slices_updated.push(page);
                    slices_map_updated.insert(id, index_current);
                    index_current += 1;
                    task = tasks.next();
                }
            };
        }

        self.slices = slices_updated;
        self.slices_map = slices_map_updated;
    }

    fn add_new_slice(
        &mut self,
        index_previous: usize,
        reserved_size_previous: u64,
        new_slice: PageSlice,
    ) {
        let mut slices_updated = Vec::with_capacity(self.slices.len() + 1);
        let mut slices_map_updated = HashMap::with_capacity(self.slices_map.len() + 1);

        let new_id = *new_slice.handle.id();
        let mut new_slice = Some(new_slice);

        let mut index_current = 0;
        for mut slice in self.slices.drain(..) {
            if index_current == index_previous {
                slice.storage.utilization.size = reserved_size_previous;
                let id = *slice.handle.id();
                slices_updated.push(slice);
                slices_map_updated.insert(id, index_current);
                index_current += 1;

                // New slice
                slices_updated.push(new_slice.take().unwrap());
                slices_map_updated.insert(new_id, index_current);
                index_current += 1;
            } else {
                let id = *slice.handle.id();
                slices_updated.push(slice);
                slices_map_updated.insert(id, index_current);
                index_current += 1;
            }
        }

        self.slices = slices_updated;
        self.slices_map = slices_map_updated;
    }

    fn memory_job(&self) -> MemoryJob {
        let mut job = MemoryJob::default();
        let mut task = MemoryTask::default();

        for (index, slice) in self.slices.iter().enumerate() {
            if slice.handle.is_free() {
                task.size += slice.storage.size();
                task.indexes.push(index);
            } else {
                task = job.add(task);
            }
        }
        job.add(task);

        job
    }
}

#[derive(Debug, PartialEq, Eq)]
struct MemoryBlock {
    is_free: bool,
    size: u64,
}

#[derive(Default, PartialEq, Eq)]
struct MemoryPageSummary {
    blocks: Vec<MemoryBlock>,
    amount_free: u64,
    amount_full: u64,
    amount_total: u64,
    num_free: usize,
    num_full: usize,
    num_total: usize,
}

impl Display for MemoryBlock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.is_free {
            true => f.write_fmt(format_args!("Free ({})", BytesFormat::new(self.size))),
            false => f.write_fmt(format_args!("Taken ({})", BytesFormat::new(self.size))),
        }
    }
}
impl Display for MemoryPageSummary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl Debug for MemoryPageSummary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n==== Memory Page Summary ====\n")?;
        f.write_str("[Info]\n")?;

        for (tag, num, amount) in [
            ("Free ", self.num_free, self.amount_free),
            ("Full ", self.num_full, self.amount_full),
            ("Total", self.num_total, self.amount_total),
        ] {
            f.write_fmt(format_args!(
                " - {tag}: {} slices ({})\n",
                num,
                BytesFormat::new(amount),
            ))?;
        }

        f.write_str("\n[Blocks]\n")?;
        let mut blocks = String::new();
        for (i, b) in self.blocks.iter().enumerate() {
            if i == 0 {
                blocks += "|";
            }
            blocks += format!(" {b} |").as_str();
        }
        let size = blocks.len();
        for _ in 0..size {
            f.write_str("-")?;
        }
        f.write_str("\n")?;
        f.write_str(&blocks)?;
        f.write_str("\n")?;
        for _ in 0..size {
            f.write_str("-")?;
        }

        f.write_str("\n=============================")?;
        f.write_str("\n")
    }
}

impl Display for MemoryPage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.summary()))
    }
}

struct PageSlice {
    handle: SliceHandle,
    storage: StorageHandle,
}

#[derive(Default, Debug, PartialEq, Eq)]
struct MemoryTask {
    indexes: Vec<usize>,
    cursor: usize,
    size: u64,
}

#[derive(Default, Debug, PartialEq, Eq)]
struct MemoryJob {
    tasks: Vec<MemoryTask>,
}

impl MemoryJob {
    fn add(&mut self, mut task: MemoryTask) -> MemoryTask {
        if task.indexes.len() < 2 {
            return task;
        }

        let mut returned = MemoryTask::default();
        core::mem::swap(&mut task, &mut returned);
        self.tasks.push(returned);
        task
    }
}

#[derive(Debug)]
enum MemoryTaskStatus {
    Merging,
    StartMerging,
    Ingoring,
    Completed,
}

impl MemoryTask {
    fn register(&mut self, index: usize) -> MemoryTaskStatus {
        let index_current = self.indexes[self.cursor];

        if index_current == index {
            self.cursor += 1;
            if self.cursor == 1 {
                return MemoryTaskStatus::StartMerging;
            }

            if self.cursor == self.indexes.len() {
                return MemoryTaskStatus::Completed;
            } else {
                return MemoryTaskStatus::Merging;
            }
        }

        MemoryTaskStatus::Ingoring
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::{StorageId, StorageUtilization};

    use super::*;

    const MB: u64 = 1024 * 1024;

    #[test]
    fn test_memory_page() {
        let mut page = new_memory_page(32 * MB);
        let slice = page
            .reserve(16 * MB)
            .expect("Enough space to allocate a new slice");

        assert_eq!(slice.is_free(), false);
        assert_eq!(slice.can_mut(), true);

        let storage = page
            .get(&slice.binding())
            .expect("To find the correct storage");

        assert_eq!(
            storage.utilization,
            StorageUtilization {
                offset: 0,
                size: 16 * MB
            },
            "Utilization to be correct"
        );

        let summary = page.summary();

        assert_eq!(
            summary,
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: true,
                        size: 16 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 16 * MB
                    }
                ],
                amount_free: 32 * MB,
                amount_full: 0,
                amount_total: 32 * MB,
                num_free: 2,
                num_full: 0,
                num_total: 2
            },
            "Summary is correct before cleanup",
        );
        page.cleanup();
        let summary = page.summary();

        assert_eq!(
            summary,
            MemoryPageSummary {
                blocks: vec![MemoryBlock {
                    is_free: true,
                    size: 32 * MB
                },],
                amount_free: 32 * MB,
                amount_full: 0,
                amount_total: 32 * MB,
                num_free: 1,
                num_full: 0,
                num_total: 1
            },
            "Summary is correct after cleanup",
        );
    }

    #[test]
    fn test_memory_job() {
        let mut page = new_memory_page(32 * MB);
        let slice = page
            .reserve(16 * MB)
            .expect("Enough space to allocate a new slice");

        core::mem::drop(slice);
        let job = page.memory_job();

        assert_eq!(
            job,
            MemoryJob {
                tasks: vec![MemoryTask {
                    indexes: vec![0, 1],
                    cursor: 0,
                    size: 32 * MB,
                }]
            }
        );
    }

    #[test]
    fn test_scenario() {
        let mut page = new_memory_page(32 * MB);

        let slice_1 = page
            .reserve(4 * MB)
            .expect("Enough space to allocate a new slice");
        let slice_2 = page
            .reserve(15 * MB)
            .expect("Enough space to allocate a new slice");
        let slice_3 = page
            .reserve(8 * MB)
            .expect("Enough space to allocate a new slice");
        let slice_4 = page
            .reserve(4 * MB)
            .expect("Enough space to allocate a new slice");

        assert_eq!(
            page.summary(),
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 15 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 8 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 1 * MB
                    }
                ],
                amount_free: 1 * MB,
                amount_full: 31 * MB,
                amount_total: 32 * MB,
                num_free: 1,
                num_full: 4,
                num_total: 5
            },
        );

        let slice_5 = page.reserve(8 * MB);
        assert!(slice_5.is_none(), "No more place");

        core::mem::drop(slice_2);
        let slice_5 = page.reserve(9 * MB);
        assert!(slice_5.is_some(), "Now we have more place");

        let slice_6 = page.reserve(9 * MB);
        assert!(slice_6.is_none(), "No more place");

        core::mem::drop(slice_3);
        let slice_6 = page.reserve(9 * MB);
        assert!(slice_6.is_none(), "No more place");

        page.cleanup();

        assert_eq!(
            page.summary(),
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 9 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 14 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 1 * MB
                    }
                ],
                amount_free: 15 * MB,
                amount_full: 17 * MB,
                amount_total: 32 * MB,
                num_free: 2,
                num_full: 3,
                num_total: 5
            },
        );

        let slice_6 = page.reserve(9 * MB);
        assert!(slice_6.is_some(), "Now we have more place");
        core::mem::drop(slice_1);
        core::mem::drop(slice_4);

        page.cleanup();

        assert_eq!(
            page.summary(),
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: false,
                        size: 9 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 9 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 14 * MB
                    }
                ],
                amount_free: 14 * MB,
                amount_full: 18 * MB,
                amount_total: 32 * MB,
                num_free: 1,
                num_full: 2,
                num_total: 3
            },
        );
    }

    fn new_memory_page(size: u64) -> MemoryPage {
        let storage = StorageHandle::new(StorageId::new(), StorageUtilization { offset: 0, size });

        MemoryPage::new(storage)
    }
}
