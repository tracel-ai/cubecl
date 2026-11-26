pub(crate) use cubecl_runtime::memory_management::*;

#[cfg(test)]
mod counting_storage_allocator;
mod direct_slice_reserver;
mod exclusive_slice_reserver;
mod page_mapping;
mod page_suballocator;
mod paged_slice_reserver;
mod persistent_slice_reserver;
mod slice;
mod slice_allocator;
mod slice_binder;
mod slice_reserver;

#[cfg(test)]
pub(crate) use counting_storage_allocator::*;
pub(crate) use direct_slice_reserver::*;
pub(crate) use exclusive_slice_reserver::*;
pub(crate) use page_mapping::*;
pub(crate) use page_suballocator::*;
pub(crate) use paged_slice_reserver::*;
pub(crate) use persistent_slice_reserver::*;
pub(crate) use slice::*;
pub(crate) use slice_allocator::*;
pub(crate) use slice_binder::*;
pub(crate) use slice_reserver::*;
