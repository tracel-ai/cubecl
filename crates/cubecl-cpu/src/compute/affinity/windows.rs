//! Windows numbers logical processors per *processor group* of at most 64 (a
//! `KAFFINITY` of bits); only a machine with more than 64 has several groups.
//! A [`CoreId`] here is `group * 64 + bit`, one flat numbering across groups,
//! and both pinning and topology go through the group-aware APIs, so such a
//! machine is covered whole rather than through its first group.

use std::{mem, ptr};

use winapi::shared::basetsd::{DWORD_PTR, KAFFINITY};
use winapi::shared::minwindef::DWORD;
use winapi::um::processthreadsapi::{GetCurrentProcess, GetCurrentThread};
use winapi::um::processtopologyapi::{GetThreadGroupAffinity, SetThreadGroupAffinity};
use winapi::um::sysinfoapi::GetLogicalProcessorInformationEx;
use winapi::um::winbase::GetProcessAffinityMask;
use winapi::um::winnt::{
    CacheData, GROUP_AFFINITY, LOGICAL_PROCESSOR_RELATIONSHIP, RelationCache,
    RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};

use super::{CoreId, ThreadAffinity};

/// One record of `GetLogicalProcessorInformationEx`.
type Record = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;

/// `CoreId`s per processor group: a group holds one `KAFFINITY` of logical
/// processors (`MAXIMUM_PROC_PER_GROUP`, 64 on 64-bit Windows).
const GROUP_STRIDE: usize = KAFFINITY::BITS as usize;

pub(super) struct Platform;

impl ThreadAffinity for Platform {
    fn active_cpus() -> Vec<CoreId> {
        let mut cpus: Vec<_> = match process_affinity() {
            Some(affinity) => cpus_of(&affinity).collect(),
            // Every processor of every group, since each belongs to one core.
            None => core_masks().iter().flat_map(cpus_of).collect(),
        };
        cpus.sort_unstable();
        if cpus.is_empty() {
            // Nothing could be read; std's count, numbered from 0, keeps the
            // pool from ending up with no core to put a worker on.
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            cpus = (0..cores).map(CoreId).collect();
        }
        cpus
    }

    fn physical_core(cpu: CoreId) -> Option<CoreId> {
        core_masks()
            .iter()
            .find(|siblings| cpus_of(siblings).any(|sibling| sibling == cpu))
            .and_then(|siblings| cpus_of(siblings).next())
    }

    fn l1d_cache_size() -> Option<usize> {
        // One record per cache instance; the smallest L1d, since hybrid parts
        // give their efficiency cores less and a stage sized to that stays
        // resident on whichever core runs it.
        records(RelationCache)
            .iter()
            .map(|record| unsafe { record.u.Cache() })
            .filter(|cache| cache.Level == 1 && cache.Type == CacheData && cache.CacheSize > 0)
            .map(|cache| cache.CacheSize as usize)
            .min()
    }

    fn pin_current(cpu: CoreId) {
        let affinity = GROUP_AFFINITY {
            Mask: 1 << (cpu.0 % GROUP_STRIDE),
            Group: (cpu.0 / GROUP_STRIDE) as u16,
            Reserved: [0; 3],
        };
        unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, ptr::null_mut()) };
    }
}

/// The SMT siblings of every physical core, one group affinity each.
fn core_masks() -> Vec<GROUP_AFFINITY> {
    records(RelationProcessorCore)
        .iter()
        // A core's logical processors always share one group, so Windows
        // documents `GroupCount` as 1 and `GroupMask[0]` is the whole core.
        .map(|record| unsafe { record.u.Processor() }.GroupMask[0])
        .collect()
}

/// The affinity of the calling process, or `None` once its threads span
/// several groups, the default past 64 logical processors, which reports an
/// all-zero mask and may run on every processor of every group.
fn process_affinity() -> Option<GROUP_AFFINITY> {
    let (mut process, mut system): (DWORD_PTR, DWORD_PTR) = (0, 0);
    let ok = unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) };
    if ok == 0 || process == 0 {
        return None;
    }
    // A non-zero mask means every thread is in one group, the caller's, whose
    // number is all this reads back; the process mask is the one to keep.
    let mut affinity: GROUP_AFFINITY = unsafe { mem::zeroed() };
    let ok = unsafe { GetThreadGroupAffinity(GetCurrentThread(), &mut affinity) };
    affinity.Mask = process;
    (ok != 0).then_some(affinity)
}

/// The CPUs a group affinity selects, lowest first.
fn cpus_of(affinity: &GROUP_AFFINITY) -> impl Iterator<Item = CoreId> {
    let (mask, group) = (affinity.Mask, affinity.Group as usize);
    (0..GROUP_STRIDE)
        .filter(move |bit| mask >> bit & 1 == 1)
        .map(move |bit| CoreId(group * GROUP_STRIDE + bit))
}

/// The records `GetLogicalProcessorInformationEx` returns for `relationship`,
/// or none when the query fails.
fn records(relationship: LOGICAL_PROCESSOR_RELATIONSHIP) -> Vec<Record> {
    // Without a buffer the call fails and reports the length needed. The
    // buffer is `u64`s so the records land at the alignment they are laid out
    // with, the parser reading their fields back out of the bytes.
    let mut len: DWORD = 0;
    unsafe { GetLogicalProcessorInformationEx(relationship, ptr::null_mut(), &mut len) };
    let mut buffer = vec![0u64; (len as usize).div_ceil(mem::size_of::<u64>())];
    let ok = unsafe {
        GetLogicalProcessorInformationEx(relationship, buffer.as_mut_ptr().cast(), &mut len)
    };
    let bytes = bytemuck::cast_slice::<u64, u8>(&buffer);
    let filled = if ok != 0 { len as usize } else { 0 };
    parse(&bytes[..filled.min(bytes.len())], relationship)
}

/// Splits a query buffer into the records matching `relationship`. Records are
/// variable-length, a header then a body ending in an array, so each is
/// copied into the fixed-size struct, which holds the array's first entry:
/// the whole of a core (one group affinity) or of a cache.
fn parse(bytes: &[u8], relationship: LOGICAL_PROCESSOR_RELATIONSHIP) -> Vec<Record> {
    let header = mem::offset_of!(Record, u);
    let mut records = Vec::new();
    let mut offset = 0;
    while offset + header <= bytes.len() {
        let size = &bytes[offset + mem::offset_of!(Record, Size)..][..mem::size_of::<DWORD>()];
        let size = DWORD::from_ne_bytes(size.try_into().unwrap()) as usize;
        // A record smaller than its own header, or reaching past the buffer,
        // leaves the rest unwalkable; a zero size would never advance either.
        if size < header || offset + size > bytes.len() {
            break;
        }
        let mut record: Record = unsafe { mem::zeroed() };
        unsafe {
            ptr::copy_nonoverlapping(
                bytes[offset..].as_ptr(),
                ptr::from_mut(&mut record).cast::<u8>(),
                size.min(mem::size_of::<Record>()),
            );
        }
        if record.Relationship == relationship {
            records.push(record);
        }
        offset += size;
    }
    records
}

/// What the tests in [`super`] need of this platform beyond the trait.
#[cfg(test)]
impl Platform {
    /// `GetLogicalProcessorInformationEx` reports a real topology here.
    pub(super) const READS_TOPOLOGY: bool = true;

    /// The CPU the calling thread is on, which a pinned thread cannot leave.
    pub(super) fn current_cpu() -> Option<CoreId> {
        let mut number: winapi::um::winnt::PROCESSOR_NUMBER = unsafe { mem::zeroed() };
        unsafe { winapi::um::processthreadsapi::GetCurrentProcessorNumberEx(&mut number) };
        Some(CoreId(
            number.Group as usize * GROUP_STRIDE + number.Number as usize,
        ))
    }
}

#[cfg(test)]
mod tests {
    use winapi::um::winnt::PROCESSOR_RELATIONSHIP;

    use super::*;

    fn put(record: &mut [u8], offset: usize, bytes: &[u8]) {
        record[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    /// A `RelationProcessorCore` record for one core, its SMT siblings the set
    /// bits of `mask` within processor group `group`.
    fn core(group: u16, mask: KAFFINITY) -> Vec<u8> {
        let body = mem::offset_of!(Record, u);
        let masks = body + mem::offset_of!(PROCESSOR_RELATIONSHIP, GroupMask);
        let size = masks + mem::size_of::<GROUP_AFFINITY>();
        let mut record = vec![0u8; size];
        let relationship = RelationProcessorCore.to_ne_bytes();
        put(
            &mut record,
            mem::offset_of!(Record, Relationship),
            &relationship,
        );
        put(
            &mut record,
            mem::offset_of!(Record, Size),
            &(size as u32).to_ne_bytes(),
        );
        let count = body + mem::offset_of!(PROCESSOR_RELATIONSHIP, GroupCount);
        put(&mut record, count, &1u16.to_ne_bytes());
        put(
            &mut record,
            masks + mem::offset_of!(GROUP_AFFINITY, Mask),
            &mask.to_ne_bytes(),
        );
        put(
            &mut record,
            masks + mem::offset_of!(GROUP_AFFINITY, Group),
            &group.to_ne_bytes(),
        );
        record
    }

    /// The multi-group machine the one running this is unlikely to be: two SMT
    /// cores in group 0 and one in group 1, then a tail too short to be a
    /// record. Everything else the platform does is checked live in `super`.
    #[test]
    fn records_split_by_size_and_number_cpus_group_major() {
        let mut bytes = [core(0, 0b0011), core(0, 0b1100), core(1, 0b0101)].concat();
        bytes.extend_from_slice(&core(1, 0b1)[..4]);

        let cores: Vec<Vec<usize>> = parse(&bytes, RelationProcessorCore)
            .iter()
            .map(|record| unsafe { record.u.Processor() }.GroupMask[0])
            .map(|siblings| cpus_of(&siblings).map(|cpu| cpu.0).collect())
            .collect();

        let g = GROUP_STRIDE;
        assert_eq!(cores, [vec![0, 1], vec![2, 3], vec![g, g + 2]]);
        assert!(parse(&bytes, RelationCache).is_empty());
    }
}
