//! What the HSA metadata of an AMDGPU code object reports about its kernels.

use cubecl_runtime::compiler::RegisterSpill;

/// The ELF section type of notes.
const SHT_NOTE: u32 = 7;
/// The note type of the HSA metadata, a `MessagePack` map.
const NT_AMDGPU_METADATA: u32 = 32;
/// The owner name of the HSA metadata note.
const AMDGPU_NOTE_NAME: &[u8] = b"AMDGPU\0";
/// The size of a note's header: its name size, descriptor size and type.
const NOTE_HEADER_BYTES: usize = 12;
/// The alignment of a note's name and descriptor.
const NOTE_ALIGNMENT_BYTES: usize = 4;

/// The registers the kernels of `code_object` spill, as its HSA metadata reports them: the
/// vector registers they spill to scratch memory together, and the most scratch memory one of
/// them reserves per unit. Scalar registers spill into the lanes of vector registers rather than
/// into memory, so they are not counted.
///
/// `None` when `code_object` is no 64-bit ELF, or holds no HSA metadata that counts spills.
pub fn register_spill(code_object: &[u8]) -> Option<RegisterSpill> {
    let metadata = metadata(code_object)?;
    let spilled = values(metadata, b".vgpr_spill_count").reduce(u64::saturating_add)?;
    let scratch_bytes = values(metadata, b".private_segment_fixed_size")
        .max()
        .unwrap_or(0);

    Some(RegisterSpill {
        spilled_registers: u32::try_from(spilled).unwrap_or(u32::MAX),
        scratch_bytes,
    })
}

/// The HSA metadata of the 64-bit ELF `elf`: the descriptor of its `AMDGPU` metadata note.
fn metadata(elf: &[u8]) -> Option<&[u8]> {
    if elf.get(..5)? != b"\x7fELF\x02" {
        return None;
    }
    let section_headers = usize::try_from(read_u64(elf, 0x28)?).ok()?;
    let header_bytes = usize::from(read_u16(elf, 0x3a)?);
    let sections = usize::from(read_u16(elf, 0x3c)?);

    (0..sections)
        .filter_map(|index| elf.get(section_headers + index * header_bytes..))
        .filter(|header| read_u32(header, 4) == Some(SHT_NOTE))
        .filter_map(|header| {
            let offset = usize::try_from(read_u64(header, 0x18)?).ok()?;
            let size = usize::try_from(read_u64(header, 0x20)?).ok()?;
            elf.get(offset..offset.checked_add(size)?)
        })
        .find_map(amdgpu_metadata_note)
}

/// The descriptor of the HSA metadata note among `notes`, the contents of one note section.
fn amdgpu_metadata_note(mut notes: &[u8]) -> Option<&[u8]> {
    while notes.len() >= NOTE_HEADER_BYTES {
        let name_bytes = usize::try_from(read_u32(notes, 0)?).ok()?;
        let descriptor_bytes = usize::try_from(read_u32(notes, 4)?).ok()?;
        let name = notes.get(NOTE_HEADER_BYTES..NOTE_HEADER_BYTES + name_bytes)?;
        let descriptor_start =
            NOTE_HEADER_BYTES + name_bytes.next_multiple_of(NOTE_ALIGNMENT_BYTES);
        let descriptor_end = descriptor_start + descriptor_bytes;
        let descriptor = notes.get(descriptor_start..descriptor_end)?;
        if read_u32(notes, 8)? == NT_AMDGPU_METADATA && name == AMDGPU_NOTE_NAME {
            return Some(descriptor);
        }
        notes = notes.get(descriptor_end.next_multiple_of(NOTE_ALIGNMENT_BYTES)..)?;
    }
    None
}

/// The unsigned integer each entry named `key` holds in the `MessagePack` `metadata`, one per
/// kernel. A key no longer than 31 bytes is a `fixstr`, and its value follows it.
fn values<'a>(metadata: &'a [u8], key: &'a [u8]) -> impl Iterator<Item = u64> + 'a {
    let fixstr = 0xa0 | key.len() as u8;
    metadata
        .windows(key.len() + 1)
        .enumerate()
        .filter(move |(_, window)| window[0] == fixstr && &window[1..] == key)
        .filter_map(move |(at, _)| unsigned_at(metadata, at + 1 + key.len()))
}

/// The `MessagePack` unsigned integer that starts at `at` in `bytes`.
fn unsigned_at(bytes: &[u8], at: usize) -> Option<u64> {
    let marker = *bytes.get(at)?;
    let width = match marker {
        0x00..=0x7f => return Some(u64::from(marker)),
        0xcc => 1,
        0xcd => 2,
        0xce => 4,
        0xcf => 8,
        _ => return None,
    };
    let big_endian = bytes.get(at + 1..at + 1 + width)?;
    Some(
        big_endian
            .iter()
            .fold(0, |value, byte| (value << 8) | u64::from(*byte)),
    )
}

/// The little-endian `u16` at `at` in `bytes`.
fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

/// The little-endian `u32` at `at` in `bytes`.
fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// The little-endian `u64` at `at` in `bytes`.
fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_is_no_elf_reports_nothing() {
        assert_eq!(register_spill(b"__CLANG_OFFLOAD_BUNDLE__"), None);
    }

    #[test]
    fn a_wide_integer_reads_big_endian() {
        assert_eq!(unsigned_at(&[0xcd, 0x01, 0x29], 0), Some(297));
    }
}
