use cubecl_core::{
    bytes::AllocationProperty,
    zspace::{Shape, Strides, striding::try_check_contiguous_row_major_strides},
};

/// How host data should be staged into pinned memory when written to the GPU.
///
/// This captures the branching that [`Command::write_to_gpu`](crate::compute::Command) used to
/// perform inline, so the (purely arithmetic) decision can be unit tested without touching HIP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StagingPolicy {
    /// Stage the data through a single buffer: a small file-backed transfer is materialized into
    /// one pinned buffer, any other backing is written directly.
    Single,
    /// Stream large file-backed data through reused pinned buffers, one [`ChunkCopy`] at a time.
    Chunked(Vec<ChunkCopy>),
}

/// A single pinned-staged copy from a source-file window to a device region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkCopy {
    /// Byte offset of the window within the source file.
    pub src_offset: usize,
    /// Number of bytes copied (also the size of the pinned staging buffer to reserve).
    pub len: usize,
    /// Destination layout in device memory.
    pub dst: ChunkDst,
}

/// The device-side layout of a [`ChunkCopy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChunkDst {
    /// Flat contiguous copy of `len` bytes at device byte `offset`.
    Flat { offset: usize },
    /// Pitched 2D copy of `rows` rows of `width` bytes at device pitch `pitch`, starting at
    /// device byte `offset`. The source rows are packed (source pitch == `width`).
    Pitched {
        offset: usize,
        pitch: usize,
        width: usize,
        rows: usize,
    },
}

impl StagingPolicy {
    /// Decides how `data` of `size` bytes should be staged to the GPU.
    ///
    /// Chunking only applies to file-backed data larger than `chunk_size` whose backing can
    /// produce zero-copy windows (`can_view`); everything else stays [`StagingPolicy::Single`].
    /// Contiguous destinations are streamed as a flat byte range, pitched destinations as
    /// row-groups.
    pub(crate) fn plan(
        property: AllocationProperty,
        size: usize,
        shape: &Shape,
        strides: &Strides,
        elem_size: usize,
        chunk_size: usize,
        can_view: bool,
    ) -> Self {
        let should_chunk =
            matches!(property, AllocationProperty::File) && size > chunk_size && can_view;
        if !should_chunk {
            return StagingPolicy::Single;
        }

        let chunks = if try_check_contiguous_row_major_strides(shape, strides).is_ok() {
            contiguous_chunks(size, chunk_size)
        } else {
            pitched_chunks(shape, strides, elem_size, chunk_size)
        };

        StagingPolicy::Chunked(chunks)
    }
}

/// Splits a contiguous (packed) transfer into flat byte windows of at most `chunk_size`.
fn contiguous_chunks(size: usize, chunk_size: usize) -> Vec<ChunkCopy> {
    let mut chunks = Vec::new();
    let mut off = 0;
    while off < size {
        let len = (size - off).min(chunk_size);
        chunks.push(ChunkCopy {
            src_offset: off,
            len,
            dst: ChunkDst::Flat { offset: off },
        });
        off += len;
    }
    chunks
}

/// Splits a pitched transfer into row-groups whose packed source size is at most `chunk_size`.
///
/// The device rows are padded to `stride_bytes` while the source file packs rows at
/// `width_bytes`, so each group is copied with a 2D transfer.
fn pitched_chunks(
    shape: &Shape,
    strides: &Strides,
    elem_size: usize,
    chunk_size: usize,
) -> Vec<ChunkCopy> {
    let rank = shape.len();
    let width_bytes = shape[rank - 1] * elem_size;
    let height = shape.iter().rev().skip(1).product::<usize>().max(1);
    let stride_bytes = strides[rank - 2] * elem_size;
    // At least one row per chunk, even if a single row exceeds `chunk_size`.
    let rows_per_chunk = (chunk_size / width_bytes.max(1)).max(1);

    let mut chunks = Vec::new();
    let mut r0 = 0;
    while r0 < height {
        let rows = rows_per_chunk.min(height - r0);
        chunks.push(ChunkCopy {
            src_offset: r0 * width_bytes,
            len: rows * width_bytes,
            dst: ChunkDst::Pitched {
                offset: r0 * stride_bytes,
                pitch: stride_bytes,
                width: width_bytes,
                rows,
            },
        });
        r0 += rows;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: usize = 100;

    #[test]
    fn non_file_data_is_single() {
        let policy = StagingPolicy::plan(
            AllocationProperty::Native,
            10_000,
            &[10_000usize].into(),
            &[1usize].into(),
            1,
            CHUNK,
            true,
        );
        assert_eq!(policy, StagingPolicy::Single);
    }

    #[test]
    fn small_file_is_single() {
        let policy = StagingPolicy::plan(
            AllocationProperty::File,
            CHUNK, // not strictly greater than the chunk size
            &[CHUNK].into(),
            &[1usize].into(),
            1,
            CHUNK,
            true,
        );
        assert_eq!(policy, StagingPolicy::Single);
    }

    #[test]
    fn unviewable_file_is_single() {
        let policy = StagingPolicy::plan(
            AllocationProperty::File,
            10_000,
            &[10_000usize].into(),
            &[1usize].into(),
            1,
            CHUNK,
            false,
        );
        assert_eq!(policy, StagingPolicy::Single);
    }

    #[test]
    fn contiguous_file_is_chunked_flat() {
        let size = 250;
        let policy = StagingPolicy::plan(
            AllocationProperty::File,
            size,
            &[size].into(),
            &[1usize].into(),
            1,
            CHUNK,
            true,
        );

        let expected = vec![
            ChunkCopy {
                src_offset: 0,
                len: 100,
                dst: ChunkDst::Flat { offset: 0 },
            },
            ChunkCopy {
                src_offset: 100,
                len: 100,
                dst: ChunkDst::Flat { offset: 100 },
            },
            // Last chunk is the smaller remainder.
            ChunkCopy {
                src_offset: 200,
                len: 50,
                dst: ChunkDst::Flat { offset: 200 },
            },
        ];
        assert_eq!(policy, StagingPolicy::Chunked(expected));
    }

    #[test]
    fn contiguous_multi_dim_is_treated_as_flat() {
        // A packed [3, 200] f32 tensor: contiguous, so chunked as a flat byte stream.
        let elem_size = 4;
        let size = 3 * 200 * elem_size; // 2400 bytes
        let policy = StagingPolicy::plan(
            AllocationProperty::File,
            size,
            &[3usize, 200].into(),
            &[200usize, 1].into(),
            elem_size,
            1024,
            true,
        );

        match policy {
            StagingPolicy::Chunked(chunks) => {
                assert!(
                    chunks
                        .iter()
                        .all(|c| matches!(c.dst, ChunkDst::Flat { .. }))
                );
                assert_eq!(chunks.iter().map(|c| c.len).sum::<usize>(), size);
                assert_eq!(chunks.first().unwrap().src_offset, 0);
            }
            other => panic!("expected chunked, got {other:?}"),
        }
    }

    #[test]
    fn pitched_file_is_chunked_by_rows() {
        // [4, 10] f32 rows packed in the file (width 40 B) but padded to stride 16 elems
        // (64 B) on device. Chunk size fits 2 rows.
        let elem_size = 4;
        let width_bytes = 10 * elem_size; // 40
        let stride_bytes = 16 * elem_size; // 64
        let height = 4;
        let size = height * width_bytes; // packed source size
        let chunk = 2 * width_bytes; // exactly 2 rows

        let policy = StagingPolicy::plan(
            AllocationProperty::File,
            size,
            &[height, 10].into(),
            &[16usize, 1].into(),
            elem_size,
            chunk,
            true,
        );

        let expected = vec![
            ChunkCopy {
                src_offset: 0,
                len: 2 * width_bytes,
                dst: ChunkDst::Pitched {
                    offset: 0,
                    pitch: stride_bytes,
                    width: width_bytes,
                    rows: 2,
                },
            },
            ChunkCopy {
                src_offset: 2 * width_bytes,
                len: 2 * width_bytes,
                dst: ChunkDst::Pitched {
                    offset: 2 * stride_bytes,
                    pitch: stride_bytes,
                    width: width_bytes,
                    rows: 2,
                },
            },
        ];
        assert_eq!(policy, StagingPolicy::Chunked(expected));
    }

    #[test]
    fn pitched_keeps_at_least_one_row_per_chunk() {
        // A single row (40 B) is wider than the chunk budget (16 B): still one row per chunk.
        let elem_size = 4;
        let width_bytes = 10 * elem_size; // 40
        let height = 3;
        let size = height * width_bytes;

        let policy = StagingPolicy::plan(
            AllocationProperty::File,
            size,
            &[height, 10].into(),
            &[16usize, 1].into(),
            elem_size,
            16, // smaller than one row
            true,
        );

        match policy {
            StagingPolicy::Chunked(chunks) => {
                assert_eq!(chunks.len(), height);
                assert!(
                    chunks
                        .iter()
                        .all(|c| matches!(c.dst, ChunkDst::Pitched { rows: 1, .. }))
                );
            }
            other => panic!("expected chunked, got {other:?}"),
        }
    }
}
