//! Metadata helpers to easily get offsets etc.
//!
//! Conceptually, metadata is represented like this:
//! ```rust
//! struct Metadata<const N: usize> {
//! ranks: [u32; N],
//! buffer_lengths: [u32; N],
//! logical_lengths: [u32; N],
//! shape_offsets: [usize; N],
//! stride_offsets: [usize; N],
//! shapes: Vec<u32>,
//! strides: Vec<u32>
//! }
//! ```
//! where `Vec` isn't an actual `Vec`, just a dynamically sized series of values.
//!
//! Ranks and lengths have a constant offset, while shapes/strides involve loading the tensor's
//! offset, then adding `dim` to the offset to get each shape/stride.

const RANK: u32 = 0;
const BUFFER_LEN: u32 = 1;
const LENGTH: u32 = 2;
const SHAPE_OFFSETS: u32 = 3;
const STRIDE_OFFSETS: u32 = 4;

pub struct Metadata {
    num_buffers: u32,
}

impl Metadata {
    pub fn new(num_buffers: u32) -> Self {
        Self { num_buffers }
    }

    fn offset_of(&self, id: u32) -> u32 {
        self.num_buffers * id
    }

    pub fn rank_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of(RANK) + buffer_idx
    }

    pub fn buffer_len_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of(BUFFER_LEN) + buffer_idx
    }

    pub fn len_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of(LENGTH) + buffer_idx
    }

    pub fn shape_offset_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of(SHAPE_OFFSETS) + buffer_idx
    }

    pub fn stride_offset_index(&self, buffer_idx: u32) -> u32 {
        self.offset_of(STRIDE_OFFSETS) + buffer_idx
    }
}

#[derive(Default)]
pub struct MetadataBuilder {
    ranks: Vec<u32>,
    buffer_lens: Vec<u32>,
    lengths: Vec<u32>,
    shapes: Vec<Vec<u32>>,
    strides: Vec<Vec<u32>>,
}

impl MetadataBuilder {
    pub fn with_array(mut self, buffer_len: u32, len: u32) -> Self {
        self.ranks.push(1);
        self.buffer_lens.push(buffer_len);
        self.lengths.push(len);
        self.shapes.push(Vec::new());
        self.strides.push(Vec::new());
        self
    }

    pub fn with_tensor(
        mut self,
        rank: u32,
        buffer_len: u32,
        shape: Vec<u32>,
        strides: Vec<u32>,
    ) -> Self {
        self.ranks.push(rank);
        self.buffer_lens.push(buffer_len);
        self.lengths.push(shape.iter().product());
        self.shapes.push(shape);
        self.strides.push(strides);
        self
    }

    pub fn finish(self) -> Vec<u32> {
        let num_buffers = self.ranks.len();
        let mut shape_offsets = Vec::with_capacity(num_buffers);
        let mut strides_offsets = Vec::with_capacity(num_buffers);

        let mut current_offset = num_buffers as u32 * 5; // Total fields in static portion

        for shape in self.shapes.iter() {
            shape_offsets.push(current_offset);
            current_offset += shape.len() as u32;
        }

        for stride in self.strides.iter() {
            strides_offsets.push(current_offset);
            current_offset += stride.len() as u32;
        }

        let shapes = self.shapes.into_iter().flatten();
        let strides = self.strides.into_iter().flatten();

        let mut meta = self.ranks;
        meta.extend(self.buffer_lens);
        meta.extend(self.lengths);
        meta.extend(shape_offsets);
        meta.extend(strides_offsets);
        meta.extend(shapes);
        meta.extend(strides);
        meta
    }
}
