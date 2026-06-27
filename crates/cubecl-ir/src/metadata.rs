use alloc::vec::Vec;
use pliron::context::Context;

use crate::{ContextExt, ElemType};

pub const INFO_ALIGN: usize = size_of::<u64>();

// Metadata
const BUFFER_LEN: usize = 0;
pub const METADATA_BASE_LEN: usize = 1;

// Extended Metadata
const SHAPE_OFFSETS: usize = 0;
const STRIDE_OFFSETS: usize = 1;
pub const METADATA_EXT_LEN: usize = 2;

/// Helper to calculate metadata offsets based on buffer count and position
#[derive(Clone, Copy, Debug, Default)]
pub struct Metadata {
    num_meta: usize,
    num_extended_meta: usize,
}

impl Metadata {
    pub fn new(num_meta: usize, num_extended_meta: usize) -> Self {
        Self {
            num_meta,
            num_extended_meta,
        }
    }

    fn offset_of(&self, id: usize) -> usize {
        self.num_meta * id
    }

    fn base_len(&self) -> usize {
        self.num_meta * METADATA_BASE_LEN
    }

    pub fn static_len(&self) -> usize {
        self.num_meta * METADATA_BASE_LEN + self.num_extended_meta * METADATA_EXT_LEN
    }

    pub fn num_meta(&self) -> usize {
        self.num_meta
    }

    pub fn num_extended_meta(&self) -> usize {
        self.num_extended_meta
    }

    fn offset_of_extended(&self, id: usize) -> usize {
        self.base_len() + self.num_extended_meta * id
    }

    pub fn buffer_len_index(&self, buffer_idx: usize) -> usize {
        self.offset_of(BUFFER_LEN) + buffer_idx
    }

    pub fn shape_offset_index(&self, buffer_idx: usize) -> usize {
        self.offset_of_extended(SHAPE_OFFSETS) + buffer_idx
    }

    pub fn stride_offset_index(&self, buffer_idx: usize) -> usize {
        self.offset_of_extended(STRIDE_OFFSETS) + buffer_idx
    }
}

/// Helper to calculate info struct fields
#[derive(Clone, Debug, Default)]
pub struct Info {
    pub scalars: Vec<SizedInfoField>,
    pub sized_meta: Option<SizedInfoField>,
    pub has_dynamic_meta: bool,
    pub dynamic_meta_offset: usize,
    pub metadata: Metadata,
}

#[derive(Clone, Copy, Debug)]
pub struct SizedInfoField {
    pub ty: ElemType,
    pub count: usize,
    pub offset: usize,
}

impl SizedInfoField {
    pub fn padded_size(&self, ctx: &Context) -> usize {
        let padding_factor = INFO_ALIGN / self.ty.expand_size(ctx.address_type());
        self.count.next_multiple_of(padding_factor)
    }
}

impl Info {
    pub fn has_info(&self) -> bool {
        !self.scalars.is_empty() || self.sized_meta.is_some()
    }
}
