use alloc::{vec, vec::Vec};

use cubecl_ir::{AddressType, StorageType};
use cubecl_runtime::{kernel::ScalarKernelArg, server::MetadataBindingInfo};

use crate::{Metadata, MetadataBuilder, ScalarBuilder};

pub(crate) const INFO_ALIGN: usize = size_of::<u64>();

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
    pub ty: StorageType,
    pub size: usize,
    pub offset: usize,
}

impl SizedInfoField {
    pub fn padded_size(&self) -> usize {
        let padding_factor = INFO_ALIGN / self.ty.size();
        self.size.next_multiple_of(padding_factor)
    }
}

impl Info {
    pub fn new(scalars: &[ScalarKernelArg], metadata: Metadata, address_type: StorageType) -> Self {
        let mut scalar_fields = Vec::with_capacity(scalars.len());
        let mut sized_meta = None;

        let mut offset = 0;

        for scalar in scalars {
            scalar_fields.push(SizedInfoField {
                ty: scalar.ty,
                size: scalar.count,
                offset,
            });
            offset += (scalar.ty.size() * scalar.count).next_multiple_of(INFO_ALIGN);
        }

        if metadata.static_len() > 0 {
            let size = metadata.static_len() as usize;
            sized_meta = Some(SizedInfoField {
                ty: address_type,
                size,
                offset,
            });
            offset += (address_type.size() * size).next_multiple_of(INFO_ALIGN);
        }

        Info {
            scalars: scalar_fields,
            sized_meta,
            has_dynamic_meta: metadata.num_extended_meta() > 0,
            dynamic_meta_offset: offset,
            metadata,
        }
    }

    pub fn has_info(&self) -> bool {
        !self.scalars.is_empty() || self.sized_meta.is_some()
    }
}

#[derive(Default)]
pub struct InfoBuilder {
    pub scalars: ScalarBuilder,
    pub metadata: MetadataBuilder,
}

impl InfoBuilder {
    pub fn finish(&mut self, address_type: AddressType) -> MetadataBindingInfo {
        let addr_packing = INFO_ALIGN / address_type.size();

        let scalars_size = self.scalars.len_aligned();
        let static_len = self.metadata.static_len(address_type);
        let static_size = static_len.div_ceil(addr_packing);
        let dynamic_len = self.metadata.dynamic_len(address_type);
        let dynamic_size = dynamic_len.div_ceil(addr_packing);

        let mut out = vec![0; scalars_size + static_size + dynamic_size];
        self.scalars.finish(&mut out[..scalars_size]);
        self.metadata
            .finish(address_type, out[scalars_size..].split_at_mut(static_size));

        MetadataBindingInfo {
            data: out,
            dynamic_metadata_offset: scalars_size + static_size,
        }
    }
}
