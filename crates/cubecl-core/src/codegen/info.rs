use alloc::vec;

use cubecl_ir::AddressType;
use cubecl_runtime::server::MetadataBindingInfo;

use crate::{MetadataBuilder, ScalarBuilder};

#[derive(Default)]
pub struct InfoBuilder {
    pub scalars: ScalarBuilder,
    pub metadata: MetadataBuilder,
}

impl InfoBuilder {
    pub fn finish(&mut self, address_type: AddressType) -> MetadataBindingInfo {
        let addr_per_u64 = size_of::<u64>() / address_type.size();

        let scalars_size = self.scalars.len_u64();
        let static_len = self.metadata.static_len(address_type);
        let static_size = static_len.div_ceil(addr_per_u64);
        let dynamic_len = self.metadata.dynamic_len(address_type);
        let dynamic_size = dynamic_len.div_ceil(addr_per_u64);
        let mut out = vec![0; scalars_size + static_size + dynamic_size];
        self.scalars.finish(&mut out[..scalars_size]);
        self.metadata.finish(address_type, &mut out[scalars_size..]);
        MetadataBindingInfo {
            data: out,
            static_metadata_len: static_len,
            dynamic_metadata_offset: scalars_size + static_size,
        }
    }
}
