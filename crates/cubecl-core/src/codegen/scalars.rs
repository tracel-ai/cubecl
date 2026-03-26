use alloc::vec::Vec;

use cubecl_ir::StorageType;

use crate::{INFO_ALIGN, ScalarArgType};

/// Stores the data and type for a scalar arg
pub type ScalarValues = Vec<u8>;

#[derive(Default)]
pub struct ScalarBuilder {
    /// Sorted list of scalars, should be faster than `BTreeMap` for this purpose. Benchmark later.
    scalars: Vec<(StorageType, ScalarValues)>,
}

impl ScalarBuilder {
    /// Add a new scalar value to the state.
    pub fn push<T: ScalarArgType>(&mut self, val: T) {
        let val = [val];
        let bytes = T::as_bytes(&val);
        self.get_or_insert_mut(T::cube_type())
            .extend(bytes.iter().copied());
    }

    /// Add a new raw value to the state.
    pub fn push_raw(&mut self, bytes: &[u8], dtype: StorageType) {
        self.get_or_insert_mut(dtype).extend(bytes.iter().copied());
    }

    fn get_or_insert_mut(&mut self, ty: StorageType) -> &mut ScalarValues {
        let pos = self.scalars.iter().position(|(k, _)| *k >= ty);

        match pos {
            Some(i) if self.scalars[i].0 == ty => &mut self.scalars[i].1,
            Some(i) => {
                self.scalars.insert(i, (ty, Vec::new()));
                &mut self.scalars[i].1
            }
            None => {
                self.scalars.push((ty, Vec::new()));
                &mut self.scalars.last_mut().unwrap().1
            }
        }
    }

    pub fn len_aligned(&self) -> usize {
        self.scalars
            .iter()
            .map(|(_, v)| v.len().div_ceil(INFO_ALIGN))
            .sum()
    }

    pub fn finish(&mut self, out: &mut [u64]) {
        let mut out_u8 = bytemuck::cast_slice_mut::<u64, u8>(out);

        for (_, values) in self.scalars.iter_mut().filter(|(_, v)| !v.is_empty()) {
            let len_padded = values.len().next_multiple_of(INFO_ALIGN);

            out_u8[0..values.len()].copy_from_slice(values);
            out_u8 = &mut out_u8[len_padded..];
            values.clear();
        }
    }
}
