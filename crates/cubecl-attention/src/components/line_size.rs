use std::fmt::Debug;

use cubecl_core::{LineSizeError, Runtime, ir::Elem, tensor_line_size_parallel};

use crate::components::{AttentionSetupError, Ident};

#[derive(Debug)]
/// Line size used for each tensor in global memory accesses.
/// Represents the numbe of elements processed per SIMD load/store.
pub struct AttentionLineSizes {
    pub query: u8,
    pub key: u8,
    pub value: u8,
    pub mask: u8,
    pub out: u8,
}

#[derive(Clone, Debug)]
/// Candidate line sizes supported for each tensor.
///
/// These lists begin with compiler-supported sizes and are progressively
/// filtered based on problem shape divisibility and hardware constraints.
pub struct AvailableLineSizes {
    pub query: Vec<u8>,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub mask: Vec<u8>,
    pub out: Vec<u8>,
}

impl AvailableLineSizes {
    pub fn from_elem_types<R: Runtime>(elem_in: &Elem, elem_mask: &Elem, elem_out: &Elem) -> Self {
        let in_available: Vec<u8> = R::line_size_elem(elem_in).collect();
        let mask_available: Vec<u8> = R::line_size_elem(elem_mask).collect();
        let out_available = R::line_size_elem(elem_out).collect();

        AvailableLineSizes {
            query: in_available.clone(),
            key: in_available.clone(),
            value: in_available,
            mask: mask_available,
            out: out_available,
        }
    }

    /// Filter available line sizes considering tensor shapes and strides for ident
    pub fn filter_with_tensor(self, ident: Ident, strides: &[usize], shape: &[usize]) -> Self {
        let rank = strides.len();

        let iter = match ident {
            Ident::Query => self.query.iter().copied(),
            Ident::Key => self.key.iter().copied(),
            Ident::Value => self.value.iter().copied(),
            Ident::Mask => self.mask.iter().copied(),
            Ident::Out => self.out.iter().copied(),
        };

        let target = tensor_line_size_parallel(iter, shape, strides, rank - 1);

        self.filter(move |x| *x == target, ident)
    }

    /// Filter available line sizes for ident
    pub fn filter<F>(mut self, mut pred: F, ident: Ident) -> Self
    where
        F: FnMut(&u8) -> bool,
    {
        match ident {
            Ident::Query => {
                self.query = self.query.into_iter().filter(&mut pred).collect();
            }
            Ident::Key => {
                self.key = self.key.into_iter().filter(&mut pred).collect();
            }
            Ident::Value => {
                self.value = self.value.into_iter().filter(&mut pred).collect();
            }
            Ident::Mask => {
                self.mask = self.mask.into_iter().filter(&mut pred).collect();
            }
            Ident::Out => {
                self.out = self.out.into_iter().filter(&mut pred).collect();
            }
        }
        self
    }

    /// Pick the largest remaining line size for each tensor
    pub fn pick_max(self) -> Result<AttentionLineSizes, AttentionSetupError> {
        let pick = |v: Vec<u8>| {
            v.into_iter().max().ok_or(AttentionSetupError::LineSize(
                LineSizeError::NoValidLineSize,
            ))
        };

        Ok(AttentionLineSizes {
            query: pick(self.query)?,
            key: pick(self.key)?,
            value: pick(self.value)?,
            mask: pick(self.mask)?,
            out: pick(self.out)?,
        })
    }
}
