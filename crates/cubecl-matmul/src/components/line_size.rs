use cubecl_core::{LineSizeError, Runtime, ir::Elem};

use crate::kernels::MatmulSetupError;
use std::fmt::Debug;

#[derive(Debug)]
pub struct MatmulLineSizes {
    pub lhs: u8,
    pub rhs: u8,
    pub out: u8,
}

impl From<MatmulLineSizes> for (u8, u8, u8) {
    fn from(line_sizes: MatmulLineSizes) -> Self {
        (line_sizes.lhs, line_sizes.rhs, line_sizes.out)
    }
}

#[derive(Clone, Debug)]
pub struct AvailableLineSizes {
    pub lhs: Vec<u8>,
    pub rhs: Vec<u8>,
    pub out: Vec<u8>,
}

impl AvailableLineSizes {
    pub fn from_elem_types<R: Runtime>(elem_in: &Elem, elem_out: &Elem) -> Self {
        let in_available: Vec<u8> = R::line_size_elem(elem_in).collect();
        let out_available = R::line_size_elem(elem_out).collect();
        AvailableLineSizes {
            lhs: in_available.clone(),
            rhs: in_available,
            out: out_available,
        }
    }

    pub fn filter_lhs<F>(self, pred: F) -> Self
    where
        F: FnMut(&u8) -> bool,
    {
        Self {
            lhs: self.lhs.iter().copied().filter(pred).collect(),
            rhs: self.rhs,
            out: self.out,
        }
    }

    pub fn filter_rhs<F>(self, pred: F) -> Self
    where
        F: FnMut(&u8) -> bool,
    {
        Self {
            lhs: self.lhs,
            rhs: self.rhs.iter().copied().filter(pred).collect(),
            out: self.out,
        }
    }

    pub fn filter_out<F>(self, pred: F) -> Self
    where
        F: FnMut(&u8) -> bool,
    {
        Self {
            lhs: self.lhs,
            rhs: self.rhs,
            out: self.out.iter().copied().filter(pred).collect(),
        }
    }

    pub fn commit(self) -> Result<MatmulLineSizes, MatmulSetupError> {
        let pick = |v: Vec<u8>| {
            v.into_iter()
                .max()
                .ok_or(MatmulSetupError::LineSize(LineSizeError::NoValidLineSize))
        };

        Ok(MatmulLineSizes {
            lhs: pick(self.lhs)?,
            rhs: pick(self.rhs)?,
            out: pick(self.out)?,
        })
    }
}
