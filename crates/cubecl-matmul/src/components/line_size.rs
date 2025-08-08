use cubecl_core::{LineSizeError, Runtime, ir::Elem, tensor_line_size_parallel};

use crate::components::{MatrixLayout, error::MatmulSetupError};
use std::fmt::Debug;

#[derive(Debug)]
/// Line size used for each tensor in global memory accesses.
/// Represents the number of elements processed per SIMD load/store.
pub struct MatmulLineSizes {
    pub lhs: u8,
    pub rhs: u8,
    pub out: u8,
}

#[derive(Clone, Debug)]
/// Candidate line sizes supported for each tensor.
///
/// These lists begin with compiler-supported sizes and are progressively
/// filtered based on problem shape divisibility and hardware constraints.
pub struct AvailableLineSizes {
    pub lhs: Vec<u8>,
    pub rhs: Vec<u8>,
    pub out: Vec<u8>,
}

impl AvailableLineSizes {
    pub fn from_elem_types<R: Runtime>(elem_lhs: &Elem, elem_rhs: &Elem, elem_out: &Elem) -> Self {
        AvailableLineSizes {
            lhs: R::line_size_elem(elem_lhs).collect(),
            rhs: R::line_size_elem(elem_rhs).collect(),
            out: R::line_size_elem(elem_out).collect(),
        }
    }

    /// Filter available line sizes considering tensor shapes and strides for Lhs
    pub fn filter_lhs_with_tensor(
        self,
        strides: &[usize],
        shape: &[usize],
        layout: MatrixLayout,
    ) -> Self {
        let lhs_vec: Vec<u8> = self.lhs.to_vec();
        let rank = strides.len();

        let target = tensor_line_size_parallel(
            lhs_vec.iter().copied(),
            shape,
            strides,
            match layout {
                MatrixLayout::RowMajor => rank - 1,
                MatrixLayout::ColMajor => rank - 2,
            },
        );

        self.filter_lhs(move |x| *x == target)
    }

    /// Filter available line sizes considering tensor shapes and strides for Rhs
    pub fn filter_rhs_with_tensor(
        self,
        strides: &[usize],
        shape: &[usize],
        layout: MatrixLayout,
    ) -> Self {
        let rhs_vec: Vec<u8> = self.rhs.to_vec();
        let rank = strides.len();

        let target = tensor_line_size_parallel(
            rhs_vec.iter().copied(),
            shape,
            strides,
            match layout {
                MatrixLayout::RowMajor => rank - 1,
                MatrixLayout::ColMajor => rank - 2,
            },
        );

        self.filter_rhs(move |x| *x == target)
    }

    /// Filter available line sizes considering tensor shapes and strides for output
    pub fn filter_out_with_tensor(self, strides: &[usize], shape: &[usize]) -> Self {
        let out_vec: Vec<u8> = self.out.to_vec();
        let rank = strides.len();

        let target = tensor_line_size_parallel(out_vec.iter().copied(), shape, strides, rank - 1);

        self.filter_out(move |x| *x == target)
    }

    /// Filter available line sizes for Lhs
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

    /// Filter available line sizes for Rhs
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

    /// Filter available line sizes for output
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

    /// Pick the largest remaining line size for each tensor
    pub fn pick_max(self) -> Result<MatmulLineSizes, MatmulSetupError> {
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
