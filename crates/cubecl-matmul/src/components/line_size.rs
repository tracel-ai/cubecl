use cubecl_core::{LineSizeError, Runtime, ir::Elem, tensor_line_size_parallel};

use crate::{components::MatrixLayout, kernels::MatmulSetupError};
use std::fmt::Debug;

#[derive(Debug)]
pub struct MatmulLineSizes {
    pub lhs: u8,
    pub rhs: u8,
    pub out: u8,
}

#[derive(Clone, Debug)]
pub struct AvailableLineSizes {
    pub lhs: Vec<u8>,
    pub rhs: Vec<u8>,
    pub out: Vec<u8>,
}

#[derive(Copy, Clone, Debug)]
pub struct MatmulLayouts {
    pub lhs: MatrixLayout,
    pub rhs: MatrixLayout,
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

    pub fn filter_out_with_tensor(self, strides: &[usize], shape: &[usize]) -> Self {
        let out_vec: Vec<u8> = self.out.to_vec();
        let rank = strides.len();

        let target = tensor_line_size_parallel(out_vec.iter().copied(), shape, strides, rank - 1);

        self.filter_out(move |x| *x == target)
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
