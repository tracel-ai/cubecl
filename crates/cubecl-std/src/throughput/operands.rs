use cubecl_core::ir::{ElemType, FloatKind};
use cubecl_runtime::{client::Client, throughput::ComputeCmmaConfig};

/// The operand types and vector widths an arithmetic ceiling is measured in.
pub(super) struct Arithmetic;

impl Arithmetic {
    /// Without an fma of its own for `dtype` a device emulates one per
    /// operation, and a kernel converts and accumulates in f32 rather than pay
    /// that, so both are measured and the faster kept.
    pub(super) fn dtypes(client: &Client, dtype: ElemType) -> alloc::vec::Vec<ElemType> {
        let mut dtypes = alloc::vec![dtype];

        if let Some(accumulator) = Self::promoted(dtype)
            && client.properties().features.supports_type(accumulator)
        {
            dtypes.push(accumulator);
        }

        dtypes
    }

    /// `io_optimized_vector_sizes` is ordered for the loads and stores this
    /// probe issues none of, and its widest is not the fastest on every device.
    pub(super) fn widths(client: &Client, dtype: ElemType) -> alloc::vec::Vec<usize> {
        let widths: alloc::vec::Vec<usize> =
            client.io_optimized_vector_sizes(dtype.size()).collect();

        if widths.is_empty() {
            alloc::vec![1]
        } else {
            widths
        }
    }

    /// What a kernel with `dtype` operands accumulates in when the device has no
    /// arithmetic of its own for them, or `None` where `dtype` is already the
    /// widest of the two.
    fn promoted(dtype: ElemType) -> Option<ElemType> {
        let accumulator = ElemType::Float(FloatKind::F32);
        let narrower_float =
            matches!(dtype, ElemType::Float(_)) && dtype.size() < accumulator.size();

        narrower_float.then_some(accumulator)
    }
}

/// The cooperative matrix a compute-cmma probe launches.
pub(super) struct CooperativeMatrix;

impl CooperativeMatrix {
    /// A non-empty capability list says the device has tensor hardware, not this
    /// shape of it. `mma` is not consulted: the probe issues `cmma::execute`.
    pub(super) fn implemented(client: &Client, dtype: ElemType, config: ComputeCmmaConfig) -> bool {
        client.properties().features.matmul.cmma.iter().any(|it| {
            it.a_type == dtype
                && it.b_type == dtype
                && it.cd_type == config.accumulator_type
                && it.m as usize == config.cmma_dims.m
                && it.n as usize == config.cmma_dims.n
                && it.k as usize == config.cmma_dims.k
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_core::ir::{IntKind, UIntKind};

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// Where the device does the arithmetic itself, its rate is the ceiling and
    /// a second measurement of the same thing costs a probe for nothing.
    #[test]
    fn nothing_as_wide_as_the_accumulator_is_promoted() {
        assert_eq!(Arithmetic::promoted(F32), None);
        assert_eq!(Arithmetic::promoted(ElemType::Float(FloatKind::F64)), None);
    }

    /// The probe retires a multiply for an integer, which no float rate bounds.
    #[test]
    fn an_integer_is_not_promoted() {
        assert_eq!(Arithmetic::promoted(ElemType::Int(IntKind::I8)), None);
        assert_eq!(Arithmetic::promoted(ElemType::UInt(UIntKind::U16)), None);
    }

    #[test]
    fn every_float_narrower_than_the_accumulator_is_promoted() {
        for dtype in [FloatKind::F16, FloatKind::BF16, FloatKind::E4M3] {
            assert_eq!(Arithmetic::promoted(ElemType::Float(dtype)), Some(F32));
        }
    }
}
