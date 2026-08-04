use alloc::vec;
use alloc::vec::Vec;
use core::{default::Default, ops::Deref};
use serde::{Deserialize, Serialize};

/// Describes a quantization scheme/configuration.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QuantScheme {
    /// The logical data type of quantized input values (e.g., `QInt8`).
    ///
    /// This defines how values are interpreted during computation, independent of how they're stored.
    pub value: QuantValue,
    /// Precision used for quantization parameters (e.g., scale and biases).
    ///
    /// This is the only param a one-level scheme has. [`QuantLevel::BlockTensor`] adds a second one
    /// for its per-tensor scale, so a consumer that reads this field alone will miss that factor.
    pub param: QuantParam,
    /// Data type used for storing quantized values.
    pub store: QuantStore,
    /// Granularity level of quantization (e.g., per-tensor).
    pub level: QuantLevel,
    /// Quantization mode (e.g., symmetric).
    pub mode: QuantMode,
}

impl Default for QuantScheme {
    fn default() -> Self {
        Self {
            value: QuantValue::Q8F,
            param: QuantParam::F32,
            store: QuantStore::PackedU32(0),
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
        }
    }
}

impl QuantScheme {
    /// Set the quantization level.
    pub fn with_level(mut self, level: QuantLevel) -> Self {
        self.level = level;
        self
    }

    /// Set the quantization mode.
    pub fn with_mode(mut self, mode: QuantMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the data type used for quantized values.
    pub fn with_value(mut self, value: QuantValue) -> Self {
        self.value = value;
        self
    }

    /// Set the data type used to store quantized values.
    pub fn with_store(mut self, store: QuantStore) -> Self {
        self.store = store;
        self
    }

    /// Set the precision used for quantization parameters
    pub fn with_param(mut self, param: QuantParam) -> Self {
        self.param = param;
        self
    }

    /// Returns the size of the quantization storage type in bits.
    pub fn size_bits_stored(&self) -> usize {
        self.store.size_bits(&self.value)
    }

    /// Returns the size of the quantization storage type in bits.
    pub fn size_bits_value(&self) -> usize {
        self.value.size_bits()
    }

    /// Returns the number of quantized values stored in a single element.
    pub fn num_quants(&self) -> usize {
        self.size_bits_stored() / self.value.size_bits()
    }

    /// Returns the native packing factor for the values. When native packing > 1, the packed
    /// representation stores `num_quants` elements grouped into packs of `native_packing` size.
    pub fn native_packing(&self) -> usize {
        self.value.native_packing()
    }

    /// Returns the packing dim for the store.
    pub fn packing_dim(&self) -> Option<usize> {
        self.store.packing_dim()
    }

    /// Swaps the packing dim if it's either of `dim0` or `dim1`.
    /// Executes the corresponding update to `shape.swap(dim0, dim1)`.
    pub fn swap_packing_dim(&mut self, dim0: usize, dim1: usize) {
        if let QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim) =
            &mut self.store
        {
            if *packed_dim == dim0 {
                *packed_dim = dim1;
            } else if *packed_dim == dim1 {
                *packed_dim = dim0;
            }
        }
    }
}

/// Level or granularity of quantization.
///
/// Append new variants, never insert. Some transports serialize this with a format that encodes
/// variants by position rather than by name, so inserting one silently reinterprets streams and
/// stored schemes written by an older build.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantLevel {
    /// Quantize the whole tensor using a single tensor.
    Tensor,
    /// Quantize a tensor using multiple blocks.
    Block(BlockSize),
    /// Quantize a tensor using multiple blocks whose scales are themselves normalized by a single
    /// per-tensor scale.
    ///
    /// See [`QuantLevel::block_tensor`] for what that buys and what it does not.
    BlockTensor {
        /// Size of each block. The block scales use [`QuantScheme::param`].
        block: BlockSize,
        /// Precision of the per-tensor scale. Only a param with more range than the block scales
        /// use is meaningful.
        global: QuantParam,
    },
}

impl QuantLevel {
    /// Converting constructor for [`QuantLevel::Block`]
    pub fn block(values: impl AsRef<[u8]>) -> Self {
        QuantLevel::Block(BlockSize::new(values))
    }

    /// Converting constructor for [`QuantLevel::BlockTensor`].
    ///
    /// The per-tensor scale absorbs the tensor's dynamic range, which is what lets the block scales
    /// live in a narrow type. Without it a block scale has to cover that range on its own, and a
    /// type like [`QuantParam::UE4M3`] underflows to zero for small values.
    ///
    /// What the block param covers is then the spread between blocks, which is still bounded. A
    /// block whose scale falls further below the largest one than the block param can express is
    /// stored at that param's smallest value, far too coarse for it, and every value in the block
    /// quantizes to zero. [`QuantParam::UE4M3`] spans about 2^18 this way, from its smallest
    /// subnormal to 448, so a tensor holding a genuine outlier can lose its ordinary values.
    ///
    /// The quantized view in `cubecl-std` reads the per-tensor scale as a binding of its own, and
    /// rejects a launch whose bindings disagree with the level rather than reconstruct values short
    /// by that factor.
    pub fn block_tensor(values: impl AsRef<[u8]>, global: QuantParam) -> Self {
        QuantLevel::BlockTensor {
            block: BlockSize::new(values),
            global,
        }
    }

    /// The block size, for the levels that quantize in blocks.
    pub fn block_size(&self) -> Option<BlockSize> {
        match self {
            QuantLevel::Tensor => None,
            QuantLevel::Block(block) | QuantLevel::BlockTensor { block, .. } => Some(*block),
        }
    }

    /// The precision of the per-tensor scale, for the levels that have one.
    pub fn global_param(&self) -> Option<QuantParam> {
        match self {
            QuantLevel::Tensor | QuantLevel::Block(_) => None,
            QuantLevel::BlockTensor { global, .. } => Some(*global),
        }
    }
}

impl QuantParam {
    /// The largest finite value representable by the parameter type.
    ///
    /// A two-level scheme picks its per-tensor scale so that the largest block scale lands here,
    /// which is what keeps the block scales inside the range their type can express. That recipe
    /// only holds for a block param narrower than the scale it divides: dividing by
    /// [`QuantParam::F32`]'s or [`QuantParam::UE8M0`]'s maximum drives the per-tensor scale
    /// subnormal and the renormalized block scales to infinity. A two-level scheme has nothing to
    /// gain from those params anyway, since their block scales already reach the full range.
    pub fn max_representable(&self) -> f32 {
        match self {
            QuantParam::F32 => f32::MAX,
            QuantParam::F16 => half::f16::MAX.to_f32(),
            QuantParam::BF16 => half::bf16::MAX.to_f32(),
            // Spelled out because `ue8m0` and `e4m3` sit behind the `fp8` feature and this
            // function is not gated. The tests check both against those types when it is on.
            QuantParam::UE8M0 => f32::from_bits(0x7F00_0000), // 2^127
            QuantParam::UE4M3 => 448.0,
        }
    }

    /// The smallest value representable by the parameter type that is not below `scale`.
    ///
    /// Storing a quantization scale wants this rather than the nearest value. Rounding down puts
    /// the scale below what calibration asked for, so every value at the block maximum clips to
    /// the quantization range; rounding up costs one step of coarseness instead. Backends have to
    /// agree on this, or a tensor quantized on one reconstructs differently on another.
    ///
    /// This is not a cast. Conversion to these types rounds to nearest, which is what a cast
    /// should do; this is the storage policy for a scale specifically.
    ///
    /// `scale` must not be negative. Symmetric quantization only produces non-negative scales,
    /// and the stepping below walks away from zero for a negative input.
    ///
    /// [`QuantParam::UE8M0`] answers [`None`]. Its minimum is 2^-127, subnormal in f32, where the
    /// grid below no longer holds.
    pub fn round_up(&self, scale: f32) -> Option<f32> {
        match self {
            QuantParam::F32 => {
                return Some(scale);
            }
            QuantParam::UE8M0 => {
                return None;
            }
            _ => {}
        }
        if scale.is_nan() {
            return Some(scale);
        }
        debug_assert!(scale >= 0.0, "a quantization scale is never negative");

        // Nothing representable sits above the maximum, and converting past it yields an infinity
        // for the params that have one, which would make every reconstructed value NaN.
        let max = self.max_representable();
        if scale >= max {
            return Some(max);
        }

        let grid = self.f32_grid();

        if let Some(subnormals) = grid.subnormals
            && scale < subnormals.min_normal
        {
            // Below the minimum normal the spacing stops halving, so the answer is a count of steps.
            // Qualified call: the inherent `f32::ceil` lives in std, and this crate builds no_std.
            return Some(num_traits::Float::ceil(scale / subnormals.spacing) * subnormals.spacing);
        }

        Some(f32::from_bits(
            (scale.to_bits() + grid.round_up_bias()) & grid.truncate_mask(),
        ))
    }

    /// The param's grid, expressed on the f32 bit pattern. See [`F32Grid`].
    ///
    /// bf16 reports no subnormal range because it does not need the separate treatment: its pattern
    /// is f32's top half all the way down, so the bit step stays right where the others stop. Its
    /// own subnormals start at 2^-133, which is subnormal in f32 too and flushed to zero by most
    /// backends.
    ///
    /// # Panics
    ///
    /// For [`QuantParam::F32`], which is the grid itself, and [`QuantParam::UE8M0`], which is not
    /// yet supported.
    pub fn f32_grid(&self) -> F32Grid {
        /// One f32 ulp per param ulp: the mantissa bits f32 carries and the param does not.
        const fn bit_step(mantissa_digits: u32) -> u32 {
            1 << (f32::MANTISSA_DIGITS - mantissa_digits)
        }

        match self {
            QuantParam::F16 => F32Grid {
                bit_step: bit_step(half::f16::MANTISSA_DIGITS),
                subnormals: Some(SubnormalRange {
                    min_normal: half::f16::MIN_POSITIVE.to_f32(),
                    spacing: half::f16::MIN_POSITIVE_SUBNORMAL.to_f32(),
                }),
            },
            QuantParam::BF16 => F32Grid {
                bit_step: bit_step(half::bf16::MANTISSA_DIGITS),
                subnormals: None,
            },
            // Spelled out rather than read off `e4m3`, which sits behind the `fp8` feature while
            // this is not gated. The tests check them against that type when it is on.
            QuantParam::UE4M3 => F32Grid {
                bit_step: bit_step(4),
                subnormals: Some(SubnormalRange {
                    min_normal: 0.015625, // 2^-6
                    spacing: 0.001953125, // 2^-9
                }),
            },
            QuantParam::F32 => {
                unimplemented!("F32 is the grid, it has no narrower one to round onto")
            }
            QuantParam::UE8M0 => unimplemented!("UE8M0 scales are not yet supported"),
        }
    }
}

/// A narrower float format's grid, laid over the f32 bit pattern.
///
/// f32 carries every param this exists for exactly, so the grid can be walked there rather than
/// through the storage type. A value representable in the param leaves the low f32 mantissa bits
/// zero, so one param ulp is an increment at that position and the carry into the exponent falls
/// out on its own. Working in f32 also keeps the grid available to backends with no narrow integer,
/// and to builds without the `fp8` feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F32Grid {
    /// One step up in the normal range, as an increment on the f32 bit pattern.
    pub bit_step: u32,
    /// The param's subnormals, for the formats whose subnormals land in f32's normal range.
    pub subnormals: Option<SubnormalRange>,
}

/// Where a format's subnormals begin and how far apart they are, in f32.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubnormalRange {
    /// The smallest normal value, below which the spacing stops halving.
    pub min_normal: f32,
    /// The constant distance between neighbouring subnormals.
    pub spacing: f32,
}

impl F32Grid {
    /// Clears the mantissa bits the param does not carry, truncating a bit pattern onto the grid.
    pub fn truncate_mask(&self) -> u32 {
        !(self.bit_step - 1)
    }

    /// Added to a bit pattern before [`truncate_mask`](Self::truncate_mask) to turn that truncation
    /// into a round up. The carry it can produce is only safe below the param's maximum, which is
    /// why callers saturate there first.
    pub fn round_up_bias(&self) -> u32 {
        self.bit_step - 1
    }
}

/// Data type used to represent quantized values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantValue {
    /// 8-bit quantization with full range.
    Q8F,
    /// 8-bit floating point, e5m2 format.
    E5M2,
    /// 8-bit floating point, e4m3 format.
    E4M3,
    /// 4-bit quantization with full range.
    Q4F,
    /// 4-bit floating point, e2m1 format.
    E2M1,
    /// 2-bit quantization with full range.
    Q2F,
    /// 8-bit quantization with symmetric range.
    Q8S,
    /// 4-bit quantization with symmetric range.
    Q4S,
    /// 2-bit quantization with symmetric range.
    Q2S,
}

impl QuantValue {
    /// Returns the size of the quantization input type in bits.
    pub fn size_bits(&self) -> usize {
        match self {
            QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2 => 8,
            QuantValue::Q4F | QuantValue::Q4S | QuantValue::E2M1 => 4,
            QuantValue::Q2F | QuantValue::Q2S => 2,
        }
    }

    /// Packing factor for the native representation used for intermediate values. If > 1, values
    /// should always be processed in `native_packing` sized chunks.
    pub fn native_packing(&self) -> usize {
        match self {
            QuantValue::E2M1 => 2,
            _ => 1,
        }
    }

    /// The possible range of values allowed by the quant value.
    pub fn range(&self) -> (f32, f32) {
        match self {
            QuantValue::Q8F => (i8::MIN as f32, i8::MAX as f32),
            QuantValue::Q4F => (-8.0, 7.0),
            QuantValue::Q2F => (-2.0, 1.0),
            QuantValue::Q8S => (-i8::MAX as f32, i8::MAX as f32),
            QuantValue::Q4S => (-7.0, 7.0),
            QuantValue::Q2S => (-1.0, 1.0),
            QuantValue::E4M3 => (-448.0, 448.0),
            QuantValue::E5M2 => (-57344.0, 57344.0),
            QuantValue::E2M1 => (-6.0, 6.0), // Hardcoded because of no-std
        }
    }

    /// If the range of values is symmetric around zero.
    pub fn is_symmetric(&self) -> bool {
        match self {
            Self::Q8F | Self::Q4F | Self::Q2F | Self::E4M3 | Self::E5M2 | Self::E2M1 => false,
            Self::Q8S | Self::Q4S | Self::Q2S => true,
        }
    }
}

impl QuantStore {
    /// Returns the size of the quantization input type in bits.
    pub fn size_bits(&self, value: &QuantValue) -> usize {
        match self {
            QuantStore::Native => value.size_bits(),
            QuantStore::PackedNative(_) => value.size_bits() * value.native_packing(),
            QuantStore::PackedU32(_) => 32,
        }
    }

    fn packing_dim(&self) -> Option<usize> {
        match self {
            QuantStore::Native => None,
            QuantStore::PackedNative(packing_dim) | QuantStore::PackedU32(packing_dim) => {
                Some(*packing_dim)
            }
        }
    }
}

/// Data type used to stored quantized values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantStore {
    /// Native quantization doesn't require packing and unpacking.
    Native,
    /// Store packed quantized values in a natively supported packing format (i.e. e2m1x2).
    /// Argument is the dimension the tensor is packed on, starting from the innermost dimension.
    PackedNative(usize),
    /// Store packed quantized values in a 4-byte unsigned integer.
    /// Argument is the dimension the tensor is packed on, starting from the innermost dimension.
    PackedU32(usize),
    // /// Store packed quantized values in a 8-bit unsigned integer.
    // U8,
}

/// Strategy used to quantize values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantMode {
    /// Symmetric or scale quantization.
    Symmetric,
}

/// Quantization floating-point precision.
///
/// This is used to represent the floating-point precision of quantization parameters like the scale(s)
/// or the accumulation precision used during operations like matrix multiplication.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantParam {
    /// Full precision.
    F32,
    /// Half precision.
    F16,
    /// bfloat16 precision.
    BF16,
    /// unsigned floating point, e8m0 format.
    UE8M0,
    /// unsigned floating point, e4m3 format.
    UE4M3,
}

const MAX_DIMS: usize = 5;

/// Copyable block size, specialized version of `SmallVec`.
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockSize {
    storage: [u8; MAX_DIMS],
    len: u8,
}

impl core::fmt::Debug for BlockSize {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "BlockSize({:?})", self.as_slice())
    }
}

impl BlockSize {
    /// Max number of dimensions for block size
    pub const MAX_DIMS: usize = MAX_DIMS;

    /// Create a new blocksize from a set of values. The number of values must be `<= MAX_DIMS`.
    pub fn new(values: impl AsRef<[u8]>) -> Self {
        let values = values.as_ref();
        debug_assert!(
            values.len() <= MAX_DIMS,
            "Tried creating a block size larger than the cap"
        );
        let len = values.len().min(MAX_DIMS);
        let mut storage = [1; MAX_DIMS];
        storage[..len].copy_from_slice(&values[..len]);
        Self {
            storage,
            len: len as u8,
        }
    }

    /// Create a new blocksize from a set of values. The number of values must be `<= MAX_DIMS`.
    /// Trims any leading zeros.
    pub fn new_trim(values: impl AsRef<[u8]>) -> Self {
        let values = values.as_ref();
        let first_value = values.iter().position(|s| *s != 1).unwrap_or(0);
        Self::new(&values[first_value..])
    }

    /// Return a slice of only the initialized values
    pub fn as_slice(&self) -> &[u8] {
        &self.storage[..self.len as usize]
    }

    /// Return a vec of only the initialized values
    pub fn to_vec(&self) -> Vec<u8> {
        self.storage[..self.len as usize].to_vec()
    }

    /// Returns `N` dimensions, unsqueezing if necessary.
    pub fn as_dim<const N: usize>(&self) -> [u8; N] {
        let data_len = N.min(self.len as usize);
        let data_start = N - data_len;
        let mut out = [1; N];
        out[data_start..].copy_from_slice(&self.storage[..data_len]);
        out
    }

    /// Returns a vector of `len` dimensions, unsqueezing if necessary.
    pub fn to_dim_vec(&self, len: usize) -> Vec<u8> {
        let data_len = len.min(self.len as usize);
        let data_start = len - data_len;
        let mut out = vec![1; len];
        out[data_start..].copy_from_slice(&self.storage[..data_len]);
        out
    }

    /// Create an iterator over all stored dimensions
    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        self.as_slice().iter()
    }

    /// Returns the total number of elements in each block
    pub fn num_elements(&self) -> usize {
        self.iter().map(|it| *it as usize).product()
    }
}

impl Deref for BlockSize {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: AsRef<[u8]>> From<T> for BlockSize {
    fn from(value: T) -> Self {
        BlockSize::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up_never_lands_below_the_scale() {
        for param in [QuantParam::F16, QuantParam::BF16, QuantParam::UE4M3] {
            for exp in -12..8 {
                for step in 1..17 {
                    let scale = (step as f32 / 16.0) * 2f32.powi(exp);
                    let up = param.round_up(scale).unwrap();
                    assert!(
                        up >= scale,
                        "{param:?}: {up} is below {scale}, which clips the block maximum"
                    );
                }
            }
        }
    }

    #[test]
    fn round_up_saturates_rather_than_stepping_off_the_top() {
        for param in [QuantParam::F16, QuantParam::BF16, QuantParam::UE4M3] {
            let max = param.max_representable();
            assert_eq!(param.round_up(max).unwrap(), max);
            assert!(param.round_up(max * 2.0).unwrap().is_finite());
        }
    }

    /// Every variant is dispatched somewhere, so none of them may panic here.
    #[test]
    fn round_up_answers_for_every_param() {
        for param in [
            QuantParam::F32,
            QuantParam::F16,
            QuantParam::BF16,
            QuantParam::UE8M0,
            QuantParam::UE4M3,
        ] {
            assert_eq!(
                param.round_up(0.3).is_some(),
                param != QuantParam::UE8M0,
                "{param:?}"
            );
        }
    }

    #[test]
    fn round_up_is_the_identity_for_f32() {
        for scale in [1.0e-30, 0.1, 1.0, 12345.678, f32::MAX] {
            assert_eq!(QuantParam::F32.round_up(scale).unwrap(), scale);
        }
    }

    /// The checks that need the real storage types to compare against.
    #[cfg(feature = "fp8")]
    mod storage_types {
        use super::*;

        #[test]
        fn round_up_is_the_nearest_representable_value_not_below() {
            // Rounding up must not overshoot: stepping down from the answer has to land below.
            for param in [QuantParam::F16, QuantParam::BF16, QuantParam::UE4M3] {
                for exp in -8..6 {
                    let scale = 1.7 * 2f32.powi(exp);
                    let up = param.round_up(scale).unwrap();
                    assert_eq!(
                        up,
                        param.round_up(up).unwrap(),
                        "{param:?}: not idempotent at {scale}"
                    );
                    assert!(
                        step(param, up, -1) < scale,
                        "{param:?}: {up} overshoots {scale} by at least a step"
                    );
                }
            }
        }

        /// `round_up` reads the grid instead of converting through the storage type, so a wrong
        /// constant there is only visible against the type itself. Nothing else in this file would
        /// catch one: a grid finer than the real thing still lands above the scale, still steps
        /// down below it, and still looks idempotent.
        #[test]
        fn f32_grid_matches_the_storage_types() {
            for param in [QuantParam::F16, QuantParam::BF16, QuantParam::UE4M3] {
                let grid = param.f32_grid();

                // bf16 deliberately reports no subnormal range, since its bit step covers them too.
                if let Some(subnormals) = grid.subnormals {
                    assert_eq!(
                        subnormals.min_normal,
                        min_normal(param),
                        "{param:?}: minimum normal"
                    );
                    assert_eq!(
                        subnormals.spacing,
                        step(param, 0.0, 1),
                        "{param:?}: subnormal spacing"
                    );
                }

                // Walk the whole normal range: one step on the f32 pattern has to be one step in
                // the type, at every exponent.
                let mut value = min_normal(param);
                let max = param.max_representable();
                while value < max {
                    let stepped = f32::from_bits(value.to_bits() + grid.bit_step);
                    assert_eq!(
                        stepped,
                        step(param, value, 1),
                        "{param:?}: step above {value}"
                    );
                    value = stepped;
                }
                assert_eq!(
                    value, max,
                    "{param:?}: the grid has to land exactly on the maximum"
                );
            }
        }

        #[test]
        fn max_representable_matches_the_e4m3_type() {
            assert_eq!(
                QuantParam::UE4M3.max_representable(),
                crate::e4m3::MAX.to_f32()
            );
        }

        /// The other limit spelled out as a literal. `ue8m0` is exponent only, so its maximum is
        /// the power of two the hex literal encodes.
        #[test]
        fn max_representable_matches_the_e8m0_type() {
            assert_eq!(
                QuantParam::UE8M0.max_representable(),
                crate::ue8m0::MAX as f32
            );
        }

        /// `offset` representable steps from `value` in `param`, for positive values. Counted on
        /// the storage type's own bit pattern, so this is an oracle independent of the grid under
        /// test.
        fn step(param: QuantParam, value: f32, offset: i32) -> f32 {
            match param {
                QuantParam::F16 => half::f16::from_bits(
                    (half::f16::from_f32(value).to_bits() as i32 + offset) as u16,
                )
                .to_f32(),
                QuantParam::BF16 => half::bf16::from_bits(
                    (half::bf16::from_f32(value).to_bits() as i32 + offset) as u16,
                )
                .to_f32(),
                QuantParam::UE4M3 => crate::e4m3::from_bits(
                    (crate::e4m3::from_f32(value).to_bits() as i32 + offset) as u8,
                )
                .to_f32(),
                QuantParam::F32 | QuantParam::UE8M0 => unreachable!(),
            }
        }

        fn min_normal(param: QuantParam) -> f32 {
            match param {
                QuantParam::F16 => half::f16::MIN_POSITIVE.to_f32(),
                QuantParam::BF16 => half::bf16::MIN_POSITIVE.to_f32(),
                QuantParam::UE4M3 => crate::e4m3::MIN_POSITIVE.to_f32(),
                QuantParam::F32 | QuantParam::UE8M0 => unreachable!(),
            }
        }
    }
}
