use alloc::vec;
use alloc::vec::Vec;
use core::{default::Default, ops::Deref};
use serde::{Deserialize, Serialize};

/// Describes a quantization scheme/configuration.
///
/// Scales come at up to two levels, each an optional field set through
/// [`per_tensor`](Self::per_tensor) and [`per_block`](Self::per_block) in any order:
///
/// ```
/// # use cubecl_common::quant::scheme::{QuantScheme, ScaleDtype};
/// // One scale for the whole tensor, stored as f32. Also what a scheme with no level resolves to.
/// QuantScheme::default().per_tensor(ScaleDtype::F32);
///
/// // One scale per block of 32 values.
/// QuantScheme::default().per_block([32], ScaleDtype::F32);
///
/// // Two levels: ue4m3 block scales, normalized by a single per-tensor f32 scale.
/// QuantScheme::default()
///     .per_block([16], ScaleDtype::UE4M3)
///     .per_tensor(ScaleDtype::F32);
/// ```
///
/// A two-level scheme exists so block scales can live in a narrow type: the global per-tensor scale
/// absorbs the tensor's dynamic range, and the block dtype only covers the spread between blocks.
/// That spread is still bounded: a block whose scale falls further below the largest one than the
/// block dtype can express is stored at that dtype's smallest value, far too coarse for it, and
/// every value in the block quantizes to zero. [`ScaleDtype::UE4M3`] spans about 2^18 this way, so
/// a tensor holding a genuine outlier can lose its ordinary values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QuantScheme {
    /// The logical data type of quantized input values (e.g., [`QuantValue::Q8F`]).
    ///
    /// This defines how values are interpreted during computation, independent of how they're stored.
    pub value: QuantValue,
    /// Data type used for storing quantized values.
    pub store: QuantStore,
    /// Quantization mode (e.g., symmetric).
    pub mode: QuantMode,
    /// The per-tensor scale level. Private with [`tensor_scale`](Self::tensor_scale) as the
    /// reader, which resolves a scheme storing no level at all to a per-tensor f32 scale.
    tensor: Option<ScaleDtype>,
    /// The per-block scale level, the innermost when both levels are present.
    block: Option<BlockScale>,
}

impl Default for QuantScheme {
    fn default() -> Self {
        Self {
            value: QuantValue::Q8F,
            store: QuantStore::PackedU32(0),
            mode: QuantMode::Symmetric,
            tensor: None,
            block: None,
        }
    }
}

impl QuantScheme {
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

    /// Set the per-tensor scale level, stored as `dtype`.
    pub fn per_tensor(mut self, dtype: ScaleDtype) -> Self {
        self.tensor = Some(dtype);
        self
    }

    /// Set the per-block scale level: one scale per block of `block` values, stored as `dtype`.
    pub fn per_block(mut self, block: impl AsRef<[u8]>, dtype: ScaleDtype) -> Self {
        self.block = Some(BlockScale {
            size: BlockSize::new(block),
            dtype,
        });
        self
    }

    /// The per-tensor scale level, the global level when a block level is present.
    ///
    /// A scheme storing no level at all resolves here to a per-tensor f32 scale; the resolution
    /// is not stored, so such a scheme compares equal to [`Default`], not to an explicit
    /// `per_tensor(F32)`.
    pub fn tensor_scale(&self) -> Option<ScaleDtype> {
        if self.tensor.is_none() && self.block.is_none() {
            return Some(ScaleDtype::F32);
        }
        self.tensor
    }

    /// The per-block scale level, the innermost when both levels are present.
    pub fn block_scale(&self) -> Option<BlockScale> {
        self.block
    }

    /// The number of scale levels: as many scale tensors ride along with the values.
    pub fn num_levels(&self) -> usize {
        self.block_scale().is_some() as usize + self.tensor_scale().is_some() as usize
    }

    /// The innermost level's scale dtype, the type the per-position scales are stored in.
    pub fn scale_dtype(&self) -> ScaleDtype {
        self.block
            .map(|block| block.dtype)
            .or(self.tensor)
            .unwrap_or(ScaleDtype::F32)
    }

    /// The block level's size, or [`None`] for per-tensor quantization.
    pub fn block_size(&self) -> Option<BlockSize> {
        self.block.map(|block| block.size)
    }

    /// Swap two tensor dimensions in the block level, mirroring `shape.swap(dim0, dim1)`. The
    /// per-tensor level is unaffected.
    ///
    /// `dim0`/`dim1` are bare indices on purpose, mirroring `[T]::swap`'s own signature.
    pub fn swap_block_dims(&mut self, rank: usize, dim0: usize, dim1: usize) {
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(dim0, dim1);
        self.permute_block_dims(rank, &axes);
    }

    /// Permute the block level, mirroring a permutation of the tensor's axes. The per-tensor
    /// level is unaffected.
    pub fn permute_block_dims(&mut self, rank: usize, axes: &[usize]) {
        if let Some(block) = &mut self.block {
            let dims = block.size.to_dim_vec(rank);
            let permuted: Vec<u8> = axes.iter().map(|&axis| dims[axis]).collect();
            block.size = BlockSize::new(permuted);
        }
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

/// The per-block scale level of a [`QuantScheme`]: one scale per block of values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockScale {
    /// The block of values sharing one scale.
    pub size: BlockSize,
    /// The dtype the level's scales are stored in.
    pub dtype: ScaleDtype,
}

impl ScaleDtype {
    /// The largest finite value representable by the dtype.
    ///
    /// A two-level scheme picks its per-tensor scale so that the largest block scale lands here,
    /// which is what keeps the block scales inside the range their type can express. That recipe
    /// only holds for a block dtype narrower than the scale it divides: dividing by
    /// [`ScaleDtype::F32`]'s or [`ScaleDtype::UE8M0`]'s maximum drives the per-tensor scale
    /// subnormal and the renormalized block scales to infinity. A two-level scheme has nothing to
    /// gain from those params anyway, since their block scales already reach the full range.
    pub fn max_representable(&self) -> f32 {
        match self {
            ScaleDtype::F32 => f32::MAX,
            ScaleDtype::F16 => half::f16::MAX.to_f32(),
            ScaleDtype::BF16 => half::bf16::MAX.to_f32(),
            // Spelled out because `ue8m0` and `e4m3` sit behind the `fp8` feature and this
            // function is not gated. The tests check both against those types when it is on.
            ScaleDtype::UE8M0 => f32::from_bits(0x7F00_0000), // 2^127
            ScaleDtype::UE4M3 => 448.0,
        }
    }

    /// The smallest value representable by the dtype that is not below `scale`.
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
    /// Every dtype answers. [`ScaleDtype::UE8M0`] takes its own path rather than the shared grid
    /// below: its range runs to 2^-127, which is subnormal in f32, so the two ends need clamping
    /// before the bit stepping is meaningful. Between them the rule is the same one — a ue8m0
    /// value is a bare exponent, so rounding up to it is rounding up to a power of two.
    pub fn round_up(&self, scale: f32) -> Option<f32> {
        match self {
            ScaleDtype::F32 => {
                return Some(scale);
            }
            ScaleDtype::UE8M0 => {
                return Some(round_up_to_power_of_two(scale));
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

    /// The dtype's grid, expressed on the f32 bit pattern. See [`F32Grid`].
    ///
    /// bf16 reports no subnormal range because it does not need the separate treatment: its pattern
    /// is f32's top half all the way down, so the bit step stays right where the others stop. Its
    /// own subnormals start at 2^-133, which is subnormal in f32 too and flushed to zero by most
    /// backends.
    ///
    /// # Panics
    ///
    /// For [`ScaleDtype::F32`], which is the grid itself.
    pub fn f32_grid(&self) -> F32Grid {
        /// One f32 ulp per dtype ulp: the mantissa bits f32 carries and the dtype does not.
        const fn bit_step(mantissa_digits: u32) -> u32 {
            1 << (f32::MANTISSA_DIGITS - mantissa_digits)
        }

        match self {
            ScaleDtype::F16 => F32Grid {
                bit_step: bit_step(half::f16::MANTISSA_DIGITS),
                subnormals: Some(SubnormalRange {
                    min_normal: half::f16::MIN_POSITIVE.to_f32(),
                    spacing: half::f16::MIN_POSITIVE_SUBNORMAL.to_f32(),
                }),
            },
            ScaleDtype::BF16 => F32Grid {
                bit_step: bit_step(half::bf16::MANTISSA_DIGITS),
                subnormals: None,
            },
            // Spelled out rather than read off `e4m3`, which sits behind the `fp8` feature while
            // this is not gated. The tests check them against that type when it is on.
            ScaleDtype::UE4M3 => F32Grid {
                bit_step: bit_step(4),
                subnormals: Some(SubnormalRange {
                    min_normal: 0.015625, // 2^-6
                    spacing: 0.001953125, // 2^-9
                }),
            },
            ScaleDtype::F32 => {
                unimplemented!("F32 is the grid, it has no narrower one to round onto")
            }
            // No mantissa at all: the grid is the powers of two, so the step clears every f32
            // mantissa bit. `subnormals` stays `None` because ue8m0 has no subnormal *ladder* —
            // its bottom is the single value 2^-127, which the callers clamp to.
            ScaleDtype::UE8M0 => F32Grid {
                bit_step: bit_step(1),
                subnormals: None,
            },
        }
    }

    /// The smallest and largest values [`ScaleDtype::UE8M0`] represents: 2^-127 and 2^127.
    ///
    /// The minimum is subnormal in f32 and the maximum is the largest power of two it holds, so
    /// both are spelled as bit patterns rather than computed.
    pub const UE8M0_MIN: f32 = f32::from_bits(0x0040_0000);
    /// See [`ScaleDtype::UE8M0_MIN`].
    pub const UE8M0_MAX: f32 = f32::from_bits(0x7F00_0000);
}

/// A `ue8m0` scale as its stored byte: the code is the exponent, biased by 127.
///
/// Rounds up, which is both the storage rule for a scale and what the host `ue8m0` codec and
/// CUDA's `__nv_cvt_bfloat16raw_to_e8m0` (at `cudaRoundPosInf`) already do — a scale rounded down
/// puts the block's largest value outside the quantization range.
///
/// Lives here rather than on the `ue8m0` type so it is available without the `float4` feature:
/// serialization needs it, and `ue8m0` is a bare exponent, so the byte is the whole of it.
pub fn f32_to_ue8m0(scale: f32) -> u8 {
    let rounded = round_up_to_power_of_two(scale);
    if rounded.is_nan() {
        return 0xFF;
    }
    // Codes 1..=254 are f32's own exponent field; the clamping above keeps the shift in range,
    // and 2^-127 is subnormal in f32, so it lands on the exponent field 0 that code 0 names.
    (rounded.to_bits() >> 23) as u8
}

/// The value a `ue8m0` byte stands for. Inverse of [`f32_to_ue8m0`] on every code it produces.
pub fn ue8m0_to_f32(code: u8) -> f32 {
    match code {
        // f32 has no exponent field 0 to spare: its own is the subnormals.
        0 => ScaleDtype::UE8M0_MIN,
        0xFF => f32::NAN,
        code => f32::from_bits((code as u32) << 23),
    }
}

/// The smallest power of two not below `scale`, saturated into ue8m0's range.
///
/// A ue8m0 code *is* an exponent, so this is the whole storage rule for that dtype. Clamping both
/// ends first is what lets the middle be the same mantissa-clearing step the other dtypes use:
/// below 2^-127 there is nothing to round onto, and above 2^127 the step would carry into f32's
/// infinity and take every value scaled by it with it. Zero clamps up to the minimum — ue8m0 has
/// no zero, and a zero scale reconstructs an all-zero block correctly at any scale.
fn round_up_to_power_of_two(scale: f32) -> f32 {
    if scale.is_nan() {
        return scale;
    }
    debug_assert!(scale >= 0.0, "a quantization scale is never negative");

    if scale <= ScaleDtype::UE8M0_MIN {
        return ScaleDtype::UE8M0_MIN;
    }
    if scale >= ScaleDtype::UE8M0_MAX {
        return ScaleDtype::UE8M0_MAX;
    }

    let grid = ScaleDtype::UE8M0.f32_grid();
    f32::from_bits((scale.to_bits() + grid.round_up_bias()) & grid.truncate_mask())
}

/// A narrower float format's grid, laid over the f32 bit pattern.
///
/// f32 carries every dtype this exists for exactly, so the grid can be walked there rather than
/// through the storage type. A value representable in the dtype leaves the low f32 mantissa bits
/// zero, so one dtype ulp is an increment at that position and the carry into the exponent falls
/// out on its own. Working in f32 also keeps the grid available to backends with no narrow integer,
/// and to builds without the `fp8` feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F32Grid {
    /// One step up in the normal range, as an increment on the f32 bit pattern.
    pub bit_step: u32,
    /// The dtype's subnormals, for the formats whose subnormals land in f32's normal range.
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
    /// Clears the mantissa bits the dtype does not carry, truncating a bit pattern onto the grid.
    pub fn truncate_mask(&self) -> u32 {
        !(self.bit_step - 1)
    }

    /// Added to a bit pattern before [`truncate_mask`](Self::truncate_mask) to turn that truncation
    /// into a round up. The carry it can produce is only safe below the dtype's maximum, which is
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
    /// The stored field is an index into a lookup table of `2^bits` floats, not a number: a read
    /// reconstructs `table[field] * scale`. (Known as a codebook in the quantization literature —
    /// NF4, K-quants, and vector quantizers all decode this way.) The table travels as its own
    /// binding beside the values and scales; only the field's bit width is read from
    /// [`QuantScheme::value`], since an index has no sign or float semantics of its own.
    Lookup,
}

/// The data type a scale level stores its scales in.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ScaleDtype {
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
#[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockSize {
    storage: [u8; MAX_DIMS],
    len: u8,
}

/// Hand-written: `storage` precedes `len`, so a derived `Ord` would compare filler bytes before
/// length.
impl PartialOrd for BlockSize {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BlockSize {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.len, self.as_slice()).cmp(&(other.len, other.as_slice()))
    }
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
    ///
    /// The result is canonical, so equal rank-relative blocks compare and hash equal however they
    /// are spelled: leading unit dimensions are dropped, since the missing-dimension fill restates
    /// them. In particular, `[1, 32]` canonicalizes to `[32]`. Whole-tensor granularity is a
    /// scheme's per-tensor level, not a block size.
    pub fn new(values: impl AsRef<[u8]>) -> Self {
        Self::canonicalize(values.as_ref())
    }

    fn canonicalize(values: &[u8]) -> Self {
        let skip = values
            .iter()
            .position(|&value| value != 1)
            .unwrap_or(values.len());
        let values = &values[skip..];
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

    /// Return a slice of only the initialized values
    pub fn as_slice(&self) -> &[u8] {
        &self.storage[..self.len as usize]
    }

    /// Return a vec of only the initialized values
    pub fn to_vec(&self) -> Vec<u8> {
        self.storage[..self.len as usize].to_vec()
    }

    /// Returns `N` dimensions, unsqueezing if necessary. Missing leading dimensions fill with `1`.
    pub fn as_dim<const N: usize>(&self) -> [u8; N] {
        let data_len = N.min(self.len as usize);
        let data_start = N - data_len;
        let mut out = [1; N];
        out[data_start..].copy_from_slice(&self.storage[..data_len]);
        out
    }

    /// Returns a vector of `len` dimensions, unsqueezing if necessary. Missing leading dimensions
    /// fill with `1`.
    pub fn to_dim_vec(&self, len: usize) -> Vec<u8> {
        let data_len = len.min(self.len as usize);
        let data_start = len - data_len;
        let mut out = vec![1; len];
        out[data_start..].copy_from_slice(&self.storage[..data_len]);
        out
    }

    /// How many blocks cover each dimension of `shape`, which is the shape of the scale grid:
    /// one scale per block.
    pub fn num_blocks(&self, shape: &[usize]) -> Vec<usize> {
        self.to_dim_vec(shape.len())
            .into_iter()
            .zip(shape)
            .map(|(block, &dim)| dim.div_ceil(block as usize))
            .collect()
    }

    /// Create an iterator over all stored dimensions
    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        self.as_slice().iter()
    }

    /// Returns the total number of elements in each block.
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
    fn blocks_remain_rank_relative() {
        assert_ne!(BlockSize::new([32]), BlockSize::new([32, 32]));
        assert_eq!(BlockSize::new([32]).to_dim_vec(2), vec![1, 32]);
        assert_eq!(BlockSize::new([32, 32]).to_dim_vec(2), vec![32, 32]);
    }

    #[test]
    fn leading_unit_dimensions_canonicalize_away() {
        assert_eq!(BlockSize::new([1, 32]), BlockSize::new([32]));
    }

    #[test]
    fn leading_unit_dimensions_beyond_the_cap_still_canonicalize() {
        assert_eq!(
            BlockSize::new([1, 1, 8, 4, 2, 3]),
            BlockSize::new([8, 4, 2, 3])
        );
    }

    #[test]
    fn there_is_one_block_per_scale() {
        assert_eq!(BlockSize::new([32]).num_blocks(&[8, 64]), vec![8, 2]);
        assert_eq!(BlockSize::new([4, 32]).num_blocks(&[8, 64]), vec![2, 2]);
        assert_eq!(BlockSize::new([32]).num_blocks(&[4, 8, 64]), vec![4, 8, 2]);
    }

    #[test]
    fn a_partial_block_still_takes_a_scale() {
        assert_eq!(BlockSize::new([32]).num_blocks(&[8, 70]), vec![8, 3]);
    }

    #[test]
    fn the_default_scheme_resolves_to_per_tensor_f32() {
        let scheme = QuantScheme::default();
        assert_eq!(scheme.tensor_scale(), Some(ScaleDtype::F32));
        assert_eq!(scheme.block_scale(), None);
        assert_eq!(scheme.scale_dtype(), ScaleDtype::F32);
        assert_eq!(scheme.block_size(), None);
        assert_eq!(scheme.num_levels(), 1);
    }

    #[test]
    fn a_block_level_stands_alone() {
        let scheme = QuantScheme::default().per_block([32], ScaleDtype::F16);
        assert_eq!(scheme.tensor_scale(), None);
        assert_eq!(scheme.scale_dtype(), ScaleDtype::F16);
        assert_eq!(scheme.block_size(), Some(BlockSize::new([32])));
        assert_eq!(scheme.num_levels(), 1);
    }

    #[test]
    fn both_levels_nest_the_block_inside_the_tensor() {
        let scheme = QuantScheme::default()
            .per_block([16], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32);
        assert_eq!(scheme.scale_dtype(), ScaleDtype::UE4M3);
        assert_eq!(scheme.tensor_scale(), Some(ScaleDtype::F32));
        assert_eq!(scheme.num_levels(), 2);
    }

    #[test]
    fn levels_set_in_any_order_are_the_same_scheme() {
        assert_eq!(
            QuantScheme::default()
                .per_block([16], ScaleDtype::UE4M3)
                .per_tensor(ScaleDtype::F32),
            QuantScheme::default()
                .per_tensor(ScaleDtype::F32)
                .per_block([16], ScaleDtype::UE4M3),
        );
    }

    #[test]
    fn swapping_dims_rewrites_the_block_and_leaves_the_tensor_level_alone() {
        let mut scheme = QuantScheme::default()
            .per_block([4, 32], ScaleDtype::F16)
            .per_tensor(ScaleDtype::F32);
        scheme.swap_block_dims(2, 0, 1);
        assert_eq!(
            scheme,
            QuantScheme::default()
                .per_block([32, 4], ScaleDtype::F16)
                .per_tensor(ScaleDtype::F32)
        );

        let mut per_tensor = QuantScheme::default();
        per_tensor.swap_block_dims(2, 0, 1);
        assert_eq!(per_tensor, QuantScheme::default());
    }

    #[test]
    fn swapping_dims_canonicalizes_the_block() {
        let mut scheme = QuantScheme::default().per_block([32, 1], ScaleDtype::F32);
        scheme.swap_block_dims(2, 0, 1);
        assert_eq!(scheme.block_size(), Some(BlockSize::new([32])));
    }

    #[test]
    fn permuting_dims_rewrites_the_block() {
        let mut scheme = QuantScheme::default().per_block([1, 4, 32], ScaleDtype::F16);
        scheme.permute_block_dims(3, &[2, 0, 1]);
        assert_eq!(scheme.block_size(), Some(BlockSize::new([32, 1, 4])));
    }

    #[test]
    fn round_up_never_lands_below_the_scale() {
        for dtype in [ScaleDtype::F16, ScaleDtype::BF16, ScaleDtype::UE4M3] {
            for exp in -12..8 {
                for step in 1..17 {
                    let scale = (step as f32 / 16.0) * 2f32.powi(exp);
                    let up = dtype.round_up(scale).unwrap();
                    assert!(
                        up >= scale,
                        "{dtype:?}: {up} is below {scale}, which clips the block maximum"
                    );
                }
            }
        }
    }

    #[test]
    fn round_up_saturates_rather_than_stepping_off_the_top() {
        for dtype in [ScaleDtype::F16, ScaleDtype::BF16, ScaleDtype::UE4M3] {
            let max = dtype.max_representable();
            assert_eq!(dtype.round_up(max).unwrap(), max);
            assert!(dtype.round_up(max * 2.0).unwrap().is_finite());
        }
    }

    /// Every variant is dispatched somewhere, so none of them may panic here.
    #[test]
    fn round_up_answers_for_every_param() {
        for dtype in [
            ScaleDtype::F32,
            ScaleDtype::F16,
            ScaleDtype::BF16,
            ScaleDtype::UE8M0,
            ScaleDtype::UE4M3,
        ] {
            assert!(dtype.round_up(0.3).is_some(), "{dtype:?}");
        }
    }

    /// `ue8m0` stores a bare exponent, so rounding up to it is rounding up to a power of two.
    #[test]
    fn ue8m0_rounds_up_to_a_power_of_two() {
        for exp in -120..120 {
            let power = 2f32.powi(exp);
            // Already a power of two: nothing to round.
            assert_eq!(ScaleDtype::UE8M0.round_up(power).unwrap(), power, "2^{exp}");
            // Anything above it goes to the next one up, however little above.
            for scale in [power * 1.0001, power * 1.5, power * 1.9999] {
                assert_eq!(
                    ScaleDtype::UE8M0.round_up(scale).unwrap(),
                    power * 2.0,
                    "{scale} (2^{exp} scaled)"
                );
            }
        }
    }

    /// Both ends saturate. The bottom is the reason `ue8m0` needs its own path at all: 2^-127 is
    /// subnormal in f32, and zero — which a fully-zero block calibrates to — is not a `ue8m0`
    /// value in the first place.
    #[test]
    fn ue8m0_saturates_at_both_ends() {
        let min = ScaleDtype::UE8M0_MIN;
        let max = ScaleDtype::UE8M0_MAX;

        for scale in [0.0, f32::MIN_POSITIVE * 0.5, min * 0.5, min] {
            assert_eq!(ScaleDtype::UE8M0.round_up(scale).unwrap(), min, "{scale:e}");
        }
        for scale in [max, max * 2.0, f32::MAX, f32::INFINITY] {
            assert_eq!(ScaleDtype::UE8M0.round_up(scale).unwrap(), max, "{scale:e}");
        }
    }

    /// Every byte stands for a value that encodes back to it — the codec is a bijection on the
    /// codes, which is what serializing a scale and reading it back depends on.
    #[test]
    fn every_ue8m0_code_round_trips() {
        for code in 0..=0xFEu8 {
            let value = ue8m0_to_f32(code);
            assert_eq!(
                f32_to_ue8m0(value),
                code,
                "code {code} decoded to {value:e}"
            );
        }
        assert!(ue8m0_to_f32(0xFF).is_nan());
    }

    /// The codec agrees with `round_up`, so a scale stored through either lands on the same value.
    #[test]
    fn the_ue8m0_codec_agrees_with_the_round_up_rule() {
        for exp in -130..130 {
            for factor in [1.0, 1.3, 1.9] {
                let scale = factor * 2f32.powi(exp);
                assert_eq!(
                    ue8m0_to_f32(f32_to_ue8m0(scale)),
                    ScaleDtype::UE8M0.round_up(scale).unwrap(),
                    "{scale:e}"
                );
            }
        }
    }

    /// The answer is always representable, so rounding it again changes nothing.
    #[test]
    fn ue8m0_round_up_is_idempotent() {
        for exp in -130..130 {
            for factor in [1.0, 1.3, 1.7] {
                let scale = factor * 2f32.powi(exp);
                let up = ScaleDtype::UE8M0.round_up(scale).unwrap();
                assert_eq!(
                    ScaleDtype::UE8M0.round_up(up).unwrap(),
                    up,
                    "not idempotent at {scale:e}"
                );
                assert!(up >= scale.min(ScaleDtype::UE8M0_MAX), "{up:e} < {scale:e}");
            }
        }
    }

    #[test]
    fn round_up_is_the_identity_for_f32() {
        for scale in [1.0e-30, 0.1, 1.0, 12345.678, f32::MAX] {
            assert_eq!(ScaleDtype::F32.round_up(scale).unwrap(), scale);
        }
    }

    /// The checks that need the real storage types to compare against.
    #[cfg(feature = "fp8")]
    mod storage_types {
        use super::*;

        #[test]
        fn round_up_is_the_nearest_representable_value_not_below() {
            // Rounding up must not overshoot: stepping down from the answer has to land below.
            for dtype in [ScaleDtype::F16, ScaleDtype::BF16, ScaleDtype::UE4M3] {
                for exp in -8..6 {
                    let scale = 1.7 * 2f32.powi(exp);
                    let up = dtype.round_up(scale).unwrap();
                    assert_eq!(
                        up,
                        dtype.round_up(up).unwrap(),
                        "{dtype:?}: not idempotent at {scale}"
                    );
                    assert!(
                        step(dtype, up, -1) < scale,
                        "{dtype:?}: {up} overshoots {scale} by at least a step"
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
            for dtype in [ScaleDtype::F16, ScaleDtype::BF16, ScaleDtype::UE4M3] {
                let grid = dtype.f32_grid();

                // bf16 deliberately reports no subnormal range, since its bit step covers them too.
                if let Some(subnormals) = grid.subnormals {
                    assert_eq!(
                        subnormals.min_normal,
                        min_normal(dtype),
                        "{dtype:?}: minimum normal"
                    );
                    assert_eq!(
                        subnormals.spacing,
                        step(dtype, 0.0, 1),
                        "{dtype:?}: subnormal spacing"
                    );
                }

                // Walk the whole normal range: one step on the f32 pattern has to be one step in
                // the type, at every exponent.
                let mut value = min_normal(dtype);
                let max = dtype.max_representable();
                while value < max {
                    let stepped = f32::from_bits(value.to_bits() + grid.bit_step);
                    assert_eq!(
                        stepped,
                        step(dtype, value, 1),
                        "{dtype:?}: step above {value}"
                    );
                    value = stepped;
                }
                assert_eq!(
                    value, max,
                    "{dtype:?}: the grid has to land exactly on the maximum"
                );
            }
        }

        #[test]
        fn max_representable_matches_the_e4m3_type() {
            assert_eq!(
                ScaleDtype::UE4M3.max_representable(),
                crate::e4m3::MAX.to_f32()
            );
        }

        /// The other limit spelled out as a literal. `ue8m0` is exponent only, so its maximum is
        /// the power of two the hex literal encodes.
        #[test]
        fn max_representable_matches_the_e8m0_type() {
            assert_eq!(
                ScaleDtype::UE8M0.max_representable(),
                crate::ue8m0::MAX.to_f32()
            );
        }

        /// `offset` representable steps from `value` in `dtype`, for positive values. Counted on
        /// the storage type's own bit pattern, so this is an oracle independent of the grid under
        /// test.
        fn step(dtype: ScaleDtype, value: f32, offset: i32) -> f32 {
            match dtype {
                ScaleDtype::F16 => half::f16::from_bits(
                    (half::f16::from_f32(value).to_bits() as i32 + offset) as u16,
                )
                .to_f32(),
                ScaleDtype::BF16 => half::bf16::from_bits(
                    (half::bf16::from_f32(value).to_bits() as i32 + offset) as u16,
                )
                .to_f32(),
                ScaleDtype::UE4M3 => crate::e4m3::from_bits(
                    (crate::e4m3::from_f32(value).to_bits() as i32 + offset) as u8,
                )
                .to_f32(),
                ScaleDtype::F32 | ScaleDtype::UE8M0 => unreachable!(),
            }
        }

        fn min_normal(dtype: ScaleDtype) -> f32 {
            match dtype {
                ScaleDtype::F16 => half::f16::MIN_POSITIVE.to_f32(),
                ScaleDtype::BF16 => half::bf16::MIN_POSITIVE.to_f32(),
                ScaleDtype::UE4M3 => crate::e4m3::MIN_POSITIVE.to_f32(),
                ScaleDtype::F32 | ScaleDtype::UE8M0 => unreachable!(),
            }
        }
    }
}
