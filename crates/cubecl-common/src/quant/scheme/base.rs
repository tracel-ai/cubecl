use alloc::vec::Vec;
use core::default::Default;
use serde::{Deserialize, Serialize};

use super::code::PackedFields;
use super::saved::SavedScheme;
use super::{BlockScale, BlockSize};
use crate::{e2m1, ue8m0};

/// Describes a quantization scheme/configuration.
///
/// Scales come at one or two levels, a [`finest`](Self::finest) one and a
/// [`coarser`](Self::coarser) one over it, which [`per_block`](Self::per_block) and
/// [`coarser_per_block`](Self::coarser_per_block) set ([`per_tensor`](Self::per_tensor) sets
/// either). A value's scale is the product of the scales of the blocks it is in, one block a
/// level:
///
/// ```
/// # use cubecl_common::quant::scheme::{BlockExtent, BlockSize, QuantScheme, ScaleDtype};
/// // One scale for the whole tensor, stored as f32: the default scheme's level.
/// QuantScheme::default().per_tensor(ScaleDtype::F32);
///
/// // One scale per block of 32 values.
/// QuantScheme::default().per_block([32], ScaleDtype::F32);
///
/// // Two levels: ue4m3 block scales, normalized by a single per-tensor f32 scale.
/// QuantScheme::default()
///     .per_block([16], ScaleDtype::UE4M3)
///     .per_tensor(ScaleDtype::F32);
///
/// // One scale per row, over the whole last axis, however long it is.
/// let row = BlockSize::ROW;
/// QuantScheme::default().per_block(row, ScaleDtype::F32);
///
/// // Two levels: ue8m0 block scales under one f32 scale per row.
/// QuantScheme::default()
///     .per_block([32], ScaleDtype::UE8M0)
///     .coarser_per_block(row, ScaleDtype::F32);
///
/// // Two levels: one f32 scale per 64 values along the last axis, four blocks of 16.
/// QuantScheme::default()
///     .per_block([16], ScaleDtype::UE4M3)
///     .coarser_per_block([64], ScaleDtype::F32);
/// ```
///
/// A two-level scheme exists so block scales can live in a narrow type: the coarser scale absorbs
/// the dynamic range of the values it covers, and the block dtype only covers the spread between
/// their blocks. That spread is still bounded: a block whose scale falls further below the largest
/// one than the block dtype can express is stored at that dtype's smallest value, far too coarse
/// for it, and every value in the block quantizes to zero. [`ScaleDtype::UE4M3`] spans about 2^18
/// this way, so a tensor holding a genuine outlier can lose its ordinary values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "SavedScheme")]
pub struct QuantScheme {
    /// The logical data type of quantized input values (e.g., [`QuantValue::Q8F`]).
    ///
    /// This defines how values are interpreted during computation, independent of how they're stored.
    pub value: QuantValue,
    /// Data type used for storing quantized values.
    pub store: QuantStore,
    /// The level whose scales lie beside the values, one per block of its size.
    pub finest: BlockScale,
    /// A level over the finest one, where the scheme has one: each of its blocks covers whole
    /// blocks of the finest level.
    ///
    /// One optional level rather than a list only because no format has a third yet, and this
    /// keeps the scheme, which every quantized dtype carries, at the size two levels take. A third
    /// is this field becoming an array of them, coarsest last, read by [`levels`](Self::levels),
    /// [`to_code`](Self::to_code) and the saved form.
    pub coarser: Option<BlockScale>,
}

impl Default for QuantScheme {
    fn default() -> Self {
        Self {
            value: QuantValue::Q8F,
            store: QuantStore::PackedU32(0),
            finest: BlockScale {
                size: BlockSize::TENSOR,
                dtype: ScaleDtype::F32,
            },
            coarser: None,
        }
    }
}

impl QuantScheme {
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

    /// Set one scale for the whole tensor, stored as `dtype`: the coarser level over a finest level
    /// of blocks, and otherwise the finest level itself. So the order counts:
    /// `per_tensor(..).per_block(..)` replaces the tensor's scale by the blocks'.
    pub fn per_tensor(mut self, dtype: ScaleDtype) -> Self {
        let level = BlockScale {
            size: BlockSize::TENSOR,
            dtype,
        };
        match self.finest.size == BlockSize::TENSOR {
            true => self.finest = level,
            false => self.coarser = Some(level),
        }
        self
    }

    /// Set the finest level: one scale per block of `size`, stored as `dtype`. The coarser level
    /// is kept, so a scale for the whole tensor is set after the blocks it covers.
    ///
    /// # Panics
    ///
    /// As [`BlockSize::new`], for a block given as its values.
    pub fn per_block(mut self, size: impl Into<BlockSize>, dtype: ScaleDtype) -> Self {
        self.finest = BlockScale {
            size: size.into(),
            dtype,
        };
        self
    }

    /// Set the coarser level: one scale per block of `size`, stored as `dtype`.
    ///
    /// # Assumptions
    ///
    /// Each block of `size` covers whole blocks of the finest level.
    ///
    /// # Panics
    ///
    /// As [`BlockSize::new`], for a block given as its values.
    pub fn coarser_per_block(mut self, size: impl Into<BlockSize>, dtype: ScaleDtype) -> Self {
        self.coarser = Some(BlockScale {
            size: size.into(),
            dtype,
        });
        self
    }

    /// The shape of each level's grid of scales over a tensor of `shape`, the finest first: one
    /// scale per block, the last block of an axis possibly in part.
    pub fn scale_grids(&self, shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + use<> {
        let shape = shape.to_vec();
        self.levels()
            .map(move |level| level.size.num_blocks(&shape))
    }

    /// The scale levels, the finest first: as many scale tensors ride along with the values.
    pub fn levels(&self) -> impl Iterator<Item = BlockScale> + use<> {
        core::iter::once(self.finest).chain(self.coarser)
    }

    /// The finest of the [`levels`](Self::levels), the one whose scales are laid out beside the
    /// values.
    pub fn finest(&self) -> BlockScale {
        self.finest
    }

    /// Swap two tensor dimensions in the scale levels, mirroring `shape.swap(dim0, dim1)`.
    ///
    /// `dim0`/`dim1` are bare indices on purpose, mirroring `[T]::swap`'s own signature.
    ///
    /// # Panics
    ///
    /// Where a dimension is past `rank`, and as [`BlockSize::permuted`].
    pub fn swap_block_dims(&mut self, rank: usize, dim0: usize, dim1: usize) {
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(dim0, dim1);
        self.permute_block_dims(&axes);
    }

    /// Permute the scale levels, mirroring a permutation of the tensor's axes.
    ///
    /// # Panics
    ///
    /// As [`BlockSize::permuted`].
    pub fn permute_block_dims(&mut self, axes: &[usize]) {
        self.finest.size = self.finest.size.permuted(axes);
        if let Some(coarser) = &mut self.coarser {
            coarser.size = coarser.size.permuted(axes);
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

    /// Swap two tensor dimensions in the scale levels and the packing dim, mirroring
    /// `shape.swap(dim0, dim1)`.
    ///
    /// # Panics
    ///
    /// As [`swap_block_dims`](Self::swap_block_dims), and where the packing dim is past `rank`.
    pub fn swap_dims(&mut self, rank: usize, dim0: usize, dim1: usize) {
        self.swap_block_dims(rank, dim0, dim1);
        self.swap_packing_dim(rank, dim0, dim1);
    }

    /// Permute the scale levels and the packing dim, mirroring a permutation of the tensor's axes.
    ///
    /// # Panics
    ///
    /// As [`BlockSize::permuted`], and where the packing dim is not one of `axes`.
    pub fn permute_dims(&mut self, axes: &[usize]) {
        self.permute_block_dims(axes);
        self.permute_packing_dim(axes);
    }

    /// Update the packing dim for a reshape to `rank` dimensions: a packing dim the reshape drops
    /// becomes the innermost one. Only the index changes, so a backend holding codes packed in
    /// words repacks them itself.
    pub fn reshape_packing_dim(&mut self, rank: usize) {
        if let QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim) =
            &mut self.store
            && *packed_dim >= rank.max(1)
        {
            *packed_dim = 0;
        }
    }
}

impl ScaleDtype {
    /// The largest finite value representable by the dtype.
    pub fn max_representable(&self) -> f32 {
        match self {
            ScaleDtype::F32 => f32::MAX,
            ScaleDtype::F16 => half::f16::MAX.to_f32(),
            ScaleDtype::BF16 => half::bf16::MAX.to_f32(),
            ScaleDtype::UE8M0 => ue8m0::MAX.to_f32(),
            // Spelled out because `e4m3` sits behind the `fp8` feature and this function is not
            // gated. The tests check it against that type when it is on.
            ScaleDtype::UE4M3 => 448.0,
        }
    }

    /// Where a two-level scheme puts the largest finest-level scale of a coarser block, when the
    /// finest level is stored as this dtype: the coarser scale is that largest one divided by
    /// this value, and every finest scale under it is stored relative to it.
    ///
    /// A dtype narrower than f32's range puts it at its top, which leaves the most room below for
    /// the smaller ones. A dtype whose range reaches f32's puts it at one, so the coarser scale is
    /// the largest one itself: at its top, the coarser scale would go subnormal and the others
    /// would overflow. For [`ScaleDtype::UE8M0`] that also makes each stored scale a power of two
    /// at most one, the offset of its block below the largest.
    pub fn largest_under_coarser(&self) -> f32 {
        match self {
            ScaleDtype::F16 | ScaleDtype::UE4M3 => self.max_representable(),
            ScaleDtype::F32 | ScaleDtype::BF16 | ScaleDtype::UE8M0 => 1.0,
        }
    }

    /// The smallest value representable by the dtype that is not below `scale`.
    ///
    /// Storing a quantization scale wants this rather than the nearest value. Rounding down puts
    /// the scale below what calibration asked for, so every value at the block maximum clips to
    /// the quantization range; rounding up costs one step of coarseness instead. Backends have to
    /// agree on this, or a tensor quantized on one reconstructs differently on another.
    ///
    /// This is not a cast: conversion to these types rounds to nearest, which is what a cast
    /// should do, and this is the storage policy for a scale. [`ScaleDtype::UE8M0`] is the
    /// exception, rounded by [`ue8m0`]'s own conversion, which rounds up because that format
    /// exists to store scales.
    ///
    /// `scale` must not be negative. Symmetric quantization only produces non-negative scales,
    /// and the stepping below walks away from zero for a negative input.
    pub fn round_up(&self, scale: f32) -> f32 {
        if scale.is_nan() {
            return scale;
        }
        debug_assert!(scale >= 0.0, "a quantization scale is never negative");

        match self {
            ScaleDtype::F32 => {
                return scale;
            }
            ScaleDtype::UE8M0 => {
                return ue8m0::from_f32(scale).to_f32();
            }
            _ => {}
        }

        // Nothing representable sits above the maximum, and converting past it yields an infinity
        // for the params that have one, which would make every reconstructed value NaN.
        let max = self.max_representable();
        if scale >= max {
            return max;
        }

        let grid = self.f32_grid();

        if let Some(subnormals) = grid.subnormals
            && scale < subnormals.min_normal
        {
            // Below the minimum normal the spacing stops halving, so the answer is a count of steps.
            // Qualified call: the inherent `f32::ceil` lives in std, and this crate builds no_std.
            return num_traits::Float::ceil(scale / subnormals.spacing) * subnormals.spacing;
        }

        f32::from_bits((scale.to_bits() + grid.round_up_bias()) & grid.truncate_mask())
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
    pub const UE8M0_MIN: f32 = ue8m0::MIN.to_f32();
    /// See [`ScaleDtype::UE8M0_MIN`].
    pub const UE8M0_MAX: f32 = ue8m0::MAX.to_f32();
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

/// The format of each stored value: a number (an integer or a float) the scales multiply, or an
/// index into a lookup table (`table[field] * scale`).
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
    /// A 32-bit float, one a stored element: a float operand's format.
    F32,
    /// A 16-bit float, one a stored element: a float operand's format.
    F16,
    /// A bfloat16, one a stored element: a float operand's format.
    BF16,
    /// An 8-bit index into a lookup table of `2^8` floats, not a number: a read reconstructs
    /// `table[field] * scale`. (Known as a codebook: NF4, K-quants and vector quantizers decode
    /// this way.) The table travels as its own binding beside the values and scales.
    Index8,
    /// A 4-bit index into a lookup table of `2^4` floats, read as [`Index8`](Self::Index8) is.
    Index4,
    /// A 2-bit index into a lookup table of `2^2` floats, read as [`Index8`](Self::Index8) is.
    Index2,
}

impl QuantValue {
    /// Returns the size of the quantization input type in bits.
    pub fn size_bits(&self) -> usize {
        match self {
            QuantValue::Q8F
            | QuantValue::Q8S
            | QuantValue::E4M3
            | QuantValue::E5M2
            | QuantValue::Index8 => 8,
            QuantValue::Q4F | QuantValue::Q4S | QuantValue::E2M1 | QuantValue::Index4 => 4,
            QuantValue::Q2F | QuantValue::Q2S | QuantValue::Index2 => 2,
            QuantValue::F32 => 32,
            QuantValue::F16 | QuantValue::BF16 => 16,
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
            QuantValue::E2M1 => (e2m1::MIN.to_f32(), e2m1::MAX.to_f32()),
            QuantValue::F32 => (f32::MIN, f32::MAX),
            QuantValue::F16 => (half::f16::MIN.to_f32(), half::f16::MAX.to_f32()),
            QuantValue::BF16 => (half::bf16::MIN.to_f32(), half::bf16::MAX.to_f32()),
            QuantValue::Index8 => (0.0, 255.0),
            QuantValue::Index4 => (0.0, 15.0),
            QuantValue::Index2 => (0.0, 3.0),
        }
    }

    /// If each value is a float read from its sign, exponent and mantissa bits, a minifloat or an
    /// IEEE float, rather than an integer.
    pub fn is_float(&self) -> bool {
        match self {
            Self::E4M3 | Self::E5M2 | Self::E2M1 | Self::F32 | Self::F16 | Self::BF16 => true,
            Self::Q8F
            | Self::Q8S
            | Self::Q4F
            | Self::Q4S
            | Self::Q2F
            | Self::Q2S
            | Self::Index8
            | Self::Index4
            | Self::Index2 => false,
        }
    }

    /// If each value is an index into a lookup table rather than a number.
    pub fn is_index(&self) -> bool {
        matches!(self, Self::Index8 | Self::Index4 | Self::Index2)
    }

    /// If the values are integers whose range leaves out the most negative value their bits hold.
    pub fn is_symmetric(&self) -> bool {
        match self {
            Self::Q8F
            | Self::Q4F
            | Self::Q2F
            | Self::E4M3
            | Self::E5M2
            | Self::E2M1
            | Self::F32
            | Self::F16
            | Self::BF16
            | Self::Index8
            | Self::Index4
            | Self::Index2 => false,
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

/// A fieldless enum's `to_code` and `from_code` from one table, so the two cannot disagree and a
/// variant added to the enum fails the build until it has a code.
macro_rules! codes {
    ($ty:ty { $($variant:ident => $code:literal),+ $(,)? }) => {
        impl $ty {
            pub(super) fn to_code(self) -> u64 {
                match self {
                    $(Self::$variant => $code,)+
                }
            }

            pub(super) fn from_code(code: u64) -> Option<Self> {
                match code {
                    $($code => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

codes!(QuantValue {
    Q8F => 0,
    E5M2 => 1,
    E4M3 => 2,
    Q4F => 3,
    E2M1 => 4,
    Q2F => 5,
    Q8S => 6,
    Q4S => 7,
    Q2S => 8,
    F32 => 9,
    F16 => 10,
    BF16 => 11,
    Index8 => 12,
    Index4 => 13,
    Index2 => 14,
});

// `0` is no code: a level's bits read `0` when the scheme has no such level.
codes!(ScaleDtype {
    F32 => 1,
    F16 => 2,
    BF16 => 3,
    UE8M0 => 4,
    UE4M3 => 5,
});

/// A [`QuantScheme`] packed into one `u64`, for an IR that carries a scheme as a scalar: a remote
/// backend's custom operation crosses the wire as tensors and scalars only.
///
/// The layout, from the low bits: the value (4), the store's kind (2) and packing axis (6), then
/// each level the scheme states, the finest first: its dtype (3, `0` where there is
/// no such level), whether the rest of the axes are whole (1), how many of the last axes it lists
/// (3) and the extent along each of them, the last one first (8 each, `0` for the whole axis).
impl QuantScheme {
    /// This scheme as one `u64`; [`from_code`](Self::from_code) reads it back. `None` for a scheme
    /// no code holds: a packing axis past 63, a level whose rest is neither one value nor the
    /// whole axis, or levels listing more extents than the bits left hold, five for one level and
    /// four for two.
    pub fn to_code(&self) -> Option<u64> {
        let (store, axis) = match self.store {
            QuantStore::Native => (0, 0),
            QuantStore::PackedNative(axis) => (1, axis),
            QuantStore::PackedU32(axis) => (2, axis),
        };
        let mut fields = PackedFields::default()
            .with(self.value.to_code(), 4)?
            .with(store, 2)?
            .with(axis as u64, 6)?;
        fields = self.finest.pack(fields)?;
        if let Some(coarser) = self.coarser {
            fields = coarser.pack(fields)?;
        }
        Some(fields.bits)
    }

    /// The scheme [`to_code`](Self::to_code) packed into `code`; `None` for a code no scheme packs
    /// to, so decoding is exactly the inverse of packing.
    pub fn from_code(code: u64) -> Option<Self> {
        let fields = PackedFields {
            bits: code,
            used_bits: 0,
        };
        let (value, fields) = fields.split_bits(4)?;
        let (store, fields) = fields.split_bits(2)?;
        let (axis, fields) = fields.split_bits(6)?;
        let store = match store {
            0 => QuantStore::Native,
            1 => QuantStore::PackedNative(axis as usize),
            2 => QuantStore::PackedU32(axis as usize),
            _ => return None,
        };
        let (finest, fields) = BlockScale::unpack(fields)?;
        let (coarser, _) = BlockScale::unpack(fields)?;
        // Read fields alone accept codes no scheme packs to: bits a field of this scheme does not
        // use, a listed extent that says the rest again. Packing the result again keeps the one
        // code each scheme has.
        Some(Self {
            value: QuantValue::from_code(value)?,
            store,
            finest: finest?,
            coarser,
        })
        .filter(|scheme| scheme.to_code() == Some(code))
    }
}

impl QuantScheme {
    /// Swap two tensor dimensions in the packing dim, mirroring `shape.swap(dim0, dim1)` on a
    /// tensor of `rank` dimensions: the packing dim counts from the innermost dimension, so it
    /// takes the rank to know which dimension it is.
    fn swap_packing_dim(&mut self, rank: usize, dim0: usize, dim1: usize) {
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(dim0, dim1);
        self.permute_packing_dim(&axes);
    }

    /// Permute the packing dim, mirroring a permutation of the tensor's axes.
    fn permute_packing_dim(&mut self, axes: &[usize]) {
        // A scalar keeps its packing dim, as `reshape_packing_dim` leaves it.
        if let QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim) =
            &mut self.store
            && !axes.is_empty()
        {
            let rank = axes.len();
            let packed_axis = rank
                .checked_sub(*packed_dim + 1)
                .expect("the packing dim should be one of the permuted axes");
            let new_axis = axes
                .iter()
                .position(|axis| *axis == packed_axis)
                .expect("the permutation should contain the packed axis");
            *packed_dim = rank - new_axis - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::BlockExtent;
    use super::*;
    use alloc::vec;

    fn row() -> BlockSize {
        BlockSize::ROW
    }

    fn level(size: impl Into<BlockSize>, dtype: ScaleDtype) -> BlockScale {
        BlockScale {
            size: size.into(),
            dtype,
        }
    }

    #[test]
    fn the_default_scheme_resolves_to_per_tensor_f32() {
        let levels: Vec<BlockScale> = QuantScheme::default().levels().collect();
        assert_eq!(levels, [level(BlockSize::TENSOR, ScaleDtype::F32)]);
    }

    #[test]
    fn a_stated_level_is_the_only_one() {
        let scheme = QuantScheme::default().per_block([32], ScaleDtype::F16);
        let levels: Vec<BlockScale> = scheme.levels().collect();
        assert_eq!(levels, [level([32], ScaleDtype::F16)]);
    }

    #[test]
    fn the_default_scheme_is_a_stated_scale_for_the_tensor() {
        assert_eq!(
            QuantScheme::default(),
            QuantScheme::default().per_tensor(ScaleDtype::F32)
        );
    }

    #[test]
    fn a_tensors_scale_set_after_blocks_is_the_coarser_level() {
        let scheme = QuantScheme::default()
            .per_block([16], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32);
        let levels: Vec<BlockScale> = scheme.levels().collect();
        assert_eq!(
            levels,
            [
                level([16], ScaleDtype::UE4M3),
                level(BlockSize::TENSOR, ScaleDtype::F32)
            ]
        );
    }

    #[test]
    fn blocks_set_after_a_tensors_scale_replace_it() {
        let scheme = QuantScheme::default()
            .per_tensor(ScaleDtype::F16)
            .per_block([16], ScaleDtype::UE4M3);
        let levels: Vec<BlockScale> = scheme.levels().collect();
        assert_eq!(levels, [level([16], ScaleDtype::UE4M3)]);
    }

    #[test]
    fn setting_blocks_again_replaces_the_finest_level() {
        let scheme = QuantScheme::nvfp4().per_block([32, 1], ScaleDtype::F16);
        assert_eq!(scheme.finest, level([32, 1], ScaleDtype::F16));
        assert_eq!(
            scheme.coarser,
            Some(level(BlockSize::TENSOR, ScaleDtype::F32))
        );
    }

    #[test]
    fn setting_a_tensors_scale_replaces_the_coarser_level() {
        let scheme = QuantScheme::mxfp4_row_factored().per_tensor(ScaleDtype::F32);
        assert_eq!(
            scheme.coarser,
            Some(level(BlockSize::TENSOR, ScaleDtype::F32))
        );
    }

    #[test]
    fn coarser_blocks_set_the_coarser_level_under_the_finest() {
        let scheme = QuantScheme::default()
            .per_block([32], ScaleDtype::UE8M0)
            .coarser_per_block(row(), ScaleDtype::F32);
        assert_eq!(scheme.finest, level([32], ScaleDtype::UE8M0));
        assert_eq!(scheme.coarser, Some(level(row(), ScaleDtype::F32)));
    }

    /// Blocks of 16 over `[4, 64]` lay a `[4, 4]` grid; blocks of 32 a `[4, 2]` one.
    #[test]
    fn each_level_lays_its_own_grid() {
        let scheme = QuantScheme::default()
            .per_block([16], ScaleDtype::UE4M3)
            .coarser_per_block([32], ScaleDtype::F32);
        let grids: Vec<Vec<usize>> = scheme.scale_grids(&[4, 64]).collect();
        assert_eq!(grids, [vec![4, 4], vec![4, 2]]);
    }

    #[test]
    fn a_coarser_scale_for_each_row_covers_its_blocks() {
        let grids: Vec<Vec<usize>> = QuantScheme::mxfp4_row_factored()
            .scale_grids(&[3, 100])
            .collect();
        assert_eq!(grids, [vec![3, 4], vec![3, 1]]);
    }

    #[test]
    fn a_narrow_block_dtype_lands_its_largest_scale_at_its_top() {
        assert_eq!(ScaleDtype::UE4M3.largest_under_coarser(), 448.0);
    }

    /// At its top, the coarser scale would go subnormal: `ue8m0` lands at one, the coarser scale
    /// being the largest block scale itself.
    #[test]
    fn a_block_dtype_reaching_f32s_range_lands_its_largest_scale_at_one() {
        assert_eq!(ScaleDtype::UE8M0.largest_under_coarser(), 1.0);
    }

    #[test]
    fn swapping_dims_moves_every_level() {
        let mut scheme = QuantScheme::mxfp4_row_factored();
        scheme.swap_block_dims(2, 0, 1);
        let column = BlockSize::with_rest(
            &[BlockExtent::WholeAxis, BlockExtent::VALUE],
            BlockExtent::VALUE,
        );
        assert_eq!(scheme.finest, level([32, 1], ScaleDtype::UE8M0));
        assert_eq!(scheme.coarser, Some(level(column, ScaleDtype::F32)));
    }

    #[test]
    fn a_scale_for_the_tensor_does_not_move() {
        let mut scheme = QuantScheme::default().per_tensor(ScaleDtype::F32);
        scheme.swap_block_dims(2, 0, 1);
        assert_eq!(scheme, QuantScheme::default().per_tensor(ScaleDtype::F32));
    }

    #[test]
    fn permuting_dims_moves_the_packing_dim_and_the_block_with_their_axes() {
        let mut scheme = QuantScheme::default()
            .with_store(QuantStore::PackedU32(0))
            .per_block([1, 2, 4], ScaleDtype::F32);
        scheme.permute_dims(&[2, 0, 1]);
        assert_eq!(scheme.store, QuantStore::PackedU32(2));
        assert_eq!(scheme.finest().size, BlockSize::new([4, 1, 2]));
    }

    #[test]
    fn swapping_dims_moves_the_packing_dim_only_when_its_dim_is_swapped() {
        let mut scheme = QuantScheme::default().with_store(QuantStore::PackedU32(0));
        scheme.swap_dims(3, 0, 1);
        assert_eq!(scheme.store, QuantStore::PackedU32(0));
        scheme.swap_dims(3, 1, 2);
        assert_eq!(scheme.store, QuantStore::PackedU32(1));
    }

    #[test]
    fn a_scalar_keeps_its_packing_dim() {
        let mut scheme = QuantScheme::default().with_store(QuantStore::PackedU32(0));
        scheme.reshape_packing_dim(0);
        scheme.permute_dims(&[]);
        assert_eq!(scheme.store, QuantStore::PackedU32(0));
    }

    #[test]
    fn a_reshape_that_drops_the_packing_dim_packs_innermost() {
        let mut scheme = QuantScheme::default().with_store(QuantStore::PackedNative(1));
        scheme.reshape_packing_dim(2);
        assert_eq!(scheme.store, QuantStore::PackedNative(1));
        scheme.reshape_packing_dim(1);
        assert_eq!(scheme.store, QuantStore::PackedNative(0));
    }

    #[test]
    fn round_up_never_lands_below_the_scale() {
        for dtype in [ScaleDtype::F16, ScaleDtype::BF16, ScaleDtype::UE4M3] {
            for exp in -12..8 {
                for step in 1..17 {
                    let scale = (step as f32 / 16.0) * 2f32.powi(exp);
                    let up = dtype.round_up(scale);
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
            assert_eq!(dtype.round_up(max), max);
            assert!(dtype.round_up(max * 2.0).is_finite());
        }
    }

    /// Every variant is dispatched somewhere, so none of them may panic here, and every answer is
    /// a scale something can be divided by.
    #[test]
    fn round_up_answers_for_every_param() {
        for dtype in [
            ScaleDtype::F32,
            ScaleDtype::F16,
            ScaleDtype::BF16,
            ScaleDtype::UE8M0,
            ScaleDtype::UE4M3,
        ] {
            let up = dtype.round_up(0.3);
            assert!(up.is_finite() && up > 0.0, "{dtype:?} answered {up:e}");
        }
    }

    /// 2^`exp`, from the exponent field rather than through `powi`.
    ///
    /// `powi` is only ever approximate — Miri returns a value a few ulp off, deliberately, so
    /// nothing comes to depend on one host's precision — and a power of two that is a hair off is
    /// a different power of two once its mantissa is cleared. Only for exponents f32 has a normal
    /// for: -126..=127.
    fn power_of_two(exp: i32) -> f32 {
        f32::from_bits(((exp + 127) as u32) << 23)
    }

    /// `ue8m0` stores a bare exponent, so rounding up to it is rounding up to a power of two.
    #[test]
    fn ue8m0_rounds_up_to_a_power_of_two() {
        for exp in -120..120 {
            let power = power_of_two(exp);
            // Already a power of two: nothing to round.
            assert_eq!(ScaleDtype::UE8M0.round_up(power), power, "2^{exp}");
            // Anything above it goes to the next one up, however little above.
            for scale in [power * 1.0001, power * 1.5, power * 1.9999] {
                assert_eq!(
                    ScaleDtype::UE8M0.round_up(scale),
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
            assert_eq!(ScaleDtype::UE8M0.round_up(scale), min, "{scale:e}");
        }
        for scale in [max, max * 2.0, f32::MAX, f32::INFINITY] {
            assert_eq!(ScaleDtype::UE8M0.round_up(scale), max, "{scale:e}");
        }
    }

    /// The answer is always representable, so rounding it again changes nothing.
    #[test]
    fn ue8m0_round_up_is_idempotent() {
        for exp in -130..130 {
            for factor in [1.0, 1.3, 1.7] {
                let scale = factor * 2f32.powi(exp);
                let up = ScaleDtype::UE8M0.round_up(scale);
                assert_eq!(
                    ScaleDtype::UE8M0.round_up(up),
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
            assert_eq!(ScaleDtype::F32.round_up(scale), scale);
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
                    let up = dtype.round_up(scale);
                    assert_eq!(
                        up,
                        dtype.round_up(up),
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

    #[test]
    fn a_float_format_stores_one_float_an_element() {
        for (value, bits) in [
            (QuantValue::F32, 32),
            (QuantValue::F16, 16),
            (QuantValue::BF16, 16),
        ] {
            let scheme = QuantScheme::default()
                .with_value(value)
                .with_store(QuantStore::Native);
            assert_eq!(scheme.size_bits_value(), bits, "{value:?}");
            assert_eq!(scheme.num_quants(), 1, "{value:?}");
            assert!(value.is_float() && !value.is_symmetric(), "{value:?}");
        }
        assert_eq!(QuantValue::F16.range(), (-65504.0, 65504.0));
    }

    /// Every scheme a code holds packs into one that reads back as itself: every value, store (at
    /// both ends of the axis) and dtype, and levels of every extent and rest, alone and
    /// together. The variants come from the code tables, which the build holds to every variant.
    #[test]
    #[cfg_attr(miri, ignore = "safe code only, and too slow under miri")]
    fn every_scheme_round_trips_through_its_code() {
        let values: Vec<_> = (0..1 << 4).filter_map(QuantValue::from_code).collect();
        let dtypes: Vec<_> = (0..1 << 3).filter_map(ScaleDtype::from_code).collect();
        let stores = [
            QuantStore::Native,
            QuantStore::PackedNative(0),
            QuantStore::PackedNative(63),
            QuantStore::PackedU32(0),
            QuantStore::PackedU32(63),
        ];
        let sizes = [
            BlockSize::TENSOR,
            row(),
            BlockSize::new([]),
            BlockSize::new([32]),
            BlockSize::new([255, 1]),
            BlockSize::with_rest(&[BlockExtent::from(16)], BlockExtent::WholeAxis),
        ];
        let longest = BlockSize::new([2, 4, 8, 16, 255]);
        let levels: Vec<BlockScale> = dtypes
            .iter()
            .flat_map(|&dtype| sizes.map(|size| level(size, dtype)))
            .collect();
        let coarser: Vec<Option<BlockScale>> = [None]
            .into_iter()
            .chain(levels.iter().copied().map(Some))
            .collect();
        let pairs: Vec<(BlockScale, Option<BlockScale>)> = levels
            .iter()
            .flat_map(|&finest| coarser.iter().map(move |&coarser| (finest, coarser)))
            .chain([(level(longest, ScaleDtype::F32), None)])
            .collect();
        let mut codes = alloc::collections::BTreeSet::new();
        let mut count = 0;
        for value in values.iter().copied() {
            for store in stores {
                for (finest, coarser) in pairs.iter().copied() {
                    let scheme = QuantScheme {
                        value,
                        store,
                        finest,
                        coarser,
                    };
                    let code = scheme.to_code().unwrap();
                    assert_eq!(QuantScheme::from_code(code), Some(scheme), "{scheme:?}");
                    codes.insert(code);
                    count += 1;
                }
            }
        }
        assert_eq!(codes.len(), count, "two schemes share a code");
    }

    /// A code no scheme packs to reads back as no scheme: a reserved value in any field, and a
    /// bit the scheme it would decode to does not use.
    #[test]
    fn a_code_no_scheme_packs_to_is_refused() {
        let canonical = QuantScheme::default()
            .with_store(QuantStore::PackedU32(1))
            .per_block([32], ScaleDtype::F16)
            .to_code()
            .unwrap();
        let native = QuantScheme::default()
            .with_store(QuantStore::Native)
            .to_code()
            .unwrap();
        let refused = [
            ("value 15", (canonical & !0xF) | 15),
            ("store 3", canonical | (3 << 4)),
            ("a dtype of 7", canonical | (7 << 12)),
            ("six listed extents", (canonical & !(7 << 16)) | (6 << 16)),
            ("an axis on a native store", native | (5 << 6)),
            ("no finest level", canonical & !(7 << 12)),
            ("an extent past the listed ones", canonical | (8 << 30)),
            (
                "a listed extent that is the rest",
                (canonical & !(0xFF << 19)) | (1 << 19),
            ),
            ("bit 63", canonical | (1 << 63)),
        ];
        for (case, code) in refused {
            assert_eq!(QuantScheme::from_code(code), None, "{case}");
        }
    }

    /// A scheme the layout cannot hold has no code rather than a code that reads back as another.
    #[test]
    fn a_scheme_the_layout_cannot_hold_has_no_code() {
        let pairs = BlockSize::with_rest(&[], BlockExtent::from(2));
        let unheld = [
            (
                "a packing axis of 64",
                QuantScheme::default().with_store(QuantStore::PackedU32(64)),
            ),
            (
                "a rest of two values",
                QuantScheme::default().per_block(pairs, ScaleDtype::F16),
            ),
            (
                "five extents in two levels",
                QuantScheme::default()
                    .per_block([2, 4, 8, 16, 32], ScaleDtype::F16)
                    .per_tensor(ScaleDtype::F32),
            ),
        ];
        for (case, scheme) in unheld {
            assert_eq!(scheme.to_code(), None, "{case}");
        }
    }
}
