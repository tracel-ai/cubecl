use alloc::vec;
use alloc::vec::Vec;
use core::{default::Default, ops::Deref};
use serde::{Deserialize, Serialize};

/// Describes a quantization scheme/configuration.
///
/// Field order is part of the wire contract: some transports serialize structs positionally, so
/// fields must never be reordered or inserted, only appended.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct QuantScheme {
    /// The logical data type of quantized input values (e.g., `QInt8`).
    ///
    /// This defines how values are interpreted during computation, independent of how they're stored.
    pub value: QuantValue,
    /// Data type used for storing quantized values.
    pub store: QuantStore,
    /// Quantization mode (e.g., symmetric).
    pub mode: QuantMode,
    /// The scale levels, innermost first. Private so that ordering and nesting hold by
    /// construction: go through [`ScaleLevels`] to build them.
    levels: ScaleLevels,
}

impl Default for QuantScheme {
    fn default() -> Self {
        Self {
            value: QuantValue::Q8F,
            store: QuantStore::PackedU32(0),
            mode: QuantMode::Symmetric,
            levels: ScaleLevels::tensor(QuantParam::F32),
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

    /// Set the scale levels.
    pub fn with_scales(mut self, levels: ScaleLevels) -> Self {
        self.levels = levels;
        self
    }

    /// The scale levels, innermost first.
    pub fn levels(&self) -> &[ScaleLevel] {
        self.levels.levels()
    }

    /// The scale levels as the owned, fixed-capacity value.
    pub fn scale_levels(&self) -> ScaleLevels {
        self.levels
    }

    /// The innermost level's scale precision, the type block scales are stored in.
    pub fn param(&self) -> QuantParam {
        self.levels()[0].param
    }

    /// The innermost level's scale granularity.
    pub fn granularity(&self) -> ScaleGranularity {
        self.levels()[0].granularity
    }

    /// The innermost level's block size, or [`None`] for per-tensor quantization.
    pub fn block_size(&self) -> Option<BlockSize> {
        self.granularity().block_size()
    }

    /// Swap two tensor dimensions in every block granularity, mirroring
    /// `shape.swap(dim0, dim1)`. Tensor granularities are unchanged.
    ///
    /// `dim0`/`dim1` are bare indices on purpose, mirroring `[T]::swap`'s own signature.
    pub fn swap_block_dims(&mut self, rank: usize, dim0: usize, dim1: usize) {
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(dim0, dim1);
        self.permute_block_dims(rank, &axes);
    }

    /// Permute every block granularity, mirroring a permutation of the tensor's axes. Tensor
    /// granularities are unchanged.
    ///
    /// Kept here rather than in each consumer so a level can never be missed: hand-written copies
    /// of this rewrite have silently skipped the outer level before.
    pub fn permute_block_dims(&mut self, rank: usize, axes: &[usize]) {
        for level in self.levels.levels_mut() {
            if let ScaleGranularity::Block(block) = &mut level.granularity {
                let dims = block.to_dim_vec(rank);
                let permuted: Vec<u8> = axes.iter().map(|&axis| dims[axis]).collect();
                *block = BlockSize::new(permuted);
            }
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

/// Maximum number of scale levels in a scheme. An implementation constant, not API: the committed
/// shape is the nonempty list, and raising this alongside consumer support is a non-breaking
/// change.
const MAX_LEVELS: usize = 2;

/// The region of a tensor that shares one scale at a level.
///
/// Tensor granularity is explicit rather than encoded as a special block size. A [`BlockSize`]
/// stays rank-relative as a result: `[FULL]` always means the full trailing dimension, while
/// `Tensor` covers every dimension at any rank.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ScaleGranularity {
    /// One scale for the whole tensor.
    Tensor,
    /// One scale per block.
    Block(BlockSize),
}

impl ScaleGranularity {
    /// The block size, or [`None`] when this granularity covers the whole tensor.
    pub fn block_size(&self) -> Option<BlockSize> {
        match self {
            Self::Tensor => None,
            Self::Block(block) => Some(*block),
        }
    }

    /// The shape of this granularity's scale tensor for a value tensor of `shape`.
    pub fn grid(&self, shape: &[usize]) -> Vec<usize> {
        match self {
            Self::Tensor => vec![1; shape.len()],
            Self::Block(block) => block
                .to_dim_vec(shape.len())
                .into_iter()
                .zip(shape)
                .map(|(block, &dim)| {
                    if block == BlockSize::FULL {
                        1
                    } else {
                        dim.div_ceil(block as usize)
                    }
                })
                .collect(),
        }
    }

    /// Whether every region of `inner` sits inside one region of `self`.
    fn contains(&self, inner: &Self) -> bool {
        match (self, inner) {
            (Self::Tensor, _) => true,
            (Self::Block(_), Self::Tensor) => false,
            (Self::Block(outer), Self::Block(inner)) => outer.contains(inner),
        }
    }
}

/// One level of scales: values inside each region share one scale, stored as `param`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScaleLevel {
    /// The region covered by one scale.
    pub granularity: ScaleGranularity,
    /// The precision the level's scales are stored in.
    pub param: QuantParam,
}

impl ScaleLevel {
    /// The shape of the level's scale tensor for a value tensor of `shape`.
    pub fn grid(&self, shape: &[usize]) -> Vec<usize> {
        self.granularity.grid(shape)
    }
}

/// The scale levels of a [`QuantScheme`], innermost first: each level's scales are normalized
/// against the next level's, and reconstruction multiplies them back together.
///
/// Nesting and ordering hold by construction. Order is never specified by the caller: it derives
/// from containment, since a level whose regions do not contain the inner level's regions is
/// invalid no matter how it is written.
///
/// # Examples
///
/// One f16 scale per block of 32 values, a single level:
///
/// ```
/// # use cubecl_common::quant::scheme::{BlockSize, QuantParam, ScaleGranularity, ScaleLevels};
/// let levels = ScaleLevels::block([32], QuantParam::F16);
/// // storage: [{ granularity: Block([32]), param: F16 }, { granularity: Tensor, param: F32 }]
/// // len: 1                                                   ^ unused slot, canonical filler
///
/// let [level] = levels.levels() else { unreachable!() };
/// assert_eq!(level.granularity, ScaleGranularity::Block(BlockSize::new([32])));
/// assert_eq!(level.param, QuantParam::F16);
/// assert_eq!(level.grid(&[8, 64]), vec![8, 2]); // one scale per block of a [8, 64] tensor
/// ```
///
/// NVFP4-style two levels: ue4m3 block scales normalized against one per-tensor f32 scale. The
/// list stores innermost first however it was built:
///
/// ```
/// # use cubecl_common::quant::scheme::{QuantParam, ScaleGranularity, ScaleLevels};
/// let levels = ScaleLevels::block([16], QuantParam::UE4M3).and_tensor(QuantParam::F32);
/// // storage: [{ granularity: Block([16]), param: UE4M3 }, { granularity: Tensor, param: F32 }]
/// // len: 2                                                     ^ both slots in use
/// assert_eq!(
///     levels,
///     ScaleLevels::tensor(QuantParam::F32).and_block([16], QuantParam::UE4M3),
/// );
///
/// let [blocks, tensor] = levels.levels() else { unreachable!() };
/// assert_eq!(blocks.param, QuantParam::UE4M3);
/// assert_eq!(tensor.granularity, ScaleGranularity::Tensor);
/// assert_eq!(tensor.grid(&[8, 64]), vec![1, 1]); // one scale for the whole tensor
/// ```
///
/// A two-level scheme exists so block scales can live in a narrow type: the outer per-tensor scale
/// absorbs the tensor's dynamic range, and the block param only covers the spread between blocks.
/// That spread is still bounded: a block whose scale falls further below the largest one than the
/// block param can express is stored at that param's smallest value, far too coarse for it, and
/// every value in the block quantizes to zero. [`QuantParam::UE4M3`] spans about 2^18 this way, so
/// a tensor holding a genuine outlier can lose its ordinary values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ScaleLevels {
    /// Slots past `len` hold [`ScaleLevels::FILLER`] so derived comparisons stay truthful.
    storage: [ScaleLevel; MAX_LEVELS],
    len: u8,
}

/// Hand-written: `storage` precedes `len`, so a derived `Ord` would compare filler bytes before
/// level count.
impl PartialOrd for ScaleLevels {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScaleLevels {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.len, self.levels()).cmp(&(other.len, other.levels()))
    }
}

impl ScaleLevels {
    /// One scale for the whole tensor.
    pub fn tensor(param: QuantParam) -> Self {
        Self::one(ScaleLevel {
            granularity: ScaleGranularity::Tensor,
            param,
        })
    }

    /// One scale per block of `block` values.
    pub fn block(block: impl AsRef<[u8]>, param: QuantParam) -> Self {
        Self::one(ScaleLevel {
            granularity: ScaleGranularity::Block(BlockSize::new(block)),
            param,
        })
    }

    /// Add a per-tensor scale level.
    ///
    /// # Panics
    ///
    /// When no valid order exists: a duplicated granularity, blocks that do not nest, or more
    /// levels than the supported capacity.
    pub fn and_tensor(self, param: QuantParam) -> Self {
        self.and(ScaleLevel {
            granularity: ScaleGranularity::Tensor,
            param,
        })
    }

    /// Add a per-block scale level.
    ///
    /// # Panics
    ///
    /// When no valid order exists: a duplicated granularity, blocks that do not nest, or more
    /// levels than the supported capacity.
    pub fn and_block(self, block: impl AsRef<[u8]>, param: QuantParam) -> Self {
        self.and(ScaleLevel {
            granularity: ScaleGranularity::Block(BlockSize::new(block)),
            param,
        })
    }

    /// Build from levels in any order, sorted innermost-first by block containment. For dynamic
    /// inputs like checkpoint headers; literal schemes read better through the named constructors.
    pub fn try_new(levels: &[ScaleLevel]) -> Result<Self, InvalidScaleLevels> {
        let (&first, rest) = levels.split_first().ok_or(InvalidScaleLevels::Empty)?;
        let mut out = Self::one(first);
        for &level in rest {
            out.insert(level)?;
        }
        Ok(out)
    }

    /// The levels, innermost first.
    pub fn levels(&self) -> &[ScaleLevel] {
        &self.storage[..self.len as usize]
    }

    /// The number of levels.
    #[allow(clippy::len_without_is_empty, reason = "never empty by construction")]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// These levels without the outermost one, for a consumer that folds an outer scale away and
    /// serves the rest as a shallower scheme. [`None`] for a single level, which has no inner
    /// levels to serve.
    pub fn inner(&self) -> Option<Self> {
        if self.len == 1 {
            return None;
        }
        let mut storage = self.storage;
        storage[self.len as usize - 1] = Self::FILLER;
        Some(Self {
            storage,
            len: self.len - 1,
        })
    }

    /// The canonical value of unused storage slots.
    const FILLER: ScaleLevel = ScaleLevel {
        granularity: ScaleGranularity::Tensor,
        param: QuantParam::F32,
    };

    /// The list of exactly one level.
    fn one(level: ScaleLevel) -> Self {
        Self {
            storage: [level, Self::FILLER],
            len: 1,
        }
    }

    /// [`insert`](Self::insert) for the chaining constructors, where an invalid level is a
    /// programmer error rather than a condition to handle.
    fn and(mut self, level: ScaleLevel) -> Self {
        self.insert(level)
            .unwrap_or_else(|invalid| panic!("{invalid}"));
        self
    }

    /// Insert one level at the position containment assigns it.
    fn insert(&mut self, level: ScaleLevel) -> Result<(), InvalidScaleLevels> {
        let len = self.len as usize;
        if len == MAX_LEVELS {
            return Err(InvalidScaleLevels::TooMany);
        }
        let mut index = len;
        for (i, existing) in self.levels().iter().enumerate() {
            if existing.granularity == level.granularity {
                return Err(InvalidScaleLevels::Duplicate);
            } else if existing.granularity.contains(&level.granularity) {
                // Containment is transitive, so everything past the first container contains the
                // level too and needs no check of its own.
                index = i;
                break;
            } else if !level.granularity.contains(&existing.granularity) {
                return Err(InvalidScaleLevels::NotNested);
            }
        }
        let mut storage = self.storage;
        storage.copy_within(index..len, index + 1);
        storage[index] = level;
        let new_len = len + 1;
        // Every level but the innermost must cover the whole tensor; checked on a copy so a
        // rejected insert leaves `self` untouched.
        if storage[1..new_len]
            .iter()
            .any(|l| l.granularity != ScaleGranularity::Tensor)
        {
            return Err(InvalidScaleLevels::OuterNotTensor);
        }
        self.storage = storage;
        self.len = new_len as u8;
        Ok(())
    }

    /// Callers must preserve nesting and canonical order, which a uniform rewrite of every
    /// level's block does.
    fn levels_mut(&mut self) -> &mut [ScaleLevel] {
        &mut self.storage[..self.len as usize]
    }
}

/// Why a set of levels cannot form a [`ScaleLevels`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidScaleLevels {
    /// No levels at all: a scheme has at least one.
    Empty,
    /// More levels than the supported capacity.
    TooMany,
    /// No containment order exists: for every pair, one level's regions must contain the other's.
    NotNested,
    /// Two levels share the same granularity, so one of them is redundant.
    Duplicate,
    /// A level other than the innermost does not cover the whole tensor. Only the innermost
    /// level's scales are addressed per position, so every other level must be per-tensor.
    OuterNotTensor,
}

impl core::fmt::Display for InvalidScaleLevels {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            InvalidScaleLevels::Empty => write!(f, "a scheme has at least one scale level"),
            InvalidScaleLevels::TooMany => {
                write!(
                    f,
                    "more scale levels than the supported capacity {MAX_LEVELS}"
                )
            }
            InvalidScaleLevels::NotNested => {
                write!(
                    f,
                    "each scale level's regions must contain the previous level's"
                )
            }
            InvalidScaleLevels::Duplicate => {
                write!(f, "two scale levels share the same granularity")
            }
            InvalidScaleLevels::OuterNotTensor => {
                write!(
                    f,
                    "every scale level but the innermost must cover the whole tensor"
                )
            }
        }
    }
}

impl core::error::Error for InvalidScaleLevels {}

impl Serialize for ScaleLevels {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.levels())
    }
}

impl<'de> Deserialize<'de> for ScaleLevels {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let levels = Vec::<ScaleLevel>::deserialize(deserializer)?;
        Self::try_new(&levels).map_err(serde::de::Error::custom)
    }
}

/// The `QuantLevel` enum shape legacy stored schemes carry, kept so those artifacts still load.
/// Deserialization only. Delete together with the legacy branch of [`QuantScheme`]'s
/// `Deserialize` once stored schemes have migrated.
#[derive(Deserialize)]
enum LegacyQuantLevel {
    Tensor,
    Block(BlockSize),
    BlockTensor {
        block: BlockSize,
        global: QuantParam,
    },
}

impl LegacyQuantLevel {
    /// The legacy shape splits the innermost param off into its own scheme field, passed here.
    fn into_levels(self, param: QuantParam) -> Result<ScaleLevels, InvalidScaleLevels> {
        match self {
            LegacyQuantLevel::Tensor => Ok(ScaleLevels::tensor(param)),
            LegacyQuantLevel::Block(block) => Ok(ScaleLevels::one(ScaleLevel {
                granularity: ScaleGranularity::Block(block),
                param,
            })),
            LegacyQuantLevel::BlockTensor { block, global } => ScaleLevels::try_new(&[
                ScaleLevel {
                    granularity: ScaleGranularity::Block(block),
                    param,
                },
                ScaleLevel {
                    granularity: ScaleGranularity::Tensor,
                    param: global,
                },
            ]),
        }
    }
}

/// Hand-written to also accept the legacy shape (`value, param, store, level, mode`), which
/// stored artifacts still carry. Legacy data only exists in name-keyed formats, so that path
/// lives in `visit_map` alone; positional formats carry in-flight messages between same-version
/// peers and only need the current shape.
impl<'de> Deserialize<'de> for QuantScheme {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SchemeVisitor;

        impl<'de> serde::de::Visitor<'de> for SchemeVisitor {
            type Value = QuantScheme;

            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "a quantization scheme")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<QuantScheme, A::Error> {
                use serde::de::Error;
                Ok(QuantScheme {
                    value: seq
                        .next_element()?
                        .ok_or_else(|| Error::invalid_length(0, &"value"))?,
                    store: seq
                        .next_element()?
                        .ok_or_else(|| Error::invalid_length(1, &"store"))?,
                    mode: seq
                        .next_element()?
                        .ok_or_else(|| Error::invalid_length(2, &"mode"))?,
                    levels: seq
                        .next_element()?
                        .ok_or_else(|| Error::invalid_length(3, &"levels"))?,
                })
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<QuantScheme, A::Error> {
                use serde::de::Error;

                let mut value = None;
                let mut store = None;
                let mut mode = None;
                let mut levels: Option<ScaleLevels> = None;
                let mut param: Option<QuantParam> = None;
                let mut level: Option<LegacyQuantLevel> = None;

                while let Some(key) = map.next_key::<alloc::borrow::Cow<str>>()? {
                    match key.as_ref() {
                        "value" => value = Some(map.next_value()?),
                        "store" => store = Some(map.next_value()?),
                        "mode" => mode = Some(map.next_value()?),
                        "levels" => levels = Some(map.next_value()?),
                        "param" => param = Some(map.next_value()?),
                        "level" => level = Some(map.next_value()?),
                        _ => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                let levels = match (levels, level, param) {
                    (Some(levels), None, None) => levels,
                    (None, Some(level), Some(param)) => {
                        level.into_levels(param).map_err(Error::custom)?
                    }
                    (None, None, None) => return Err(Error::missing_field("levels")),
                    (None, Some(_), None) => return Err(Error::missing_field("param")),
                    (None, None, Some(_)) => return Err(Error::missing_field("level")),
                    (Some(_), _, _) => {
                        return Err(Error::custom(
                            "a scheme carries either levels or the old level and param, not both",
                        ));
                    }
                };

                Ok(QuantScheme {
                    value: value.ok_or_else(|| Error::missing_field("value"))?,
                    store: store.ok_or_else(|| Error::missing_field("store"))?,
                    mode: mode.ok_or_else(|| Error::missing_field("mode"))?,
                    levels,
                })
            }
        }

        deserializer.deserialize_struct(
            "QuantScheme",
            &["value", "store", "mode", "levels"],
            SchemeVisitor,
        )
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
#[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize)]
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

    /// The dimension value meaning the block covers that axis entirely, resolved against the
    /// tensor shape by [`resolved_dims`](Self::resolved_dims). Never valid as a literal extent.
    pub const FULL: u8 = 0;

    /// Create a new blocksize from a set of values. The number of values must be `<= MAX_DIMS`.
    ///
    /// A dimension of [`FULL`](Self::FULL) covers that axis entirely.
    ///
    /// The result is canonical, so equal rank-relative blocks compare and hash equal however they
    /// are spelled: leading unit dimensions are dropped, since the missing-dimension fill restates
    /// them. In particular, `[1, FULL]` canonicalizes to `[FULL]`; both mean a block spanning the
    /// complete trailing dimension. Whole-tensor granularity is represented separately by
    /// [`ScaleGranularity::Tensor`].
    pub fn new(values: impl AsRef<[u8]>) -> Self {
        let values = values.as_ref();
        debug_assert!(
            values.len() <= MAX_DIMS,
            "Tried creating a block size larger than the cap"
        );
        Self::canonicalize(values)
    }

    fn canonicalize(values: &[u8]) -> Self {
        let skip = values
            .iter()
            .position(|&value| value != 1)
            .unwrap_or(values.len());
        let values = &values[skip..];
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

    /// The block's extent along every dimension of `shape`, with full dimensions resolved to the
    /// shape's. This is the only reading of a block size that may enter arithmetic: the raw
    /// dimensions can hold [`FULL`](Self::FULL), which divides and multiplies as a zero.
    pub fn resolved_dims(&self, shape: &[usize]) -> Vec<usize> {
        self.to_dim_vec(shape.len())
            .into_iter()
            .zip(shape)
            .map(|(block, &dim)| {
                if block == Self::FULL {
                    dim
                } else {
                    block as usize
                }
            })
            .collect()
    }

    /// Whether every block of `inner` sits inside one block of `self`, the nesting scale levels
    /// require. A full dimension contains everything along its axis; a finite extent contains the
    /// extents that divide it.
    pub fn contains(&self, inner: &BlockSize) -> bool {
        let rank = self.len.max(inner.len) as usize;
        self.to_dim_vec(rank)
            .into_iter()
            .zip(inner.to_dim_vec(rank))
            .all(|(outer, inner)| {
                outer == Self::FULL || (inner != Self::FULL && outer.is_multiple_of(inner))
            })
    }

    /// Create an iterator over all stored dimensions
    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        self.as_slice().iter()
    }

    /// Returns the total number of elements in each block.
    ///
    /// Meaningless for a block with full dimensions, whose element count depends on the tensor:
    /// resolve through [`resolved_dims`](Self::resolved_dims) instead.
    pub fn num_elements(&self) -> usize {
        // Not a debug_assert!: a release build must not silently multiply FULL as zero.
        assert!(
            !self.as_slice().contains(&Self::FULL),
            "a full dimension has no element count without a shape"
        );
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

/// Hand-written to canonicalize stored bytes: artifacts can carry non-canonical spellings
/// (leading unit dimensions), and letting one deserialize raw would break the promise that equal
/// granularities compare and hash equal.
impl<'de> Deserialize<'de> for BlockSize {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename = "BlockSize")]
        struct Raw {
            storage: [u8; MAX_DIMS],
            len: u8,
        }

        let raw = Raw::deserialize(deserializer)?;
        let len = raw.len as usize;
        if len > MAX_DIMS {
            return Err(serde::de::Error::custom(
                "block size length exceeds the dimension cap",
            ));
        }
        Ok(BlockSize::canonicalize(&raw.storage[..len]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_dimensions_remain_rank_relative() {
        assert_ne!(BlockSize::new([0]), BlockSize::new([0, 0]));
        assert_eq!(BlockSize::new([0]).to_dim_vec(2), vec![1, 0]);
        assert_eq!(BlockSize::new([0, 0]).to_dim_vec(2), vec![0, 0]);
    }

    #[test]
    fn unit_dimensions_ahead_of_a_full_block_canonicalize_away() {
        assert_eq!(BlockSize::new([1, 0]), BlockSize::new([0]));
    }

    #[test]
    fn full_dimensions_resolve_to_the_shape() {
        assert_eq!(BlockSize::new([0]).resolved_dims(&[8, 64]), vec![1, 64]);
        assert_eq!(BlockSize::new([0, 0]).resolved_dims(&[8, 64]), vec![8, 64]);
        assert_eq!(BlockSize::new([0, 32]).resolved_dims(&[8, 64]), vec![8, 32]);
        assert_eq!(BlockSize::new([32]).resolved_dims(&[8, 64]), vec![1, 32]);
    }

    #[test]
    fn the_grid_is_one_scale_per_region() {
        let level = |granularity| ScaleLevel {
            granularity,
            param: QuantParam::F32,
        };
        assert_eq!(level(ScaleGranularity::Tensor).grid(&[8, 64]), vec![1, 1]);
        assert_eq!(
            level(ScaleGranularity::Block(BlockSize::new([32]))).grid(&[8, 64]),
            vec![8, 2]
        );
        assert_eq!(
            level(ScaleGranularity::Block(BlockSize::new([0, 32]))).grid(&[8, 64]),
            vec![1, 2]
        );
    }

    #[test]
    fn an_all_full_block_is_not_rank_free_tensor_granularity() {
        let block = ScaleGranularity::Block(BlockSize::new([0, 0]));
        assert_eq!(block.grid(&[8, 64]), vec![1, 1]);
        assert_eq!(block.grid(&[4, 8, 64]), vec![4, 1, 1]);
        assert_eq!(ScaleGranularity::Tensor.grid(&[4, 8, 64]), vec![1, 1, 1]);
    }

    /// Leading unit dimensions restate the missing-dimension fill, so equal granularities spelled
    /// at different ranks must land on one representation.
    #[test]
    fn leading_unit_dimensions_canonicalize_away() {
        assert_eq!(BlockSize::new([1, 32]), BlockSize::new([32]));
        assert_eq!(BlockSize::new([1, 0, 32]), BlockSize::new([0, 32]));
        assert_eq!(BlockSize::new([1, 1]), BlockSize::new([1]));
        assert_ne!(BlockSize::new([32, 1]), BlockSize::new([32]));
    }

    #[test]
    #[should_panic(expected = "share the same granularity")]
    fn equal_granularities_spelled_differently_are_still_duplicates() {
        ScaleLevels::block([1, 32], QuantParam::F16).and_block([32], QuantParam::F32);
    }

    #[test]
    fn containment_is_divisibility_with_full_on_top() {
        assert!(BlockSize::new([0]).contains(&BlockSize::new([32])));
        assert!(!BlockSize::new([32]).contains(&BlockSize::new([0])));
        assert!(BlockSize::new([32]).contains(&BlockSize::new([16])));
        assert!(!BlockSize::new([16]).contains(&BlockSize::new([32])));
        assert!(!BlockSize::new([12]).contains(&BlockSize::new([8])));
    }

    #[test]
    fn levels_sort_innermost_first_whatever_the_input_order() {
        let block = ScaleLevel {
            granularity: ScaleGranularity::Block(BlockSize::new([16])),
            param: QuantParam::UE4M3,
        };
        let tensor = ScaleLevel {
            granularity: ScaleGranularity::Tensor,
            param: QuantParam::F32,
        };
        let sorted = ScaleLevels::try_new(&[tensor, block]).unwrap();
        assert_eq!(sorted.levels(), &[block, tensor]);
        assert_eq!(
            sorted,
            ScaleLevels::block([16], QuantParam::UE4M3).and_tensor(QuantParam::F32)
        );
    }

    #[test]
    fn levels_without_a_containment_order_are_rejected() {
        let a = ScaleLevel {
            granularity: ScaleGranularity::Block(BlockSize::new([8])),
            param: QuantParam::F32,
        };
        let b = ScaleLevel {
            granularity: ScaleGranularity::Block(BlockSize::new([12])),
            param: QuantParam::F32,
        };
        assert_eq!(
            ScaleLevels::try_new(&[a, b]),
            Err(InvalidScaleLevels::NotNested)
        );
        assert_eq!(
            ScaleLevels::try_new(&[a, a]),
            Err(InvalidScaleLevels::Duplicate)
        );
        assert_eq!(ScaleLevels::try_new(&[]), Err(InvalidScaleLevels::Empty));
    }

    #[test]
    #[should_panic(expected = "share the same granularity")]
    fn two_levels_with_the_same_granularity_are_rejected() {
        ScaleLevels::tensor(QuantParam::F16).and_tensor(QuantParam::F32);
    }

    #[test]
    fn levels_added_in_any_order_sort_by_containment() {
        assert_eq!(
            ScaleLevels::tensor(QuantParam::F32).and_block([16], QuantParam::UE4M3),
            ScaleLevels::block([16], QuantParam::UE4M3).and_tensor(QuantParam::F32),
        );
    }

    #[test]
    fn inner_drops_the_outermost_level() {
        let two = ScaleLevels::block([16], QuantParam::UE4M3).and_tensor(QuantParam::F32);
        assert_eq!(
            two.inner(),
            Some(ScaleLevels::block([16], QuantParam::UE4M3))
        );
    }

    #[test]
    fn a_single_level_has_no_inner() {
        assert_eq!(ScaleLevels::tensor(QuantParam::F32).inner(), None);
    }

    #[test]
    fn swapping_dims_rewrites_every_block_and_leaves_tensor_granularity_alone() {
        let mut scheme = QuantScheme::default()
            .with_scales(ScaleLevels::block([4, 32], QuantParam::F16).and_tensor(QuantParam::F32));
        scheme.swap_block_dims(2, 0, 1);
        assert_eq!(
            scheme.levels(),
            ScaleLevels::block([32, 4], QuantParam::F16)
                .and_tensor(QuantParam::F32)
                .levels()
        );

        let mut per_tensor = QuantScheme::default();
        per_tensor.swap_block_dims(2, 0, 1);
        assert_eq!(per_tensor, QuantScheme::default());
    }

    #[test]
    fn swapping_dims_preserves_partial_full_blocks() {
        let mut scheme = QuantScheme::default()
            .with_scales(ScaleLevels::block([BlockSize::FULL, 1], QuantParam::F32));
        scheme.swap_block_dims(2, 0, 1);
        assert_eq!(scheme.block_size(), Some(BlockSize::new([BlockSize::FULL])));
        assert_eq!(
            scheme.block_size().unwrap().resolved_dims(&[8, 64]),
            vec![1, 64]
        );
    }

    #[test]
    fn permuting_dims_rewrites_every_level() {
        let mut scheme =
            QuantScheme::default().with_scales(ScaleLevels::block([1, 4, 32], QuantParam::F16));
        scheme.permute_block_dims(3, &[2, 0, 1]);
        assert_eq!(scheme.block_size(), Some(BlockSize::new([32, 1, 4])));
    }

    #[test]
    fn the_default_scheme_is_per_tensor_f32() {
        let scheme = QuantScheme::default();
        assert_eq!(
            scheme.levels(),
            ScaleLevels::tensor(QuantParam::F32).levels()
        );
        assert_eq!(scheme.param(), QuantParam::F32);
        assert_eq!(scheme.granularity(), ScaleGranularity::Tensor);
        assert_eq!(scheme.block_size(), None);
    }

    mod serde_compat {
        use super::*;
        use serde::Serialize;

        /// The scheme shape before scale levels, byte-for-byte as stored artifacts carry it:
        /// same field names, order, and variant names as the old derives produced.
        ///
        /// Declared independently of `LegacyQuantLevel` on purpose: serializing through the shim's
        /// own type would test it against itself, and a name it gets wrong would pass unnoticed.
        #[derive(Serialize)]
        struct OldScheme {
            value: QuantValue,
            param: QuantParam,
            store: QuantStore,
            level: OldLevel,
            mode: QuantMode,
        }

        #[derive(Serialize)]
        enum OldLevel {
            #[allow(dead_code)]
            Tensor,
            Block(OldBlockSize),
            BlockTensor {
                block: OldBlockSize,
                global: QuantParam,
            },
        }

        /// The raw stored shape of a block size, free to spell non-canonical bytes the way old
        /// artifacts do.
        #[derive(Serialize)]
        #[serde(rename = "BlockSize")]
        struct OldBlockSize {
            storage: [u8; 5],
            len: u8,
        }

        fn old_block(values: &[u8]) -> OldBlockSize {
            let mut storage = [1; 5];
            storage[..values.len()].copy_from_slice(values);
            OldBlockSize {
                storage,
                len: values.len() as u8,
            }
        }

        fn cbor_round_trip<T: Serialize, O: for<'de> Deserialize<'de>>(value: &T) -> O {
            let mut bytes = alloc::vec::Vec::new();
            ciborium::ser::into_writer(value, &mut bytes).unwrap();
            ciborium::de::from_reader(bytes.as_slice()).unwrap()
        }

        #[test]
        fn the_current_shape_round_trips() {
            for scheme in [
                QuantScheme::default(),
                QuantScheme::default()
                    .with_value(QuantValue::E2M1)
                    .with_store(QuantStore::PackedNative(0))
                    .with_scales(
                        ScaleLevels::block([16], QuantParam::UE4M3).and_tensor(QuantParam::F32),
                    ),
            ] {
                assert_eq!(cbor_round_trip::<_, QuantScheme>(&scheme), scheme);
            }
        }

        #[test]
        fn an_old_per_tensor_scheme_still_loads() {
            let old = OldScheme {
                value: QuantValue::Q8S,
                param: QuantParam::F32,
                store: QuantStore::Native,
                level: OldLevel::Tensor,
                mode: QuantMode::Symmetric,
            };
            let loaded: QuantScheme = cbor_round_trip(&old);
            assert_eq!(
                loaded,
                QuantScheme::default()
                    .with_value(QuantValue::Q8S)
                    .with_store(QuantStore::Native)
                    .with_scales(ScaleLevels::tensor(QuantParam::F32))
            );
        }

        #[test]
        fn an_old_block_scheme_still_loads() {
            let old = OldScheme {
                value: QuantValue::Q8F,
                param: QuantParam::F16,
                store: QuantStore::PackedU32(0),
                level: OldLevel::Block(old_block(&[32])),
                mode: QuantMode::Symmetric,
            };
            let spelled_wide = OldScheme {
                level: OldLevel::Block(old_block(&[1, 32])),
                ..old
            };
            // A stored non-canonical spelling lands on the canonical value, so the two hash and
            // compare equal after loading.
            assert_eq!(
                cbor_round_trip::<_, QuantScheme>(&spelled_wide),
                cbor_round_trip::<_, QuantScheme>(&old)
            );
            let loaded: QuantScheme = cbor_round_trip(&old);
            assert_eq!(
                loaded,
                QuantScheme::default().with_scales(ScaleLevels::block([32], QuantParam::F16))
            );
        }

        #[test]
        fn an_old_two_level_scheme_still_loads() {
            let old = OldScheme {
                value: QuantValue::Q8F,
                param: QuantParam::UE4M3,
                store: QuantStore::PackedU32(0),
                level: OldLevel::BlockTensor {
                    block: old_block(&[16]),
                    global: QuantParam::F32,
                },
                mode: QuantMode::Symmetric,
            };
            let loaded: QuantScheme = cbor_round_trip(&old);
            assert_eq!(
                loaded,
                QuantScheme::default().with_scales(
                    ScaleLevels::block([16], QuantParam::UE4M3,).and_tensor(QuantParam::F32)
                )
            );
        }
    }

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
                crate::ue8m0::MAX.to_f32()
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
