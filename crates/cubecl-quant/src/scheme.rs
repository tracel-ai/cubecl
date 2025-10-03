use alloc::vec;
use alloc::vec::Vec;
use core::{default::Default, ops::Deref};
use cubecl_common::{e4m3, e5m2};
use serde::{Deserialize, Serialize};

/// Describes a quantization scheme/configuration.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QuantScheme {
    /// The logical data type of quantized input values (e.g., QInt8).
    ///
    /// This defines how values are interpreted during computation, independent of how they're stored.
    pub value: QuantValue,
    /// Precision used for quantization parameters (e.g., scale and biases).
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
            store: QuantStore::U32,
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
        // Assume native packing if store type is < 8 bits
        self.store.size_bits(&self.value).max(8)
    }

    /// Returns the size of the quantization storage type in bits.
    pub fn size_bits_value(&self) -> usize {
        self.value.size_bits()
    }

    /// Returns the number of quantized values stored in a single element.
    pub fn num_quants(&self) -> usize {
        self.size_bits_stored() / self.value.size_bits()
    }
}

/// Level or granularity of quantization.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantLevel {
    /// Quantize the whole tensor using a single tensor.
    Tensor,
    /// Quantize a tensor using multiple blocks.
    Block(BlockSize),
}

impl QuantLevel {
    pub fn block(values: impl AsRef<[u8]>) -> Self {
        QuantLevel::Block(BlockSize::new(values))
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

    /// The possible range of values allowed by the quant value.
    pub fn range(&self) -> (f32, f32) {
        match self {
            QuantValue::Q8F => (i8::MIN as f32, i8::MAX as f32),
            QuantValue::Q4F => (-8.0, 7.0),
            QuantValue::Q2F => (-2.0, 1.0),
            QuantValue::Q8S => (-i8::MAX as f32, i8::MAX as f32),
            QuantValue::Q4S => (-7.0, 7.0),
            QuantValue::Q2S => (-1.0, 1.0),
            QuantValue::E4M3 => (e4m3::MIN as f32, e4m3::MAX as f32),
            QuantValue::E5M2 => (e5m2::MIN as f32, e5m2::MAX as f32),
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
            QuantStore::U32 => 32,
        }
    }
}

/// Data type used to stored quantized values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantStore {
    /// Native quantization doesn't require packing and unpacking.
    Native,
    /// Store packed quantized values in a 4-byte unsigned integer.
    U32,
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockSize {
    storage: [u8; MAX_DIMS],
    len: u8,
}

impl BlockSize {
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

    /// Return a slice of only the initialized valeus
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

    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        self.as_slice().iter()
    }

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
