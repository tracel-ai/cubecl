use alloc::string::String;
use core::fmt::{Debug, Display};
use core::hash::Hash;

/// What a value needs to key an autotune: to be hashed and compared in memory, displayed in the
/// logs, and, where tunes persist, serialized into the table.
///
/// # Guarantees
///
/// - Every implementation's equal keys pose problems one decision is measured for: a tune
///   measures one of them and keeps its tune for every equal key.
/// - Where tunes persist, a key serialized and read back is equal to the key it was written from.
pub trait AutotuneKeyable:
    Clone + Debug + PartialEq + Eq + Hash + Display + Persistable + Send + Sync + 'static
{
}

/// What a key needs to be stored in a table, where tunes persist: nothing otherwise.
#[cfg(serializable)]
pub trait Persistable: serde::Serialize + serde::de::DeserializeOwned {}

#[cfg(serializable)]
impl<T: serde::Serialize + serde::de::DeserializeOwned> Persistable for T {}

/// [`Persistable`] where nothing persists: nothing.
#[cfg(not(serializable))]
pub trait Persistable {}

#[cfg(not(serializable))]
impl<T> Persistable for T {}

impl AutotuneKeyable for String {}
