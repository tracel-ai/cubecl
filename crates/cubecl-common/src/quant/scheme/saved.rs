use core::num::NonZeroU8;

use serde::Deserialize;

use super::{
    BlockExtent, BlockScale, BlockSize, QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype,
};

/// A scheme as it may be read back: written with its finest and coarser levels, with the list of
/// blocks it held before them, or with the per-tensor, per-row and per-block levels it held before
/// that list.
#[derive(Deserialize)]
pub(super) struct SavedScheme {
    pub value: QuantValue,
    pub store: QuantStore,
    pub mode: QuantMode,
    #[serde(default)]
    pub finest: Option<BlockScale>,
    #[serde(default)]
    pub coarser: Option<BlockScale>,
    #[serde(default)]
    pub blocks: [Option<BlockScale>; 2],
    #[serde(default)]
    pub tensor: Option<ScaleDtype>,
    #[serde(default)]
    pub row: Option<ScaleDtype>,
    #[serde(default)]
    pub block: Option<SavedBlockScale>,
}

/// A per-block level.
#[derive(Deserialize)]
pub(super) struct SavedBlockScale {
    pub size: SavedBlockSize,
    pub dtype: ScaleDtype,
}

/// The values a block covers along the last axes, the last one at `len - 1`.
#[derive(Deserialize)]
pub(super) struct SavedBlockSize {
    pub storage: [NonZeroU8; BlockSize::MAX_DIMS],
    pub len: usize,
}

impl TryFrom<SavedScheme> for QuantScheme {
    type Error = &'static str;

    fn try_from(saved: SavedScheme) -> Result<Self, Self::Error> {
        let block = saved.block.map(BlockScale::try_from).transpose()?;
        let row = saved.row.map(|dtype| BlockScale {
            size: BlockSize::with_rest(&[BlockExtent::WholeAxis], BlockExtent::VALUE),
            dtype,
        });
        let tensor = saved.tensor.map(|dtype| BlockScale {
            size: BlockSize::TENSOR,
            dtype,
        });
        let mut levels = [saved.finest, saved.coarser, block, row, tensor]
            .into_iter()
            .chain(saved.blocks)
            .flatten();
        let scheme = Self {
            value: saved.value,
            store: saved.store,
            mode: saved.mode,
            ..Self::default()
        };
        let scheme = match levels.next() {
            Some(finest) => Self {
                finest,
                coarser: levels.next(),
                ..scheme
            },
            None => scheme,
        };
        match levels.next() {
            Some(_) => Err("a scheme holds at most two scale levels"),
            None => Ok(scheme),
        }
    }
}

impl TryFrom<SavedBlockScale> for BlockScale {
    type Error = &'static str;

    fn try_from(saved: SavedBlockScale) -> Result<Self, Self::Error> {
        let extents = saved.size.storage.map(BlockExtent::Values);
        let listed = extents
            .get(..saved.size.len)
            .ok_or("a block lists more axes than it holds")?;
        Ok(Self {
            size: BlockSize::with_rest(listed, BlockExtent::VALUE),
            dtype: saved.dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// A scheme as it was written before it listed its blocks, with these levels.
    fn written(levels: Value) -> Value {
        let mut scheme = json!({
            "value": "E2M1",
            "store": { "PackedU32": 0 },
            "mode": "Symmetric",
            "tensor": null,
            "block": null,
        });
        let levels = levels.as_object().unwrap().clone();
        scheme.as_object_mut().unwrap().extend(levels);
        scheme
    }

    fn block_of_16() -> Value {
        json!({ "size": { "storage": [16, 1, 1, 1, 1], "len": 1 }, "dtype": "UE4M3" })
    }

    fn base() -> QuantScheme {
        QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .with_store(QuantStore::PackedU32(0))
    }

    #[test]
    fn a_scheme_reads_back_as_itself() {
        for scheme in [
            QuantScheme::default(),
            QuantScheme::nvfp4(),
            QuantScheme::fp8_per_row(),
            QuantScheme::mxfp4_row_factored(),
        ] {
            let json = serde_json::to_value(scheme).unwrap();
            assert_eq!(serde_json::from_value::<QuantScheme>(json).unwrap(), scheme);
        }
    }

    /// The form published models hold, with and without the per-row level it had last.
    #[test]
    fn a_scheme_written_in_levels_reads_back_as_its_blocks() {
        let row = BlockSize::with_rest(&[BlockExtent::WholeAxis], BlockExtent::VALUE);
        let cases = [
            (json!({}), base()),
            (
                json!({ "tensor": "F32" }),
                base().per_tensor(ScaleDtype::F32),
            ),
            (
                json!({ "block": block_of_16(), "row": null }),
                base().per_block([16], ScaleDtype::UE4M3),
            ),
            (
                json!({ "tensor": "F32", "block": block_of_16() }),
                base()
                    .per_block([16], ScaleDtype::UE4M3)
                    .per_tensor(ScaleDtype::F32),
            ),
            (
                json!({ "row": "F32" }),
                base().per_block(row, ScaleDtype::F32),
            ),
            (
                json!({ "row": "F32", "block": block_of_16() }),
                QuantScheme::mxfp4_row_factored().per_block([16], ScaleDtype::UE4M3),
            ),
        ];
        for (levels, scheme) in cases {
            assert_eq!(
                serde_json::from_value::<QuantScheme>(written(levels)).unwrap(),
                scheme
            );
        }
    }

    /// The form schemes were written in while they listed their blocks.
    #[test]
    fn a_scheme_written_as_a_list_of_blocks_reads_back_as_its_levels() {
        let level = |size: BlockSize, dtype| json!({ "size": size, "dtype": dtype });
        let cases = [
            (json!([null, null]), base()),
            (
                json!([
                    level(BlockSize::new([16]), "UE4M3"),
                    level(BlockSize::TENSOR, "F32")
                ]),
                base()
                    .per_block([16], ScaleDtype::UE4M3)
                    .per_tensor(ScaleDtype::F32),
            ),
        ];
        for (blocks, scheme) in cases {
            assert_eq!(
                serde_json::from_value::<QuantScheme>(written(json!({ "blocks": blocks })))
                    .unwrap(),
                scheme
            );
        }
    }

    #[test]
    fn a_scheme_written_with_three_levels_is_refused() {
        let levels = json!({ "tensor": "F32", "row": "F32", "block": block_of_16() });
        assert!(serde_json::from_value::<QuantScheme>(written(levels)).is_err());
    }

    #[test]
    fn a_block_written_past_its_axes_is_refused() {
        let block = json!({ "size": { "storage": [16, 1, 1, 1, 1], "len": 6 }, "dtype": "F32" });
        assert!(serde_json::from_value::<QuantScheme>(written(json!({ "block": block }))).is_err());
    }
}
