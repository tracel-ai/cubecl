use alloc::vec::Vec;
use core::num::NonZeroU8;

use serde::Deserialize;

use super::{BlockExtent, BlockScale, BlockSize, QuantScheme, QuantStore, QuantValue, ScaleDtype};

/// A scheme as it may be read back: written with its level slots, with the finest and coarser
/// levels it held before them, with the list of blocks it held before those, or with the
/// per-tensor, per-row and per-block levels it held before that list.
#[derive(Deserialize)]
pub(super) struct SavedScheme {
    pub value: QuantValue,
    pub store: QuantStore,
    #[serde(default)]
    pub mode: Option<SavedMode>,
    #[serde(default)]
    pub levels: Option<[Option<BlockScale>; QuantScheme::LEVEL_SLOTS]>,
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

/// How a scheme written before its value said so read its stored fields: as numbers, or as indices
/// into a lookup table.
#[derive(Deserialize)]
pub(super) enum SavedMode {
    Symmetric,
    Lookup,
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
        let value = match saved.mode {
            Some(SavedMode::Lookup) => index_of_width(saved.value)?,
            Some(SavedMode::Symmetric) | None => saved.value,
        };
        let scheme = Self::new(value, saved.store);
        let block = saved.block.map(BlockScale::try_from).transpose()?;
        let row = saved
            .row
            .map(|dtype| BlockScale::per_block(BlockSize::ROW, dtype));
        let tensor = saved.tensor.map(BlockScale::per_tensor);
        let older: Vec<BlockScale> = [saved.finest, saved.coarser, block, row, tensor]
            .into_iter()
            .chain(saved.blocks)
            .flatten()
            .collect();
        match (saved.levels, older.as_slice()) {
            (Some(levels), []) => Ok(Self { levels, ..scheme }),
            (Some(_), _) => Err("a scheme is written with its levels in one form"),
            // Before a scheme stated every level, one with none had one f32 scale for the tensor.
            (None, []) => Ok(scheme.with_levels([BlockScale::per_tensor(ScaleDtype::F32)])),
            (None, older) if older.len() <= QuantScheme::LEVEL_SLOTS => {
                let mut levels = [None; QuantScheme::LEVEL_SLOTS];
                for (slot, level) in levels.iter_mut().zip(older) {
                    *slot = Some(*level);
                }
                Ok(Self { levels, ..scheme })
            }
            (None, _) => Err("a scheme holds at most four scale levels"),
        }
    }
}

/// The index format as wide as `value`, which a scheme written with a lookup mode stored its
/// indices in.
fn index_of_width(value: QuantValue) -> Result<QuantValue, &'static str> {
    match value {
        QuantValue::Q8F | QuantValue::Q8S => Ok(QuantValue::Index8),
        QuantValue::Q4F | QuantValue::Q4S => Ok(QuantValue::Index4),
        QuantValue::Q2F | QuantValue::Q2S => Ok(QuantValue::Index2),
        _ => Err("a lookup scheme stores its indices in an integer value"),
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
            QuantScheme::new(QuantValue::BF16, QuantStore::Native),
            QuantScheme {
                levels: [Some(block_16()), None, Some(tensor_f32()), None],
                ..base()
            },
        ] {
            let json = serde_json::to_value(scheme).unwrap();
            assert_eq!(serde_json::from_value::<QuantScheme>(json).unwrap(), scheme);
        }
    }

    /// A scheme written with a lookup mode stored indices in an integer value as wide.
    #[test]
    fn a_lookup_mode_reads_back_as_the_index_of_its_width() {
        let lookup =
            |value: &str| json!({ "value": value, "store": { "PackedU32": 0 }, "mode": "Lookup" });
        for (value, index) in [
            ("Q8S", QuantValue::Index8),
            ("Q4F", QuantValue::Index4),
            ("Q2F", QuantValue::Index2),
        ] {
            let scheme = serde_json::from_value::<QuantScheme>(lookup(value)).unwrap();
            assert_eq!(scheme.value, index, "{value}");
        }
        assert!(serde_json::from_value::<QuantScheme>(lookup("E4M3")).is_err());
    }

    fn block_16() -> BlockScale {
        BlockScale::per_block([16], ScaleDtype::UE4M3)
    }

    fn tensor_f32() -> BlockScale {
        BlockScale::per_tensor(ScaleDtype::F32)
    }

    fn row_f32() -> BlockScale {
        BlockScale::per_block(BlockSize::ROW, ScaleDtype::F32)
    }

    fn reads(json: Value) -> QuantScheme {
        serde_json::from_value::<QuantScheme>(json).unwrap()
    }

    /// The form published models hold, from the innermost level out.
    #[test]
    fn a_scheme_written_in_levels_reads_back_in_slots() {
        let cases = [
            (
                json!({ "tensor": "F32" }),
                base().with_levels([tensor_f32()]),
            ),
            (
                json!({ "block": block_of_16(), "row": null }),
                base().with_levels([block_16()]),
            ),
            (
                json!({ "tensor": "F32", "block": block_of_16() }),
                base().with_levels([block_16(), tensor_f32()]),
            ),
            (json!({ "row": "F32" }), base().with_levels([row_f32()])),
            (
                json!({ "row": "F32", "block": block_of_16() }),
                base().with_levels([block_16(), row_f32()]),
            ),
            (
                json!({ "tensor": "F32", "row": "F32", "block": block_of_16() }),
                base().with_levels([block_16(), row_f32(), tensor_f32()]),
            ),
        ];
        for (levels, scheme) in cases {
            assert_eq!(reads(written(levels)), scheme);
        }
    }

    /// Before a scheme stated every level, one with none had one f32 scale for the tensor.
    #[test]
    fn a_scheme_written_with_no_level_reads_back_with_one_for_the_tensor() {
        assert_eq!(
            reads(written(json!({}))),
            base().with_levels([tensor_f32()])
        );
        assert_eq!(
            reads(written(json!({ "blocks": [null, null] }))),
            base().with_levels([tensor_f32()])
        );
    }

    /// The form schemes were written in while they listed their blocks.
    #[test]
    fn a_scheme_written_as_a_list_of_blocks_reads_back_in_slots() {
        let level = |size: BlockSize, dtype| json!({ "size": size, "dtype": dtype });
        let blocks = json!([
            level(BlockSize::new([16]), "UE4M3"),
            level(BlockSize::TENSOR, "F32")
        ]);
        assert_eq!(
            reads(written(json!({ "blocks": blocks }))),
            base().with_levels([block_16(), tensor_f32()])
        );
    }

    /// The form schemes were written in while they had a finest level and a coarser one.
    #[test]
    fn a_scheme_written_finest_and_coarser_reads_back_in_slots() {
        let finest = serde_json::to_value(BlockScale::per_block([32], ScaleDtype::UE8M0)).unwrap();
        let coarser = serde_json::to_value(row_f32()).unwrap();
        let json = json!({
            "value": "E2M1",
            "store": { "PackedU32": 0 },
            "mode": "Symmetric",
            "finest": finest,
            "coarser": coarser,
        });
        assert_eq!(
            reads(json),
            base().with_levels([BlockScale::per_block([32], ScaleDtype::UE8M0), row_f32()])
        );
    }

    /// Four empty slots are a scheme with no scale, not the old form with no level.
    #[test]
    fn empty_slots_read_back_as_no_scale() {
        let json = json!({
            "value": "BF16",
            "store": "Native",
            "levels": [null, null, null, null],
        });
        assert_eq!(
            reads(json),
            QuantScheme::new(QuantValue::BF16, QuantStore::Native)
        );
    }

    #[test]
    fn a_scheme_written_with_its_levels_in_two_forms_is_refused() {
        let mut json = serde_json::to_value(QuantScheme::nvfp4()).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("tensor".into(), json!("F32"));
        assert!(serde_json::from_value::<QuantScheme>(json).is_err());
    }

    #[test]
    fn a_block_written_past_its_axes_is_refused() {
        let block = json!({ "size": { "storage": [16, 1, 1, 1, 1], "len": 6 }, "dtype": "F32" });
        assert!(serde_json::from_value::<QuantScheme>(written(json!({ "block": block }))).is_err());
    }
}
