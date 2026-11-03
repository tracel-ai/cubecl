//! The block-scaled manual mma: packed codes under block scales, contracted by the device's
//! block-scaled instruction of the form the blueprint states ([`BlockScaledOperands`]): NVFP4's
//! `e2m1` under `e4m3` scales of sixteen, MXFP4's and MXFP8's under `ue8m0` scales of
//! thirty-two, or any other the device takes.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    ir::{ElemType, FloatKind},
    prelude::*,
    quant::scheme::QuantValue,
};

use cubecl_std::tensor::layout::CoordsDyn;

use super::widths::*;
use crate::tile::kind::plane::load_matrix::LDMATRIX_ROW_BYTES;
use crate::*;

/// Words one `ldmatrix` row holds.
const LDMATRIX_ROW_WORDS: usize = LDMATRIX_ROW_BYTES / size_of::<u32>();

/// Bind a block-scaled fragment's widths in role `ident`: its values' and its scales'.
#[cube]
pub(crate) fn register_block_scaled_sizes(
    def: &MmaDefinition<LhsCodes, RhsCodes, f32>,
    #[comptime] ident: MatrixIdent,
) {
    let values = def.vector_size(ident);
    let scales = def.scales_vector_size();
    intrinsic!(|scope| {
        match ident {
            MatrixIdent::A => scope.register_size::<NLB>(values),
            MatrixIdent::B => scope.register_size::<NRB>(values),
            MatrixIdent::Accumulator => {
                panic!("MmaData::block_scaled: an accumulator carries no scales")
            }
        }
        scope.register_size::<NSB>(scales);
    });
}

/// Bind the elements `form` names: each role's codes and the scales', so either fragment's
/// instruction names both inputs.
#[cube]
pub(crate) fn register_block_scaled_elements(#[comptime] form: BlockScaledOperands) {
    intrinsic!(|scope| {
        scope.register_type::<LhsCodes>(ElemType::Float(form.lhs));
        scope.register_type::<RhsCodes>(ElemType::Float(form.rhs));
        scope.register_type::<BlockScale>(ElemType::Float(form.scales));
    });
}

/// The device's block-scaled instruction of `m × n × k` over `form`'s codes under its scales,
/// summing in `CD`: one scale every `form.block` values of `k`.
#[cube]
pub(crate) fn block_scaled_definition<CD: Numeric>(
    #[comptime] form: BlockScaledOperands,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> MmaDefinition<LhsCodes, RhsCodes, CD> {
    MmaDefinition::<LhsCodes, RhsCodes, CD>::new_scaled::<BlockScale>(
        m,
        n,
        k,
        comptime!(k / form.block),
    )
}

/// Refuses `src`, loaded into a fragment of `T` in role `ident`, as an operand of the device's
/// block-scaled instruction of `m × n × k` over `form` ([`MmaOperands::BlockScaled`]) unless it
/// is one: packed codes in shared memory, the ones a register of the instruction holds for
/// `ident` and as many to a word, under one level of scales, one every `form.block` values of
/// `k`, served as stored, on a device that offers the instruction.
/// A fragment already loaded passes: it was refused or admitted as it loaded, and holds the
/// instruction's registers rather than the values it loaded them from.
///
/// The leaf reads each block's scale at its first value and hands it over as `form.scales`, so
/// the scales must cover `form.block` values of `k` or a multiple of it, each exact in that type.
/// Scales stored as [words](Packing::Words) are handed over a word at a time, so a word must be
/// of the instruction's scale type and hold the scales of one step of it. A per-tensor factor
/// above them is the product's to apply: the instruction holds one level.
#[cube]
pub(crate) fn assert_block_scaled<T: Numeric>(
    src: &Tile,
    #[comptime] form: BlockScaledOperands,
    #[comptime] ident: MatrixIdent,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    match &src.kind {
        TileKind::PlaneTile(_) => {}
        TileKind::Memory(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_)
        | TileKind::Lines(_)
        | TileKind::PlanePartition(_) => refuse_unless_block_scaled::<T>(src, form, ident, m, n, k),
    }
}

/// [`assert_block_scaled`] of an operand still in memory.
#[cube]
fn refuse_unless_block_scaled<T: Numeric>(
    src: &Tile,
    #[comptime] form: BlockScaledOperands,
    #[comptime] ident: MatrixIdent,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let packing = src.packing();
    let shared = src.is_shared();
    let levels = src.factor_levels();
    let word_field = src.innermost_factor().word_field();
    let offered = offers_block_scaled(form, m, n, k);
    let as_stored = served_as_stored::<T>();
    let registers = codes_per_register(form, ident, m, n, k);
    comptime!({
        let held = form.codes(ident);
        let stored = match packing {
            Packing::Packed { field } => field.code().map(|code| (code, field.per_word())),
            Packing::Plain | Packing::Words { .. } => None,
        };
        let scale_field = Field::of_float(form.scales);
        let step_scales = k / form.block;
        let refusal = if !offered {
            Some(format!(
                "the device offers no block-scaled instruction of {m}x{n}x{k} over {form:?}"
            ))
        } else if ElemType::Float(held).size_bits() != 8 {
            Some(format!(
                "a register of the instruction holds its {held:?} codes one a byte, which no \
                 packed word stores"
            ))
        } else if stored.map(|(code, _)| code) != code_of(held) {
            Some(format!(
                "the operand is stored as {packing:?}, not packed {held:?} codes"
            ))
        } else if stored.map(|(_, per_word)| per_word) != Some(registers) {
            Some(format!(
                "the operand stores {packing:?} words, and a register of the instruction holds \
                 {registers} {held:?} codes"
            ))
        } else if !as_stored {
            Some(
                "the operand is served at a 16-bit float, which a fragment decodes the codes into \
                 as it loads; state the instruction's operands `MmaOperands::Plain`"
                    .to_string(),
            )
        } else if !shared {
            Some("the operand is not in shared memory; stage it as stored".to_string())
        } else if levels != 1 {
            Some(format!(
                "the operand carries {levels} levels of scales and the instruction reads one; \
                 take the coarser ones out for the sum (`Tile::factor`)"
            ))
        } else if !k.is_multiple_of(form.block) {
            Some(format!(
                "a depth of {k} is not whole blocks of {}",
                form.block
            ))
        } else if word_field.is_some_and(|field| field != scale_field) {
            Some(format!(
                "the scales are words of {word_field:?}, and the instruction reads {:?}",
                form.scales
            ))
        } else if word_field.is_some_and(|field| field.per_word() != step_scales) {
            Some(format!(
                "a word holds {} scales, and one step of the instruction reads {step_scales}",
                scale_field.per_word()
            ))
        } else {
            None
        };
        if let Some(refusal) = refusal {
            panic!("MmaOperands::BlockScaled: {refusal}");
        }
    });
}

/// The code a register element of a block-scaled instruction holds: `e2m1x2` two `e2m1`. `None`
/// for one no stored field holds.
fn code_of(kind: FloatKind) -> Option<QuantValue> {
    match kind {
        FloatKind::E2M1x2 | FloatKind::E2M1 => Some(QuantValue::E2M1),
        FloatKind::E4M3 => Some(QuantValue::E4M3),
        FloatKind::E5M2 => Some(QuantValue::E5M2),
        _ => None,
    }
}

/// Codes one register of the block-scaled instruction of `m × n × k` over `form` holds in role
/// `ident`.
#[cube]
fn codes_per_register(
    #[comptime] form: BlockScaledOperands,
    #[comptime] ident: MatrixIdent,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(usize) {
    register_block_scaled_elements(form);
    let def = block_scaled_definition::<f32>(form, m, n, k);
    let elements = def.vector_size(ident);
    comptime!(elements * BlockScaledOperands::values_per_element(form.codes(ident)))
}

/// Whether a fragment of `T` holds the codes as they are stored, which the block-scaled
/// instruction reads: anything but a float of 16 bits or more, which a fragment decoding the
/// codes into a 16-bit instruction's operand holds instead.
// `T` is read by the expansion, which is where a type has its element.
#[allow(clippy::extra_unused_type_parameters)]
#[cube]
fn served_as_stored<T: Numeric>() -> comptime_type!(bool) {
    intrinsic!(|scope| {
        !matches!(
            T::elem_type(scope),
            ElemType::Float(FloatKind::F16 | FloatKind::BF16 | FloatKind::F32 | FloatKind::TF32)
        )
    })
}

/// Whether the device offers the block-scaled instruction of `m × n × k` over `form`, read off
/// its properties at expansion.
#[cube]
fn offers_block_scaled(
    #[comptime] form: BlockScaledOperands,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(bool) {
    intrinsic!(|scope| {
        let wanted = form.instruction(m, n, k);
        scope
            .state()
            .device_properties
            .as_ref()
            .is_some_and(|properties| properties.features.matmul.scaled_mma.contains(&wanted))
    })
}

/// Load a block-scaled operand of `form`: each unit's words of `src`'s stored codes where the
/// instruction places its registers, each a run of a word's codes along `k`, and the scales of the
/// row (for `A`) or column (for `B`) its scale register serves, one every `form.block` values of
/// the instruction's `k`. `layout` is the window's: an `A` read along its rows, a `B` along its
/// columns where it lies col-major, the stored words running along `k` either way.
///
/// The words come in one `ldmatrix` for the whole fragment where `io` lets the role load through
/// it and a stage in shared memory holds them ([`load_block_scaled_ldmatrix`]); a word at a time
/// otherwise.
///
/// Scales stored as [words](Packing::Words) of a step's scales, as a checkpoint packs them, are one
/// load each, straight into the register; scales stored one to a value are read and narrowed one
/// at a time.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_block_scaled<C: Scalar, NV: Size, NS: Size>(
    src: &Tile,
    values: &mut Array<Vector<C, NV>>,
    scales: &mut Array<Vector<BlockScale, NS>>,
    #[comptime] form: BlockScaledOperands,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIo,
    #[comptime] shape: (usize, usize, usize),
) {
    let (m, n, k) = comptime!(shape);
    let def = block_scaled_definition::<f32>(form, m, n, k);
    let space = comptime!(src.place.space.clone());
    // The window's rows are the served axis, its columns the contraction, however many axes
    // make it up: a block-split space states `k` as its blocks and the position inside one.
    let rows = comptime!(match ident {
        MatrixIdent::A => m,
        MatrixIdent::B => n,
        MatrixIdent::Accumulator => {
            panic!("MmaData::load_block_scaled: an accumulator carries no scales")
        }
    });
    let axes = comptime!(MatrixAxes::new(&space, rows, k).unwrap_or_else(|no| {
        panic!("MmaData::load_block_scaled: a block-scaled window is its rows by `k`: {no}")
    }));
    // The window's own `(row, col)` of the matrix's `(row, col)`: a col-major window is the
    // operand's matrix transposed.
    let transposed = comptime!(match layout {
        MatrixLayout::RowMajor => false,
        MatrixLayout::ColMajor => true,
        MatrixLayout::Undefined => {
            panic!("MmaData::load_block_scaled: a block-scaled window is row- or col-major")
        }
    });
    comptime!(assert!(
        (ident == MatrixIdent::A) != transposed,
        "MmaData::load_block_scaled: the instruction reads both operands along `k`, so a \
         block-scaled window holds its words along `k`: an `A` row-major and a `B` col-major"
    ));
    // A register of the instruction is one stored word (`assert_block_scaled`); the window may be
    // read a whole line of words at a time, up to the sixteen bytes one unit reads at once.
    let packing = src.packing();
    let per_word = comptime!(match packing {
        Packing::Packed { field } => field.per_word(),
        Packing::Plain | Packing::Words { .. } => {
            panic!("MmaData::load_block_scaled: a block-scaled operand is packed codes")
        }
    });
    let load = src.vector_tile();
    let line_words = comptime!(load.values() / per_word);
    comptime!(assert!(
        load.values().is_multiple_of(per_word) && LDMATRIX_ROW_WORDS.is_multiple_of(line_words),
        "MmaData::load_block_scaled: a register of the instruction is one stored word, {per_word} \
         codes, read out of lines of whole words up to sixteen bytes; this window is read {} \
         values a load",
        load.values()
    ));
    let size!(WP) = line_words;
    let words = src.nd_words::<WP>(comptime!(Guard::Checked));
    let unit = UNIT_POS_PLANE;
    let offered = loads_words_as_matrix();
    let shared = src.is_shared();
    let gathered = src.gathered();
    let method = comptime!(match offered && shared && !gathered {
        true => io.load_method(ident),
        false => LoadMethod::Manual,
    });
    match method {
        LoadMethod::LoadMatrix => {
            load_block_scaled_ldmatrix(
                src, &words, values, &def, form, ident, axes, transposed, line_words, per_word,
            );
        }
        LoadMethod::Manual => {
            // An element of the register holds `per_element` codes: an `e2m1x2` two.
            let per_register = def.vector_size(ident);
            let per_element = comptime!(BlockScaledOperands::values_per_element(form.codes(ident)));
            let registers = def.vectors_per_lane(ident);
            #[unroll]
            for i in 0..registers {
                let (row, col) = def.position_of_nth(
                    unit,
                    comptime!((i * per_register * per_element) as u32),
                    ident,
                );
                let (window_row, window_col) = if comptime!(transposed) {
                    (col, row)
                } else {
                    (row, col)
                };
                let at =
                    TileMatrix::value_coords(window_row, window_col, 0usize, &space, axes, 1usize);
                // The window's columns run along `k` whichever its layout.
                let line = words.read(load.index(&at, &space));
                let word = window_col / comptime!(per_word as u32) % comptime!(line_words as u32);
                values[i] =
                    Vector::<C, NV>::reinterpret(line.extract_dynamic(word.cast::<usize>()));
            }
        }
    }
    // The row of `A` or the column of `B` this unit's scale register serves, its scales along
    // `k` one a block.
    let served = def.scales_index(unit, ident);
    let factor = src.innermost_factor();
    let packed_scales = factor.holds_words();
    if comptime!(packed_scales) {
        // The step's scales as the operand stores them: one word, the register itself
        // (`assert_block_scaled` holds a word to a step's scales).
        let at = TileMatrix::value_coords(served, 0u32.runtime(), 0usize, &space, axes, 1usize);
        let word = factor.word_at(&at, comptime!(space.clone()));
        scales[0] = Vector::<BlockScale, NS>::reinterpret(word);
    } else {
        let mut register = Vector::<BlockScale, NS>::empty();
        #[unroll]
        for b in 0..comptime!(k / form.block) {
            // The window's rows are the served axis whichever its layout: an `A`'s rows, a
            // `B`'s columns lying col-major.
            let along_k = comptime!((b * form.block) as u32);
            let at =
                TileMatrix::value_coords(served, along_k.runtime(), 0usize, &space, axes, 1usize);
            let scale = factor.at_coords(&at, comptime!(space.clone()), comptime!(Guard::Checked));
            register.insert(b, BlockScale::cast_from(scale));
        }
        scales[0] = register;
    }
}

/// The words of a block-scaled fragment in one `ldmatrix`: the instruction's registers lie as a
/// 16-bit instruction's do, a register one 8×8 matrix of 16-bit cells, eight rows of four words
/// along `k`, so unit `l` addresses row `l % 8` of matrix `l / 8` and the instruction hands each
/// unit its register's word. The matrices lie where the registers do
/// ([`MmaDefinition::position_of_nth`] of unit 0), so the rows a unit addresses are the rows the
/// word-at-a-time load reads, sixteen bytes at a time, and no matrix is transposed: both
/// operands' windows hold their words along `k`, as the registers run.
///
/// The address is the window's own arrangement of the row ([`Masked::line_slice`]), so a swizzled
/// stage is read where its fill wrote it.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_block_scaled_ldmatrix<C: Scalar, WP: Size, NV: Size>(
    src: &Tile,
    words: &Masked<'_, Vector<u32, WP>, CoordsDyn>,
    values: &mut Array<Vector<C, NV>>,
    def: &MmaDefinition<LhsCodes, RhsCodes, f32>,
    #[comptime] form: BlockScaledOperands,
    #[comptime] ident: MatrixIdent,
    #[comptime] axes: MatrixAxes,
    #[comptime] transposed: bool,
    #[comptime] line_words: usize,
    #[comptime] per_word: usize,
) {
    let space = comptime!(src.place.space.clone());
    let load = src.vector_tile();
    let rank = comptime!(space.rank());
    // An element of the register holds `per_element` codes, an `e2m1x2` two; a register is one
    // word.
    let per_register = def.vector_size(ident);
    let per_element = comptime!(BlockScaledOperands::values_per_element(form.codes(ident)));
    let registers = def.vectors_per_lane(ident);
    let unit = UNIT_POS_PLANE;
    let row_in_matrix = unit % 8;
    let nth_matrix = unit / 8 % comptime!(registers as u32);
    let (row, col) = def.position_of_nth(
        0,
        nth_matrix * comptime!((per_register * per_element) as u32),
        ident,
    );
    // The window's rows are the served axis whichever its layout, its words along `k`.
    let (window_row, window_col) = if comptime!(transposed) {
        (col + row_in_matrix, row)
    } else {
        (row + row_in_matrix, col)
    };
    let at = TileMatrix::value_coords(window_row, window_col, 0usize, &space, axes, 1usize);
    let mut run = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        // One row of a matrix: sixteen bytes of words along `k`, the innermost axis in its lines.
        let extent = comptime!({
            let values = axes.col_run(&space, LDMATRIX_ROW_WORDS * per_word)[p];
            match p == rank - 1 {
                true => (values / (line_words * per_word)) as u32,
                false => values as u32,
            }
        });
        run.push(extent.runtime());
    }
    let row_slice = words.line_slice(load.index(&at, &space), run);
    let regs = def.load_matrix::<Vector<u32, WP>, Const<1>>(row_slice, ident, registers, false);
    #[unroll]
    for i in 0..registers {
        values[i] = Vector::<C, NV>::reinterpret(regs[i].extract(0usize));
    }
}

/// Whether the device offers `ldmatrix`, which moves 16-bit cells: words of codes move as pairs of
/// them. Read off its properties at expansion.
#[cube]
fn loads_words_as_matrix() -> comptime_type!(bool) {
    intrinsic!(|scope| {
        scope
            .state()
            .device_properties
            .as_ref()
            .is_some_and(|properties| {
                properties
                    .features
                    .matmul
                    .ldmatrix
                    .contains(&ElemType::Float(FloatKind::F16))
            })
    })
}

/// `acc += lhs · rhs` over two block-scaled operand fragments via
/// `MmaDefinition::execute_scaled`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mma_execute_block_scaled<A: Numeric>(
    lhs: &Array<Vector<LhsCodes, NLB>>,
    lhs_scales: &Array<Vector<BlockScale, NSB>>,
    rhs: &Array<Vector<RhsCodes, NRB>>,
    rhs_scales: &Array<Vector<BlockScale, NSB>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] form: BlockScaledOperands,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    register_block_scaled_elements(form);
    let def = block_scaled_definition::<A>(form, m, n, k);
    let out = def.execute_scaled(lhs, rhs, &*acc, lhs_scales[0], rhs_scales[0]);
    let num = def.vectors_per_lane(MatrixIdent::Accumulator);
    #[unroll]
    for i in 0..num {
        acc[i] = out[i];
    }
}
