//! The block-scaled manual mma: `e2m1` operands under `e4m3` scales, NVFP4's, contracted by the
//! device's block-scaled instruction.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    e2m1x2, e4m3,
    ir::{ElemType, FloatKind, features::ScaledMmaConfig},
    prelude::*,
    quant::scheme::QuantValue,
};

use cubecl_std::tensor::layout::CoordsDyn;

use super::widths::*;
use crate::tile::kind::plane::load_matrix::LDMATRIX_ROW_BYTES;
use crate::*;

/// `e2m1` values one stored word holds, and so one register of the block-scaled instruction.
pub(crate) const E2M1_PER_WORD: usize = 8;

/// Words one `ldmatrix` row holds.
const LDMATRIX_ROW_WORDS: usize = LDMATRIX_ROW_BYTES / size_of::<u32>();

/// Values one `e2m1` block scale covers along the contraction under the instruction this
/// encoding runs: NVFP4's block, a `ue4m3` scale every sixteen values.
pub(crate) const E2M1_SCALE_BLOCK: usize = 16;

/// Bind a block-scaled fragment's widths in role `ident`: its values' and its scales'.
#[cube]
pub(crate) fn register_block_scaled_sizes(
    def: &MmaDefinition<e2m1x2, e2m1x2, f32>,
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

/// The device's block-scaled instruction of `m × n × k` over `e2m1` operands under `e4m3` scales,
/// summing in `CD`: one scale every [`E2M1_SCALE_BLOCK`] values of `k`.
#[cube]
pub(crate) fn block_scaled_definition<CD: Numeric>(
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> MmaDefinition<e2m1x2, e2m1x2, CD> {
    MmaDefinition::<e2m1x2, e2m1x2, CD>::new_scaled::<e4m3>(
        m,
        n,
        k,
        comptime!(k / E2M1_SCALE_BLOCK),
    )
}

/// Refuses `src`, loaded into a fragment of `T`, as an operand of the device's block-scaled
/// instruction of `m × n × k` ([`MmaOperands::BlockScaled`]) unless it is one: packed `e2m1` values
/// in shared memory under one level of scales, one every [`E2M1_SCALE_BLOCK`] values of `k`, served
/// as stored, on a device that offers the instruction.
/// A fragment already loaded passes: it was refused or admitted as it loaded, and holds the
/// instruction's registers rather than the values it loaded them from.
///
/// What the instruction takes is NVFP4: the leaf reads each block's scale at its first value and
/// hands it over as `e4m3`, so the scales must cover sixteen values of `k` or a multiple of
/// sixteen, each exact in `e4m3`. A per-tensor factor above them is the product's to apply: the
/// instruction holds one level.
#[cube]
pub(crate) fn assert_block_scaled<T: Numeric>(
    src: &Tile,
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
        | TileKind::PlanePartition(_) => refuse_unless_block_scaled::<T>(src, m, n, k),
    }
}

/// [`assert_block_scaled`] of an operand still in memory.
#[cube]
fn refuse_unless_block_scaled<T: Numeric>(
    src: &Tile,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let packing = src.packing();
    let shared = src.is_shared();
    let levels = src.factor_levels();
    let offered = offers_block_scaled(m, n, k);
    let as_stored = served_as_stored::<T>();
    comptime!({
        let e2m1 = matches!(
            packing,
            Packing::Packed {
                field: Field::Quant(QuantValue::E2M1) | Field::ConvertedE2M1
            }
        );
        let refusal = if !offered {
            Some(format!(
                "the device offers no block-scaled instruction of {m}x{n}x{k}"
            ))
        } else if !e2m1 {
            Some(format!(
                "the operand is stored as {packing:?}, not packed e2m1 values"
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
        } else if !k.is_multiple_of(E2M1_SCALE_BLOCK) {
            Some(format!(
                "a depth of {k} is not whole blocks of {E2M1_SCALE_BLOCK}"
            ))
        } else {
            None
        };
        if let Some(refusal) = refusal {
            panic!("MmaOperands::BlockScaled: {refusal}");
        }
    });
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

/// Whether the device offers the block-scaled instruction of `m × n × k` over `e2m1` operands
/// under `e4m3` scales, read off its properties at expansion.
#[cube]
fn offers_block_scaled(
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(bool) {
    intrinsic!(|scope| {
        let fp4 = ElemType::Float(FloatKind::E2M1x2);
        let wanted = ScaledMmaConfig {
            a_type: fp4,
            b_type: fp4,
            cd_type: ElemType::Float(FloatKind::F32),
            scales_type: ElemType::Float(FloatKind::E4M3),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            scales_factor: (k / E2M1_SCALE_BLOCK) as u32,
        };
        scope
            .state()
            .device_properties
            .as_ref()
            .is_some_and(|properties| properties.features.matmul.scaled_mma.contains(&wanted))
    })
}

/// Load a block-scaled operand: each unit's words of `src`'s stored `e2m1` values where the
/// instruction places its registers, each a run of eight values along `k`, and the four scales
/// of the row (for `A`) or column (for `B`) its scale register serves, one every
/// [`E2M1_SCALE_BLOCK`] values of the instruction's `k`. `layout` is the window's: an `A` read
/// along its rows, a `B` along its columns where it lies col-major, the stored words running along
/// `k` either way.
///
/// The words come in one `ldmatrix` for the whole fragment where `io` lets the role load through
/// it and a stage in shared memory holds them ([`load_block_scaled_ldmatrix`]); a word at a time
/// otherwise.
///
/// Scales stored as words of four, as an NVFP4 checkpoint packs them, are one load each, straight
/// into the register; scales stored one to a value are read and narrowed one at a time.
#[cube]
pub(crate) fn load_block_scaled<NV: Size, NS: Size>(
    src: &Tile,
    values: &mut Array<Vector<e2m1x2, NV>>,
    scales: &mut Array<Vector<e4m3, NS>>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIo,
    #[comptime] shape: (usize, usize, usize),
) {
    let (m, n, k) = comptime!(shape);
    let def = block_scaled_definition::<f32>(m, n, k);
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
    // A register of the instruction is one stored word, eight values; the window may be read a
    // whole line of words at a time, up to the sixteen bytes one unit reads at once.
    let load = src.vector_tile();
    let line_words = comptime!(load.values() / E2M1_PER_WORD);
    comptime!(assert!(
        load.values().is_multiple_of(E2M1_PER_WORD)
            && LDMATRIX_ROW_WORDS.is_multiple_of(line_words),
        "MmaData::load_block_scaled: a register of the instruction is one stored word, eight \
         `e2m1` values, read out of lines of whole words up to sixteen bytes; this window is read \
         {} values a load",
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
                src, &words, values, &def, ident, axes, transposed, line_words,
            );
        }
        LoadMethod::Manual => {
            // An `e2m1x2` holds two values.
            let per_register = def.vector_size(ident);
            let registers = def.vectors_per_lane(ident);
            #[unroll]
            for i in 0..registers {
                let (row, col) =
                    def.position_of_nth(unit, comptime!((i * per_register * 2) as u32), ident);
                let (window_row, window_col) = if comptime!(transposed) {
                    (col, row)
                } else {
                    (row, col)
                };
                let at =
                    TileMatrix::value_coords(window_row, window_col, 0usize, &space, axes, 1usize);
                // The window's columns run along `k` whichever its layout.
                let line = words.read(load.index(&at, &space));
                let word =
                    window_col / comptime!(E2M1_PER_WORD as u32) % comptime!(line_words as u32);
                values[i] =
                    Vector::<e2m1x2, NV>::reinterpret(line.extract_dynamic(word.cast::<usize>()));
            }
        }
    }
    // The row of `A` or the column of `B` this unit's scale register serves, its scales along
    // `k` one a block.
    let served = def.scales_index(unit, ident);
    let factor = src.innermost_factor();
    let packed_scales = factor.holds_words();
    if comptime!(packed_scales) {
        // The step's four scales as the operand stores them: one word, the register itself.
        comptime!(assert!(
            k / E2M1_SCALE_BLOCK == 4,
            "MmaData::load_block_scaled: a word holds four block scales, and this step reads {}",
            k / E2M1_SCALE_BLOCK
        ));
        let at = TileMatrix::value_coords(served, 0u32.runtime(), 0usize, &space, axes, 1usize);
        let word = factor.word_at(&at, comptime!(space.clone()));
        scales[0] = Vector::<e4m3, NS>::reinterpret(word);
    } else {
        let mut register = Vector::<e4m3, NS>::empty();
        #[unroll]
        for b in 0..comptime!(k / E2M1_SCALE_BLOCK) {
            // The window's rows are the served axis whichever its layout: an `A`'s rows, a
            // `B`'s columns lying col-major.
            let along_k = comptime!((b * E2M1_SCALE_BLOCK) as u32);
            let at =
                TileMatrix::value_coords(served, along_k.runtime(), 0usize, &space, axes, 1usize);
            let scale = factor.at_coords(&at, comptime!(space.clone()));
            register.insert(b, e4m3::cast_from(scale));
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
fn load_block_scaled_ldmatrix<WP: Size, NV: Size>(
    src: &Tile,
    words: &Masked<'_, Vector<u32, WP>, CoordsDyn>,
    values: &mut Array<Vector<e2m1x2, NV>>,
    def: &MmaDefinition<e2m1x2, e2m1x2, f32>,
    #[comptime] ident: MatrixIdent,
    #[comptime] axes: MatrixAxes,
    #[comptime] transposed: bool,
    #[comptime] line_words: usize,
) {
    let space = comptime!(src.place.space.clone());
    let load = src.vector_tile();
    let rank = comptime!(space.rank());
    // An `e2m1x2` holds two values; a register is one word.
    let per_register = def.vector_size(ident);
    let registers = def.vectors_per_lane(ident);
    let unit = UNIT_POS_PLANE;
    let row_in_matrix = unit % 8;
    let nth_matrix = unit / 8 % comptime!(registers as u32);
    let (row, col) =
        def.position_of_nth(0, nth_matrix * comptime!((per_register * 2) as u32), ident);
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
            let values = axes.col_run(&space, LDMATRIX_ROW_WORDS * E2M1_PER_WORD)[p];
            match p == rank - 1 {
                true => (values / (line_words * E2M1_PER_WORD)) as u32,
                false => values as u32,
            }
        });
        run.push(extent.runtime());
    }
    let row_slice = words.line_slice(load.index(&at, &space), run);
    let regs = def.load_matrix::<Vector<u32, WP>, Const<1>>(row_slice, ident, registers, false);
    #[unroll]
    for i in 0..registers {
        values[i] = Vector::<e2m1x2, NV>::reinterpret(regs[i].extract(0usize));
    }
}

/// Whether the device offers `ldmatrix`, which moves 16-bit cells: words of `e2m1` values move as
/// pairs of them. Read off its properties at expansion.
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
pub(crate) fn mma_execute_block_scaled<A: Numeric>(
    lhs: &Array<Vector<e2m1x2, NLB>>,
    lhs_scales: &Array<Vector<e4m3, NSB>>,
    rhs: &Array<Vector<e2m1x2, NRB>>,
    rhs_scales: &Array<Vector<e4m3, NSB>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let def = block_scaled_definition::<A>(m, n, k);
    let out = def.execute_scaled(lhs, rhs, &*acc, lhs_scales[0], rhs_scales[0]);
    let num = def.vectors_per_lane(MatrixIdent::Accumulator);
    #[unroll]
    for i in 0..num {
        acc[i] = out[i];
    }
}
