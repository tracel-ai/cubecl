//! The `ldmatrix` transport of the manual-mma encoding: a unit hands the instruction its row's
//! address, where the window's layout arranges it, so padded and swizzled stages alike are
//! read in 16-byte rows rather than a cell at a time.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    prelude::*,
};

use crate::*;

/// Bytes one row of an `ldmatrix` 8×8 matrix holds, which one unit addresses.
pub(super) const LDMATRIX_ROW_BYTES: usize = 16;

/// `ldmatrix` load: each unit hands the instruction the address of one 16-byte row of one of the
/// fragment's 8×8 matrices, and the instruction hands every unit its cells. Unit `l` addresses
/// row `l % 8` of matrix `l / 8`, and the matrices lie where the fragment's registers do
/// ([`MmaDefinition::position_of_nth`] of unit 0), so the rows a unit addresses are the rows the
/// manual load would read its cells from, taken 16 bytes at a time.
///
/// The address is the window's own arrangement of that row ([`Masked::line_slice`]): a padded or a
/// swizzled stage is read where its fill wrote it, which a base and a row stride could not say of
/// a swizzled one. The instruction transposes where the window's order is not the one the
/// fragment's register vectors run along: a col weight bound as `{n, k}` beside a `B` whose
/// vectors run along `k` needs none.
///
/// A row is a whole number of the window's lines, and the window lies in shared memory, which a
/// stage's 16-byte alignment ([`Memory::smem_owned`]) keeps every row's address aligned in: the
/// manual-mma fragment load reads any other window a cell at a time.
#[cube]
pub(super) fn load_ldmatrix<T: Numeric, W: Size, N: Size, A: Numeric, B: Numeric, CD: Numeric>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] edges: (usize, usize),
    #[comptime] k_offset: usize,
) {
    let width = src.vector_size();
    let elem_size = T::size().comptime();
    let row_cells = comptime!(LDMATRIX_ROW_BYTES / elem_size);
    let (rows, cols) = comptime!(edges);
    let col_major_window = comptime!(match layout {
        MatrixLayout::RowMajor => false,
        MatrixLayout::ColMajor => true,
        MatrixLayout::Undefined => {
            panic!("MmaData::load: an ldmatrix load reads a row- or col-major window")
        }
    });
    let view = if comptime!(col_major_window) {
        src.fragment_matrix_packed::<T, W>(cols, rows)
    } else {
        src.fragment_matrix_packed::<T, W>(rows, cols)
    };

    let num_regs = def.vectors_per_lane(ident);
    let vector_size = def.vector_size(ident);
    let vector_layout = def.vector_layout(ident);
    // Whether the instruction transposes each 8×8 matrix on its way to the registers.
    let instruction_transposes = comptime!(vector_layout != layout);
    // Units address a matrix eight at a time, wrapping where the fragment holds fewer than four.
    let unit = UNIT_POS_PLANE;
    let row_in_matrix = unit % 8;
    let nth_matrix = unit / 8 % comptime!(num_regs as u32);
    let (row, col) = def.position_of_nth(0, nth_matrix * comptime!(vector_size as u32), ident);
    // The step this fragment loads, where the window is several instructions deep.
    let (row, col) = along_k(row, col, ident, k_offset);
    // The unit's row runs along the window's lines, and the next unit's starts a line-row on.
    let (line_row, cell) = if comptime!(col_major_window) {
        (col + row_in_matrix, row)
    } else {
        (row + row_in_matrix, col)
    };
    let row_slice = view.line_slice(
        (line_row, cell / comptime!(width as u32)),
        (
            1u32.runtime(),
            comptime!((row_cells / width) as u32).runtime(),
        ),
    );
    let regs =
        def.load_matrix::<Vector<T, W>, N>(row_slice, ident, num_regs, instruction_transposes);
    #[unroll]
    for i in 0..num_regs {
        fragment[i] = regs[i];
    }
}

/// `(row, col)`, a cell of an `ident` operand one instruction deep, moved `k_offset` along `k`:
/// where it lies in a window several instructions deep, at the step `k_offset` starts.
#[cube]
pub(crate) fn along_k(
    row: u32,
    col: u32,
    #[comptime] ident: MatrixIdent,
    #[comptime] k_offset: usize,
) -> (u32, u32) {
    match comptime!(ident) {
        MatrixIdent::A => (row, col + comptime!(k_offset as u32)),
        MatrixIdent::B => (row + comptime!(k_offset as u32), col),
        MatrixIdent::Accumulator => (row, col),
    }
}
