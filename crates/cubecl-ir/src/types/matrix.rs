use cubecl_macros_internal::TypeHash;
use derive_more::Display;
use derive_new::new;
use pliron::derive::{format, pliron_type, type_interface_impl};

use crate::{
    interfaces::{AlignedType, HasElementType, TypedExt},
    prelude::*,
};

#[allow(missing_docs)]
#[derive(new, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "cube.matrix",
    format = "$ident `<` $scope `, ` $elem_ty `, ` $shape `, ` $layout `>`",
    generate_get = true,
    verifier = "succ"
)]
pub struct MatrixType {
    pub ident: MatrixIdent,
    pub shape: MatrixShape,
    pub elem_ty: TypeHandle,
    pub layout: MatrixLayout,
    pub scope: MatrixScope,
}

impl MatrixType {
    /// Size of the unpacked matrix elements, in bits
    pub fn unpacked_elem_size_bits(&self, ctx: &Context) -> usize {
        let size_bits = self.elem_ty.size(ctx) * 8;
        size_bits / self.elem_ty.packing_factor(ctx)
    }
}

#[type_interface_impl]
impl AlignedType for MatrixType {
    fn align(&self, ctx: &Context) -> usize {
        self.elem_ty.align(ctx)
    }
}

#[type_interface_impl]
impl HasElementType for MatrixType {
    fn element_type(&self, ctx: &Context) -> Option<TypeHandle> {
        type_cast::<dyn HasElementType>(&*self.elem_ty.deref(ctx))?.element_type(ctx)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[format("$m `x` $n `x` $k")]
pub struct MatrixShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl MatrixShape {
    pub fn num_elems(&self, ident: MatrixIdent) -> usize {
        match ident {
            MatrixIdent::A => self.m * self.k,
            MatrixIdent::B => self.k * self.n,
            MatrixIdent::Accumulator => self.m * self.n,
        }
    }
}

impl From<(usize, usize, usize)> for MatrixShape {
    fn from((m, n, k): (usize, usize, usize)) -> Self {
        Self { m, n, k }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, Display)]
#[format]
#[allow(missing_docs)]
pub enum MatrixIdent {
    #[display("IdentA")]
    A,
    #[display("IdentB")]
    B,
    #[display("IdentAcc")]
    Accumulator,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, Display)]
#[display(rename_all = "snake_case")]
#[format]
#[allow(missing_docs)]
pub enum MatrixLayout {
    ColMajor,
    RowMajor,
    Undefined,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, Display)]
#[display(rename_all = "snake_case")]
#[format]
#[allow(missing_docs)]
pub enum MatrixScope {
    Plane,
    Cube,
}

/// Whether a form of `ldmatrix` or `stmatrix` may, must, or cannot transpose the matrices it moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum MatrixMoveTransposition {
    Optional,
    Required,
    Unavailable,
}

impl MatrixMoveTransposition {
    /// Whether a move that does or does not transpose is valid for this form.
    pub fn admits(self, transpose: bool) -> bool {
        match self {
            MatrixMoveTransposition::Optional => true,
            MatrixMoveTransposition::Required => transpose,
            MatrixMoveTransposition::Unavailable => !transpose,
        }
    }
}

/// The shape of each matrix an `ldmatrix` loads, and how its elements are laid out in shared
/// memory and in the 32-bit registers they land in. Every row is 16 bytes of shared memory whose
/// address one unit supplies, eight units per 8 rows.
///
/// The `B4x16P64` and `B6x16P32` forms read sixteen packed 4-bit (6-bit) values followed by 64 (32)
/// bits of padding per row, and widen each value into the low bits of its own byte in the
/// registers. An `e2m1` operand of the `mma` kinds that take 8-bit containers sits in bits 2 to 5
/// of its byte instead, so a 4-bit load must be shifted left by two before it feeds one.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[format]
pub enum LdMatrixForm {
    /// 8x8 16-bit elements, transposition optional.
    M8N8B16,
    /// 8x16 4-bit elements widened to bytes, never transposed.
    M8N16B4x16P64,
    /// 8x16 6-bit elements widened to bytes, never transposed.
    M8N16B6x16P32,
    /// 16x16 8-bit elements, always transposed.
    M16N16B8,
    /// 16x16 4-bit elements widened to bytes, always transposed.
    M16N16B4x16P64,
    /// 16x16 6-bit elements widened to bytes, always transposed.
    M16N16B6x16P32,
}

impl LdMatrixForm {
    /// The 32-bit registers each unit receives per matrix.
    pub fn registers_per_matrix(self) -> usize {
        if self.is_16x16() { 2 } else { 1 }
    }

    /// Whether this form may, must, or cannot transpose.
    pub fn transposition(self) -> MatrixMoveTransposition {
        match self {
            LdMatrixForm::M8N8B16 => MatrixMoveTransposition::Optional,
            LdMatrixForm::M8N16B4x16P64 | LdMatrixForm::M8N16B6x16P32 => {
                MatrixMoveTransposition::Unavailable
            }
            LdMatrixForm::M16N16B8
            | LdMatrixForm::M16N16B4x16P64
            | LdMatrixForm::M16N16B6x16P32 => MatrixMoveTransposition::Required,
        }
    }

    /// Whether one instruction of this form can load `num_matrices` matrices.
    pub fn admits_matrix_count(self, num_matrices: usize) -> bool {
        if self.is_16x16() {
            matches!(num_matrices, 1 | 2)
        } else {
            matches!(num_matrices, 1 | 2 | 4)
        }
    }

    fn is_16x16(self) -> bool {
        matches!(
            self,
            LdMatrixForm::M16N16B8 | LdMatrixForm::M16N16B4x16P64 | LdMatrixForm::M16N16B6x16P32
        )
    }
}

/// The shape of each matrix an `stmatrix` stores, and the width of its elements. Every form takes
/// one 32-bit register per matrix from each unit.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[format]
pub enum StMatrixForm {
    /// 8x8 16-bit elements, transposition optional.
    M8N8B16,
    /// 16x8 8-bit elements, always transposed.
    M16N8B8,
}

impl StMatrixForm {
    /// Whether this form may or must transpose.
    pub fn transposition(self) -> MatrixMoveTransposition {
        match self {
            StMatrixForm::M8N8B16 => MatrixMoveTransposition::Optional,
            StMatrixForm::M16N8B8 => MatrixMoveTransposition::Required,
        }
    }
}

/// Which scale a block-scaled `mma` takes from the scale registers of each quad of units, PTX's
/// `{byte-id, thread-id}`. The default, byte 0 of the first units of the quad, is what the scaled
/// `mma` reads when no selector is given.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[format("`{` $byte_id `,` $thread_id `}`")]
pub struct BlockScaleSelector {
    /// The first byte of the scale register read: any byte under `scale_vec::1X`, 0 or 2 under
    /// `2X`, 0 under `4X`.
    pub byte_id: usize,
    /// The units of the quad whose registers are read: for A, 0 selects units 0 and 1 and 1
    /// selects units 2 and 3; for B, the unit's index within the quad.
    pub thread_id: usize,
}

impl BlockScaleSelector {
    /// Whether this selector names scales of the `ident` operand of a block-scaled `mma` with
    /// `scales_factor` scales per row or column. PTX leaves any other selector undefined.
    pub fn is_valid_for(self, ident: MatrixIdent, scales_factor: usize) -> bool {
        let byte_is_valid = match scales_factor {
            1 => self.byte_id < 4,
            2 => matches!(self.byte_id, 0 | 2),
            _ => self.byte_id == 0,
        };
        let thread_is_valid = match ident {
            MatrixIdent::A => self.thread_id < 2,
            MatrixIdent::B => self.thread_id < 4,
            MatrixIdent::Accumulator => false,
        };
        byte_is_valid && thread_is_valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The valid selectors are exactly PTX's table: any byte under `1X`, byte 0 or 2 under `2X`,
    /// byte 0 under `4X`; thread 0 or 1 for A, 0 to 3 for B; none for the accumulator.
    #[test]
    fn block_scale_selectors_valid_exactly_where_the_ptx_table_says() {
        let valid_bytes = [(1, &[0, 1, 2, 3][..]), (2, &[0, 2][..]), (4, &[0][..])];
        let valid_threads = [
            (MatrixIdent::A, &[0, 1][..]),
            (MatrixIdent::B, &[0, 1, 2, 3][..]),
            (MatrixIdent::Accumulator, &[][..]),
        ];
        for (scales_factor, bytes) in valid_bytes {
            for (ident, threads) in valid_threads {
                for byte_id in 0..5 {
                    for thread_id in 0..5 {
                        let selector = BlockScaleSelector { byte_id, thread_id };
                        assert_eq!(
                            selector.is_valid_for(ident, scales_factor),
                            bytes.contains(&byte_id) && threads.contains(&thread_id),
                            "{selector:?} for {ident:?} with {scales_factor} scales"
                        );
                    }
                }
            }
        }
    }

    /// The 16x16 loads take one or two matrices, two registers each, and must transpose; the
    /// 8x16 loads take one, two or four, one register each, and cannot.
    #[test]
    fn ldmatrix_forms_admit_the_counts_and_transposition_ptx_allows() {
        let sixteen_by_sixteen = [
            LdMatrixForm::M16N16B8,
            LdMatrixForm::M16N16B4x16P64,
            LdMatrixForm::M16N16B6x16P32,
        ];
        for form in sixteen_by_sixteen {
            assert_eq!(form.registers_per_matrix(), 2);
            assert!(form.admits_matrix_count(1) && form.admits_matrix_count(2));
            assert!(!form.admits_matrix_count(4) && !form.admits_matrix_count(3));
            assert!(form.transposition().admits(true) && !form.transposition().admits(false));
        }
        for form in [LdMatrixForm::M8N16B4x16P64, LdMatrixForm::M8N16B6x16P32] {
            assert_eq!(form.registers_per_matrix(), 1);
            assert!(
                [1, 2, 4]
                    .iter()
                    .all(|count| form.admits_matrix_count(*count))
            );
            assert!(!form.admits_matrix_count(3));
            assert!(!form.transposition().admits(true) && form.transposition().admits(false));
        }
        let m8n8 = LdMatrixForm::M8N8B16;
        assert_eq!(m8n8.registers_per_matrix(), 1);
        assert!(
            [1, 2, 4]
                .iter()
                .all(|count| m8n8.admits_matrix_count(*count))
        );
        assert!(m8n8.transposition().admits(true) && m8n8.transposition().admits(false));
    }

    /// `stmatrix` `m16n8 .b8` must transpose; `m8n8 .b16` may or may not.
    #[test]
    fn stmatrix_forms_admit_the_transposition_ptx_allows() {
        let m16n8 = StMatrixForm::M16N8B8.transposition();
        assert!(m16n8.admits(true) && !m16n8.admits(false));
        let m8n8 = StMatrixForm::M8N8B16.transposition();
        assert!(m8n8.admits(true) && m8n8.admits(false));
    }
}
