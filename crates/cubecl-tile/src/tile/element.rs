//! The element a tile serves its values at, held as data ([`Tile::element`](crate::Tile)), and
//! named as a type where code needs one.
//!
//! A function defines a type for a tile's element where it first meets the tile as data
//! (`let define!(Served) = self.element;`), and hands the type to what it calls as a generic. A
//! public op does it for the tiles it is handed; inside the crate, so does the code that holds
//! tiles apart from any op: a pipeline's payloads and sources, a meeting's stages, a factor's
//! scales, a codebook's table.
//!
//! The type a function defines is its own: no caller can pass it in, and nothing the function
//! calls names it but through a generic, so no function goes on at an element another one
//! named. A function re-entered on a tile of another element gets its type back when the inner
//! call returns.

use cubecl::define_scalar;

// The element a value is held at, apart from any name: a cmma fragment, a warpgroup accumulator
// or an erased call is kept at it and named at its element where it is used, and a read that
// converts a value defines it for as long as the read expands.
define_scalar!(pub(crate) Held);
