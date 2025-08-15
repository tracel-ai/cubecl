use cubecl_matmul::components::tile::TileMatmul;

use crate::components::AttentionPrecision;

pub trait ScoreMatmul<AP: AttentionPrecision>: TileMatmul<AP::ES, AP::ES, AP::EA> {}
impl<AP, T> ScoreMatmul<AP> for T
where
    AP: AttentionPrecision,
    T: TileMatmul<AP::ES, AP::ES, AP::EA>,
{
}

pub trait ValueMatmul<AP: AttentionPrecision>: TileMatmul<AP::ES, AP::ES, AP::EA> {}
impl<AP, T> ValueMatmul<AP> for T
where
    AP: AttentionPrecision,
    T: TileMatmul<AP::ES, AP::ES, AP::EA>,
{
}
