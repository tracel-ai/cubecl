use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use cubecl_zspace::metadata_next::{Indices, Layout, Metadata, MetadataHandle};

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct MetadataVec {
    pub shape: Vec<usize>,
    pub layout: LayoutVec,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum LayoutVec {
    #[default]
    Contiguous,
    Strided {
        strides: Vec<usize>,
    },
}

fn bench_metadata(c: &mut Criterion) {
    let shape_data_8 = [1, 2, 3, 4, 5, 6, 7, 8];
    let shape_8 = Indices::new(&shape_data_8);
    let strides_8 = Indices::new(&shape_data_8);
    let metadata_8 = Metadata {
        shape: shape_8.clone(),
        layout: Layout::Strided {
            strides: strides_8.clone(),
        },
    };

    let shape_data_1 = [1];
    let shape_1 = Indices::new(&shape_data_1);
    let metadata_1 = Metadata {
        shape: shape_1.clone(),
        layout: Layout::Contiguous,
    };

    let metadata_vec_8 = MetadataVec {
        shape: shape_data_8.to_vec(),
        layout: LayoutVec::Strided {
            strides: shape_data_8.to_vec(),
        },
    };

    let handle = MetadataHandle::new();
    let mut group = c.benchmark_group("Metadata");
    let arc_metadat = Arc::new(Metadata {
        shape: Indices::new(&[1, 2, 3, 4, 5, 6, 7, 8]),
        layout: Layout::Contiguous,
    });
    let box_metadat = Box::new(Metadata {
        shape: Indices::new(&[1, 2, 3, 4, 5, 6, 7, 8]),
        layout: Layout::Contiguous,
    });

    group.bench_function("clone_smallvec_rank8", |b| {
        b.iter(|| black_box(metadata_8.clone()))
    });

    group.bench_function("clone_smallvec_rank1", |b| {
        b.iter(|| black_box(metadata_1.clone()))
    });

    group.bench_function("clone_vec_rank8_baseline", |b| {
        b.iter(|| {
            let cloned = metadata_vec_8.clone();
            black_box(cloned)
        })
    });

    group.bench_function("mutate_smallvec_rank8", |b| {
        b.iter(|| {
            let mut new_metadata = metadata_8.clone();
            new_metadata.shape = Indices::new(black_box(&[8, 7, 6, 5, 4, 3, 2, 1]));
            black_box(new_metadata)
        })
    });

    group.bench_function("handle_new_pool", |b| {
        b.iter(|| black_box(MetadataHandle::new()))
    });

    group.bench_function("new_box_baseline", |b| {
        b.iter(|| {
            black_box(Box::new(Metadata {
                shape: Indices::new(&[1, 2, 3, 4, 5, 6, 7, 8]),
                layout: Layout::Contiguous,
            }))
        })
    });

    group.bench_function("box_baseline_clone", |b| {
        b.iter(|| black_box(box_metadat.clone()))
    });

    group.bench_function("new_arc_baseline", |b| {
        b.iter(|| {
            black_box(Arc::new(Metadata {
                shape: Indices::new(&[1, 2, 3, 4, 5, 6, 7, 8]),
                layout: Layout::Contiguous,
            }))
        })
    });

    group.bench_function("arc_baseline_clone", |b| {
        b.iter(|| black_box(arc_metadat.clone()))
    });

    group.bench_function("arc_mut_clone", |b| {
        b.iter(|| {
            let mut new = arc_metadat.as_ref().clone();
            new.layout = Layout::Strided {
                strides: Indices::new(&[45, 45, 45]),
            };
            black_box(Arc::new(new))
        })
    });
    group.bench_function("arc_access", |b| {
        b.iter(|| {
            let shape_0 = arc_metadat.shape[0];
            black_box(Arc::new(shape_0))
        })
    });
    group.bench_function("smallvec_access", |b| {
        b.iter(|| {
            let shape_0 = arc_metadat.shape[0];
            black_box(Arc::new(shape_0))
        })
    });

    let (handle, mut entry) = MetadataHandle::new();
    *entry = metadata_8.clone();
    drop(entry);

    group.bench_function("handle_access_pool", |b| {
        b.iter(|| {
            let m = handle.metadata();
            black_box(m.shape.rank())
        })
    });

    group.bench_function("handle_clone", |b| b.iter(|| black_box(handle.clone())));

    group.bench_function("handle_mutate_pool", |b| {
        b.iter(|| {
            let (new_handle, mut entry) = MetadataHandle::new();
            *entry = metadata_8.clone();
            black_box(new_handle)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_metadata);
criterion_main!(benches);
