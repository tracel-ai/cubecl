//! A clean, launchable [`Tile`] input for tests. Describe a tile as a [`Space`]
//! with a stack of `.split`/`.tile` levels and get a launchable tile: the buffer
//! is a plain `[grid…, tile…]` strided tensor (or `[grid…, level1…, …]` when
//! recursively tiled), presented in its logical space via an explicit
//! `tiled_view`. No `Tiler`, no semantic-view juggling.

use cubecl::{
    bytes::Bytes,
    client::Client,
    prelude::TensorArg,
    prelude::TensorBinding,
    quant::scheme::{QuantScheme, QuantValue},
    zspace::{Shape, Tiling},
};
use cubecl::{frontend::Scalar, std::tensor::TensorHandle};
use cubecl_tile::{
    Axis, Launcher, Space, TileArgLaunch, TileSpec,
    kind::Field,
    launch::{Arg, Labelled},
};

use crate::support::{TestInput, TestInputBuilder};

/// A tile-shaped test input: the device buffer plus the logical [`Space`] it's
/// viewed in. The sub-tile sizes live in the buffer's trailing dims, so the view
/// reads them from there.
pub struct TileInput {
    handle: TensorHandle,
    space: Space,
    /// The axes the tile spans, in buffer order.
    axes: Vec<Axis>,
    levels: usize,
}

impl TileInput {
    /// Start building a tile over `space`. Stack tiling levels coarse→fine with
    /// [`split`](TileInputBuilder::split) (by count) or
    /// [`tile`](TileInputBuilder::tile) (by element edge): chain them for
    /// recursion, or [`untiled`](TileInputBuilder::untiled) for none, then a data
    /// finalizer ([`arange`](TileInputBuilder::arange) /
    /// [`zeros`](TileInputBuilder::zeros)).
    pub fn builder(client: &Client, space: Space) -> TileInputBuilder {
        TileInputBuilder {
            client: client.clone(),
            space,
            levels: None,
        }
    }

    /// The tile `handle` holds over `space`, stored `levels` levels deep.
    fn new(handle: TensorHandle, space: Space, levels: usize) -> Self {
        let axes = (0..space.rank()).map(|i| space.axis_at(i)).collect();
        TileInput {
            handle,
            space,
            axes,
            levels,
        }
    }

    /// Launch arg for this tile's raw global buffer: a plain [`TensorArg`] over the
    /// `[grid…, tile…]` buffer, optionally re-lined by `vector_size` along the inner
    /// axis (so a kernel reading raw `Vector<E, S>` slices lands on contiguous lines).
    /// `vector_size == 1` is the plain buffer. For a `TileArg` pass the plain buffer: the
    /// argument states the width it is bound at and the metadata stays scalar-unit.
    pub fn tensor_arg(&self, vector_size: usize) -> TensorArg {
        if vector_size <= 1 {
            return self.handle.clone().binding().into_tensor_arg();
        }
        let shape = self.handle.shape();
        let strides = self.handle.strides();
        let inner = shape.len() - 1;
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s / vector_size } else { s })
            .collect();
        let new_strides: Vec<usize> = strides
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s } else { s / vector_size })
            .collect();
        TensorHandle::new(
            self.handle.handle.clone(),
            new_shape,
            new_strides,
            self.handle.dtype,
        )
        .binding()
        .into_tensor_arg()
    }

    /// The tile as one operand of `launcher`, served at the element it holds, bound `width` wide.
    /// The launch derives its spec: the axes its tiles reach past are masked, and a stored tiling
    /// is read as the buffer lays it out.
    ///
    /// # Panics
    ///
    /// Where the launch refuses the binding.
    pub fn arg(&self, launcher: &Launcher, width: usize) -> TileArgLaunch {
        self.bind(launcher)
            .vectorize(width)
            .build()
            .unwrap_or_else(|refusal| panic!("TileInput::arg: {refusal}"))
            .arg()
    }

    /// [`arg`](Self::arg) before it is built: the launch's binding of this tile, labelled with
    /// its axes, for a caller that states more of it (a delivery, a boundary).
    pub fn bind<'a>(&'a self, launcher: &'a Launcher) -> Arg<'a, Labelled> {
        let mut binding = self.handle.clone().binding();
        binding.tiling = self.tiling();
        launcher.arg(binding, self.handle.dtype).axes(&self.axes)
    }

    /// How the buffer stores each axis: once untiled, else once per level and once for the grid.
    fn tiling(&self) -> Tiling {
        match self.levels {
            0 => Tiling::UNTILED,
            levels => Tiling::new(&vec![levels + 1; self.space.rank()])
                .expect("TileInput: a tiling the buffer's dims hold"),
        }
    }

    /// The semantic space the tile lives in.
    pub fn space(&self) -> Space {
        self.space.clone()
    }

    /// The device handle, for reading an output back.
    pub fn handle(&self) -> TensorHandle {
        self.handle.clone()
    }
}

/// One tiling level, added coarse→fine. [`Split`](TileLevel::Split) and
/// [`Tile`](TileLevel::Tile) are duals against the running tile edge: `Tile(e)`
/// sets the current tile to `e` elements; `Split(n)` divides it into `n`.
enum TileLevel {
    /// Divide the current tile into this many sub-tiles per axis.
    Split(Vec<usize>),
    /// Set the current tile to this many elements per axis.
    Tile(Vec<usize>),
}

/// Fluent builder for a [`TileInput`]: a [`Space`], a coarse→fine stack of tiling
/// levels (each a [`split`](Self::split) or [`tile`](Self::tile)), and a data
/// finalizer that fills the `[grid…, level…, finest…]` buffer.
pub struct TileInputBuilder {
    client: Client,
    space: Space,
    levels: Option<Vec<TileLevel>>,
}

impl TileInputBuilder {
    /// Divide the current tile into `counts[axis]` sub-tiles per axis: a finer
    /// level. Chain for recursion: `.split(&[4, 4]).split(&[2, 2])`.
    pub fn split(mut self, counts: &[usize]) -> Self {
        self.levels
            .get_or_insert_with(Vec::new)
            .push(TileLevel::Split(counts.to_vec()));
        self
    }

    /// Set the current tile to `edges[axis]` elements per axis: a finer level.
    /// The dual of [`split`](Self::split) (it divides the current edge down to
    /// `edges`), so `.tile(&[16, 16]).tile(&[8, 8])` ≡ `.tile(&[16, 16]).split(&[2, 2])`.
    pub fn tile(mut self, edges: &[usize]) -> Self {
        self.levels
            .get_or_insert_with(Vec::new)
            .push(TileLevel::Tile(edges.to_vec()));
        self
    }

    /// No sub-tiling: the buffer is the logical shape itself, row-major (zero tile
    /// levels: the view is the identity).
    pub fn untiled(mut self) -> Self {
        self.levels = Some(Vec::new());
        self
    }

    /// Arange `0, 1, 2, …` written straight onto the physical buffer; the
    /// `tiled_view` then presents it in logical coordinates.
    pub fn arange(self) -> TileInput {
        self.build(TestInputBuilder::arange)
    }

    /// All-zeros physical buffer: e.g. a matmul output.
    pub fn zeros(self) -> TileInput {
        self.build(TestInputBuilder::zeros)
    }

    /// Uniformly distributed random values in `[lo, hi)` range, seeded by `seed`.
    pub fn uniform(self, seed: u64, lo: f32, hi: f32) -> TileInput {
        self.build(|b: TestInputBuilder| b.uniform(seed, lo, hi))
    }

    /// Continue as a packed-u32 quantized input: a *container* step, like the tiling ones.
    /// the tile will be declared **in values** (its shape and strides count them) while the
    /// buffer holds the packed words, the binding convention the tile launch expects for a
    /// packed operand. Content comes from the quantized builder's own finalizer (e.g.
    /// [`arange`](QuantizedTileInputBuilder::arange)), which also mints the scales: a
    /// quantized tensor is one thing (data + scales + scheme). Untiled only: packed storage
    /// has no physically tiled layout.
    ///
    /// The launch binds the two as one operand, [`values_arg`] under its scales
    /// ([`scales_arg`]), and the kernel states the decode where it happens.
    ///
    /// [`values_arg`]: QuantizedTileInput::values_arg
    /// [`scales_arg`]: QuantizedTileInput::scales_arg
    pub fn packed(self, scheme: &QuantScheme) -> QuantizedTileInputBuilder {
        let levels = self
            .levels
            .expect("TileInput: set .untiled() before .packed");
        assert!(
            levels.is_empty(),
            "TileInput::packed: a packed input is untiled"
        );
        QuantizedTileInputBuilder {
            client: self.client,
            space: self.space,
            scheme: *scheme,
        }
    }

    /// Build the `[grid…, level…, finest…]` device buffer, filled by `fill` (a
    /// `TestInput` finalizer like `arange`/`zeros`) in physical row-major order.
    /// Walking coarse→fine, each level becomes one block of `rank` dims and the
    /// leftover edge is the finest block: `(levels + 1) * rank` dims, the layout
    /// the `tiled_view` reads back.
    fn build(self, fill: impl FnOnce(TestInputBuilder) -> TestInput) -> TileInput {
        let levels = self
            .levels
            .expect("TileInput: set .split/.tile(...) or .untiled() before a finalizer");
        let rank = self.space.rank();

        let mut current: Vec<usize> = (0..rank)
            .map(|i| self.space.extent(self.space.axis_at(i)))
            .collect();
        let mut blocks: Vec<Vec<usize>> = Vec::with_capacity(levels.len() + 1);
        for level in &levels {
            let (values, is_split) = match level {
                TileLevel::Split(values) => (values, true),
                TileLevel::Tile(values) => (values, false),
            };
            assert_eq!(
                values.len(),
                rank,
                "TileInput: a tile level needs one value per axis (rank {rank})"
            );
            let block: Vec<usize> = (0..rank)
                .map(|i| {
                    let (edge, value) = (current[i], values[i]);
                    assert!(
                        value != 0 && edge % value == 0,
                        "TileInput: tile value {value} does not divide the current edge {edge} on axis {i}"
                    );
                    let (dim, next) = if is_split {
                        (value, edge / value)
                    } else {
                        (edge / value, value)
                    };
                    current[i] = next;
                    dim
                })
                .collect();
            blocks.push(block);
        }
        blocks.push(current);

        let mut dims = Vec::with_capacity(blocks.len() * rank);
        for block in &blocks {
            dims.extend_from_slice(block);
        }
        let builder = TestInput::builder(self.client, Shape::from(dims));
        TileInput::new(
            fill(builder).generate_without_host_data(),
            self.space,
            levels.len(),
        )
    }
}

/// The quantized continuation of a [`TileInputBuilder`]: the container is fixed
/// (packed-u32, values-unit declaration, the scheme's block grid for scales); a content
/// finalizer fills it and mints the values tile and its scales together: a quantized
/// tensor is one thing (data, scales, scheme).
pub struct QuantizedTileInputBuilder {
    client: Client,
    space: Space,
    scheme: QuantScheme,
}

impl QuantizedTileInputBuilder {
    /// `arange`, wrapped into the scheme's representable range: values run
    /// `lo, lo+1, …, hi, lo, …`, so every representable value (and its sign extension)
    /// appears. Scales are a distinct-per-block ramp (`0.05 · (block + 1)`) on the scheme's
    /// block grid.
    ///
    /// Integer value types only. The walk is over whole numbers packed as two's complement, and
    /// [`q`](QuantizedTileInput::q) is handed to tests as the magnitude each field stands for —
    /// both of which hold only where a format's codes are its magnitudes. Under a minifloat the
    /// fields would be codes, so a test multiplying `q · scale` would compare the kernel against the
    /// integer grid the format exists to avoid.
    pub fn arange(self) -> QuantizedTileInput {
        assert!(
            integer_valued(self.scheme.value),
            "QuantizedTileInputBuilder::arange: {:?} is a minifloat, whose fields are codes rather \
             than magnitudes; state the codes with `pack_q_values` and fold the format's grid into \
             the test's own reference",
            self.scheme.value
        );

        let rank = self.space.rank();
        let shape: Vec<usize> = (0..rank)
            .map(|i| self.space.extent(self.space.axis_at(i)))
            .collect();

        let (lo, hi) = self.scheme.value.range();
        let (lo, hi) = (lo as i32, hi as i32);
        let span = hi - lo + 1;
        let q: Vec<i32> = (0..shape.iter().product())
            .map(|i| lo + (i as i32 % span))
            .collect();
        let words = crate::support::stubs::quant::pack_q_values(&q, &self.scheme);
        let handle = self.client.create(Bytes::from_elems(words));

        let block = crate::support::stubs::quant::block_dims(&self.scheme, &shape);
        let grid = crate::support::stubs::quant::scales_grid(&shape, &block);
        let scale_values: Vec<f32> = (0..grid.iter().product())
            .map(|g| 0.05 * (g + 1) as f32)
            .collect();
        let scales = TestInput::builder(self.client, Shape::from(grid))
            .custom(scale_values.clone())
            .generate_without_host_data();

        QuantizedTileInput {
            tile: TileInput::new(
                TensorHandle::new_contiguous(shape, handle, u32::elem_type_native()),
                self.space,
                0,
            ),
            scales,
            scheme: self.scheme,
            q,
            scale_values,
        }
    }
}

/// Whether the format's codes are the integers they stand for, which is what makes an integer
/// walk over the fields a walk over the values. Mirrors `rounds_to_integers` in the quant stub.
fn integer_valued(value: QuantValue) -> bool {
    match value {
        QuantValue::Q8F
        | QuantValue::Q8S
        | QuantValue::Q4F
        | QuantValue::Q4S
        | QuantValue::Q2F
        | QuantValue::Q2S => true,
        QuantValue::E5M2 | QuantValue::E4M3 | QuantValue::E2M1 => false,
    }
}

/// A packed-quantized input born whole: a quantized tensor is one thing (data, scales,
/// scheme), so the [`quantized builder`](QuantizedTileInputBuilder) mints the pair together,
/// plus the exact numbers behind both for host references.
pub struct QuantizedTileInput {
    pub tile: TileInput,
    scales: TensorHandle,
    scheme: QuantScheme,
    /// The quant values, row-major in the logical shape.
    pub q: Vec<i32>,
    /// One scale per block, row-major over the scheme's block grid.
    pub scale_values: Vec<f32>,
}

impl QuantizedTileInput {
    /// Binding for the innermost level's scales tensor.
    pub fn scales_binding(&self) -> TensorBinding {
        self.scales.clone().binding()
    }

    /// The field each stored value occupies: the scheme's value, or under a lookup scheme an
    /// index as wide, which names a table entry rather than a number.
    pub fn field(&self) -> Field {
        match self.scheme.mode {
            cubecl::quant::scheme::QuantMode::Lookup => Field::Index {
                bits: self.scheme.size_bits_value(),
            },
            _ => Field::Quant(self.scheme.value),
        }
    }

    /// The packed values as an operand, bound `width` words wide: `u32` words of
    /// [`field`](Self::field)s, addressed as `spec` says (declared in values), under none of their
    /// scales.
    pub fn values_arg(&self, width: usize, spec: TileSpec) -> TileArgLaunch {
        TileArgLaunch::new(
            self.tile.tensor_arg(1),
            self.tile.handle.dtype,
            width,
            spec.packed(self.field()),
        )
    }

    /// The innermost level's scales as an operand, addressed as `spec` says: one per block of
    /// the scheme's grid, row-major.
    pub fn scales_arg(&self, width: usize, spec: TileSpec) -> TileArgLaunch {
        TileArgLaunch::new(
            self.scales_binding().into_tensor_arg(),
            self.scales.dtype,
            width,
            spec,
        )
    }
}
