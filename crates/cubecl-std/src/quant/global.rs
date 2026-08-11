use cubecl::prelude::*;
use cubecl_common::quant::scheme::{QuantLevel, QuantParam};
use cubecl_core as cubecl;

/// The per-tensor scale of a two-level scheme, absent for a one-level scheme.
///
/// The scale crosses each stage as one of these rather than as an option threaded by hand:
/// [`GlobalScaleArg`] at launch, [`GlobalScaleCompilationArg`] at registration, and this cube type
/// in the kernel, holding the value already read from its binding.
#[derive(CubeType, Clone)]
pub struct GlobalScale {
    value: ComptimeOption<f32>,
}

#[cube]
impl GlobalScale {
    /// The empty scale of a one-level scheme.
    pub fn none() -> GlobalScale {
        GlobalScale {
            value: ComptimeOption::new_None(),
        }
    }

    /// Wrap a per-tensor scale already read into the kernel.
    pub fn some(value: f32) -> GlobalScale {
        GlobalScale {
            value: ComptimeOption::new_Some(value),
        }
    }

    /// The scale a read multiplies its values by, as the served vector type.
    ///
    /// The two levels multiply in f32: a block scale is normalized against the per-tensor scale, so
    /// on its own it overflows a narrow `F` by orders of magnitude before the per-tensor scale can
    /// bring the product back into a range `F` holds.
    pub fn effective<S: CubePrimitive, F: Numeric, NF: Size>(&self, scale: S) -> Vector<F, NF> {
        if comptime!(self.value.is_some()) {
            let global = self.value.unwrap();
            Vector::<F, NF>::cast_from(global * f32::cast_from(scale))
        } else {
            Vector::<F, NF>::cast_from(scale)
        }
    }
}

impl GlobalScale {
    /// Panic when this scale and the level disagree. See [`GlobalScaleArg::validate`].
    pub fn validate(&self, level: QuantLevel) {
        check(level, self.value.is_some());
    }
}

impl Clone for GlobalScaleExpand {
    fn clone(&self) -> Self {
        Self { value: self.value }
    }
}

impl GlobalScaleExpand {
    /// The empty scale of a one-level scheme.
    pub fn none() -> Self {
        Self {
            value: ComptimeOptionExpand::None,
        }
    }

    /// Wrap a per-tensor scale already read into the kernel.
    pub fn some(value: NativeExpand<f32>) -> Self {
        Self {
            value: ComptimeOptionExpand::Some(value),
        }
    }

    /// Panic when this scale and the level disagree. See [`GlobalScaleArg::validate`].
    pub fn validate(&self, level: QuantLevel) {
        check(level, self.value.is_some());
    }
}

/// Launch-side per-tensor scale: the buffer it binds from, absent for a one-level scheme.
pub struct GlobalScaleArg<R: Runtime> {
    buffer: Option<BufferArg<R>>,
}

impl<R: Runtime> GlobalScaleArg<R> {
    /// The empty scale of a one-level scheme.
    pub fn none() -> Self {
        Self { buffer: None }
    }

    /// The buffer holding a two-level scheme's per-tensor scale in its first element.
    pub fn new(buffer: BufferArg<R>) -> Self {
        Self {
            buffer: Some(buffer),
        }
    }

    /// Panic when this scale and the level disagree.
    ///
    /// The per-tensor scale binds as a buffer of its own, so nothing ties it to the level: a
    /// missing one is dropped from the reconstruction and every value comes back short by that
    /// factor, an extra one is a caller quantizing differently than the scheme it passed.
    ///
    /// The binding is f32, and a level storing the scale in anything else is rejected rather than
    /// read as f32 bytes. There is one per-tensor scale for a whole tensor, so a narrower type
    /// saves nothing and only reintroduces rounding error.
    pub fn validate(&self, level: QuantLevel) {
        check(level, self.buffer.is_some());
    }

    /// Registered as f32 to match the element type [`GlobalScaleCompilationArg::expand`] reads it
    /// back with.
    pub fn register(self, launcher: &mut KernelLauncher<R>) -> GlobalScaleCompilationArg {
        GlobalScaleCompilationArg {
            buffer: self
                .buffer
                .map(|buffer| <[f32] as LaunchArg>::register(buffer, launcher)),
        }
    }
}

/// The per-tensor scale between registration and expansion.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct GlobalScaleCompilationArg {
    buffer: Option<BufferCompilationArg>,
}

impl GlobalScaleCompilationArg {
    /// Panic when this scale and the level disagree. See [`GlobalScaleArg::validate`].
    pub fn validate(&self, level: QuantLevel) {
        check(level, self.buffer.is_some());
    }

    /// Read the per-tensor scale into the scope the view is built in.
    ///
    /// One read for the whole kernel: the scale is a single value for the entire tensor, and a
    /// read per element would be a global load the optimizer cannot hoist back out of a loop.
    /// Reading it as f32 is what keeps the two levels multiplying in f32 later, since a block
    /// scale alone can overflow a narrow compute type.
    pub fn expand(&self, builder: &mut KernelBuilder) -> GlobalScaleExpand {
        match &self.buffer {
            Some(global) => {
                let buffer = <[f32] as LaunchArg>::expand(global, builder);
                let pos = NativeExpand::<usize>::from_lit(&builder.scope, 0);
                GlobalScaleExpand::some(*buffer.__expand_index_method(&builder.scope, pos))
            }
            None => GlobalScaleExpand::none(),
        }
    }
}

impl LaunchArg for GlobalScale {
    type RuntimeArg<R: Runtime> = GlobalScaleArg<R>;
    type CompilationArg = GlobalScaleCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        arg.register(launcher)
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> GlobalScaleExpand {
        arg.expand(builder)
    }
}

/// The one copy of the binding contract every stage's `validate` goes through.
fn check(level: QuantLevel, global_provided: bool) {
    match (level.global_param(), global_provided) {
        (None, false) | (Some(QuantParam::F32), true) => {}
        (Some(_), false) => {
            panic!("{level:?} takes a per-tensor scale, but no global was provided")
        }
        (None, true) => {
            panic!("global was provided, but {level:?} does not take a per-tensor scale")
        }
        (Some(param), true) => {
            panic!("the per-tensor scale binds as f32, but {level:?} stores it as {param:?}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::check;
    use cubecl_common::quant::scheme::{QuantLevel, QuantParam};

    #[test]
    fn a_one_level_scheme_takes_no_global() {
        check(QuantLevel::Tensor, false);
        check(QuantLevel::block([32]), false);
    }

    #[test]
    fn a_two_level_scheme_takes_an_f32_global() {
        check(QuantLevel::block_tensor([32], QuantParam::F32), true);
    }

    /// The binding is f32, so a level naming another param would have its scale read as f32 bytes.
    #[test]
    #[should_panic(expected = "binds as f32, but")]
    fn a_two_level_scheme_storing_the_global_narrower_is_rejected() {
        check(QuantLevel::block_tensor([32], QuantParam::BF16), true);
    }

    #[test]
    #[should_panic(expected = "takes a per-tensor scale, but no global was provided")]
    fn a_two_level_scheme_without_a_global_is_rejected() {
        // Would otherwise dequantize against the block scales alone, dropping the per-tensor factor.
        check(QuantLevel::block_tensor([32], QuantParam::F32), false);
    }

    #[test]
    #[should_panic(expected = "does not take a per-tensor scale")]
    fn a_one_level_scheme_with_a_global_is_rejected() {
        check(QuantLevel::Tensor, true);
    }
}
