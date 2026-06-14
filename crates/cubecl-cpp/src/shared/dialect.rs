use std::{collections::HashSet, fmt::Debug};
use std::{fmt::Display, hash::Hash};

use cubecl_core::ir::Processor;

use crate::shared::{
    Builtin, FmtLeft, IndexedValue, MmaShape, SupportedMmaCombinations,
    SupportedScaledMmaCombinations, reduce_comparison, reduce_exclusive, reduce_inclusive,
    reduce_operator, reduce_quantifier,
    unary::{Neg, Unary},
};

use super::{
    Architecture, Body, Component, CubeIndexFlags, Elem, Flags, FragmentIdent, FragmentLayout,
    FragmentType, Instruction, Item, KernelArg, SharedMemory, Value, WarpInstruction,
    WmmaInstruction,
};

// Base dialect

pub trait Dialect:
    DialectIncludes<Self>
    + DialectTypes<Self>
    + DialectBindings<Self>
    + DialectWarpReduceCompiler<Self>
    + DialectCubeBuiltins<Self>
    + DialectInstructions<Self>
    + DialectWmmaCompiler<Self>
    + DialectProcessors<Self>
    + Default
    + Clone
    + Copy
    + Debug
    + Send
    + Sync
    + Eq
    + Hash
    + 'static
{
    type Architecture: Architecture;
}

// Includes

pub trait DialectIncludes<D: Dialect> {
    type Extension: Debug + Clone + Sync + Send;

    fn compile_includes(f: &mut std::fmt::Formatter<'_>, flags: &Flags<D>) -> std::fmt::Result;
    fn compile_extensions(
        f: &mut std::fmt::Formatter<'_>,
        extensions: &[Self::Extension],
    ) -> std::fmt::Result;
    fn register_instruction_extension(
        extensions: &mut Vec<Self::Extension>,
        instruction: &Instruction<D>,
    );
    fn register_warp_instruction_extension(
        extensions: &mut Vec<Self::Extension>,
        instruction: &WarpInstruction<D>,
    );
    #[allow(unused_variables)]
    fn register_wmma_instruction_extension(
        extensions: &mut Vec<Self::Extension>,
        instruction: &WmmaInstruction<D>,
    ) {
    }
}

// Types

pub trait DialectTypes<D: Dialect> {
    fn item_can_be_optimized() -> bool;
    fn compile_elem(
        f: &mut std::fmt::Formatter<'_>,
        elem: &Elem<D>,
        word: bool,
    ) -> std::fmt::Result;

    fn compile_item(f: &mut std::fmt::Formatter<'_>, item: &Item<D>) -> std::fmt::Result;
    fn compile_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        items: &HashSet<Item<D>>,
        scalars: &[(Elem<D>, usize)],
        info: &cubecl_core::Info,
        flags: &Flags<D>,
    ) -> std::fmt::Result;
    fn compile_local_memory_qualifier(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_shared_memory_declaration(
        f: &mut std::fmt::Formatter<'_>,
        shared: &SharedMemory<D>,
    ) -> std::fmt::Result {
        let SharedMemory { ptr, offset, .. } = shared;
        let ptr_ty = ptr.item();
        let size_bytes = shared.size();
        writeln!(f, "// Shared value size: {size_bytes} bytes")?;
        writeln!(
            f,
            "{ptr_ty} {ptr} = reinterpret_cast<{ptr_ty}>(&dynamic_shared_mem[{offset}]);"
        )
    }
    fn compile_polyfills(_f: &mut std::fmt::Formatter<'_>, _flags: &Flags<D>) -> std::fmt::Result {
        Ok(())
    }
    /// Address space (for Metal dialect only).
    fn address_space_for_value(_value: &Value<D>) -> String {
        "".to_string()
    }
}

// Kernel argument bindings

pub trait DialectBindings<D: Dialect> {
    fn compile_kernel_signature(
        f: &mut std::fmt::Formatter<'_>,
        kernel_name: &str,
        tensor_maps: &[KernelArg<D>],
        buffers: &[KernelArg<D>],
        flags: &Flags<D>,
    ) -> std::fmt::Result;
    fn compile_bindings_body(
        _f: &mut std::fmt::Formatter<'_>,
        _body: &Body<D>,
    ) -> std::fmt::Result {
        Ok(())
    }
}

// Cube builtins dialect

pub trait DialectCubeBuiltins<D: Dialect> {
    /// Depending on the dialect available built-ins the
    /// inclusion rules might change.
    /// For instance in metal we have a built-in for the Unit plane position
    /// but in other dialects there is none so we have to compute it using
    /// other built-ins.
    fn builtin_rules(flags: &CubeIndexFlags) -> CubeIndexFlags {
        let unit_pos_plane = flags.unit_pos_plane;
        let plane_dim_checked = flags.plane_dim_checked;
        let plane_dim = flags.plane_dim || plane_dim_checked || unit_pos_plane;
        let plane_pos = flags.plane_pos;
        let absolute_pos = flags.absolute_pos || unit_pos_plane;
        let absolute_pos_tuple = flags.absolute_pos_tuple || absolute_pos;
        let cube_dim = flags.cube_dim;
        let cube_dim_tuple = flags.cube_dim_tuple || cube_dim || absolute_pos || plane_dim_checked;
        let unit_pos = flags.unit_pos;
        let unit_pos_tuple = flags.unit_pos_tuple || unit_pos;
        let cube_count = flags.cube_count;
        let cube_count_tuple = flags.cube_count_tuple || absolute_pos;
        let cube_pos = flags.cube_pos;
        let cube_pos_tuple = flags.cube_pos_tuple || cube_pos;
        let cluster_group = flags.cluster_pos;

        CubeIndexFlags {
            absolute_pos,
            absolute_pos_tuple,
            cube_count,
            cube_count_tuple,
            cube_dim,
            cube_dim_tuple,
            cube_pos,
            cube_pos_tuple,
            plane_dim,
            plane_dim_checked,
            plane_pos,
            unit_pos_tuple,
            unit_pos,
            unit_pos_plane,
            cluster_pos: cluster_group,
        }
    }

    fn compile_absolute_pos_tuple_computation(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = Builtin::<D>::AbsolutePosBaseName;
        let ty = value.item();
        let cube_pos_x = Builtin::<D>::CubePosX;
        let cube_pos_y = Builtin::<D>::CubePosY;
        let cube_pos_z = Builtin::<D>::CubePosZ;
        let cube_dim_x = Builtin::<D>::CubeDimX;
        let cube_dim_y = Builtin::<D>::CubeDimY;
        let cube_dim_z = Builtin::<D>::CubeDimZ;
        let unit_pos_x = Builtin::<D>::UnitPosX;
        let unit_pos_y = Builtin::<D>::UnitPosY;
        let unit_pos_z = Builtin::<D>::UnitPosZ;
        writeln!(
            f,
            "{ty} {value} = make_{ty}(
    {cube_pos_x} * {cube_dim_x} + {unit_pos_x},
    {cube_pos_y} * {cube_dim_y} + {unit_pos_y},
    {cube_pos_z} * {cube_dim_z} + {unit_pos_z}
);"
        )
    }

    fn compile_absolute_pos_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("absoluteIdx")
    }

    fn compile_absolute_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("idxGlobal")
    }

    fn compile_absolute_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_absolute_pos_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_absolute_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_absolute_pos_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_absolute_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_absolute_pos_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_cube_count_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("gridDim")
    }

    fn compile_cube_count(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("gridDimGlobal")
    }

    fn compile_cube_count_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_count_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_cube_count_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_count_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_cube_count_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_count_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_cube_dim_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockDim")
    }

    fn compile_cube_dim(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockDimGlobal")
    }

    fn compile_cube_dim_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_dim_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_cube_dim_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_dim_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_cube_dim_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_dim_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_cube_pos_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockIdx")
    }

    fn compile_cube_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockIdxGlobal")
    }

    fn compile_cube_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_pos_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_cube_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_pos_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_cube_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_pos_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_unit_pos_computation(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = Builtin::<D>::UnitPos;
        let ty = value.item();
        let cube_dim_x = Builtin::<D>::CubeDimX;
        let cube_dim_y = Builtin::<D>::CubeDimY;
        let unit_pos_x = Builtin::<D>::UnitPosX;
        let unit_pos_y = Builtin::<D>::UnitPosY;
        let unit_pos_z = Builtin::<D>::UnitPosZ;
        writeln!(
            f,
            "{ty} {value} = {unit_pos_x} + {unit_pos_y} * {cube_dim_x} + {unit_pos_z} * ({cube_dim_x} * {cube_dim_y});"
        )
    }

    fn compile_unit_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("threadIdxGlobal")
    }

    fn compile_unit_pos_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("threadIdx")
    }

    fn compile_unit_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_unit_pos_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_unit_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_unit_pos_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_unit_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_unit_pos_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_plane_dim(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("warpSize")
    }

    fn compile_plane_dim_checked(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("warpSizeChecked")
    }

    fn compile_plane_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let unit_pos_x = Builtin::<D>::UnitPosX;
        let plane_dim = Builtin::<D>::PlaneDim;
        write!(f, "{unit_pos_x} / {plane_dim}")
    }

    fn compile_unit_pos_plane(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let absolute_pos = Builtin::<D>::AbsolutePos(Elem::U32);
        let plane_dim = Builtin::<D>::PlaneDim;
        let ty = Item::<D>::Scalar(Elem::U32);
        write!(f, "{ty}({absolute_pos}) % {plane_dim}")
    }

    fn compile_cluster_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0")
    }
    fn compile_cluster_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0")
    }
    fn compile_cluster_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0")
    }
    fn compile_cluster_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0")
    }
}

// Instructions

pub trait DialectInstructions<D: Dialect> {
    // atomics
    fn compile_atomic_add(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Value<D>,
        rhs: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let rhs = rhs.ensure_lvalue(f)?;

        let optimized = Value::optimized_args([*lhs, rhs, *out]);
        let [lhs, rhs, out_optimized] = optimized.args;

        let addr_space = D::address_space_for_value(out);
        let out_item = out.item();
        let out = out.fmt_left();

        match out_optimized.item() {
            Item::Scalar(Elem::I64) => writeln!(
                f,
                "{out} = atomicAdd(reinterpret_cast<{uint}*>({lhs}), {uint}({rhs}));",
                uint = Elem::<D>::U64
            ),
            Item::Vector(inner, vectorization) if matches!(inner.elem(), Elem::F32) => {
                let vec_ty = Item::NativeVector(*inner.elem(), vectorization);
                let out_tmp = Value::tmp(out_optimized.item());
                writeln!(
                    f,
                    "{vec_ty} {out_tmp} = atomicAdd(
                    reinterpret_cast<{addr_space}{vec_ty}*>({lhs}),
                    reinterpret_cast<const {addr_space}{vec_ty}&>({rhs}));",
                )?;
                writeln!(
                    f,
                    "{out} = reinterpret_cast<{addr_space}{out_item}&>({out_tmp});"
                )
            }
            Item::Scalar(Elem::F16x2) | Item::Scalar(Elem::BF16x2) => {
                let out_tmp = Value::tmp(out_optimized.item());
                writeln!(
                    f,
                    "{} = atomicAdd(
                    reinterpret_cast<{}>({lhs}),
                    reinterpret_cast<const {addr_space}{}&>({rhs}));",
                    out_tmp.fmt_left(),
                    lhs.item(),
                    rhs.item()
                )?;
                writeln!(
                    f,
                    "{out} = reinterpret_cast<{addr_space}{out_item}&>({out_tmp});"
                )
            }
            _ => writeln!(f, "{out} = atomicAdd({lhs}, {rhs});"),
        }
    }

    fn compile_atomic_and(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Value<D>,
        rhs: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicAnd({lhs}, {rhs});")
    }

    fn compile_atomic_cas(
        f: &mut std::fmt::Formatter<'_>,
        input: &Value<D>,
        cmp: &Value<D>,
        val: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let addr_space = D::address_space_for_value(out);
        let out_item = out.item();
        let out = out.fmt_left();

        match val.item() {
            // vec4 is automatically supported by the new 128-bit template version
            Item::Vector(inner, 2) if matches!(inner.elem(), Elem::F32) => {
                let cmp = cmp.ensure_lvalue(f)?;
                let val = val.ensure_lvalue(f)?;
                let u64 = Item::Scalar(Elem::<D>::U64);
                let out_tmp = Value::tmp(u64);
                writeln!(
                    f,
                    "{} = atomicCAS(
                reinterpret_cast<{addr_space}{u64}*>({input}),
                reinterpret_cast<{u64}&>({cmp}),
                reinterpret_cast<{u64}&>({val}));",
                    out_tmp.fmt_left()
                )?;
                writeln!(f, "{out} = reinterpret_cast<{out_item}&>({out_tmp});")
            }
            Item::Vector(inner, 2) if matches!(inner.elem(), Elem::F16 | Elem::BF16) => {
                let cmp = cmp.ensure_lvalue(f)?;
                let val = val.ensure_lvalue(f)?;
                let u32 = Item::Scalar(Elem::<D>::U32);
                let out_tmp = Value::tmp(u32);
                writeln!(
                    f,
                    "{} = atomicCAS(
                reinterpret_cast<{addr_space}{u32}*>({input}),
                reinterpret_cast<{u32}&>({cmp}),
                reinterpret_cast<{u32}&>({val}));",
                    out_tmp.fmt_left()
                )?;
                writeln!(f, "{out} = reinterpret_cast<{out_item}&>({out_tmp});")
            }
            _ => writeln!(f, "{out} = atomicCAS({input}, {cmp}, {val});"),
        }
    }

    fn compile_atomic_load(
        f: &mut std::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let out_item = out.item();
        let out = out.fmt_left();

        let Item::Pointer(_, class) = input.item() else {
            unreachable!()
        };

        let unsigned_ty = match out_item.size() {
            1 => Item::Scalar(Elem::<D>::U8),
            2 => Item::Scalar(Elem::<D>::U16),
            4 => Item::Scalar(Elem::<D>::U32),
            8 => Item::Scalar(Elem::<D>::U64),
            // Hacky, but it's CUDA only for now. We should really migrate to a more modern API in
            // general
            16 => {
                let out_tmp = Value::tmp(out_item);
                writeln!(f, "{};", out_tmp.fmt_left())?;
                writeln!(
                    f,
                    "__nv_atomic_load({input}, &{out_tmp}, __NV_ATOMIC_RELAXED);"
                )?;
                return writeln!(f, "{out} = {out_tmp};");
            }
            _ => unreachable!(),
        };
        let unsigned_ptr_ty = Item::Pointer(unsigned_ty.intern(), class);

        let ptr_tmp = Value::tmp(unsigned_ptr_ty);
        let out_tmp = Value::tmp(unsigned_ty);
        writeln!(
            f,
            "volatile {} = reinterpret_cast<volatile {unsigned_ptr_ty}>({input});",
            ptr_tmp.fmt_left()
        )?;
        writeln!(f, "{} = *{ptr_tmp};", out_tmp.fmt_left())?;
        writeln!(f, "{out} = reinterpret_cast<const {out_item}&>({out_tmp});")
    }

    fn compile_atomic_max(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Value<D>,
        rhs: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicMax({lhs}, {rhs});")
    }

    fn compile_atomic_min(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Value<D>,
        rhs: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicMin({lhs}, {rhs});")
    }

    fn compile_atomic_or(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Value<D>,
        rhs: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicOr({lhs}, {rhs});")
    }

    fn compile_atomic_store(
        f: &mut std::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let tmp = Value::tmp(input.item());
        Self::compile_atomic_swap(f, out, input, &tmp)
    }

    fn compile_atomic_sub(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Value<D>,
        rhs: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let tmp = Value::tmp(rhs.item());
        Neg::format(f, rhs, &tmp)?;
        Self::compile_atomic_add(f, lhs, &tmp, out)
    }

    fn compile_atomic_swap(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Value<D>,
        rhs: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let out_item = out.item();
        let out = out.fmt_left();

        let unsigned_elem = match rhs.item().size() {
            1 => Elem::<D>::U8,
            2 => Elem::<D>::U16,
            4 => Elem::<D>::U32,
            8 => Elem::<D>::U64,
            // 128-bit wide uses a generic template that accepts arbitrary types
            _ => return writeln!(f, "{out} = atomicExch({lhs}, {rhs});"),
        };

        let rhs = rhs.ensure_lvalue(f)?;
        let Item::Pointer(_, class) = lhs.item() else {
            unreachable!()
        };
        let unsigned_ty = Item::Scalar(unsigned_elem);
        let unsigned_ptr_ty = Item::Pointer(unsigned_ty.intern(), class);

        let out_tmp = Value::tmp(unsigned_ty);
        writeln!(
            f,
            "{} = atomicExch(
                    reinterpret_cast<{unsigned_ptr_ty}>({lhs}),
                    reinterpret_cast<const {unsigned_ty}&>({rhs}));",
            out_tmp.fmt_left()
        )?;
        writeln!(f, "{out} = reinterpret_cast<const {out_item}&>({out_tmp});")
    }

    fn compile_atomic_xor(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Value<D>,
        rhs: &Value<D>,
        out: &Value<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicXor({lhs}, {rhs});")
    }

    fn compile_saturating_add(
        f: &mut std::fmt::Formatter<'_>,
        lhs: impl Display,
        rhs: impl Display,
        item: Item<D>,
    ) -> std::fmt::Result;

    fn compile_saturating_sub(
        f: &mut std::fmt::Formatter<'_>,
        lhs: impl Display,
        rhs: impl Display,
        item: Item<D>,
    ) -> std::fmt::Result;

    // debug
    fn compile_instruction_printf(
        f: &mut std::fmt::Formatter<'_>,
        format_string: &str,
        args: &[Value<D>],
    ) -> std::fmt::Result {
        let args = args.iter().map(|arg| format!("{arg}")).collect::<Vec<_>>();
        let args = match args.is_empty() {
            true => "".to_string(),
            false => format!(", {}", args.join(",")),
        };
        writeln!(f, "printf({format_string:?}{args});")
    }

    // logs
    fn compile_instruction_log1p_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
    ) -> std::fmt::Result {
        let elem = input.elem();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                write!(f, "{elem}(log1p(float({input})))")
            }
            _ => write!(f, "log1p({input})"),
        }
    }

    // exp
    fn compile_instruction_expm1_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
    ) -> std::fmt::Result {
        let elem = input.elem();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                write!(f, "{elem}(expm1(float({input})))")
            }
            _ => write!(f, "expm1({input})"),
        }
    }

    // sync
    fn compile_instruction_sync_threads(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_instruction_sync_warp(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_instruction_thread_fence(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    // trigo
    fn compile_instruction_tanh_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
    ) -> std::fmt::Result {
        let elem = input.elem();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                write!(f, "{elem}(tanh(float({input})))")
            }
            _ => write!(f, "tanh({input})"),
        }
    }

    // unary
    fn compile_instruction_find_first_set<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<D>,
    ) -> std::fmt::Result;
    fn compile_instruction_leading_zeros_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<D>,
    ) -> std::fmt::Result;

    fn compile_instruction_trailing_zeros_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<D>,
    ) -> std::fmt::Result;

    fn compile_instruction_popcount_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<D>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(")?;
        match input.elem() {
            Elem::I32 => write!(f, "__popc({}({input}))", Elem::<D>::U32),
            Elem::U32 => write!(f, "__popc({input})"),
            Elem::I64 => write!(f, "__popcll({}({input}))", Elem::<D>::U64),
            Elem::U64 => write!(f, "__popcll({input})"),
            _ => write!(f, "__popc({})", super::unary::zero_extend(input)),
        }?;
        write!(f, ")")
    }

    fn compile_instruction_reverse_bits_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<D>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(")?;
        match out_elem {
            Elem::I32 => write!(f, "__brev({}({input}))", Elem::<D>::U32),
            Elem::U32 => write!(f, "__brev({input})"),
            Elem::I64 => write!(f, "__brevll({}({input}))", Elem::<D>::U64),
            Elem::U64 => write!(f, "__brevll({input})"),
            _ => write!(
                f,
                "__brev({}) >> {}",
                super::unary::zero_extend(input),
                (size_of::<u32>() - out_elem.size()) * 8
            ),
        }?;
        write!(f, ")")
    }

    // others
    fn compile_instruction_max_function_name(
        f: &mut std::fmt::Formatter<'_>,
        item: Item<D>,
    ) -> std::fmt::Result;

    fn compile_instruction_min_function_name(
        f: &mut std::fmt::Formatter<'_>,
        item: Item<D>,
    ) -> std::fmt::Result;

    fn compile_instruction_powf(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &str,
        rhs: &str,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        match elem {
            Elem::F32 => write!(f, "powf({lhs}, {rhs})"),
            Elem::F64 => write!(f, "pow({lhs}, {rhs})"),
            _ => write!(f, "#error Unsupported type for powf: {elem}"),
        }
    }

    fn compile_instruction_hypot(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &str,
        rhs: &str,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        match elem {
            Elem::F32 => write!(f, "hypotf({lhs}, {rhs})"),
            Elem::F64 => write!(f, "hypot({lhs}, {rhs})"),
            _ => write!(f, "#error Unsupported type for hypot: {elem}"),
        }
    }

    fn compile_instruction_rhypot(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &str,
        rhs: &str,
        elem: Elem<D>,
    ) -> std::fmt::Result {
        match elem {
            Elem::F32 => write!(f, "rhypotf({lhs}, {rhs})"),
            Elem::F64 => write!(f, "rhypot({lhs}, {rhs})"),
            _ => write!(f, "#error Unsupported type for rhypot: {elem}"),
        }
    }

    fn compile_instruction_half_function_name_prefix() -> &'static str {
        "h"
    }

    fn compile_instruction_half2_function_name_prefix() -> &'static str {
        "h2"
    }

    /// Remaps a math-function name for the dialect (default: unchanged), e.g. fast-math
    /// intrinsics to each dialect's spelling (`__expf` -> `fast::exp`).
    fn compile_fast_math_function_name(name: &'static str) -> &'static str {
        name
    }

    // warp
    fn compile_warp_shuffle(
        f: &mut std::fmt::Formatter<'_>,
        val: &str,
        source: &str,
    ) -> std::fmt::Result;
    fn compile_warp_shuffle_xor(
        f: &mut std::fmt::Formatter<'_>,
        val: &str,
        elem: &Elem<D>,
        offset: &str,
    ) -> std::fmt::Result;
    fn compile_warp_shuffle_up(
        f: &mut std::fmt::Formatter<'_>,
        val: &str,
        offset: &str,
    ) -> std::fmt::Result;
    fn compile_warp_shuffle_down(
        f: &mut std::fmt::Formatter<'_>,
        val: &str,
        offset: &str,
    ) -> std::fmt::Result;
    fn compile_warp_all<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result;
    fn compile_warp_any<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result;
    fn compile_warp_ballot(
        f: &mut std::fmt::Formatter<'_>,
        input: &Value<D>,
        out_elem: &Elem<D>,
    ) -> std::fmt::Result;
    fn compile_warp_elect(f: &mut std::fmt::Formatter<'_>, out: &str) -> std::fmt::Result {
        write!(
            f,
            "
unsigned int mask = __activemask();
unsigned int leader = __ffs(mask) - 1;
{out} = threadIdx.x % warpSize == leader;
            "
        )
    }
    fn compile_unreachable(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

#[derive(Debug, Clone, Copy, new)]
pub struct ManualMma<'a, D: Dialect> {
    pub shape: MmaShape<D>,
    pub frag_a: &'a Value<D>,
    pub frag_b: &'a Value<D>,
    pub frag_c: &'a Value<D>,
    pub frag_d: &'a Value<D>,
}

pub trait DialectWarpReduceCompiler<D: Dialect>:
    Default + Clone + Copy + Debug + Send + Sync + Eq + Hash + 'static
{
    fn warp_reduce_sum(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_operator(f, input, out, "+=")
    }
    fn warp_reduce_prod(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_operator(f, input, out, "*=")
    }
    fn warp_reduce_max(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_comparison(f, input, out, D::compile_instruction_max_function_name)
    }
    fn warp_reduce_min(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_comparison(f, input, out, D::compile_instruction_min_function_name)
    }
    fn warp_reduce_all(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_quantifier(f, input, out, D::compile_warp_all::<IndexedValue<D>>)
    }
    fn warp_reduce_any(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_quantifier(f, input, out, D::compile_warp_any::<IndexedValue<D>>)
    }
    fn warp_reduce_sum_inclusive(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_inclusive(f, input, out, "+=")
    }
    fn warp_reduce_prod_inclusive(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_inclusive(f, input, out, "*=")
    }
    fn warp_reduce_sum_exclusive(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_exclusive(f, input, out, "+=", "0")
    }
    fn warp_reduce_prod_exclusive(
        f: &mut core::fmt::Formatter<'_>,
        input: &Value<D>,
        out: &Value<D>,
    ) -> core::fmt::Result {
        reduce_exclusive(f, input, out, "*=", "1")
    }
}

pub trait DialectWmmaCompiler<D: Dialect>:
    Default + Clone + Copy + Debug + Send + Sync + Eq + Hash + 'static
{
    #[allow(unused_variables)]
    fn compile_wmma_includes(
        f: &mut std::fmt::Formatter<'_>,
        flags: &Flags<D>,
    ) -> std::fmt::Result {
        Ok(())
    }
    #[allow(unused_variables)]
    fn compile_wmma_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        flags: &Flags<D>,
    ) -> std::fmt::Result {
        Ok(())
    }
    #[allow(unused_variables)]
    fn compile_wmma_local_variables(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
    #[allow(unused_variables)]
    fn compile_wwma_fragment_ident(
        f: &mut std::fmt::Formatter<'_>,
        ident: &FragmentIdent<D>,
    ) -> std::fmt::Result {
        Ok(())
    }
    #[allow(unused_variables)]
    fn compile_wmma_fragment_layout(
        f: &mut std::fmt::Formatter<'_>,
        layout: &FragmentLayout<D>,
    ) -> std::fmt::Result {
        Ok(())
    }
    #[allow(unused_variables)]
    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &FragmentType<D>,
    ) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        val: &Value<D>,
        value_ty: &Item<D>,
    ) -> std::fmt::Result;

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<D>,
    ) -> std::fmt::Result;
    fn compile_manual_mma(f: &mut std::fmt::Formatter<'_>, mma: ManualMma<D>) -> std::fmt::Result;
    fn compile_scaled_mma(
        f: &mut std::fmt::Formatter<'_>,
        mma: ManualMma<D>,
        scales_a: Value<D>,
        scales_b: Value<D>,
        scales_factor: u32,
    ) -> std::fmt::Result;
    fn supported_wmma_combinations(arch: &D::Architecture) -> SupportedMmaCombinations;
    fn supported_mma_combinations(arch: &D::Architecture) -> SupportedMmaCombinations;
    fn supported_scaled_mma_combinations(
        _arch: &D::Architecture,
    ) -> SupportedScaledMmaCombinations {
        Vec::new()
    }
}

/// IR Processors to be applied to the scopes during processing. ``CheckedIO`` is always applied
/// by default, so these are only for target specific processors like MMA index processors.
pub trait DialectProcessors<D: Dialect> {
    fn processors() -> Vec<Box<dyn Processor>>;
}
