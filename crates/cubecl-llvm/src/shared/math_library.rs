//! Rewriting the float intrinsics a GPU backend cannot lower into calls to its vendor's math
//! library.
//!
//! Neither GPU has a libm behind `llvm.atan2` and friends: an intrinsic the backend cannot
//! select is not a slow kernel but a failed compilation, and on NVPTX one that aborts the
//! process. What each vendor ships instead is a bitcode library — `ROCm`'s OCML,
//! CUDA's `libdevice` — holding the same functions under its own names. The walk that finds
//! the intrinsics, the lane-wise wrappers for vectors and the declaration bookkeeping are the
//! same either way, so they are here; the two things that differ are behind [`MathLibrary`].

use std::ffi::CStr;
use std::ffi::CString;

use llvm_sys::LLVMTypeKind;
use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMBuilderRef, LLVMModuleRef, LLVMTypeRef, LLVMValueRef};

/// A float width, which is what a math library indexes its entry points by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatWidth {
    F16,
    F32,
    F64,
}

impl FloatWidth {
    /// The suffix LLVM mangles this width to, which is also OCML's.
    pub fn suffix(self) -> &'static str {
        match self {
            FloatWidth::F16 => "f16",
            FloatWidth::F32 => "f32",
            FloatWidth::F64 => "f64",
        }
    }

    fn of(kind: LLVMTypeKind) -> Option<Self> {
        match kind {
            LLVMTypeKind::LLVMHalfTypeKind => Some(FloatWidth::F16),
            LLVMTypeKind::LLVMFloatTypeKind => Some(FloatWidth::F32),
            LLVMTypeKind::LLVMDoubleTypeKind => Some(FloatWidth::F64),
            _ => None,
        }
    }

    /// # Safety
    /// `ctx` must be a live LLVM context.
    unsafe fn llvm_ty(self, ctx: llvm_sys::prelude::LLVMContextRef) -> LLVMTypeRef {
        unsafe {
            match self {
                FloatWidth::F16 => LLVMHalfTypeInContext(ctx),
                FloatWidth::F32 => LLVMFloatTypeInContext(ctx),
                FloatWidth::F64 => LLVMDoubleTypeInContext(ctx),
            }
        }
    }
}

/// One vendor's math library: which intrinsics it has to stand in for, and what it calls them.
pub trait MathLibrary {
    /// Whether an intrinsic of `base` over `width` has to be redirected: either nothing at all
    /// is behind it, or what is behind it gives the wrong answer.
    fn needs_redirect(&self, base: &str, width: FloatWidth) -> bool;

    /// The symbol implementing `base`, and the width it is implemented at.
    ///
    /// That width may be wider than the one asked for: `libdevice` has no half entry points at
    /// all, so a half intrinsic is answered at single precision with the conversions around it,
    /// which is what CUDA's own half math does. `None` for a base the library does not have,
    /// which leaves the intrinsic to the backend.
    fn symbol(&self, base: &str, width: FloatWidth) -> Option<(String, FloatWidth)>;
}

/// The intrinsic base name of `llvm.<base>.<mangled type>`, e.g. `sinh` of `llvm.sinh.v4f32`.
///
/// Overloaded intrinsics mangle one suffix per overloaded parameter, and the bases here take
/// only floats of a single type, so everything past the first suffix is type mangling too.
fn intrinsic_base(name: &str) -> Option<&str> {
    let rest = name.strip_prefix("llvm.")?;
    let base = rest.split('.').next()?;
    (!base.is_empty()).then_some(base)
}

/// Points every call of an unsupported float intrinsic in `module` at `library`.
///
/// Returns whether anything was redirected, i.e. whether the library has to be linked in
/// behind it.
///
/// # Safety
/// `module` must be a live LLVM module.
pub unsafe fn redirect_intrinsics(
    module: LLVMModuleRef,
    library: &dyn MathLibrary,
) -> Result<bool, String> {
    unsafe {
        // Collected first: the loop below adds functions to the module being walked.
        let mut candidates = Vec::new();
        let mut func = LLVMGetFirstFunction(module);
        while !func.is_null() {
            if let Some(candidate) = classify(func, library) {
                candidates.push(candidate);
            }
            func = LLVMGetNextFunction(func);
        }

        let mut redirected = false;
        for candidate in candidates {
            let Some(replacement) = replacement(module, &candidate, library)? else {
                continue;
            };
            LLVMReplaceAllUsesWith(candidate.func, replacement);
            LLVMDeleteFunction(candidate.func);
            redirected = true;
        }
        Ok(redirected)
    }
}

/// An intrinsic declaration that has to become an OCML call.
struct Candidate {
    func: LLVMValueRef,
    /// Type of the intrinsic itself, which the replacement must match exactly.
    fn_ty: LLVMTypeRef,
    /// `sinh`, `atan2`, ...
    base: String,
    /// Width of the element type.
    width: FloatWidth,
    /// Number of lanes, `1` for a scalar.
    lanes: u32,
    /// How many operands the intrinsic takes, all of them the same float type.
    arity: u32,
}

/// Whether `func` is an unsupported float intrinsic, and what it takes.
///
/// # Safety
/// `func` must be a live LLVM function.
unsafe fn classify(func: LLVMValueRef, library: &dyn MathLibrary) -> Option<Candidate> {
    unsafe {
        if !LLVMGetFirstBasicBlock(func).is_null() {
            return None;
        }

        let mut len = 0;
        let name = CStr::from_ptr(LLVMGetValueName2(func, &mut len))
            .to_str()
            .ok()?;
        let base = intrinsic_base(name)?.to_string();

        let fn_ty = LLVMGlobalGetValueType(func);
        let ret_ty = LLVMGetReturnType(fn_ty);
        let (elem_ty, lanes) = match LLVMGetTypeKind(ret_ty) {
            LLVMTypeKind::LLVMVectorTypeKind => {
                (LLVMGetElementType(ret_ty), LLVMGetVectorSize(ret_ty))
            }
            _ => (ret_ty, 1),
        };
        let width = FloatWidth::of(LLVMGetTypeKind(elem_ty))?;
        if !library.needs_redirect(&base, width) {
            return None;
        }

        // Every base handled here is `float(float)` or `float(float, float)`. A parameter
        // that is not the return type — `llvm.powi`'s exponent, say — has no library mapping
        // of this shape, so leave the intrinsic to the backend rather than guess.
        let arity = LLVMCountParamTypes(fn_ty);
        let mut params = vec![std::ptr::null_mut(); arity as usize];
        LLVMGetParamTypes(fn_ty, params.as_mut_ptr());
        if !(1..=2).contains(&arity) || params.iter().any(|&p| p != ret_ty) {
            return None;
        }

        Some(Candidate {
            func,
            fn_ty,
            base,
            width,
            lanes,
            arity,
        })
    }
}

/// The function to call in place of `candidate`: the library symbol itself when it already has
/// exactly the right type, and a generated wrapper around it otherwise.
///
/// A wrapper is needed for two reasons, which compose: the intrinsic is over a vector and the
/// library is scalar, or the library has no entry point at the intrinsic's width and answers at
/// a wider one. `None` when the library has nothing for this base, which leaves the intrinsic
/// alone.
///
/// # Safety
/// `module` must be a live LLVM module and `candidate` describe one of its declarations.
unsafe fn replacement(
    module: LLVMModuleRef,
    candidate: &Candidate,
    library: &dyn MathLibrary,
) -> Result<Option<LLVMValueRef>, String> {
    unsafe {
        let Candidate {
            fn_ty,
            base,
            width,
            lanes,
            arity,
            ..
        } = candidate;

        let Some((symbol, call_width)) = library.symbol(base, *width) else {
            return Ok(None);
        };

        let ctx = LLVMGetModuleContext(module);
        let call_ty = call_width.llvm_ty(ctx);
        let mut call_params = vec![call_ty; *arity as usize];
        let call_fn_ty = LLVMFunctionType(call_ty, call_params.as_mut_ptr(), *arity, 0);

        let library_func = declare(module, &symbol, call_fn_ty)?;
        if *lanes == 1 && call_width == *width {
            return Ok(Some(library_func));
        }

        // One wrapper per (intrinsic, type), which is what the mangled intrinsic name already
        // distinguishes, so reuse it as the key.
        let mut len = 0;
        let intrinsic_name = CStr::from_ptr(LLVMGetValueName2(candidate.func, &mut len))
            .to_string_lossy()
            .replace('.', "_");
        let wrapper_name = format!("__cubecl_{intrinsic_name}");
        let wrapper = declare(module, &wrapper_name, *fn_ty)?;
        if LLVMGetFirstBasicBlock(wrapper).is_null() {
            build_wrapper(
                module,
                wrapper,
                library_func,
                call_fn_ty,
                call_ty,
                *fn_ty,
                *lanes,
                *arity,
            );
        }
        Ok(Some(wrapper))
    }
}

/// Gives `wrapper` a body calling `library_func` once per lane of its arguments, converting
/// each lane to the width the library implements and the answer back.
///
/// # Safety
/// All handles must be live, and `wrapper` must have no body yet.
#[allow(clippy::too_many_arguments)]
unsafe fn build_wrapper(
    module: LLVMModuleRef,
    wrapper: LLVMValueRef,
    library_func: LLVMValueRef,
    call_fn_ty: LLVMTypeRef,
    call_ty: LLVMTypeRef,
    fn_ty: LLVMTypeRef,
    lanes: u32,
    arity: u32,
) {
    unsafe {
        // Internal so the pipeline is free to inline it and drop it; `alwaysinline` because a
        // lane-wise call has no reason to survive as a call.
        LLVMSetLinkage(wrapper, llvm_sys::LLVMLinkage::LLVMInternalLinkage);
        add_enum_attribute(wrapper, "alwaysinline");

        let ctx = LLVMGetModuleContext(module);
        let block = LLVMAppendBasicBlockInContext(ctx, wrapper, c"entry".as_ptr());
        let builder = LLVMCreateBuilderInContext(ctx);
        LLVMPositionBuilderAtEnd(builder, block);

        let result_ty = LLVMGetReturnType(fn_ty);
        let elem_ty = if lanes == 1 {
            result_ty
        } else {
            LLVMGetElementType(result_ty)
        };
        let index_ty = LLVMInt32TypeInContext(ctx);

        let lane_result = |builder: LLVMBuilderRef, index: Option<LLVMValueRef>| {
            let mut args: Vec<LLVMValueRef> = (0..arity)
                .map(|arg| {
                    let param = LLVMGetParam(wrapper, arg);
                    let lane = match index {
                        Some(index) => LLVMBuildExtractElement(builder, param, index, c"".as_ptr()),
                        None => param,
                    };
                    convert(builder, lane, call_ty)
                })
                .collect();
            let call = LLVMBuildCall2(
                builder,
                call_fn_ty,
                library_func,
                args.as_mut_ptr(),
                arity,
                c"".as_ptr(),
            );
            convert(builder, call, elem_ty)
        };

        if lanes == 1 {
            let value = lane_result(builder, None);
            LLVMBuildRet(builder, value);
        } else {
            let mut result = LLVMGetPoison(result_ty);
            for lane in 0..lanes {
                let index = LLVMConstInt(index_ty, lane as u64, 0);
                let value = lane_result(builder, Some(index));
                result = LLVMBuildInsertElement(builder, result, value, index, c"".as_ptr());
            }
            LLVMBuildRet(builder, result);
        }

        LLVMDisposeBuilder(builder);
    }
}

/// `value` as `to`, widening or narrowing it when the two differ.
///
/// Only ever between float widths, so an extension is exact and the narrowing back is the one
/// rounding the round trip costs -- the same one CUDA's own half math pays for going through
/// single precision.
///
/// # Safety
/// All handles must be live and positioned.
unsafe fn convert(builder: LLVMBuilderRef, value: LLVMValueRef, to: LLVMTypeRef) -> LLVMValueRef {
    unsafe {
        let from = LLVMTypeOf(value);
        if from == to {
            return value;
        }
        let widths = |ty: LLVMTypeRef| match LLVMGetTypeKind(ty) {
            LLVMTypeKind::LLVMHalfTypeKind => 16,
            LLVMTypeKind::LLVMFloatTypeKind => 32,
            _ => 64,
        };
        if widths(from) < widths(to) {
            LLVMBuildFPExt(builder, value, to, c"".as_ptr())
        } else {
            LLVMBuildFPTrunc(builder, value, to, c"".as_ptr())
        }
    }
}

/// The function `name` in `module`, declared with `fn_ty` if it is not there yet.
///
/// # Safety
/// `module` must be a live LLVM module.
unsafe fn declare(
    module: LLVMModuleRef,
    name: &str,
    fn_ty: LLVMTypeRef,
) -> Result<LLVMValueRef, String> {
    unsafe {
        let c_name = CString::new(name).map_err(|_| format!("name '{name}' contains a NUL"))?;
        let existing = LLVMGetNamedFunction(module, c_name.as_ptr());
        if !existing.is_null() {
            return Ok(existing);
        }
        Ok(LLVMAddFunction(module, c_name.as_ptr(), fn_ty))
    }
}

/// Adds the valueless attribute `name` to `func`.
///
/// # Safety
/// `func` must be a live LLVM function.
unsafe fn add_enum_attribute(func: LLVMValueRef, name: &str) {
    unsafe {
        let ctx = LLVMGetTypeContext(LLVMTypeOf(func));
        let kind = LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
        let attribute = LLVMCreateEnumAttribute(ctx, kind, 0);
        LLVMAddAttributeAtIndex(func, llvm_sys::LLVMAttributeFunctionIndex, attribute);
    }
}
