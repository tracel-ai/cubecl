//! Rewriting the float intrinsics the AMDGPU backend cannot lower into OCML calls.

use std::ffi::CStr;
use std::ffi::CString;

use llvm_sys::LLVMTypeKind;
use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMModuleRef, LLVMTypeRef, LLVMValueRef};

/// Intrinsics the backend cannot give a correct answer for at any width.
const NEVER_CORRECT: [&str; 9] = [
    "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "atan2", "pow",
];

/// Intrinsics the hardware has in single precision but not in double, where the whole
/// transcendental unit is missing.
const SINGLE_PRECISION_ONLY: [&str; 8] =
    ["exp", "exp2", "exp10", "log", "log2", "log10", "sin", "cos"];

/// The OCML suffix for a float type, and the name LLVM mangles it to.
fn float_suffix(kind: LLVMTypeKind) -> Option<&'static str> {
    match kind {
        LLVMTypeKind::LLVMHalfTypeKind => Some("f16"),
        LLVMTypeKind::LLVMFloatTypeKind => Some("f32"),
        LLVMTypeKind::LLVMDoubleTypeKind => Some("f64"),
        _ => None,
    }
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

/// Whether an intrinsic of `base` over `suffix` needs the library: either nothing at all is
/// behind it, or what is behind it gives the wrong answer.
fn needs_ocml(base: &str, suffix: &str) -> bool {
    NEVER_CORRECT.contains(&base) || (suffix == "f64" && SINGLE_PRECISION_ONLY.contains(&base))
}

/// Points every call of an unsupported float intrinsic in `module` at OCML.
///
/// Returns whether anything was redirected, i.e. whether the device libraries have to be
/// linked in behind it.
///
/// # Safety
/// `module` must be a live LLVM module.
pub unsafe fn redirect_intrinsics_to_ocml(module: LLVMModuleRef) -> Result<bool, String> {
    unsafe {
        // Collected first: the loop below adds functions to the module being walked.
        let mut candidates = Vec::new();
        let mut func = LLVMGetFirstFunction(module);
        while !func.is_null() {
            if let Some(candidate) = classify(func) {
                candidates.push(candidate);
            }
            func = LLVMGetNextFunction(func);
        }

        let redirected = !candidates.is_empty();
        for candidate in candidates {
            let replacement = ocml_replacement(module, &candidate)?;
            LLVMReplaceAllUsesWith(candidate.func, replacement);
            LLVMDeleteFunction(candidate.func);
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
    /// `f16`, `f32` or `f64`, the OCML suffix of the element type.
    suffix: &'static str,
    /// Number of lanes, `1` for a scalar.
    lanes: u32,
    /// How many operands the intrinsic takes, all of them the same float type.
    arity: u32,
}

/// Whether `func` is an unsupported float intrinsic, and what it takes.
///
/// # Safety
/// `func` must be a live LLVM function.
unsafe fn classify(func: LLVMValueRef) -> Option<Candidate> {
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
        let suffix = float_suffix(LLVMGetTypeKind(elem_ty))?;
        if !needs_ocml(&base, suffix) {
            return None;
        }

        // Every base handled here is `float(float)` or `float(float, float)`. A parameter
        // that is not the return type — `llvm.powi`'s exponent, say — has no OCML mapping
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
            suffix,
            lanes,
            arity,
        })
    }
}

/// The function to call in place of `candidate`: OCML itself for a scalar, a generated
/// per-lane wrapper around it for a vector.
///
/// # Safety
/// `module` must be a live LLVM module and `candidate` describe one of its declarations.
unsafe fn ocml_replacement(
    module: LLVMModuleRef,
    candidate: &Candidate,
) -> Result<LLVMValueRef, String> {
    unsafe {
        let Candidate {
            fn_ty,
            base,
            suffix,
            lanes,
            arity,
            ..
        } = candidate;

        let scalar_ty = match LLVMGetTypeKind(LLVMGetReturnType(*fn_ty)) {
            LLVMTypeKind::LLVMVectorTypeKind => LLVMGetElementType(LLVMGetReturnType(*fn_ty)),
            _ => LLVMGetReturnType(*fn_ty),
        };
        let mut scalar_params = vec![scalar_ty; *arity as usize];
        let scalar_fn_ty = LLVMFunctionType(scalar_ty, scalar_params.as_mut_ptr(), *arity, 0);

        let ocml = declare(module, &format!("__ocml_{base}_{suffix}"), scalar_fn_ty)?;
        if *lanes == 1 {
            return Ok(ocml);
        }

        // One wrapper per (intrinsic, vector type), which is what the mangled intrinsic name
        // already distinguishes, so reuse it as the key.
        let mut len = 0;
        let intrinsic_name = CStr::from_ptr(LLVMGetValueName2(candidate.func, &mut len))
            .to_string_lossy()
            .replace('.', "_");
        let wrapper_name = format!("__cubecl_{intrinsic_name}");
        let wrapper = declare(module, &wrapper_name, *fn_ty)?;
        if LLVMGetFirstBasicBlock(wrapper).is_null() {
            build_lanewise(module, wrapper, ocml, scalar_fn_ty, *fn_ty, *lanes, *arity);
        }
        Ok(wrapper)
    }
}

/// Gives `wrapper` a body calling `ocml` once per lane of its arguments.
///
/// # Safety
/// All handles must be live, and `wrapper` must have no body yet.
unsafe fn build_lanewise(
    module: LLVMModuleRef,
    wrapper: LLVMValueRef,
    ocml: LLVMValueRef,
    scalar_fn_ty: LLVMTypeRef,
    fn_ty: LLVMTypeRef,
    lanes: u32,
    arity: u32,
) {
    unsafe {
        // Internal so the pipeline is free to inline it and drop it; `alwaysinline` because
        // a lane-wise call has no reason to survive as a call.
        LLVMSetLinkage(wrapper, llvm_sys::LLVMLinkage::LLVMInternalLinkage);
        add_enum_attribute(wrapper, "alwaysinline");

        let ctx = LLVMGetModuleContext(module);
        let block = LLVMAppendBasicBlockInContext(ctx, wrapper, c"entry".as_ptr());
        let builder = LLVMCreateBuilderInContext(ctx);
        LLVMPositionBuilderAtEnd(builder, block);

        let vec_ty = LLVMGetReturnType(fn_ty);
        let index_ty = LLVMInt32TypeInContext(ctx);
        let mut result = LLVMGetPoison(vec_ty);

        for lane in 0..lanes {
            let index = LLVMConstInt(index_ty, lane as u64, 0);
            let mut args: Vec<LLVMValueRef> = (0..arity)
                .map(|arg| {
                    LLVMBuildExtractElement(
                        builder,
                        LLVMGetParam(wrapper, arg),
                        index,
                        c"".as_ptr(),
                    )
                })
                .collect();
            let call = LLVMBuildCall2(
                builder,
                scalar_fn_ty,
                ocml,
                args.as_mut_ptr(),
                arity,
                c"".as_ptr(),
            );
            result = LLVMBuildInsertElement(builder, result, call, index, c"".as_ptr());
        }

        LLVMBuildRet(builder, result);
        LLVMDisposeBuilder(builder);
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
