//! Device math library support.

use std::ffi::CStr;
use std::ffi::CString;

use llvm_sys::LLVMTypeKind;
use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMBuilderRef, LLVMModuleRef, LLVMTypeRef, LLVMValueRef};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatWidth {
    F16,
    F32,
    F64,
}

impl FloatWidth {
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

/// Target math library requirements and symbols.
pub trait MathLibrary {
    /// Whether the intrinsic requires a library implementation.
    fn needs_redirect(&self, base: &str, width: FloatWidth) -> bool;

    /// Library symbol and evaluation width, which may exceed the requested width.
    /// Returns `None` for unsupported functions.
    fn symbol(&self, base: &str, width: FloatWidth) -> Option<(String, FloatWidth)>;
}

fn intrinsic_base(name: &str) -> Option<&str> {
    let rest = name.strip_prefix("llvm.")?;
    let base = rest.split('.').next()?;
    (!base.is_empty()).then_some(base)
}

/// # Safety
/// `module` must be a live LLVM module.
pub unsafe fn redirect_intrinsics(
    module: LLVMModuleRef,
    library: &dyn MathLibrary,
) -> Result<bool, String> {
    unsafe {
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

struct Candidate {
    func: LLVMValueRef,
    /// Required function signature.
    fn_ty: LLVMTypeRef,
    base: String,
    /// Element width.
    width: FloatWidth,
    /// Lane count; one for scalars.
    lanes: u32,
    /// Operand count. All operands use the same float type.
    arity: u32,
}

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
