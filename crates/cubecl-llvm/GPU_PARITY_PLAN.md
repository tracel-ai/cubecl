# LLVM GPU Parity Plan

Branch-scoped plan for the NVPTX and AMDGPU targets of `cubecl-llvm`. Two goals:

1. Every optimization one GPU target has that the other can use, both have, and the code
   that does it exists once.
2. NVPTX is at least as fast as the CUDA C++ backend (NVRTC). What closes that gap is
   generalized to AMDGPU, which already beats the HIP C++ backend.

The process for 2: compare NVPTX against CUDA C++ (PTX diff plus timings), add the missing
optimization to the shared lowering where possible, and carry it to AMDGPU.

Status markers: `[ ]` todo, `[x]` done, `[~]` in progress.

## Phase 1 — shared lowering, both targets, low risk

Changes in `shared/`, so both GPU targets get them at once. Each one is checked against the
generated PTX on NVIDIA; AMD is checked once, at the end of the phase.

- [x] **In-bounds GEPs.** `memory.index` lowers to a GEP without no-wrap flags. Clang emits
      `getelementptr inbounds` for every `ptr[i]`; without it LLVM cannot reassociate the
      address arithmetic or fold constant offsets into the addressing mode. The index is a
      zero-extended unsigned integer and checked modes clamp it first, so the GEP is
      `inbounds nuw`.
- [x] **NaN-ignoring float min/max.** `FMin`/`FMax` lower to `llvm.minimum`/`llvm.maximum`
      (NaN-propagating). The C++ backends emit `min`/`max` (`fminf`/`fmaxf`), the `minnum`/
      `maxnum` semantics, and SPIR-V's `FMin`/`FMax` leave a NaN undefined. `maximum` has
      no single instruction before sm_80 or before gfx12, so it expands to a max, a NaN test
      and a select in every ReLU, softmax and max-reduce.
- [x] **Device-scoped atomics.** Every atomic is `syncscope` system. CUDA's `atomicAdd` is
      device scope and HIP's is agent scope; system scope is for host-coherent memory, which
      CubeCL buffers are not.
- [x] **Transcendental polyfills on the CPU only.** Vector `exp`, `log`, `sin`, `cos` and
      `tanh` take a polynomial polyfill on every target, while scalars reach libdevice, OCML or
      the hardware (`ex2.approx`, `v_exp_f32`). On a GPU a vector is scalarized anyway, so the
      polyfill is slower and disagrees with the scalar result of the same operation.
- [x] **Native float atomic add.** LLVM expands `atomicrmw fadd` to a CAS loop on NVPTX
      unless the function flushes denormals (global `atom.add.f32` flushes them), and on
      RDNA3 and CDNA2 unless the atomic is known not to touch fine-grained or remote memory.
      NVRTC's `atomicAdd(float*)` is the native instruction. NVPTX sets
      `-nvptx-allow-ftz-atomics` once per process; AMDGPU tags every `atomicrmw` with
      `amdgpu.no.fine.grained.memory`, `amdgpu.no.remote.memory` and, for `fadd`,
      `amdgpu.ignore.denormal.mode`.

Found along the way: the CPU `exp` polyfill relied on `maximum` carrying a NaN through its
clamp, and now restores the NaN explicitly.

## Phase 2 — AMDGPU catches up with NVPTX

- [ ] **Annotate buffer parameters on AMDGPU.** Move `annotate_buffer_params` and
      `reads_atomically` from `nvptx/codegen.rs` into `shared/` and call them from both
      targets. Without `noalias` the AMDGPU backend cannot prove a uniform load is not
      clobbered, so shape and stride reads stay vector loads instead of `s_load`.
- [ ] **32-bit indices on AMDGPU.** `index_width` follows the address type on NVPTX and is
      fixed at 64 on AMDGPU; 64-bit integer arithmetic costs two or more VALU instructions.
- [ ] **`readlane` for a constant-lane broadcast** instead of `ds_bpermute`.
- [ ] **Metadata in the constant address space** (4), so every metadata read is a scalar load.
- [ ] **Grid constants on AMDGPU**: scalars and static metadata in the kernarg segment, with
      the HIP launcher packing the same layout the CUDA one does.
- [ ] **DPP / `ds_swizzle` / `permlanex16`** for shuffles with constant masks (plane
      reductions).

## Phase 3 — one GPU codegen driver

- [ ] One driver for `parse_ir`, `run_passes`, emission and target-machine creation,
      parameterized by the target (triple, attributes, emitted file kind). Today these exist
      in `nvptx/codegen.rs`, `amdgpu/codegen.rs` and `cpu/jit/engine.rs`.
- [ ] Finalize on the in-memory module: drop the two print/parse round trips per kernel.
- [ ] One builtin derivation (`derive_positions` and the pass body) for both targets.
- [ ] One word-splitting shuffle helper for both `plane.rs` files.
- [ ] Cache libdevice like the ROCm device libraries; cache parsed bitcode for both.

## Phase 4 — parity and safety

- [ ] NVPTX plane operations under divergence: `llvm.nvvm.activemask` as the member mask,
      then advertise `Plane::NonUniformControlFlow`.
- [ ] A HIP `restrict_to_llvm_backend`: no matrix features on CDNA (MFMA is not lowered), no
      i8/fp8 WMMA, and `bf16` only once the LLVM lowering has a type for it.
- [ ] Exact launch bounds: `nvvm.reqntid` and `reqd_work_group_size` /
      `uniform-work-group-size` from the compile-time cube dimensions.
- [ ] Remove the leftover `println!` in `scalar_alignment`; document NVPTX in the README.

## Measuring

NVIDIA measurements run on `cuda:3` (sm_60, Pascal). Each phase records the cubek
matmul, reduce and softmax timings for the CUDA C++ backend and for NVPTX before and after.
AMD runs happen once per phase, when the phase is finished.
