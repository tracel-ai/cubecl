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

- [x] **Annotate buffer parameters on AMDGPU.** Move `annotate_buffer_params` and
      `reads_atomically` from `nvptx/codegen.rs` into `shared/` and call them from both
      targets. Without `noalias` the AMDGPU backend cannot prove a uniform load is not
      clobbered, so shape and stride reads stay vector loads instead of `s_load`.
- [x] **32-bit indices on AMDGPU.** `index_width` follows the address type on NVPTX and is
      fixed at 64 on AMDGPU; 64-bit integer arithmetic costs two or more vector ALU instructions.
- [x] ~~**`readlane` for a constant-lane broadcast**~~ — not needed: the backend already
      rewrites a `ds_bpermute` from a constant lane into `v_readlane`, and a small constant
      XOR into DPP. `a_constant_lane_moves_without_lds` guards it.
- [x] ~~**Metadata in the constant address space**~~ — not needed: with the parameters
      annotated, the metadata reads are already `s_load` (the offline `scale` kernel's length).
- [ ] **Grid constants on AMDGPU**: scalars and static metadata in the kernarg segment, with
      the HIP launcher packing the same layout the CUDA one does.
- [x] ~~**DPP / `permlanex16` for the XOR masks**~~ — the backend already lowers a plane
      sum's butterfly to DPP, `ds_swizzle` and `v_permlanex16`; the one `ds_bpermute` left is
      the XOR-32 step on wave64, which no swizzle reaches.
- [ ] **Scalar base + 32-bit offset addressing.** A `U32` kernel still forms each address with
      a 64-bit shift and add per lane; `global_load v, v_off, s[base]` needs the byte offset
      known to fit in 32 bits.

Checked offline: `amdgpu/offline_tests.rs` compiles `#[cube]` kernels for AMDGPU without a
device and asserts on the assembly.

## Phase 3 — one GPU codegen driver

- [x] One driver for `parse_ir`, `run_passes`, emission and target-machine creation,
      parameterized by the target (triple, attributes, emitted file kind). Today these exist
      in `nvptx/codegen.rs`, `amdgpu/codegen.rs` and `cpu/jit/engine.rs`.
- [x] Finalize on the in-memory module: one of the two print/parse round trips per kernel is
      gone; the other is pliron's (see below).
- [x] One builtin derivation (`derive_positions` and the pass body) for both targets.
- [x] One word-splitting shuffle helper for both `plane.rs` files.
- [x] Cache libdevice like the ROCm device libraries. Each kernel still parses the bitcode it
      links; caching the parsed module is left.

## Phase 4 — parity and safety

- [x] NVPTX plane operations under divergence: `llvm.nvvm.activemask` as the member mask,
      then advertise `Plane::NonUniformControlFlow`. `test_plane_diverged_votes` checks the
      semantics; on Pascal it passes either way (the votes only see active lanes), so
      `plane_moves_are_native_shuffles` checks the mask in the PTX.
- [x] A HIP `restrict_to_llvm_backend`: no matrix features on CDNA (MFMA is not lowered), no
      i8/fp8 WMMA, and `bf16` only once the LLVM lowering has a type for it.
- [x] Exact launch bounds: `nvvm.reqntid` and `reqd_work_group_size` /
      `uniform-work-group-size` from the compile-time cube dimensions. A 1D cube no longer
      reads the y and z ids on either target (`a_1d_cube_*` offline tests).
- [x] Remove the leftover `println!` in `scalar_alignment`; document NVPTX in the README.

## NVPTX against CUDA C++

Measured on `cuda:3` (sm_60) with the cubek benches, both backends built from this branch.
Where LLVM stands after phases 1–4, as the geometric mean of LLVM's time over NVRTC's:

| Bench | Rows | LLVM / C++ |
| ----- | ---- | ---------- |
| reduce, Cube strategies | 96 | 0.93 |
| reduce, Plane strategies | 96 | 1.07 |
| reduce, Unit strategies | 96 | 1.55 |
| gemm, Unit strategies, f32 | 30 | ≈ 1.0, from 0.65 to 1.23 |
| gemm, Unit strategies, f16 | 30 | ≈ 1.2, from 0.65 to 5.2 |
| unary (`cos`) | 3 | 1.11 |

- [ ] **Unit reduce, 2.4× on Sum and Max.** The timed kernel is the width-1 one (a 32×8 cube,
      each unit streaming its own row). NVVM unrolls the row loop 4× and marks it
      `.pragma "nounroll"`, so ptxas leaves it at 4 loads per iteration. LLVM's runtime
      unroller unrolls 8× and marks only the epilogue, so ptxas unrolls the main loop again
      to 32 loads per iteration. Two experiments, neither shipped:
      - Runtime unrolling off (the loop left rolled for ptxas) brings Unit Sum to parity but
        costs 8.3% over all 288 reduce rows on the same GPU: TopK(2) and TopK(3) get 2.5×
        slower, the Plane and Cube strategies 8–14%.
      - Every loop marked `llvm.loop.unroll.disable` after LLVM's pipeline (so ptxas keeps
        LLVM's factor) brings Unit Max f16 from 2.45× to 0.99×, and leaves Unit Sum and
        Unit Max f32 at 2.4×.

      So the loop shape is part of it and not all of it. The next step needs hardware
      counters (L1/texture hit rate, DRAM bytes, stall reasons); `nvprof` refuses them here
      because the driver restricts profiling to administrators (`RmProfilingAdminOnly: 1`).
- [ ] **Large-k TopK** (all three strategy families) is 2–5× slower. Not yet diagnosed.
- [ ] **f16 unit matmul.** f32 gemm is at parity or ahead, but the f16 Double Unit max-tile rows
      and the col/col f16 rows are 2–5× slower. A scalar f16 multiply-add lowers to
      `fma.rn.f16`, so the difference is in the matmul kernel, not in the f16 lowering itself.
- [ ] **Scalar `cos`** is 1.11× slower: both inline libdevice's `cosf`; the difference is in
      what each optimizer makes of it (282 against 234 SASS instructions).
- [ ] **Index arithmetic without `nuw`.** Each unrolled load recomputes its address with a
      64-bit multiply-add because a 32-bit index add may wrap. NVRTC has the same limit, so
      this is a gain over it, and it needs CubeCL to say index arithmetic does not overflow.
- [ ] **The pliron module to LLVM hand-off** still prints the module and parses it again:
      pliron-llvm keeps the raw module handle private (`inner_ref`). Removing that round trip
      is a change to pliron.

## Measuring

NVIDIA measurements run on `cuda:3` (sm_60, Pascal): the cubek reduce, gemm (the Unit
strategies), unary and contiguous benches, for the CUDA C++ backend and for NVPTX before and
after. The C++ and LLVM backends are a compile-time choice (`cubecl/cuda-cpp`), so each is
its own build. AMD runs happen once per phase, when the phase is finished.

## On AMD

Everything AMDGPU on this branch is checked here only on the generated assembly (the
offline tests). What only a device can say:

1. **The offline tests** pass on the machine's LLVM and ROCm device libraries:
   `cargo test -p cubecl-llvm --features amdgpu`.
2. **Correctness.** `cargo test -p cubecl-hip --lib` on this branch and on `main`: the LLVM
   backend is the default when `cubecl-hip/cpp` is off. A test that fails here and not on
   `main` is this branch's.
3. **Speed.** In a cubek checkout, point the four `cubecl*` dependencies of the root
   `Cargo.toml` at this checkout (the "For local development" lines), then per cubecl
   checkout (this branch and `main`):

   ```bash
   cargo bench -p benchmarks --bench reduce --features cubecl/hip
   CUBEK_BENCH_STRATEGIES=simple_unit,double_unit \
   CUBEK_BENCH_PROBLEMS=square_1x6144,vecmat_2x1x4096x4096 \
     cargo bench -p benchmarks --bench gemm --features cubecl/hip
   ```

   What should move on AMD: the buffer annotation and 32-bit indices on every kernel that
   reads shapes or strides in a loop, `maxnum` on max-reduce and TopK (gfx11 and older),
   exact work-group sizes on 1D and 2D cubes, and native float atomic add on RDNA3 and
   CDNA2.
